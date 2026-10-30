"""Conservative live-state recovery helpers for three additional tasks.

These routines are intentionally narrower than replaying ``play_once``.  They
only continue objects whose grasp/support state is observable and whose pose is
inside the task workspace.  Unknown, moving, or unsupported states are rejected.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterator


MAX_SETTLED_M = 0.025
MAX_GRASP_DISTANCE_M = 0.15


def _xyz(value: Any) -> tuple[float, float, float] | None:
    try:
        raw = getattr(value, "p", value)
        point = tuple(float(v) for v in raw[:3])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    return point if len(point) == 3 and all(math.isfinite(v) for v in point) else None


def _actor_xyz(actor: Any) -> tuple[float, float, float] | None:
    try:
        return _xyz(actor.get_pose().p)
    except (AttributeError, TypeError, ValueError):
        return None


def _fp_xyz(actor: Any, point_id: int) -> tuple[float, float, float] | None:
    try:
        return _xyz(actor.get_functional_point(point_id, "pose").p)
    except (AttributeError, TypeError, ValueError):
        try:
            return _xyz(actor.get_functional_point(point_id))
        except (AttributeError, TypeError, ValueError):
            return None


def _actor_gripper_contact(task_env: Any, actor: Any, side: str) -> bool:
    names = set(getattr(task_env.robot, f"{side}_fix_gripper_name", []) or [])
    for item in getattr(task_env.robot, f"{side}_gripper", []) or []:
        try:
            names.add(item[0].child_link.get_name())
        except (AttributeError, IndexError, TypeError):
            continue
    target = getattr(actor, "actor", actor)
    try:
        target_id = target.get_id()
    except AttributeError:
        target_id = None
    for contact in task_env.scene.get_contacts():
        try:
            first, second = (body.entity for body in contact.bodies)
            first_match = first is target
            second_match = second is target
            if target_id is not None:
                first_match = first_match or getattr(first, "get_id", lambda: None)() == target_id
                second_match = second_match or getattr(second, "get_id", lambda: None)() == target_id
            if (first_match and getattr(second, "name", None) in names
                    or second_match and getattr(first, "name", None) in names):
                return True
        except (AttributeError, IndexError, TypeError):
            continue
    return False


def _contact(task_env: Any, actor: Any, other: str) -> bool:
    try:
        return bool(task_env.check_actors_contact(actor.get_name(), other))
    except (AttributeError, TypeError, ValueError):
        return False


def _gripper_state(task_env: Any, actor: Any) -> dict[str, dict[str, Any]]:
    position = _actor_xyz(actor)
    if position is None:
        return {}
    result = {}
    for side in ("left", "right"):
        try:
            ee = _xyz(task_env.get_arm_pose(side)[:3])
            closed = bool(getattr(task_env, f"is_{side}_gripper_close")())
            opened = bool(getattr(task_env, f"is_{side}_gripper_open")())
        except (AttributeError, TypeError, ValueError):
            return {}
        if ee is None:
            return {}
        result[side] = {
            "closed": closed,
            "open": opened,
            "contact": _actor_gripper_contact(task_env, actor, side),
            "distance_m": math.dist(position, ee),
        }
    return result


def _held_side(signals: dict[str, dict[str, Any]]) -> tuple[str | None, str | None]:
    if set(signals) != {"left", "right"}:
        return None, "missing_gripper_state"
    held = []
    for side in ("left", "right"):
        signal = signals[side]
        if not isinstance(signal.get("closed"), bool) or not isinstance(signal.get("contact"), bool):
            return None, "invalid_gripper_state"
        try:
            distance = float(signal["distance_m"])
        except (KeyError, TypeError, ValueError):
            return None, "missing_grasp_distance"
        if not math.isfinite(distance):
            return None, "invalid_grasp_distance"
        if signal["closed"] and signal["contact"]:
            held.append(side)
        elif signal["closed"] and distance <= MAX_GRASP_DISTANCE_M:
            return None, f"{side}_near_object_without_contact"
    if len(held) > 1:
        return None, "ambiguous_two_arm_grasp"
    return (held[0] if held else None), None


def _stable(current: dict[str, Any], previous: dict[str, Any] | None,
            key: str, index: int | None = None) -> tuple[bool, float | None]:
    if previous is None:
        return False, None
    try:
        now = current[key] if index is None else current[key][index]
        before = previous[key] if index is None else previous[key][index]
        displacement = math.dist(_xyz(now), _xyz(before))
    except (KeyError, IndexError, TypeError, ValueError):
        return False, None
    return displacement <= MAX_SETTLED_M, displacement


def _run_stage(task_env: Any, result: dict[str, Any], stage_id: int,
               label: str, build_motion: Callable[[], Any]) -> dict[str, Any]:
    task_env.current_stage_id = int(stage_id)
    task_env.current_control_source = "hil"
    task_env._hg_dagger_recovery_substage = label
    event = {"phase": "stage", "stage_id": int(stage_id), "substage": label}
    result.setdefault("executed_stage_ids", []).append(int(stage_id))
    try:
        task_env.plan_success = True
        motion = build_motion()
        if task_env.plan_success:
            if (isinstance(motion, tuple) and len(motion) == 2
                    and isinstance(motion[0], tuple)):
                task_env.move(*motion)
            else:
                task_env.move(motion)
        event["plan_success"] = bool(task_env.plan_success)
        if not event["plan_success"]:
            metadata = getattr(task_env, "episode_metadata", {})
            detail = metadata.get("expert_stage_error") if isinstance(metadata, dict) else None
            if detail:
                event["error"] = detail
    except Exception as exc:
        task_env.plan_success = False
        event["plan_success"] = False
        event["error"] = repr(exc)
    event["executed_stage_ids"] = list(result["executed_stage_ids"])
    return event


# ---------------------------------------------------------------------------
# pick_dual_bottles: the benchmark expects each bottle to remain held at its
# corresponding elevated target; putting either bottle down is not equivalent.

DUAL_TARGETS = {
    "bottle1": ("left", (-0.06, -0.105, 1.0)),
    "bottle2": ("right", (0.06, -0.105, 1.0)),
}


def dual_bottles_snapshot(task_env: Any, policy_steps: int) -> dict[str, Any]:
    bottles = {}
    for name in DUAL_TARGETS:
        actor = getattr(task_env, name)
        pose = _actor_xyz(actor)
        point = _fp_xyz(actor, 0)
        if pose is None or point is None:
            raise ValueError(f"missing or non-finite {name} pose")
        bottles[name] = {
            "pose": list(pose), "functional_point": list(point),
            "table_contact": _contact(task_env, actor, "table"),
            "grippers": _gripper_state(task_env, actor),
        }
    target_ok = {}
    for name, (expected_arm, target) in DUAL_TARGETS.items():
        state = bottles[name]
        point = state["functional_point"]
        held, error = _held_side(state["grippers"])
        target_ok[name] = bool(
            error is None and held == expected_arm
            and math.dist(point[:2], target[:2]) < 0.10 and point[2] > 0.89
        )
    return {
        "policy_steps": int(policy_steps), "bottles": bottles,
        "target_and_hold_ok": target_ok,
        "strict_physical_quality": bool(all(target_ok.values())),
        "task_success": bool(task_env.check_success()),
    }


def classify_dual_bottles_state(current: dict[str, Any],
                                previous: dict[str, Any] | None = None) -> dict[str, Any]:
    bottles = current.get("bottles")
    if not isinstance(bottles, dict) or set(bottles) != set(DUAL_TARGETS):
        return {"ready": False, "branch": "unknown", "reason": "missing_bottle_states"}
    if current.get("strict_physical_quality") is True:
        return {"ready": False, "branch": "done", "reason": "both_bottles_held_at_targets"}
    pending = []
    held_by_side: dict[str, str] = {}
    for name, state in bottles.items():
        if not isinstance(state, dict):
            return {"ready": False, "branch": "unknown", "reason": f"invalid_{name}_state"}
        held, error = _held_side(state.get("grippers") or {})
        if error:
            return {"ready": False, "branch": "unknown", "reason": f"{name}_{error}"}
        if held is not None:
            if held in held_by_side:
                return {"ready": False, "branch": "unknown",
                        "reason": f"{held}_appears_to_hold_multiple_bottles"}
            held_by_side[held] = name

    for name, (expected_arm, target) in DUAL_TARGETS.items():
        state = bottles[name]
        if not isinstance(state, dict):
            return {"ready": False, "branch": "unknown", "reason": f"invalid_{name}_state"}
        if (current.get("target_and_hold_ok") or {}).get(name) is True:
            continue
        point = _xyz(state.get("pose"))
        if point is None or not (-0.50 <= point[0] <= 0.50 and
                                 -0.35 <= point[1] <= 0.45 and 0.70 <= point[2] <= 1.35):
            return {"ready": False, "branch": "unrecoverable", "reason": f"{name}_outside_workspace"}
        held, error = _held_side(state.get("grippers") or {})
        if error:
            return {"ready": False, "branch": "unknown", "reason": f"{name}_{error}"}
        if held is not None:
            if held != expected_arm:
                return {"ready": False, "branch": "unknown", "reason": f"{name}_held_by_wrong_arm"}
            pending.append({"name": name, "arm": held, "held": True})
            continue
        if state.get("table_contact") is not True:
            return {"ready": False, "branch": "unknown", "reason": f"{name}_not_table_supported"}
        stable, displacement = _stable(
            {"pose": state.get("pose")},
            {"pose": (previous or {}).get("bottles", {}).get(name, {}).get("pose")},
            "pose",
        )
        if not stable:
            return {"ready": False, "branch": "wait_for_settle",
                    "reason": f"{name}_moving_or_no_previous_pose",
                    "displacement_m": displacement}
        pending.append({"name": name, "arm": expected_arm, "held": False})
    if not pending:
        return {"ready": False, "branch": "unknown", "reason": "physical_gate_not_repairable"}
    open_grippers = []
    for side in ("left", "right"):
        signals = [state["grippers"].get(side) for state in bottles.values()]
        if any(not isinstance(signal, dict) for signal in signals):
            return {"ready": False, "branch": "unknown",
                    "reason": f"{side}_gripper_state_missing"}
        closed = [signal.get("closed") for signal in signals]
        opened = [signal.get("open") for signal in signals]
        if any(not isinstance(value, bool) for value in closed + opened):
            return {"ready": False, "branch": "unknown",
                    "reason": f"{side}_gripper_state_invalid"}
        if len(set(closed)) != 1 or len(set(opened)) != 1:
            return {"ready": False, "branch": "unknown",
                    "reason": f"{side}_gripper_state_inconsistent"}
        if closed[0] and opened[0]:
            return {"ready": False, "branch": "unknown",
                    "reason": f"{side}_gripper_open_closed_conflict"}
        if held_by_side.get(side) is not None:
            continue
        if closed[0]:
            for signal in signals:
                distance = signal.get("distance_m")
                if (signal.get("contact") is not False or not isinstance(distance, (int, float))
                        or not math.isfinite(float(distance))
                        or float(distance) <= MAX_GRASP_DISTANCE_M):
                    return {"ready": False, "branch": "unknown",
                            "reason": f"{side}_closed_gripper_not_clear_of_bottles"}
            open_grippers.append(side)
        elif not opened[0]:
            return {"ready": False, "branch": "unknown",
                    "reason": f"{side}_gripper_not_in_stable_open_or_closed_state"}

    for item in pending:
        if item["held"]:
            continue
        side = item["arm"]
        if held_by_side.get(side) is not None:
            return {"ready": False, "branch": "unknown",
                    "reason": f"{side}_gripper_occupied_by_{held_by_side[side]}"}
        side_signal = bottles[item["name"]]["grippers"][side]
        if side_signal.get("closed") is True and side not in open_grippers:
            return {"ready": False, "branch": "unknown",
                    "reason": f"{item['name']}_required_gripper_not_safely_openable"}
    return {"ready": True, "branch": "recover_bottles", "stage_id": 1,
            "pending": pending, "open_grippers": open_grippers,
            "reposition_arms": [side for side in ("left", "right")
                                if side not in held_by_side]}


def iter_dual_bottles_expert_suffix(task_env: Any,
                                    decision: dict[str, Any]) -> Iterator[dict[str, Any]]:
    result = {"decision": decision, "executed_stage_ids": [], "success": False}
    if decision.get("ready") is not True or decision.get("branch") != "recover_bottles":
        result["reason"] = decision.get("reason", "unsupported_branch")
        yield {"phase": "done", "result": result}
        return
    from envs.utils import ArmTag

    open_grippers = list(decision.get("open_grippers", []))
    if open_grippers:
        motions = tuple(task_env.open_gripper(ArmTag(side)) for side in open_grippers)
        event = _run_stage(
            task_env, result, 1, "open_verified_free_grippers",
            lambda: motions[0] if len(motions) == 1 else motions,
        )
        yield event
        if not event["plan_success"]:
            result["reason"] = "free_gripper_open_plan_failed"
            yield {"phase": "done", "result": dict(result)}
            return
        opened_snapshot = dual_bottles_snapshot(task_env, 0)
        for side in open_grippers:
            for state in opened_snapshot["bottles"].values():
                signal = state["grippers"][side]
                if signal["open"] is not True or signal["contact"] is True:
                    result["reason"] = f"{side}_gripper_did_not_open_safely"
                    result["post_open_state"] = opened_snapshot
                    yield {"phase": "done", "result": dict(result)}
                    return

    reposition_arms = list(decision.get("reposition_arms", []))
    if reposition_arms:
        motions = tuple(task_env.back_to_origin(ArmTag(side)) for side in reposition_arms)
        event = _run_stage(
            task_env, result, 2, "retreat_free_arms_to_home",
            lambda: motions[0] if len(motions) == 1 else motions,
        )
        yield event
        if not event["plan_success"]:
            result["reason"] = "free_arm_retreat_plan_failed"
            yield {"phase": "done", "result": dict(result)}
            return

    for item in decision.get("pending", []):
        name = item["name"]
        arm_name = item["arm"]
        actor = getattr(task_env, name)
        arm = ArmTag(arm_name)
        if not item["held"]:
            event = _run_stage(task_env, result, 3, f"{name}_grasp",
                               lambda actor=actor, arm=arm: task_env.grasp_actor(
                                   actor, arm_tag=arm, pre_grasp_dis=0.08))
            yield event
            if not event["plan_success"]:
                result["reason"] = f"{name}_grasp_plan_failed"
                yield {"phase": "done", "result": dict(result)}
                return
            state = dual_bottles_snapshot(task_env, 0)["bottles"][name]
            held, _ = _held_side(state["grippers"])
            if held != arm_name:
                result["reason"] = f"{name}_not_held_after_grasp"
                result["post_grasp_state"] = state
                yield {"phase": "done", "result": dict(result)}
                return
        event = _run_stage(task_env, result, 4, f"{name}_lift",
                           lambda arm=arm: task_env.move_by_displacement(arm, z=0.1))
        yield event
        if not event["plan_success"]:
            result["reason"] = f"{name}_lift_plan_failed"
            yield {"phase": "done", "result": dict(result)}
            return
        state = dual_bottles_snapshot(task_env, 0)["bottles"][name]
        held, _ = _held_side(state["grippers"])
        if held != arm_name:
            result["reason"] = f"{name}_lost_after_lift"
            yield {"phase": "done", "result": dict(result)}
            return
        target = list(task_env.left_target_pose if name == "bottle1"
                      else task_env.right_target_pose)
        event = _run_stage(task_env, result, 5, f"{name}_move_to_target",
                           lambda actor=actor, arm=arm, target=target: task_env.place_actor(
                               actor, target_pose=target, arm_tag=arm,
                               functional_point_id=0, pre_dis=0.0, dis=0.0,
                               is_open=False))
        yield event
        if not event["plan_success"]:
            result["reason"] = f"{name}_target_plan_failed"
            yield {"phase": "done", "result": dict(result)}
            return
    final = dual_bottles_snapshot(task_env, 0)
    result["final_snapshot"] = final
    result["official_task_success"] = bool(task_env.check_success())
    result["success"] = bool(result["official_task_success"]
                              and final["strict_physical_quality"])
    if not result["success"]:
        result["reason"] = "original_check_or_bottle_hold_gate_failed"
    yield {"phase": "done", "result": result}


# ---------------------------------------------------------------------------
# put_bottles_dustbin: three independent objects can be completed in any
# order. Right-side bottles follow the task's native right-to-left handoff.

DUSTBIN_XY_TOL = (0.221, 0.325)


def _in_dustbin(position: Any) -> bool:
    point = _xyz(position)
    return bool(point is not None
                and abs(point[0] + 0.45) < DUSTBIN_XY_TOL[0]
                and abs(point[1]) < DUSTBIN_XY_TOL[1]
                and 0.2 < point[2] < 0.7)


def dustbin_snapshot(task_env: Any, policy_steps: int) -> dict[str, Any]:
    entries = []
    for index, actor in enumerate(task_env.bottles):
        position = _actor_xyz(actor)
        if position is None:
            raise ValueError(f"missing or non-finite bottle {index} pose")
        entries.append({
            "pose": list(position), "in_bin": _in_dustbin(position),
            "table_contact": _contact(task_env, actor, "table"),
            "dustbin_contact": _contact(task_env, actor, task_env.dustbin.get_name()),
            "grippers": _gripper_state(task_env, actor),
        })
    return {
        "policy_steps": int(policy_steps), "bottles": entries,
        "bottles_in_bin": [entry["in_bin"] for entry in entries],
        "strict_physical_quality": bool(
            all(entry["in_bin"] for entry in entries)
            and task_env.is_left_gripper_open() and task_env.is_right_gripper_open()
        ),
        "task_success": bool(task_env.check_success()),
    }


def classify_dustbin_state(current: dict[str, Any],
                           previous: dict[str, Any] | None = None) -> dict[str, Any]:
    bottles = current.get("bottles")
    if not isinstance(bottles, list) or len(bottles) != 3:
        return {"ready": False, "branch": "unknown", "reason": "expected_three_bottles"}
    if current.get("strict_physical_quality") is True:
        return {"ready": False, "branch": "done", "reason": "all_bottles_in_bin"}
    pending = []
    held_arms: dict[str, int] = {}
    previous_bottles = (previous or {}).get("bottles", [])
    for index, item in enumerate(bottles):
        if not isinstance(item, dict):
            return {"ready": False, "branch": "unknown", "reason": f"bottle_{index}_invalid_state"}
        held, error = _held_side(item.get("grippers") or {})
        if error:
            return {"ready": False, "branch": "unknown", "reason": f"bottle_{index}_{error}"}
        if item.get("in_bin") is True:
            if held is not None:
                return {"ready": False, "branch": "unknown",
                        "reason": f"bottle_{index}_inside_bin_but_still_held"}
            continue
        point = _xyz(item.get("pose"))
        if point is None or not (-0.80 <= point[0] <= 0.60 and
                                 -0.50 <= point[1] <= 0.55 and 0.68 <= point[2] <= 1.55):
            return {"ready": False, "branch": "unrecoverable", "reason": f"bottle_{index}_outside_workspace"}
        if held:
            held_arms[held] = held_arms.get(held, 0) + 1
            pending.append({"index": index, "arm": held, "held": True})
            continue
        if item.get("table_contact") is not True and item.get("dustbin_contact") is not True:
            return {"ready": False, "branch": "unknown", "reason": f"bottle_{index}_unsupported"}
        prev = previous_bottles[index] if index < len(previous_bottles) else None
        stable, displacement = _stable({"pose": item.get("pose")},
                                       {"pose": (prev or {}).get("pose")}, "pose")
        if not stable:
            return {"ready": False, "branch": "wait_for_settle",
                    "reason": f"bottle_{index}_moving_or_no_previous_pose",
                    "displacement_m": displacement}
        pending.append({"index": index,
                        "arm": "left" if point[0] < 0 else "right", "held": False})
    if any(count > 1 for count in held_arms.values()):
        return {"ready": False, "branch": "unknown",
                "reason": "one_gripper_appears_to_hold_multiple_bottles"}
    if not pending:
        return {"ready": False, "branch": "unknown", "reason": "remaining_quality_gate_not_repairable"}

    # An empty gripper may remain closed after a failed grasp. Opening it is
    # safe only when it has no contact and is well clear of every bottle.
    open_grippers = []
    for side in ("left", "right"):
        signals = [(item.get("grippers") or {}).get(side) for item in bottles]
        if any(not isinstance(signal, dict) for signal in signals):
            return {"ready": False, "branch": "unknown",
                    "reason": f"{side}_gripper_state_missing"}
        closed = [signal.get("closed") for signal in signals]
        opened = [signal.get("open") for signal in signals]
        if any(not isinstance(value, bool) for value in closed + opened):
            return {"ready": False, "branch": "unknown",
                    "reason": f"{side}_gripper_state_invalid"}
        if len(set(closed)) != 1 or len(set(opened)) != 1:
            return {"ready": False, "branch": "unknown",
                    "reason": f"{side}_gripper_state_inconsistent"}
        if held_arms.get(side):
            continue
        if closed[0]:
            for signal in signals:
                distance = signal.get("distance_m")
                if (signal.get("contact") is not False
                        or not isinstance(distance, (int, float))
                        or not math.isfinite(float(distance))
                        or float(distance) <= MAX_GRASP_DISTANCE_M):
                    return {"ready": False, "branch": "unknown",
                            "reason": f"{side}_closed_gripper_not_clear_of_bottles"}
            open_grippers.append(side)
        elif not opened[0]:
            return {"ready": False, "branch": "unknown",
                    "reason": f"{side}_gripper_not_stably_open_or_closed"}

    # Finish left-side work first so the left hand is free for the task's
    # native right-to-left transfer of any right-side bottle.
    pending.sort(key=lambda item: (item["arm"] != "left", not item["held"], item["index"]))
    return {"ready": True, "branch": "recover_remaining_bottles", "stage_id": 1,
            "pending": pending, "open_grippers": open_grippers}


def iter_dustbin_expert_suffix(task_env: Any,
                               decision: dict[str, Any]) -> Iterator[dict[str, Any]]:
    result = {"decision": decision, "executed_stage_ids": [], "success": False}
    if decision.get("ready") is not True or decision.get("branch") != "recover_remaining_bottles":
        result["reason"] = decision.get("reason", "unsupported_branch")
        yield {"phase": "done", "result": result}
        return
    from envs.utils import Action, ArmTag

    left_drop = Action("left", "move", [-0.35, -0.1, 0.93, 0.65, -0.25, 0.25, 0.65])

    open_grippers = list(decision.get("open_grippers", []))
    if open_grippers:
        motions = tuple(task_env.open_gripper(ArmTag(side)) for side in open_grippers)
        event = _run_stage(
            task_env, result, 1, "open_verified_free_grippers",
            lambda: motions[0] if len(motions) == 1 else motions,
        )
        yield event
        if not event["plan_success"]:
            result["reason"] = "free_gripper_open_plan_failed"
            yield {"phase": "done", "result": dict(result)}
            return
        opened_snapshot = dustbin_snapshot(task_env, 0)
        for side in open_grippers:
            for bottle in opened_snapshot["bottles"]:
                signal = bottle["grippers"][side]
                if signal["open"] is not True or signal["contact"] is True:
                    result["reason"] = f"{side}_gripper_did_not_open_safely"
                    result["post_open_state"] = opened_snapshot
                    yield {"phase": "done", "result": dict(result)}
                    return

    def still_held(actor: Any, side: str) -> bool:
        signals = _gripper_state(task_env, actor)
        return bool(signals.get(side, {}).get("closed") is True and
                    (signals.get(side, {}).get("contact") is True or
                     signals.get(side, {}).get("distance_m", math.inf) <= MAX_GRASP_DISTANCE_M))

    for item in decision["pending"]:
        index, source = int(item["index"]), item["arm"]
        actor = task_env.bottles[index]
        if source == "right":
            if not item["held"]:
                event = _run_stage(task_env, result, 1, f"bottle_{index}_right_grasp",
                                   lambda actor=actor: (
                                       task_env.grasp_actor(actor, arm_tag=ArmTag("right"),
                                                            pre_grasp_dis=0.1),
                                       task_env.back_to_origin(arm_tag=ArmTag("left")),
                                   ))
                yield event
                if not event["plan_success"] or not still_held(actor, "right"):
                    result["reason"] = f"bottle_{index}_right_grasp_failed"
                    yield {"phase": "done", "result": dict(result)}
                    return
            else:
                event = _run_stage(task_env, result, 1, f"bottle_{index}_clear_left_arm",
                                   lambda: task_env.back_to_origin(arm_tag=ArmTag("left")))
                yield event
                if not event["plan_success"]:
                    result["reason"] = f"bottle_{index}_left_arm_clear_failed"
                    yield {"phase": "done", "result": dict(result)}
                    return
            event = _run_stage(task_env, result, 2, f"bottle_{index}_right_lift",
                               lambda: task_env.move_by_displacement(ArmTag("right"), z=0.1))
            yield event
            if not event["plan_success"] or not still_held(actor, "right"):
                result["reason"] = f"bottle_{index}_right_lift_failed"
                yield {"phase": "done", "result": dict(result)}
                return
            event = _run_stage(task_env, result, 3, f"bottle_{index}_right_to_transfer",
                               lambda actor=actor: task_env.place_actor(
                                   actor, target_pose=task_env.right_middle_pose,
                                   arm_tag=ArmTag("right"), functional_point_id=0,
                                   pre_dis=0.0, dis=0.0, is_open=False,
                                   constrain="align"))
            yield event
            if not event["plan_success"] or not still_held(actor, "right"):
                result["reason"] = f"bottle_{index}_right_transfer_failed"
                yield {"phase": "done", "result": dict(result)}
                return
            def adjusted_left_grasp():
                motion = task_env.grasp_actor(actor, arm_tag=ArmTag("left"),
                                              pre_grasp_dis=0.1)
                try:
                    motion[1][0].target_pose[2] -= 0.06
                    motion[1][1].target_pose[2] -= 0.06
                except (AttributeError, IndexError, TypeError):
                    task_env.plan_success = False
                return motion
            event = _run_stage(task_env, result, 4, f"bottle_{index}_left_grasp_transfer",
                               adjusted_left_grasp)
            yield event
            if not event["plan_success"] or not still_held(actor, "left"):
                result["reason"] = f"bottle_{index}_left_transfer_grasp_failed"
                yield {"phase": "done", "result": dict(result)}
                return
            event = _run_stage(task_env, result, 4, f"bottle_{index}_release_right",
                               lambda: task_env.open_gripper(ArmTag("right")))
            yield event
            if not event["plan_success"]:
                result["reason"] = f"bottle_{index}_right_release_failed"
                yield {"phase": "done", "result": dict(result)}
                return
        else:
            if not item["held"]:
                event = _run_stage(task_env, result, 1, f"bottle_{index}_left_grasp",
                                   lambda actor=actor: task_env.grasp_actor(
                                       actor, arm_tag=ArmTag("left"), pre_grasp_dis=0.1))
                yield event
                if not event["plan_success"] or not still_held(actor, "left"):
                    result["reason"] = f"bottle_{index}_left_grasp_failed"
                    yield {"phase": "done", "result": dict(result)}
                    return
            event = _run_stage(task_env, result, 2, f"bottle_{index}_left_lift",
                               lambda: task_env.move_by_displacement(ArmTag("left"), z=0.1))
            yield event
            if not event["plan_success"] or not still_held(actor, "left"):
                result["reason"] = f"bottle_{index}_left_lift_failed"
                yield {"phase": "done", "result": dict(result)}
                return
        event = _run_stage(task_env, result, 5, f"bottle_{index}_move_over_bin",
                           lambda: (ArmTag("left"), [left_drop]))
        yield event
        if not event["plan_success"]:
            result["reason"] = f"bottle_{index}_drop_pose_plan_failed"
            yield {"phase": "done", "result": dict(result)}
            return
        event = _run_stage(task_env, result, 5, f"bottle_{index}_release_into_bin",
                           lambda: task_env.open_gripper(ArmTag("left")))
        yield event
        if not event["plan_success"]:
            result["reason"] = f"bottle_{index}_release_plan_failed"
            yield {"phase": "done", "result": dict(result)}
            return
        for _ in range(20):
            task_env.scene.step()
        latest = dustbin_snapshot(task_env, 0)
        result.setdefault("bottle_results", []).append({
            "index": index, "in_bin": latest["bottles_in_bin"][index],
            "pose": latest["bottles"][index]["pose"],
        })
        if not latest["bottles_in_bin"][index]:
            result["reason"] = f"bottle_{index}_not_inside_bin_after_release"
            yield {"phase": "done", "result": dict(result)}
            return
    final = dustbin_snapshot(task_env, 0)
    result["final_snapshot"] = final
    result["official_task_success"] = bool(task_env.check_success())
    result["success"] = bool(result["official_task_success"]
                              and final["strict_physical_quality"])
    if not result["success"]:
        result["reason"] = "not_all_bottles_in_bin_or_grippers_not_open"
    yield {"phase": "done", "result": result}


# ---------------------------------------------------------------------------
# stack_blocks_two: repair the base first, then the top block. Both contact
# and the original center-to-center geometry are required for the strict gate.

def _block_is_stacked(snapshot: dict[str, Any]) -> bool:
    try:
        base = _xyz(snapshot["block_p"][0])
        top = _xyz(snapshot["block_p"][1])
    except (KeyError, IndexError, TypeError):
        return False
    return bool(base is not None and top is not None
                and abs(top[0] - base[0]) < 0.025
                and abs(top[1] - base[1]) < 0.025
                and abs(top[2] - (base[2] + 0.05)) < 0.012
                and snapshot.get("block2_on_block1") is True)


def stack_blocks_two_snapshot(task_env: Any, policy_steps: int) -> dict[str, Any]:
    actors = [task_env.block1, task_env.block2]
    positions = [_actor_xyz(actor) for actor in actors]
    if any(point is None for point in positions):
        raise ValueError("missing or non-finite block pose")
    grippers = [_gripper_state(task_env, actor) for actor in actors]
    target = _xyz(task_env.block1_target_pose[:3])
    if target is None:
        raise ValueError("missing or non-finite base-block target")
    result = {
        "policy_steps": int(policy_steps), "block_p": [list(p) for p in positions],
        "block_grippers": grippers, "base_target": list(target),
        "table_contact": [_contact(task_env, actor, "table") for actor in actors],
        "block2_on_block1": _contact(task_env, task_env.block2, task_env.block1.get_name()),
        "grippers_open": {side: bool(getattr(task_env, f"is_{side}_gripper_open")())
                           for side in ("left", "right")},
        "task_success": bool(task_env.check_success()),
    }
    result["base_at_target"] = bool(
        result["table_contact"][0]
        and math.dist(positions[0][:2], target[:2]) <= 0.04
        and target[2] - 0.01 <= positions[0][2] <= target[2] + 0.08
    )
    result["strict_physical_quality"] = bool(
        result["task_success"] and result["base_at_target"]
        and result["block2_on_block1"]
        and all(result["grippers_open"].values())
    )
    return result


def classify_stack_blocks_two_state(current: dict[str, Any],
                                    previous: dict[str, Any] | None = None) -> dict[str, Any]:
    positions = current.get("block_p")
    signals = current.get("block_grippers")
    if not isinstance(positions, list) or len(positions) != 2 or not isinstance(signals, list) or len(signals) != 2:
        return {"ready": False, "branch": "unknown", "reason": "missing_block_states"}
    if current.get("strict_physical_quality") is True:
        return {"ready": False, "branch": "done", "reason": "strict_stack_complete"}
    target = _xyz(current.get("base_target"))
    points = [_xyz(p) for p in positions]
    if target is None or any(p is None for p in points):
        return {"ready": False, "branch": "unknown", "reason": "invalid_block_or_target_pose"}
    pending = []
    base_is_ready = bool(current.get("base_at_target") is True)
    held_names = []
    for index, name in enumerate(("block1", "block2")):
        point = points[index]
        if not (-0.55 <= point[0] <= 0.55 and -0.45 <= point[1] <= 0.45
                and 0.68 <= point[2] <= 1.35):
            return {"ready": False, "branch": "unrecoverable", "reason": f"{name}_outside_workspace"}
        held, error = _held_side(signals[index] or {})
        if error:
            return {"ready": False, "branch": "unknown", "reason": f"{name}_{error}"}
        # Do not disturb a verified, correctly placed subgoal merely because
        # its gripper is still holding it. The held object will be released
        # only after the complete stack is physically verified below.
        if index == 0 and current.get("base_at_target") is True:
            continue
        if index == 1 and _block_is_stacked(current):
            continue
        if held is not None:
            held_names.append(name)
            pending.append({"name": name, "index": index, "held": True, "arm": held})
            continue
        table_supported = (current.get("table_contact") or [False, False])[index] is True
        supported_on_base = index == 1 and current.get("block2_on_block1") is True
        if not (table_supported or supported_on_base):
            return {"ready": False, "branch": "unknown", "reason": f"{name}_unsupported"}
        object_signals = signals[index] or {}
        other_signals = signals[1 - index] or {}
        for side in ("left", "right"):
            signal = object_signals.get(side, {})
            if signal.get("open") is True:
                continue
            # A closed gripper may be safely ignored for this object only
            # when contact sensing positively identifies that same gripper
            # as holding the other tracked block, and it is not near this one.
            other_held, other_error = _held_side(other_signals)
            if (other_error is None and other_held == side
                    and signal.get("closed") is True
                    and signal.get("contact") is False
                    and float(signal.get("distance_m", 0.0)) > MAX_GRASP_DISTANCE_M):
                continue
            return {"ready": False, "branch": "unknown",
                    "reason": f"{name}_unheld_with_closed_gripper"}
        stable, displacement = _stable({"pose": positions[index]},
                                       {"pose": (previous or {}).get("block_p", [None, None])[index]},
                                       "pose")
        if not stable:
            return {"ready": False, "branch": "wait_for_settle",
                    "reason": f"{name}_moving_or_no_previous_pose",
                    "displacement_m": displacement}
        pending.append({"name": name, "index": index, "held": False,
                        "arm": "left" if point[0] < 0 else "right"})
    if not base_is_ready and "block2" in held_names:
        return {"ready": False, "branch": "unknown",
                "reason": "top_block_held_while_base_needs_repositioning"}
    if len(held_names) > 1:
        return {"ready": False, "branch": "unknown",
                "reason": "both_blocks_held_simultaneously"}
    if not pending:
        return {"ready": False, "branch": "unknown", "reason": "stack_gate_not_repairable"}
    return {"ready": True, "branch": "repair_stack", "stage_id": 1, "pending": pending}


def iter_stack_blocks_two_expert_suffix(task_env: Any,
                                        decision: dict[str, Any]) -> Iterator[dict[str, Any]]:
    result = {"decision": decision, "executed_stage_ids": [], "success": False}
    if decision.get("ready") is not True or decision.get("branch") != "repair_stack":
        result["reason"] = decision.get("reason", "unsupported_branch")
        yield {"phase": "done", "result": result}
        return
    from envs.utils import ArmTag

    pending = {item["name"]: item for item in decision.get("pending", [])}
    last_arm: Any = None

    def is_held(actor: Any, side: str) -> bool:
        held, _ = _held_side(_gripper_state(task_env, actor))
        return held == side

    for name in ("block1", "block2"):
        item = pending.get(name)
        if item is None:
            continue
        actor = getattr(task_env, name)
        arm = ArmTag(item["arm"])
        stage_id = 1 if name == "block1" else 2
        if not item["held"]:
            def grasp_motion(actor=actor, arm=arm):
                grasp = task_env.grasp_actor(actor, arm_tag=arm, pre_grasp_dis=0.09)
                if last_arm is not None and str(last_arm) != str(arm):
                    return (grasp, task_env.back_to_origin(arm_tag=arm.opposite))
                return grasp
            event = _run_stage(task_env, result, stage_id, f"{name}_grasp", grasp_motion)
            yield event
            if not event["plan_success"] or not is_held(actor, str(arm)):
                result["reason"] = f"{name}_grasp_failed_or_not_held"
                yield {"phase": "done", "result": dict(result)}
                return
        event = _run_stage(task_env, result, stage_id, f"{name}_lift",
                           lambda arm=arm: task_env.move_by_displacement(arm, z=0.07))
        yield event
        if not event["plan_success"] or not is_held(actor, str(arm)):
            result["reason"] = f"{name}_lift_failed_or_dropped"
            yield {"phase": "done", "result": dict(result)}
            return
        if name == "block1":
            target = list(task_env.block1_target_pose)
        else:
            # Recompute from the now-settled base; never use a stale target pose.
            target = task_env.block1.get_functional_point(1)
        event = _run_stage(task_env, result, stage_id, f"{name}_place_and_release",
                           lambda actor=actor, arm=arm, target=target: task_env.place_actor(
                               actor, target_pose=target, arm_tag=arm,
                               functional_point_id=0, pre_dis=0.05, dis=0.0,
                               pre_dis_axis="fp"))
        yield event
        if not event["plan_success"]:
            result["reason"] = f"{name}_place_plan_failed"
            yield {"phase": "done", "result": dict(result)}
            return
        last_arm = arm
        for _ in range(12):
            task_env.scene.step()
        after = stack_blocks_two_snapshot(task_env, 0)
        if name == "block1" and not after["base_at_target"]:
            result["reason"] = "base_block_not_stably_at_target"
            result["post_base_snapshot"] = after
            yield {"phase": "done", "result": dict(result)}
            return
        if name == "block2" and not _block_is_stacked(after):
            result["reason"] = "top_block_not_stably_on_base"
            result["post_top_snapshot"] = after
            yield {"phase": "done", "result": dict(result)}
            return
    after_repair = stack_blocks_two_snapshot(task_env, 0)
    if after_repair["base_at_target"] and _block_is_stacked(after_repair):
        # A grasped base can stabilize the placement while the top block is
        # being set down. Open only grippers with positively observed contact
        # to one of the two verified blocks, then re-check that the stack held.
        for side in ("left", "right"):
            signals = after_repair["block_grippers"]
            holding_verified_block = any(
                signals[index][side].get("closed") is True
                and signals[index][side].get("contact") is True
                for index in (0, 1)
            )
            if not holding_verified_block:
                continue
            event = _run_stage(task_env, result, 2, f"release_{side}_after_verified_stack",
                               lambda side=side: task_env.open_gripper(ArmTag(side)))
            yield event
            if not event["plan_success"]:
                result["reason"] = f"{side}_release_after_stack_failed"
                yield {"phase": "done", "result": dict(result)}
                return
            for _ in range(12):
                task_env.scene.step()
            after_repair = stack_blocks_two_snapshot(task_env, 0)
            if not after_repair["base_at_target"] or not _block_is_stacked(after_repair):
                result["reason"] = f"stack_shifted_after_{side}_release"
                result["post_release_snapshot"] = after_repair
                yield {"phase": "done", "result": dict(result)}
                return
    final = stack_blocks_two_snapshot(task_env, 0)
    result["final_snapshot"] = final
    result["official_task_success"] = bool(task_env.check_success())
    result["success"] = bool(result["official_task_success"]
                              and final["strict_physical_quality"])
    if not result["success"]:
        result["reason"] = "original_check_or_strict_contact_gate_failed"
    yield {"phase": "done", "result": result}
