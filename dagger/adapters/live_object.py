"""Shared safety/recording for single-object, live-state expert suffixes.

No simulator imports at module load time; no reset, teleport, play_once or
unrecorded scene.step. This is intentionally independent of policy inference.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import TaskAdapter


def entity(value):
    value = getattr(value, "actor", value)
    return getattr(value, "entity", value)


def same_entity(a, b):
    a, b = entity(a), entity(b)
    if a is b:
        return True
    aid = getattr(a, "get_id", lambda: None)()
    bid = getattr(b, "get_id", lambda: None)()
    return aid is not None and bid is not None and aid == bid


def touching(pairs, a, b):
    return any((same_entity(x, a) and same_entity(y, b)) or
               (same_entity(y, a) and same_entity(x, b)) for x, y in pairs)


def vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected finite vector[{size}]")
    return result


def quaternion(value):
    q = vector(value, 4)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError("Zero quaternion")
    return q / norm


def axis_z(q, axis):
    """World-z component of a local axis (wxyz quaternion)."""
    w, x, y, z = quaternion(q)
    return (2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y))[axis]


def angle_between(a, b):
    return float(2 * np.arccos(np.clip(abs(np.dot(quaternion(a), quaternion(b))), 0, 1)))


class LiveObjectAdapter(TaskAdapter):
    """Common contract; task subclasses supply geometry and stage semantics."""

    default_task_config = "demo_clean"
    default_frequency = 50
    supports_sampling = False
    requires_training_gate = True
    requires_complete_recovery_for_training = True
    # Conservative guard, not a substitute for checking the actual manifests.
    reserved_seed_ranges = ((0, 999, "SFT/expert data guard"),
                            (100000, 109999, "frozen policy evaluation guard"))
    actor_name: str
    allow_dual_grasp = False

    def actor(self, env):
        return getattr(env, self.actor_name)

    def support(self, env):
        raise NotImplementedError

    def geometry(self, env):
        """Return at_target and diagnostics, without requiring open grippers."""
        raise NotImplementedError

    def initialize_episode(self, env):
        if not isinstance(getattr(env, "episode_metadata", None), dict):
            env.episode_metadata = {}
        env.episode_metadata.update(task=self.name, adapter_version=1,
                                    seed=int(env.episode_seed),
                                    stage_names=dict(self.stage_labels),
                                    live_recovery_events=[])
        env._live_object_stability = None

    def validate_action_chunk(self, chunk):
        values = np.asarray(chunk, dtype=float)
        if values.shape != (50, 14) or not np.isfinite(values).all():
            raise ValueError(f"{self.name} requires finite absolute joint H50x14, got {values.shape}")

    def snapshot(self, env):
        actor = self.actor(env)
        p = vector(actor.get_pose().p, 3)
        q = quaternion(actor.get_pose().q)
        components = [c for c in entity(actor).get_components()
                      if hasattr(c, "linear_velocity") and hasattr(c, "angular_velocity")]
        if not components:
            raise ValueError("Missing rigid-body velocity")
        speed = float(np.linalg.norm(vector(components[0].linear_velocity, 3)))
        angular_speed = float(np.linalg.norm(vector(components[0].angular_velocity, 3)))
        pairs = [(c.bodies[0].entity, c.bodies[1].entity) for c in env.scene.get_contacts()]
        grips = {}
        for side in ("left", "right"):
            links = [item[0].child_link for item in getattr(env.robot, f"{side}_gripper") if item is not None]
            fixed_names = set(getattr(env.robot, f"{side}_fix_gripper_name", ()) or ())
            contact = False
            for a, b in pairs:
                other = b if same_entity(a, actor) else a if same_entity(b, actor) else None
                if other is not None and (any(same_entity(other, link) for link in links) or
                                          getattr(other, "name", None) in fixed_names):
                    contact = True
            grips[side] = {"open": bool(getattr(env, f"is_{side}_gripper_open")()),
                           "closed": bool(getattr(env, f"is_{side}_gripper_close")()),
                           "contact": contact}
        held = [s for s, g in grips.items() if g["closed"] and g["contact"]]
        on_table = touching(pairs, actor, env.table)
        on_target = touching(pairs, actor, self.support(env))
        settled = speed <= .04 and angular_speed <= .5
        proof = getattr(env, "_live_object_stability", None)
        if proof and getattr(env, "current_control_source", None) == "hil":
            settled |= bool((on_table or on_target) and np.linalg.norm(p - proof["p"]) <= .002
                            and angle_between(q, proof["q"]) <= .02 and speed <= .15 and angular_speed <= .8)
        return {"p": p.tolist(), "q": q.tolist(), "speed": speed, "angular_speed": angular_speed,
                "settled": bool(settled), "grippers": grips, "held": held,
                "on_table": on_table, "on_target": on_target,
                "table_z_bias": float(env.table_z_bias),
                "official_success": bool(env.check_success()), **self.geometry(env)}

    def rejection(self, state):
        x, y, z = state["p"]
        bias = state["table_z_bias"]
        if not (-.5 <= x <= .5 and -.35 <= y <= .45 and .70+bias <= z <= 1.35+bias):
            return "outside_recovery_workspace"
        if any(g["contact"] and not g["open"] and not g["closed"] for g in state["grippers"].values()):
            return "ambiguous_partial_grasp"
        if len(state["held"]) > 1 and not self.allow_dual_grasp:
            return "ambiguous_two_arm_grasp"
        if not state["held"]:
            if not (state["on_table"] or state["on_target"]):
                return "unheld_object_not_supported"
            if not state["settled"]:
                return "object_moving_wait_for_settle"
        return None

    def classify(self, state):
        raise NotImplementedError

    def infer_recovery_state(self, env):
        try:
            state = self.snapshot(env)
            reason = self.rejection(state)
            decision = {"recoverable": False, "branch": "reject", "stage_id": None,
                        "reason": reason, "snapshot": state}
            if reason:
                return decision
            return {**decision, "recoverable": True, **self.classify(state)}
        except (AttributeError, ValueError, TypeError, IndexError) as error:
            return {"recoverable": False, "branch": "reject", "stage_id": None,
                    "reason": f"unobservable_state: {error}"}

    def describe_recovery_state(self, state):
        snap = state.get("snapshot", {})
        return [f"任务={self.name} 建议阶段={state['stage_id']} branch={state['branch']}",
                f"可恢复={state['recoverable']} 抓持臂={snap.get('held')}",
                f"目标位置={snap.get('at_target')} 实际支撑={snap.get('on_target')}",
                f"拒绝原因={state.get('reason')}"]

    def recovery_stage_rejection(self, state, stage_id):
        if not state["recoverable"]:
            return state.get("reason") or "unrecoverable_live_state"
        expected = state["stage_id"]
        if stage_id is not None and stage_id != expected:
            return f"stage_precondition_mismatch: selected={stage_id}, live={expected}"
        return None

    def quality(self, state):
        return bool(state["official_success"] and state["at_target"] and state["on_target"]
                    and state["settled"] and not state["held"]
                    and all(g["open"] for g in state["grippers"].values()))

    def success_metrics(self, env):
        events = env.episode_metadata.get("live_recovery_events", [])
        try:
            state = self.snapshot(env)
            return {"official_task_success": state["official_success"],
                    "physical_quality_gate": self.quality(state),
                    "recovery_ok": bool(events) and all(e.get("ok") and e.get("verified") for e in events),
                    "snapshot": state}
        except (AttributeError, ValueError, TypeError, IndexError) as error:
            return {"official_task_success": bool(env.check_success()),
                    "physical_quality_gate": False, "recovery_ok": False, "error": str(error)}

    def settle(self, env):
        """Recorded 0.24s hold; reject support loss/drift/rotation, never force success."""
        env._live_object_stability = None
        initial = self.snapshot(env)
        if initial["held"] or not all(g["open"] for g in initial["grippers"].values()):
            raise ValueError("settle_requires_open_empty_grippers")
        dt = float(env.scene.get_timestep())
        if not 0 < dt <= .02:
            raise ValueError(f"Unsupported timestep: {dt}")
        count = int(np.ceil(.04 / dt))
        control = {"left_gripper": None, "right_gripper": None}
        for side in ("left", "right"):
            joints = vector(getattr(env.robot, f"get_{side}_arm_jointState")()[:-1], 6)
            if np.max(np.abs(joints)) > np.pi:
                raise ValueError("Illegal hold joints")
            control[side + "_arm"] = {"position": np.tile(joints, (count, 1)),
                                       "velocity": np.zeros((count, 6))}
        samples = []
        for _ in range(6):
            env.take_dense_action(control)
            state = self.snapshot(env)
            drift = float(np.linalg.norm(np.asarray(state["p"]) - initial["p"]))
            angle = angle_between(state["q"], initial["q"])
            samples.append({"drift_m": drift, "angle_rad": angle, "on_target": state["on_target"]})
            if (not state["on_target"] or not state["at_target"] or drift > .002 or angle > .02
                    or state["speed"] > .15 or state["angular_speed"] > .8):
                raise ValueError("placement_not_stably_supported")
        env._live_object_stability = {"p": state["p"], "q": state["q"], "samples": samples}
        if not self.quality(self.snapshot(env)):
            raise ValueError("official_success_or_physical_quality_failed")
        return samples


class RecoverySession:
    """One iterator's durable expert events; checks happen BEFORE handback boundaries."""

    def __init__(self, adapter, env, selected):
        self.adapter, self.env = adapter, env
        self.initial = adapter.infer_recovery_state(env)
        self.events = env.episode_metadata.setdefault("live_recovery_events", [])
        self.event_start = len(self.events)
        self.result: dict[str, Any] = {"initial_state": self.initial, "chosen_stage_id": selected,
                                      "executed_stage_ids": [], "plan_success": False, "success": False}
        self.reason = self.initial.get("reason") if not self.initial["recoverable"] else None
        if selected is not None and (selected not in adapter.stage_labels or selected != self.initial["stage_id"]):
            self.reason = f"stage_precondition_mismatch: selected={selected}, live={self.initial['stage_id']}"

    def stop(self, reason):
        self.events.append({"phase": "rejected", "ok": False, "verified": False, "reason": reason})
        self.result.update(reason=reason, success=False, plan_success=False)
        return {"phase": "done", "result": dict(self.result)}

    def require_held(self, side, *, only=False):
        state = self.adapter.snapshot(self.env)
        if side not in state["held"] or (only and state["held"] != [side]):
            raise ValueError(f"{side}_grasp_not_verified")

    def motion(self, stage, label, build, *, verify=None, pending=False):
        env = self.env
        env.current_control_source, env.current_stage_id = "hil", stage
        env._live_object_stability = None
        env.plan_success = True
        env.episode_metadata.pop("expert_stage_error", None)
        event = {"phase": "stage", "stage_id": stage, "substage": label,
                 "ok": False, "verified": False, "plan_success": False}
        try:
            commands = build()  # Catch construction errors as well as execution.
            if env.plan_success:
                env.move(*commands)
            legal, peak = env.planned_joints_legal()
            event.update(plan_success=bool(env.plan_success), joints_legal=bool(legal), joint_absmax=float(peak))
            if not env.plan_success or not legal:
                raise ValueError(f"planning_or_joint_limit_failure: {env.episode_metadata.get('expert_stage_error')}")
            if verify is not None:
                verify()
            event.update(ok=True, verified=not pending)
        except Exception as error:
            env.plan_success = False
            event.update(error=repr(error), plan_success=False)
        self.events.append(event)
        self.result["executed_stage_ids"].append(stage)
        return event

    def completion(self, stage):
        env = self.env
        env.current_control_source, env.current_stage_id = "hil", stage
        event = {"phase": "verification", "stage_id": stage, "substage": "settle",
                 "ok": False, "verified": False}
        try:
            event["samples"] = self.adapter.settle(env)
            legal, _ = env.planned_joints_legal()
            if not legal:
                raise ValueError("Illegal joints after hold")
            event.update(ok=True, verified=True)
            # Pending placement/retreat is accepted only after an actual hold.
            for previous in self.events[self.event_start:]:
                if previous.get("ok") and not previous.get("verified"):
                    previous["verified"] = True
            self.events.append(event)
            metrics = self.adapter.success_metrics(env)
            self.result.update(success=bool(metrics["physical_quality_gate"] and metrics["recovery_ok"]),
                               plan_success=True, final_metrics=metrics)
            if not self.result["success"]:
                self.result["reason"] = "prior_expert_error_or_quality_failure"
            return {"phase": "done", "result": dict(self.result)}
        except Exception as error:
            event["error"] = repr(error)
            self.events.append(event)
            return self.stop(f"terminal_verification_failed: {error}")


class PickPlaceAdapter(LiveObjectAdapter):
    """Four stages for QR/cup, with task-specific native primitive parameters."""

    preshape_gripper = None

    def classify(self, state):
        if state["at_target"] and state["on_target"]:
            return {"branch": "finish_supported", "stage_id": 4,
                    "arm": next(iter(state["held"]), None)}
        side = next(iter(state["held"]), "left" if state["p"][0] < 0 else "right")
        stage = (2 if state["on_table"] or state["on_target"] else 3) if state["held"] else 1
        return {"branch": "place_held" if state["held"] else "regrasp", "stage_id": stage, "arm": side}

    def grasp(self, env, arm):
        raise NotImplementedError

    def lift(self, env, arm):
        raise NotImplementedError

    def place(self, env, arm):
        raise NotImplementedError

    def recovery_stage_iterator(self, env, start_stage_id=None):
        session = RecoverySession(self, env, start_stage_id)
        if session.reason:
            yield session.stop(session.reason)
            return
        from envs.utils import ArmTag
        state = session.initial
        stage, side = state["stage_id"], state["arm"]
        arm = ArmTag(side) if side else None
        steps = []
        if stage == 1:
            # Match the task expert's grasp prefix from the live arm pose.  A
            # policy may have left an empty QR gripper closed; open only that
            # gripper, never move either arm to its episode-start pose.
            grip = state["snapshot"]["grippers"][side]
            if self.preshape_gripper is None and not grip["open"]:
                steps.append((1, "open_empty_gripper", lambda: [env.open_gripper(arm)], None, False))
            if self.preshape_gripper is not None:
                steps.append((1, "preshape_gripper", lambda: [env.close_gripper(arm, pos=self.preshape_gripper)], None, False))
            steps.append((1, "grasp", lambda: self.grasp(env, arm), lambda: session.require_held(side, only=True), False))
        if stage <= 2:
            def lift():
                session.require_held(side, only=True)
                return [self.lift(env, arm)]
            steps.append((2, "lift", lift, lambda: session.require_held(side, only=True), False))
        if stage <= 3:
            def place():
                session.require_held(side, only=True)
                return [self.place(env, arm)]
            steps.append((3, "place_and_release", place, None, True))
        else:
            # Open only once live contact and target geometry establish support.
            def release():
                current = self.snapshot(env)
                if not (current["on_target"] and current["at_target"]):
                    raise ValueError("release_requires_target_support")
                return [env.open_gripper(ArmTag(s)) for s in ("left", "right")]
            steps.append((4, "release_supported", release, None, True))
        if side:
            def retreat():
                current = self.snapshot(env)
                if current["held"] or not all(g["open"] for g in current["grippers"].values()):
                    raise ValueError("retreat_requires_released_object")
                return [env.move_by_displacement(arm, z=.05, move_axis="arm")]
            steps.append((4, "retreat", retreat, None, True))
        for number, label, build, verify, pending in steps:
            event = session.motion(number, label, build, verify=verify, pending=pending)
            yield event
            if not event["ok"]:
                yield session.stop(event.get("error", "expert_motion_failed"))
                return
        yield session.completion(4)


class OperatorPickPlaceAdapter(PickPlaceAdapter):
    """Opt-in live recovery for cup/QR; never changes handover_block's gates.

    State classification is advice. An explicit operator stage is authoritative;
    only a failed primitive or an illegal newly planned joint path stops execution.
    """

    requires_complete_recovery_for_training = False
    segment_training_gate = True
    operator_regrasp_stage_id = None

    def describe_recovery_state(self, state):
        lines = super().describe_recovery_state(state)
        lines.append(f"接触/支撑诊断（不拦截手选）={state.get('diagnostic_rejection')}")
        lines.append("人工阶段优先；仅规划执行失败或新关节路径非法中止专家动作。")
        return lines

    def infer_recovery_state(self, env):
        try:
            state = self.snapshot(env)
            return {"recoverable": True, "reason": None, "snapshot": state,
                    **self.classify(state), "diagnostic_rejection": self.rejection(state)}
        except (AttributeError, ValueError, TypeError, IndexError) as error:
            return {"recoverable": False, "branch": "unobservable", "stage_id": None,
                    "reason": f"unobservable_state: {error}"}

    def recovery_stage_rejection(self, state, stage_id):
        if not state.get("recoverable") and state.get("reason"):
            return state["reason"]
        if stage_id is not None and stage_id not in self.stage_labels:
            return f"unknown_stage: {stage_id}"
        if stage_id is None and state.get("stage_id") is None:
            return state.get("reason") or "choose_stage_manually"
        return None

    def success_metrics(self, env):
        try:
            state = self.snapshot(env)
            return {"official_task_success": state["official_success"],
                    "physical_quality_gate": self.quality(state), "snapshot": state}
        except (AttributeError, ValueError, TypeError, IndexError) as error:
            return {"official_task_success": bool(env.check_success()),
                    "physical_quality_gate": False, "error": str(error)}

    def recovery_stage_iterator(self, env, start_stage_id=None):
        from envs.utils import ArmTag

        initial = self.infer_recovery_state(env)
        stage = start_stage_id if start_stage_id is not None else initial.get("stage_id")
        reason = self.recovery_stage_rejection(initial, stage)
        result = {"initial_state": initial, "chosen_stage_id": stage,
                  "executed_stage_ids": [], "plan_success": False, "success": False}
        if reason:
            result["reason"] = reason
            yield {"phase": "done", "result": result}
            return

        snapshot = initial.get("snapshot", {})
        held = snapshot.get("held", [])
        side = next(iter(held), "left" if snapshot.get("p", [0])[0] < 0 else "right")
        arm = ArmTag(side)
        events = env.episode_metadata.setdefault("live_recovery_events", [])

        def motion(number, label, build):
            env.current_control_source, env.current_stage_id = "hil", number
            env.plan_success = True
            env.episode_metadata.pop("expert_stage_error", None)
            frame_start = int(env.FRAME_IDX) if hasattr(env, "FRAME_IDX") else None
            lengths = {s: len(getattr(env, f"{s}_joint_path")) for s in ("left", "right")
                       if hasattr(env, f"{s}_joint_path")}
            event = {"phase": "stage", "stage_id": number, "substage": label,
                     "ok": False, "verified": False, "frame_start": frame_start}
            try:
                commands = build()
                if env.plan_success:
                    env.move(*commands)
                if len(lengths) == 2 and hasattr(env, "joint_path_absmax"):
                    paths = [getattr(env, f"{s}_joint_path")[lengths[s]:] for s in ("left", "right")]
                    peak = max(env.joint_path_absmax(path) for path in paths)
                    finite = all(np.isfinite(np.asarray(item["position"], dtype=float)).all()
                                 for path in paths for item in path
                                 if isinstance(item, dict) and item.get("position") is not None)
                    legal = bool(finite and np.isfinite(peak) and peak <= np.pi)
                else:
                    legal, peak = env.planned_joints_legal()
                event.update(plan_success=bool(env.plan_success), joints_legal=bool(legal),
                             joint_absmax=float(peak))
                event["ok"] = event["verified"] = bool(env.plan_success and legal)
                if not event["ok"]:
                    event["error"] = env.episode_metadata.get("expert_stage_error") or "motion_failed_or_illegal_joints"
            except Exception as error:
                env.plan_success = False
                event["error"] = repr(error)
            event["frame_end_exclusive"] = int(env.FRAME_IDX) if hasattr(env, "FRAME_IDX") else None
            events.append(event)
            result["executed_stage_ids"].append(number)
            return event

        steps = []
        if stage == self.operator_regrasp_stage_id and self.operator_regrasp_stage_id is not None:
            # Operator-only correction when the object is already between the
            # fingers. Keep the arm fixed during open/close: no grasp planner.
            steps.extend(((stage, "open_for_regrasp", lambda: [env.open_gripper(arm)]),
                          (stage, "reclose_gripper", lambda: [env.close_gripper(arm, pos=0.0)]),
                          (stage, "lift", lambda: [self.lift(env, arm)]),
                          (stage, "rotate_and_replace", lambda: [self.place(env, arm)])))
        else:
            if stage == 1:
                if self.preshape_gripper is None:
                    if not snapshot.get("grippers", {}).get(side, {}).get("open", True):
                        steps.append((1, "open_gripper", lambda: [env.open_gripper(arm)]))
                else:
                    steps.append((1, "preshape_gripper", lambda: [env.close_gripper(arm, pos=self.preshape_gripper)]))
                steps.append((1, "grasp", lambda: self.grasp(env, arm)))
            if stage <= 2:
                steps.append((2, "lift", lambda: [self.lift(env, arm)]))
            if stage <= 3:
                steps.append((3, "place_and_release", lambda: [self.place(env, arm)]))
            else:
                steps.append((4, "release", lambda: [env.open_gripper(arm)]))
            steps.append((4, "retreat", lambda: [env.move_by_displacement(arm, z=.05, move_axis="arm")]))

        for number, label, build in steps:
            event = motion(number, label, build)
            yield event
            if not event["ok"]:
                result["reason"] = event.get("error", "expert_motion_failed")
                yield {"phase": "done", "result": dict(result)}
                return

        metrics = self.success_metrics(env)
        result.update(success=bool(metrics["official_task_success"]),
                      plan_success=True, final_metrics=metrics)
        if not result["success"]:
            result["reason"] = "official_task_not_successful_after_recovery"
        yield {"phase": "done", "result": dict(result)}
