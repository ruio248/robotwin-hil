"""Task-specific live-state and recovery adapters for the shared HIL loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Adapter:
    task_name: str
    snapshot: Any
    classify: Any
    recovery: Any
    quality: Any
    stage_labels: dict[int, str]


def _place_basket_adapter() -> Adapter:
    from place_can_basket_recovery import (
        basket_snapshot, classify_basket_state, iter_basket_expert_suffix,
    )

    def classify(current: dict, previous: dict | None) -> dict:
        decision = classify_basket_state(current)
        # Regrasping a can that is still on the table is only safe after it has
        # settled across two policy chunk boundaries. Do not infer stability
        # from a single instant.
        if decision.get("branch") == "regrasp_place_can":
            if previous is None:
                return {"ready": False, "branch": "wait_for_settle",
                        "reason": "no_previous_pose_for_table_can"}
            try:
                can_motion = sum(abs(float(a) - float(b)) for a, b in zip(
                    current["can_p"], previous["can_p"],
                ))
                basket_motion = sum(abs(float(a) - float(b)) for a, b in zip(
                    current["basket_p"], previous["basket_p"],
                ))
            except (KeyError, TypeError, ValueError):
                return {"ready": False, "branch": "unknown",
                        "reason": "missing_previous_object_pose"}
            if max(can_motion, basket_motion) > 0.025:
                return {"ready": False, "branch": "wait_for_settle",
                        "reason": "can_or_basket_still_moving",
                        "can_displacement_l1_m": can_motion,
                        "basket_displacement_l1_m": basket_motion}
            decision["settled_displacement_l1_m"] = {
                "can": can_motion, "basket": basket_motion,
            }
        return decision

    def quality(task_env, snapshot: dict) -> dict:
        official = bool(task_env.check_success())
        physical = bool(
            snapshot.get("can_basket_contact") is True
            and snapshot.get("can_table_contact") is False
            and snapshot.get("basket_axis_z", 0.0) > 0.5
            and snapshot.get("can_basket_l1_m", float("inf")) < 0.15
            and snapshot.get("source_gripper_open") is True
            and snapshot.get("receiver_gripper_open") is True
        )
        return {"strict_success": bool(official and physical),
                "official_task_success": official,
                "physical_quality_gate": physical,
                "can_basket_l1_m": snapshot.get("can_basket_l1_m")}

    return Adapter(
        "place_can_basket",
        lambda env, steps: basket_snapshot(env, steps),
        classify,
        iter_basket_expert_suffix,
        quality,
        {1: "重抓罐", 2: "将罐放入篮", 3: "释放并后撤", 4: "抓篮/放稳篮", 5: "抬起篮子"},
    )


def _rotate_qrcode_adapter() -> Adapter:
    from rotate_qrcode_recovery import (
        classify_qrcode_state, iter_qrcode_expert_suffix, qrcode_snapshot,
    )

    def quality(task_env, snapshot: dict) -> dict:
        official = bool(task_env.check_success())
        xy_error = float(snapshot.get("target_xy_error_m", float("inf")))
        location_ok = xy_error <= 0.08 and snapshot.get("table_contact") is True
        return {
            "strict_success": bool(official and location_ok),
            "official_task_success": official,
            "physical_quality_gate": location_ok,
            "target_xy_error_m": xy_error,
        }

    return Adapter(
        "rotate_qrcode",
        lambda env, steps: qrcode_snapshot(env, steps),
        classify_qrcode_state,
        iter_qrcode_expert_suffix,
        quality,
        {1: "重抓二维码板", 2: "抬起", 3: "对齐并放到目标", 4: "释放"},
    )


def _new_task_adapter(task_name: str) -> Adapter:
    from hg_dagger_extra_task_recovery import (
        classify_dual_bottles_state, classify_dustbin_state,
        classify_stack_blocks_two_state, dual_bottles_snapshot,
        dustbin_snapshot, iter_dual_bottles_expert_suffix,
        iter_dustbin_expert_suffix, iter_stack_blocks_two_expert_suffix,
        stack_blocks_two_snapshot,
    )

    if task_name == "pick_dual_bottles":
        return Adapter(
            task_name,
            dual_bottles_snapshot,
            classify_dual_bottles_state,
            iter_dual_bottles_expert_suffix,
            lambda env, snap: {
                "strict_success": bool(env.check_success() and
                                        snap.get("strict_physical_quality") is True),
                "official_task_success": bool(env.check_success()),
                "physical_quality_gate": snap.get("strict_physical_quality") is True,
            },
            {1: "安全张开远离瓶子的空夹爪", 2: "空闲手臂退回安全位",
             3: "抓取未完成瓶", 4: "抬升", 5: "移至各自目标并保持抓持"},
        )
    if task_name == "put_bottles_dustbin":
        return Adapter(
            task_name,
            dustbin_snapshot,
            classify_dustbin_state,
            iter_dustbin_expert_suffix,
            lambda env, snap: {
                "strict_success": bool(env.check_success() and
                                        snap.get("strict_physical_quality") is True),
                "official_task_success": bool(env.check_success()),
                "physical_quality_gate": snap.get("strict_physical_quality") is True,
                "bottles_in_bin": snap.get("bottles_in_bin"),
            },
            {1: "安全张开空夹爪/抓取或重抓", 2: "抬起", 3: "右臂转交至左臂",
             4: "转交后释放右夹爪", 5: "放入垃圾桶并逐瓶确认"},
        )
    if task_name == "stack_blocks_two":
        return Adapter(
            task_name,
            stack_blocks_two_snapshot,
            classify_stack_blocks_two_state,
            iter_stack_blocks_two_expert_suffix,
            lambda env, snap: {
                "strict_success": bool(env.check_success() and
                                        snap.get("strict_physical_quality") is True),
                "official_task_success": bool(env.check_success()),
                "physical_quality_gate": snap.get("strict_physical_quality") is True,
            },
            {1: "底块移至目标", 2: "顶块放置并释放"},
        )
    raise ValueError(f"no HG-DAgger adapter registered for {task_name!r}")


def get_adapter(task_name: str) -> Adapter:
    if task_name == "place_can_basket":
        return _place_basket_adapter()
    if task_name == "rotate_qrcode":
        return _rotate_qrcode_adapter()
    if task_name in {"pick_dual_bottles", "put_bottles_dustbin", "stack_blocks_two"}:
        return _new_task_adapter(task_name)
    raise ValueError(f"no HG-DAgger adapter registered for {task_name!r}")
