#!/usr/bin/env python3
"""Stage-six-only HIL collection, sharing the HG-DAgger control loop.

The baseline and enhanced arms intentionally use the same expert prefix and
seed order. This module also keeps the pairing/eligibility rules testable
without importing SAPIEN or a policy client.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def handoff_record(task_env: Any) -> dict[str, Any]:
    state = task_env.infer_recovery_state()
    robot = getattr(task_env, "robot", None)
    bar = getattr(task_env, "bar", None)
    tray = getattr(task_env, "tray_floor", None)
    signature: dict[str, list[float]] = {}
    if robot is not None and bar is not None and tray is not None:
        for name, values in (
            ("left_joint", robot.get_left_arm_jointState()),
            ("right_joint", robot.get_right_arm_jointState()),
            ("bar_p", bar.get_pose().p), ("bar_q", bar.get_pose().q),
            ("tray_p", tray.get_pose().p), ("tray_q", tray.get_pose().q),
        ):
            signature[name] = [float(value) for value in values]
    ready = bool(
        task_env.plan_success
        and state["receiver_holds"]
        and not state["source_holds"]
        and not state["placed"]
        and state["recoverable"]
        and state["stage_id"] == 6
    )
    return {"ready": ready, "state": state, "stage_id": 6 if ready else None,
            "signature": signature}


def handoff_max_abs_difference(a: dict[str, Any], b: dict[str, Any]) -> float:
    left, right = a.get("signature", {}), b.get("signature", {})
    required = {"left_joint", "right_joint", "bar_p", "bar_q", "tray_p", "tray_q"}
    if not required.issubset(left) or not required.issubset(right):
        raise ValueError("Paired handoff signature is incomplete")
    differences = []
    for name in sorted(required):
        if len(left[name]) != len(right[name]):
            raise ValueError(f"Paired handoff dimension mismatch for {name}")
        differences.extend(abs(float(x) - float(y)) for x, y in zip(left[name], right[name]))
    maximum = max(differences)
    if not math.isfinite(maximum):
        raise ValueError("Paired handoff contains non-finite values")
    return maximum


def handoff_pair_differences(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Compare re-executed expert prefixes; seed pairing is not exact replay."""
    handoff_max_abs_difference(a, b)  # Validate all required fields first.
    return {
        name: max(abs(float(x) - float(y)) for x, y in
                  zip(a["signature"][name], b["signature"][name]))
        for name in ("left_joint", "right_joint", "bar_p", "bar_q", "tray_p", "tray_q")
    }


def handoff_pair_ready(differences: dict[str, float]) -> bool:
    """Predeclared semantic handoff tolerances, not an exact-state guarantee."""
    tolerances = {
        "left_joint": 0.02, "right_joint": 0.02,
        "bar_p": 0.002, "bar_q": 0.01,
        "tray_p": 1e-6, "tray_q": 1e-6,
    }
    return all(differences[name] <= limit for name, limit in tolerances.items())


def is_valid_hil(*, interventions: int, hil_frames: int, expert_result: dict | None,
                 final_success: bool, joints_legal: bool) -> bool:
    return bool(
        interventions > 0
        and hil_frames > 0
        and expert_result is not None
        and expert_result.get("success")
        and final_success
        and joints_legal
    )


def baseline_seeds(path: Path, *, checkpoint: str, task_config: str,
                   frequency: int) -> list[int]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("one_stage") is not True or report.get("sampling_mode") != "off":
        raise ValueError("Paired seed source must be a completed one-stage baseline report")
    if report.get("aborted") or not report.get("records"):
        raise ValueError("Baseline session is incomplete or empty")
    for name, expected in (("checkpoint_name", checkpoint),
                           ("task_config", task_config), ("frequency", frequency)):
        if report.get(name) != expected:
            raise ValueError(f"Baseline {name} differs: {report.get(name)!r} != {expected!r}")
    records = report["records"]
    if len(records) != int(report.get("rollouts", -1)):
        raise ValueError("Baseline report has an inconsistent rollout count")
    if any(record.get("rollout_status") not in {"completed", "prefix_failed", "aborted"}
           for record in records):
        raise ValueError("Baseline contains an interrupted rollout")
    # Only completed stage-6 trials have an outcome to pair. Prefix failures
    # and operator-aborted trials remain in the baseline effort accounting.
    seeds = [int(record["seed"]) for record in records
             if record["rollout_status"] == "completed"]
    if not seeds:
        raise ValueError("Baseline contains no eligible stage-6 trials")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Baseline contains duplicate seeds")
    return seeds


def baseline_handoffs(path: Path) -> dict[int, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {int(record["seed"]): record.get("stage6_handoff") or {}
            for record in report["records"] if record.get("rollout_status") == "completed"}


def main() -> int:
    from hg_dagger_handover import main as collection_main

    sys.argv[1:1] = ["--one-stage"]
    return collection_main()


if __name__ == "__main__":
    raise SystemExit(main())
