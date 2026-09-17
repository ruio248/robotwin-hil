"""Run deterministic scripted-planner smoke tests for handover_to_tray.

The runner deliberately does not record RGB/HDF5. It checks whether the task
geometry and six-stage script are suitable for bulk expert generation first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import yaml

ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))

from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from scripts.collect_data import class_decorator, get_embodiment_config


def load_args(task_name: str, task_config: str) -> dict:
    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml"), "r", encoding="utf-8") as handle:
        args = yaml.safe_load(handle)
    args["task_name"] = task_name

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as handle:
        embodiments = yaml.safe_load(handle)

    embodiment_type = args["embodiment"]
    if len(embodiment_type) != 1:
        raise ValueError("handover_to_tray smoke currently requires one dual-arm embodiment")
    robot_file = embodiments[embodiment_type[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["left_embodiment_config"] = get_embodiment_config(robot_file)
    args["right_embodiment_config"] = get_embodiment_config(robot_file)
    args["dual_arm_embodied"] = True
    args["embodiment_name"] = str(embodiment_type[0])
    args["task_config"] = task_config
    args["save_data"] = False
    args["save_video"] = False
    args["collect_data"] = False
    args["render_freq"] = 0
    return args


def run_episode(task_name: str, args: dict, seed: int, index: int) -> dict:
    task = class_decorator(task_name)
    result = {"index": index, "seed": seed, "success": False}
    try:
        episode_args = deepcopy(args)
        episode_args["seed"] = seed
        episode_args["now_ep_num"] = index
        episode_args["need_plan"] = True
        task.setup_demo(**episode_args)
        info = task.play_once()
        joints_legal, joint_absmax = task.planned_joints_legal()
        result.update(
            {
                "plan_success": bool(task.plan_success),
                "success": bool(task.plan_success and joints_legal and task.check_success()),
                "joints_legal": bool(joints_legal),
                "joint_absmax": float(joint_absmax),
                "stage_ids": [item["stage_id"] for item in task.stage_snapshots],
                "stage_plan_success": [item["plan_success"] for item in task.stage_snapshots],
                "success_metrics": task.success_metrics(),
                "final_bar_functional_pose": task._pose_dict(task.bar.get_functional_point(0, "pose")),
                "final_bar_center_pose": task._pose_dict(task.bar.get_pose()),
                "tray_target_pose": task._pose_dict(task.tray_floor.get_functional_point(1, "pose")),
                "source_arm": str(task.source_arm_tag),
                "receiver_arm": str(task.receiver_arm_tag),
                "task_metadata": info.get("task_metadata", {}),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=5)
    finally:
        try:
            task.close_env(clear_cache=True)
        except Exception as close_exc:
            result["close_error"] = f"{type(close_exc).__name__}: {close_exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="handover_to_tray")
    parser.add_argument("--config", default="handover_to_tray_smoke")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    parser.add_argument("--report", default="/tmp/handover_to_tray_smoke.json")
    cli = parser.parse_args()

    if cli.episodes <= 0:
        raise ValueError("--episodes must be positive")
    args = load_args(cli.task, cli.config)
    results = [
        run_episode(cli.task, args, cli.seed_start + index, index)
        for index in range(cli.episodes)
    ]
    success_count = sum(item["success"] for item in results)
    report = {
        "task": cli.task,
        "config": cli.config,
        "episodes": cli.episodes,
        "seed_start": cli.seed_start,
        "successes": success_count,
        "success_rate": success_count / cli.episodes,
        "required_success_rate": cli.min_success_rate,
        "passed": success_count / cli.episodes >= cli.min_success_rate,
        "results": results,
    }
    report_path = Path(cli.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"report={report_path}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
