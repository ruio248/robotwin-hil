#!/usr/bin/env python3
"""One-seed infrastructure diagnosis; not a policy rollout or benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", str(SCRIPT_DIR.parent))).resolve()
sys.path.insert(0, str(ROBOTWIN_ROOT / "scripts"))

from eval_policy_xpolicylab import class_decorator, load_task_args, safe_close_env  # noqa: E402
from envs._base_task import Base_Task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if cli.output.exists():
        raise FileExistsError(f"refusing to overwrite prior diagnosis: {cli.output}")
    cli.output.parent.mkdir(parents=True, exist_ok=True)

    args, _ = load_task_args({
        "task_name": "move_can_pot", "task_config": "demo_clean",
        "policy_name": "Pi_05_RobotTwin", "ckpt_name": "stability_diagnostic_only",
    })
    args.update({
        "eval_instruction": "seen", "eval_mode": True, "render_freq": 0,
        "need_plan": True, "save_data": False, "save_video": False,
        "save_path": str(cli.output.parent),
    })
    task_env = class_decorator("move_can_pot")
    original_check_stable = Base_Task.check_stable
    record = {
        "task": "move_can_pot", "seed": cli.seed,
        "protocol": "repeat original check_stable once after first failure; no policy",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stability_checks": [],
    }

    def check_with_one_retry(env):
        first = original_check_stable(env)
        record["stability_checks"].append({
            "stable": bool(first[0]), "unstable_objects": list(first[1]),
        })
        if first[0]:
            return first
        second = original_check_stable(env)
        record["stability_checks"].append({
            "stable": bool(second[0]), "unstable_objects": list(second[1]),
        })
        return second

    Base_Task.check_stable = check_with_one_retry
    try:
        task_env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=False, **args)
        record["setup_success"] = True
        record["can_pose"] = {
            "p": list(map(float, task_env.can.get_pose().p)),
            "q": list(map(float, task_env.can.get_pose().q)),
        }
        record["pot_pose"] = {
            "p": list(map(float, task_env.pot.get_pose().p)),
            "q": list(map(float, task_env.pot.get_pose().q)),
        }
        return 0
    except Exception as exc:
        record["setup_success"] = False
        record["exception"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        return 2
    finally:
        Base_Task.check_stable = original_check_stable
        cli.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        safe_close_env(task_env, clear_cache=True)


if __name__ == "__main__":
    raise SystemExit(main())
