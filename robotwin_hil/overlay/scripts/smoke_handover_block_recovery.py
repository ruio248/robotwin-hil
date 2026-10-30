#!/usr/bin/env python3
"""Single expert-prefix HIL recovery smoke; never a policy success-rate test."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", str(SCRIPT_DIR.parent))).resolve()
sys.path.insert(0, str(ROBOTWIN_ROOT / "scripts"))

from eval_policy_xpolicylab import class_decorator, load_task_args, safe_close_env  # noqa: E402
from handover_block_recovery import (  # noqa: E402
    classify_handover_state,
    handover_snapshot,
    iter_expert_suffix,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after-actions", type=int,
                        help="Stop at this many successful staged actions; handback smoke only")
    parser.add_argument("--record-frames", action="store_true",
                        help="Preserve native expert recovery frames for label-integrity smoke")
    cli = parser.parse_args()
    if cli.stop_after_actions is not None and cli.stop_after_actions < 1:
        parser.error("--stop-after-actions must be positive")
    if cli.output.exists():
        raise FileExistsError(f"refusing to overwrite prior smoke: {cli.output}")
    cli.output.parent.mkdir(parents=True, exist_ok=True)

    args, _ = load_task_args({
        "task_name": "handover_block", "task_config": "demo_clean",
        "policy_name": "Pi_05_RobotTwin", "ckpt_name": "expert_only_recovery_smoke",
    })
    args.update({
        "eval_instruction": "seen", "eval_mode": True, "render_freq": 0,
        "need_plan": True, "save_data": False, "save_video": False,
        "save_path": str(cli.output.parent),
    })
    task_env = class_decorator("handover_block")
    raw_saved = False
    record = {
        "task": "handover_block", "seed": cli.seed,
        "protocol": "expert_prefix_then_staged_expert_suffix; no policy",
        "stop_after_actions": cli.stop_after_actions,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage_events": [],
    }
    try:
        task_env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=False, **args)
        if cli.record_frames and not hasattr(task_env, "control_mask"):
            raise RuntimeError(
                "this RoboTwin Base_Task lacks the per-frame control_mask interface"
            )
        from envs.utils import ArmTag

        source = ArmTag("left")
        task_env.plan_success = True
        task_env.move(task_env.grasp_actor(
            task_env.box, arm_tag=source, pre_grasp_dis=0.07,
            grasp_dis=0.0, contact_point_id=[0, 1, 2, 3],
        ))
        record["prefix_grasp_plan_success"] = bool(task_env.plan_success)
        if not task_env.plan_success:
            record["reason"] = "expert_prefix_grasp_failed"
            return 2
        task_env.move(task_env.move_by_displacement(source, z=0.10))
        record["prefix_lift_plan_success"] = bool(task_env.plan_success)
        if not task_env.plan_success:
            record["reason"] = "expert_prefix_lift_failed"
            return 2

        if cli.record_frames:
            raw_dir = cli.output.parent / f"{cli.output.stem}_raw"
            if raw_dir.exists():
                raise FileExistsError(f"refusing to overwrite raw recovery: {raw_dir}")
            task_env.save_data = True
            task_env.save_video = False
            task_env.save_dir = str(raw_dir)
            task_env.FRAME_IDX = 0
            task_env.folder_path = {}
            task_env.control_mask = []
            task_env.current_control_source = "hil"
            task_env._take_picture()

        snapshot = handover_snapshot(task_env, policy_steps=0)
        decision = classify_handover_state(snapshot)
        record["handoff_snapshot"] = snapshot
        record["decision"] = decision
        stages = iter_expert_suffix(task_env, decision)
        for event in stages:
            record["stage_events"].append(event)
            if (cli.stop_after_actions is not None and event["phase"] == "stage"
                    and len(record["stage_events"]) >= cli.stop_after_actions):
                record["handback_ready"] = bool(task_env.plan_success)
                stages.close()
                break
        record["final_check_success"] = bool(task_env.check_success())
        record["success"] = bool(
            record["stage_events"]
            and record["stage_events"][-1].get("phase") == "done"
            and record["stage_events"][-1]["result"].get("success")
            and record["final_check_success"]
        )
        if cli.record_frames:
            cache = Path(task_env.folder_path["cache"])
            frames = sorted(cache.glob("*.pkl"), key=lambda path: int(path.stem))
            if len(frames) < 2 or len(frames) != len(task_env.control_mask):
                raise RuntimeError(
                    "expert frame count and control mask do not match: "
                    f"files={len(frames)} mask={len(task_env.control_mask)} "
                    f"frame_idx={task_env.FRAME_IDX} cache={cache}"
                )
            stage_ids = []
            for frame in frames:
                with frame.open("rb") as stream:
                    stage_ids.append(int(pickle.load(stream)["task_metadata"]["stage_id"]))
            if set(task_env.control_mask) != {"hil"} or any(stage_id < 0 for stage_id in stage_ids):
                raise RuntimeError("expert frame source or stage label is invalid")
            if (raw_dir / "frames").exists():
                raise FileExistsError(f"refusing to overwrite recovery frames: {raw_dir / 'frames'}")
            shutil.move(str(cache), str(raw_dir / "frames"))
            (raw_dir / "episode.json").write_text(json.dumps({
                "task": "handover_block", "seed": cli.seed,
                "protocol": record["protocol"], "save_freq": int(task_env.save_freq or 15),
                "control_mask": list(task_env.control_mask),
                "stage_ids": stage_ids,
                "expert_actions": record["stage_events"],
                "success": record["success"],
            }, ensure_ascii=False, indent=2) + "\n")
            raw_saved = True
            record["native_recovery_frames"] = {
                "frames_dir": str(raw_dir / "frames"),
                "count": len(frames), "control_sources": sorted(set(task_env.control_mask)),
                "sampled_stage_ids": sorted(set(stage_ids)),
            }
        if cli.stop_after_actions is not None:
            return 0 if record.get("handback_ready") else 2
        return 0 if record["success"] else 2
    except Exception as exc:
        record["exception"] = repr(exc)
        raise
    finally:
        cli.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        safe_close_env(task_env, clear_cache=not raw_saved)


if __name__ == "__main__":
    raise SystemExit(main())
