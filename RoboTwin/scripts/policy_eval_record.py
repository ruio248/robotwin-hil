#!/usr/bin/env python3
"""Record per-seed policy evaluation results and failure samples.

Run the policy without expert takeover and save:
- one JSONL entry per seed (success, steps, final metrics, bar pose, grippers)
- rollout video/HDF5 for failed seeds (configurable)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))
if str(ROBOTWIN_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT / "scripts"))

from eval_policy_xpolicylab import (
    build_policy_client,
    class_decorator,
    close_policy_client,
    is_episode_end,
    load_task_args,
    normalize_action_chunk,
    prepare_policy_case,
    reset_policy,
    robotwin_obs_to_xpolicylab,
    safe_close_env,
    xpolicylab_action_to_robotwin,
)


PROMPT = "Pass the red bar from the left arm to the right arm and place it in the blue tray."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18300)
    parser.add_argument("--policy-name", default="Pi_05_RobotTwin")
    parser.add_argument("--ckpt-name", default="v2_promptfix_9999")
    parser.add_argument("--task-config", default="handover_to_tray_v2_promptfix")
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--render-freq", type=int, default=5)
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument("--step-limit", type=int, default=None)
    parser.add_argument(
        "--save-videos",
        choices=["failure", "all", "none"],
        default="failure",
        help="which rollouts to keep as HDF5/video",
    )
    parser.add_argument("--render-max-num-materials", type=int, default=128)
    parser.add_argument("--render-max-num-textures", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_runtime_args(cli: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    user_args = {
        "task_name": "handover_to_tray",
        "task_config": cli.task_config,
        "policy_name": cli.policy_name,
        "host": cli.host,
        "port": int(cli.port),
        "protocol": "ws",
        "xpolicylab_root": str(ROBOTWIN_ROOT / "XPolicyLab"),
        "ckpt_name": cli.ckpt_name,
        "ckpt_setting": cli.ckpt_name,
        "action_type": "joint",
        "evaluation_id": f"policy-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "trial_id": "policy-eval",
        "action_case_id": "policy-eval-v2",
    }
    args, _ = load_task_args(user_args)
    args["eval_instruction"] = "seen"
    args["eval_mode"] = True
    args["render_freq"] = int(cli.render_freq)
    args["need_plan"] = True
    args["save_data"] = False
    args["save_video"] = False
    args["save_path"] = str(cli.output_dir)
    args["save_freq"] = int(cli.save_freq)
    args["render_max_num_materials"] = int(cli.render_max_num_materials)
    args["render_max_num_textures"] = int(cli.render_max_num_textures)
    user_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    user_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    return user_args, args


def pose_to_dict(pose) -> dict[str, Any]:
    return {
        "position": np.asarray(pose.p, dtype=np.float64).round(8).tolist(),
        "quaternion_wxyz": np.asarray(pose.q, dtype=np.float64).round(8).tolist(),
    }


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> int:
    cli = parse_args()
    cli.output_dir = cli.output_dir.expanduser().resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    user_args, args = build_runtime_args(cli)
    task_env = class_decorator("handover_to_tray")
    model_client = build_policy_client(user_args)
    records_path = cli.output_dir / "eval_records.jsonl"
    summary_path = cli.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results: list[dict[str, Any]] = []

    try:
        for rollout_index in range(int(cli.episodes)):
            seed = int(cli.seed_start) + rollout_index
            task_env.setup_demo(
                now_ep_num=rollout_index,
                seed=seed,
                is_test=False,
                **args,
            )
            if cli.step_limit is not None:
                task_env.step_lim = int(cli.step_limit)
            task_env.set_instruction(PROMPT)
            if hasattr(task_env, "_update_render"):
                task_env._update_render()
            viewer = getattr(task_env, "viewer", None)
            if viewer is not None:
                viewer.render()
            prepare_policy_case(model_client, "handover_to_tray", seed, PROMPT, "joint")
            reset_policy(model_client)

            keep_video = cli.save_videos == "all"
            task_env.save_data = cli.save_videos != "none"
            task_env.save_video = cli.save_videos != "none"
            task_env.save_dir = str(cli.output_dir)
            task_env.FRAME_IDX = 0
            task_env.folder_path = {}
            task_env.ep_num = int(rollout_index)
            task_env.control_mask = []
            task_env.current_control_source = "policy"
            if task_env.save_data:
                task_env._take_picture()

            policy_steps = 0
            while not is_episode_end(task_env):
                observation = task_env.get_obs()
                xpl_obs = robotwin_obs_to_xpolicylab(
                    observation,
                    instruction=PROMPT,
                    env_idx=0,
                    frequency=int(cli.frequency),
                    task_env=task_env,
                )
                model_client.call(func_name="update_obs", obs=xpl_obs)
                action_chunk = normalize_action_chunk(
                    model_client.call(func_name="get_action")
                )
                if not action_chunk:
                    raise RuntimeError("Policy returned an empty action chunk.")

                for action in action_chunk:
                    flat_action, action_type = xpolicylab_action_to_robotwin(
                        action,
                        action_type="joint",
                        current_observation=observation,
                    )
                    task_env.current_control_source = "policy"
                    task_env.take_action(flat_action, action_type=action_type)
                    policy_steps += 1
                    if task_env.save_data and policy_steps % int(cli.save_freq) == 0:
                        task_env._take_picture()
                    if is_episode_end(task_env):
                        break

                    observation = task_env.get_obs()
                    xpl_obs = robotwin_obs_to_xpolicylab(
                        observation,
                        instruction=PROMPT,
                        env_idx=0,
                        frequency=int(cli.frequency),
                        task_env=task_env,
                    )
                    model_client.call(func_name="update_obs", obs=xpl_obs)

            success = bool(task_env.eval_success)
            metrics = task_env.success_metrics()
            record = {
                "seed": int(seed),
                "success": success,
                "policy_steps": int(policy_steps),
                "final_check_success": bool(task_env.check_success()),
                "final_success_metrics": metrics,
                "bar_pose": pose_to_dict(task_env.bar.get_pose()),
                "current_stage_id": int(getattr(task_env, "current_stage_id", -1)),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }

            if cli.save_videos != "none" and not success:
                keep_video = True

            if task_env.save_data:
                task_env.episode_metadata["control_mask"] = list(task_env.control_mask)
                task_env.episode_metadata["policy_eval"] = record
                if keep_video and int(task_env.FRAME_IDX) >= 2:
                    task_env.merge_pkl_to_hdf5_video(instructions=[PROMPT])
                else:
                    cache = getattr(task_env, "folder_path", {}).get("cache")
                    if cache and Path(cache).exists():
                        task_env.remove_data_cache()

            record["video_path"] = (
                f"video/episode_{rollout_index:07d}.mp4" if keep_video else None
            )
            record["hdf5_path"] = (
                f"data/episode_{rollout_index:07d}.hdf5" if keep_video else None
            )
            results.append(record)
            append_jsonl(records_path, record)
            print(
                f"[EVAL] seed={seed} success={success} steps={policy_steps} "
                f"video={keep_video}",
                flush=True,
            )
            safe_close_env(task_env)
    except Exception:
        import traceback

        traceback.print_exc()
        safe_close_env(task_env, clear_cache=True)
        return 3
    finally:
        close_policy_client(model_client)

    summary = {
        "episodes": len(results),
        "successes": sum(1 for item in results if item["success"]),
        "success_rate": (
            sum(1 for item in results if item["success"]) / len(results)
            if results
            else 0.0
        ),
        "results": results,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"records={records_path}")
    print(f"summary={summary_path}")
    print(f"success_rate={summary['success_rate']:.2f} ({summary['successes']}/{summary['episodes']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
