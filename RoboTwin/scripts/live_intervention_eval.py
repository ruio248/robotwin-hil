#!/usr/bin/env python3
"""Live intervention evaluation on real policy rollouts.

Run the policy normally, trigger a scripted-expert takeover when the episode
looks failed (bar dropped, bar out of the recoverable workspace, or a configured
step), and record whether the expert rescues the episode. This is the
intervention path of HG-DAgger measured on real test-seed failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
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
    parser.add_argument("--seed-start", type=int, default=31000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--render-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument("--step-limit", type=int, default=None)
    parser.add_argument(
        "--intervene-step",
        type=int,
        default=600,
        help="Fallback takeover step; set -1 to disable the fixed-step trigger.",
    )
    parser.add_argument(
        "--no-intervene-on-drop",
        action="store_true",
        help="Disable takeover when a previously held bar is dropped.",
    )
    parser.add_argument(
        "--no-intervene-out-of-workspace",
        action="store_true",
        help="Disable takeover when the bar leaves the recoverable workspace.",
    )
    parser.add_argument(
        "--save-videos",
        choices=["all", "none"],
        default="all",
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
        "evaluation_id": f"live-intervention-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "trial_id": "live-intervention",
        "action_case_id": "live-intervention-v2",
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
    records_path = cli.output_dir / "live_intervention_records.jsonl"
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
            was_held = False
            intervene = False
            intervene_reason = None
            intervention_state: dict[str, Any] | None = None

            while not is_episode_end(task_env) and not intervene:
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
                    try:
                        task_env.take_action(flat_action, action_type=action_type)
                    except Exception as exc:
                        intervene = True
                        intervene_reason = "policy_step_error"
                        intervention_state = {"error": repr(exc)}
                        break
                    policy_steps += 1
                    if task_env.save_data and policy_steps % int(cli.save_freq) == 0:
                        task_env._take_picture()
                    if is_episode_end(task_env):
                        break

                    try:
                        state = task_env.infer_recovery_state()
                    except Exception as exc:
                        intervene = True
                        intervene_reason = "state_error"
                        intervention_state = {"error": repr(exc)}
                        break
                    held = bool(state["source_holds"] or state["receiver_holds"])
                    if (
                        not cli.no_intervene_on_drop
                        and was_held
                        and not held
                    ):
                        intervene = True
                        intervene_reason = "bar_dropped"
                    elif (
                        not cli.no_intervene_out_of_workspace
                        and not state["recoverable"]
                    ):
                        intervene = True
                        intervene_reason = "out_of_workspace"
                    elif (
                        cli.intervene_step >= 0
                        and policy_steps >= cli.intervene_step
                    ):
                        intervene = True
                        intervene_reason = "fixed_step"
                    was_held = held

                    if intervene:
                        intervention_state = task_env.infer_recovery_state()
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

            expert_result = None
            expert_error = None
            if intervene and not is_episode_end(task_env):
                task_env.current_control_source = "hil"
                try:
                    expert_result = task_env.recover_from_current_state()
                except Exception:
                    expert_error = traceback.format_exc()

            joints_legal, joint_absmax = task_env.planned_joints_legal()
            success = bool(task_env.eval_success)
            record = {
                "seed": int(seed),
                "intervened": bool(intervene),
                "intervene_reason": intervene_reason,
                "policy_steps_before_intervention": int(policy_steps),
                "intervention_state": intervention_state,
                "expert_plan_success": bool((expert_result or {}).get("plan_success")),
                "expert_success": bool((expert_result or {}).get("success")),
                "expert_branch": (expert_result or {}).get("branch"),
                "expert_stage_ids": (expert_result or {}).get("executed_stage_ids"),
                "expert_reason": (expert_result or {}).get("reason"),
                "expert_error": expert_error,
                "final_success": success,
                "final_check_success": bool(task_env.check_success()),
                "final_success_metrics": task_env.success_metrics(),
                "final_bar_pose": pose_to_dict(task_env.bar.get_pose()),
                "joints_legal": bool(joints_legal),
                "joint_absmax": float(joint_absmax),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }

            if task_env.save_data:
                task_env.episode_metadata["live_intervention"] = record
                if int(task_env.FRAME_IDX) >= 2:
                    task_env.merge_pkl_to_hdf5_video(instructions=[PROMPT])
                else:
                    cache = getattr(task_env, "folder_path", {}).get("cache")
                    if cache and Path(cache).exists():
                        task_env.remove_data_cache()
            record["video_path"] = (
                f"video/episode_{rollout_index:07d}.mp4" if task_env.save_data else None
            )
            results.append(record)
            append_jsonl(records_path, record)
            print(
                f"[RESCUE] seed={seed} reason={intervene_reason} "
                f"steps={policy_steps} expert_success={record['expert_success']} "
                f"final_success={record['final_success']}",
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

    intervened = [r for r in results if r["intervened"]]
    summary = {
        "episodes": len(results),
        "interventions": len(intervened),
        "rescued": sum(1 for r in intervened if r["expert_success"]),
        "rescue_rate": (
            sum(1 for r in intervened if r["expert_success"]) / len(intervened)
            if intervened
            else 0.0
        ),
        "by_reason": {},
        "by_branch": {},
        "results": results,
    }
    for reason in sorted({r["intervene_reason"] for r in intervened}):
        group = [r for r in intervened if r["intervene_reason"] == reason]
        summary["by_reason"][reason] = {
            "count": len(group),
            "rescued": sum(1 for r in group if r["expert_success"]),
        }
    for branch in sorted({r["expert_branch"] for r in intervened}):
        group = [r for r in intervened if r["expert_branch"] == branch]
        summary["by_branch"][branch] = {
            "count": len(group),
            "rescued": sum(1 for r in group if r["expert_success"]),
        }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"records={records_path}")
    print(f"summary={summary_path}")
    print(
        f"rescue_rate={summary['rescue_rate']:.2f} "
        f"({summary['rescued']}/{summary['interventions']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
