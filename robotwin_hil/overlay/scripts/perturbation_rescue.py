#!/usr/bin/env python3
"""Perturbation-rescue sweep for the RoboTwin handover_to_tray task.

Run the learned policy normally for a lead-in, inject a small disturbance
(joint-action bias or a direct push on the red bar), then hand control to the
scripted expert and record whether it can rescue the episode. This is an
automated robustness probe, not a human-in-the-loop collection run.
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

from eval_policy_xpolicylab import (  # noqa: E402
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


def build_runtime_args(cli: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    user_args: dict[str, Any] = {
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
        "evaluation_id": f"perturbation-rescue-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "trial_id": "perturbation-rescue",
        "action_case_id": "perturbation-rescue-v2",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perturb a policy rollout and check whether the scripted expert can rescue it."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18300)
    parser.add_argument("--policy-name", default="Pi_05_RobotTwin")
    parser.add_argument("--ckpt-name", default="v2_promptfix_9999")
    parser.add_argument("--task-config", default="handover_to_tray_v2_promptfix")
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--render-freq", type=int, default=5)
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument("--render-max-num-materials", type=int, default=128)
    parser.add_argument("--render-max-num-textures", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--lead-in-steps", type=int, default=30)
    parser.add_argument("--bias-duration", type=int, default=15)
    parser.add_argument(
        "--perturb-mode",
        choices=["action_bias", "push_bar"],
        default="action_bias",
    )
    parser.add_argument("--bias-magnitude", type=float, default=0.05)
    parser.add_argument(
        "--bias-dims",
        default="joints",
        help="'joints' biases the 12 arm joints; otherwise a comma-separated list of indices.",
    )
    parser.add_argument("--push-dx", type=float, default=0.04)
    parser.add_argument("--push-dy", type=float, default=0.00)
    parser.add_argument("--push-dz", type=float, default=0.00)
    return parser.parse_args()


def joint_bias_dims(bias_dims: str, left_dim: int, right_dim: int) -> list[int]:
    if bias_dims.strip().lower() == "joints":
        dims = list(range(left_dim))
        dims += [left_dim + 1 + index for index in range(right_dim)]
        return dims
    return [int(value) for value in bias_dims.split(",") if value.strip() != ""]


def apply_action_bias(action: np.ndarray, dims: list[int], magnitude: float) -> np.ndarray:
    biased = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    for dim in dims:
        if 0 <= dim < biased.shape[0]:
            biased[dim] += magnitude
    return biased


def main() -> int:
    cli = parse_args()
    cli.output_dir = cli.output_dir.expanduser().resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    user_args, args = build_runtime_args(cli)
    task_env = class_decorator("handover_to_tray")
    model_client = build_policy_client(user_args)
    left_dim = int(user_args["left_arm_dim"])
    right_dim = int(user_args["right_arm_dim"])
    bias_dims = joint_bias_dims(cli.bias_dims, left_dim, right_dim)
    results: list[dict[str, Any]] = []

    print("\n" + "=" * 72)
    print("RoboTwin perturbation-rescue sweep")
    print(f"mode={cli.perturb_mode} magnitude={cli.bias_magnitude}")
    print(f"lead_in={cli.lead_in_steps} bias_duration={cli.bias_duration}")
    print("=" * 72 + "\n")

    try:
        for rollout_index in range(int(cli.episodes)):
            seed = int(cli.seed_start) + rollout_index
            task_env.setup_demo(now_ep_num=rollout_index, seed=seed, is_test=False, **args)
            task_env.set_instruction(PROMPT)
            if hasattr(task_env, "_update_render"):
                task_env._update_render()
            viewer = getattr(task_env, "viewer", None)
            if viewer is not None:
                viewer.render()
            prepare_policy_case(model_client, "handover_to_tray", seed, PROMPT, "joint")
            reset_policy(model_client)

            policy_steps = 0
            perturbation_started = False
            object_pushed = False

            print(
                f"\n\033[96m[ROLLOUT {rollout_index + 1}/{cli.episodes}] seed={seed}\033[0m"
            )

            while not is_episode_end(task_env) and policy_steps < (
                cli.lead_in_steps + cli.bias_duration
            ):
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

                    if policy_steps >= cli.lead_in_steps:
                        perturbation_started = True
                        if cli.perturb_mode == "action_bias":
                            flat_action = apply_action_bias(
                                flat_action, bias_dims, cli.bias_magnitude
                            )
                        elif cli.perturb_mode == "push_bar" and not object_pushed:
                            import sapien

                            bar = task_env.bar
                            pose = bar.get_pose()
                            new_pose = sapien.Pose(
                                np.asarray(pose.p)
                                + np.asarray([cli.push_dx, cli.push_dy, cli.push_dz]),
                                pose.q,
                            )
                            bar.set_pose(new_pose)
                            object_pushed = True
                            print("  [PERTURB] pushed bar", flush=True)

                    task_env.take_action(flat_action, action_type=action_type)
                    policy_steps += 1
                    if policy_steps >= cli.lead_in_steps + cli.bias_duration:
                        break
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

            print(
                "  [RESCUE] invoking scripted expert from the perturbed state",
                flush=True,
            )
            policy_ended_before_rescue = bool(is_episode_end(task_env))
            recovery = task_env.recover_from_current_state()
            joints_legal, joint_absmax = task_env.planned_joints_legal()
            result = {
                "rollout_index": rollout_index,
                "seed": int(seed),
                "perturb_mode": cli.perturb_mode,
                "bias_magnitude": float(cli.bias_magnitude),
                "lead_in_steps": int(cli.lead_in_steps),
                "bias_duration": int(cli.bias_duration),
                "policy_steps_before_rescue": int(policy_steps),
                "perturbation_started": bool(perturbation_started),
                "policy_reached_end_before_rescue": policy_ended_before_rescue,
                "expert_plan_success": bool(recovery.get("plan_success")),
                "expert_success": bool(recovery.get("success")),
                "joints_legal": bool(joints_legal),
                "joint_absmax": float(joint_absmax),
                "expert_recovery": recovery,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            results.append(result)
            print(
                f"  [RESULT] seed={seed} expert_success={result['expert_success']} "
                f"plan_success={result['expert_plan_success']}"
            )
            safe_close_env(task_env)
    except Exception:
        import traceback

        traceback.print_exc()
        safe_close_env(task_env, clear_cache=True)
        return 3
    finally:
        close_policy_client(model_client)

    report = {
        "mode": cli.perturb_mode,
        "episodes": len(results),
        "rescued": sum(1 for item in results if item["expert_success"]),
        "rescue_rate": (
            sum(1 for item in results if item["expert_success"]) / len(results)
            if results
            else 0.0
        ),
        "results": results,
    }
    report_path = cli.output_dir / f"perturbation_rescue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}")
    print(f"rescue_rate={report['rescue_rate']:.2f} ({report['rescued']}/{report['episodes']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
