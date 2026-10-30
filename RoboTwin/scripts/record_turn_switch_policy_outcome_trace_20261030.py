#!/usr/bin/env python3
"""Record diagnostic-only RGB/state/policy-H10 traces for turn_switch rollouts.

This deliberately mirrors the established three-task policy evaluator. Traces
are held out for outcome analysis and are never accepted as training data.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))
if str(ROBOTWIN_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT / "scripts"))

from eval_policy_xpolicylab import (  # noqa: E402
    build_policy_client, class_decorator, close_policy_client, is_episode_end,
    load_task_args, normalize_action_chunk, prepare_policy_case, reset_policy,
    robotwin_obs_to_xpolicylab, safe_close_env,
    xpolicylab_action_to_robotwin,
)

TASK = "turn_switch"
PROMPT = "Turn the switch to its final position."
CAMERAS = (("head_camera", "cam_high"), ("left_camera", "cam_left_wrist"),
           ("right_camera", "cam_right_wrist"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def capture_query(trace_dir: Path, index: int, observation: dict,
                  action_response: Any, policy_steps: int) -> dict:
    state = np.asarray(observation.get("joint_action", {}).get("vector"), dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"expected finite 14D joint_action.vector, got {state.shape}")
    converted = [xpolicylab_action_to_robotwin(
        action, action_type="joint", current_observation=observation
    ) for action in normalize_action_chunk(action_response)]
    if len(converted) != 10 or any(kind != "qpos" or value.shape != (14,)
                                   or not np.isfinite(value).all()
                                   for value, kind in converted):
        shapes = [(np.asarray(value).shape, kind) for value, kind in converted]
        raise ValueError(f"expected a finite absolute-joint H10x14 response, got {shapes}")
    action = np.stack([value for value, _ in converted]).astype(np.float32)

    trace_dir.mkdir(parents=True, exist_ok=True)
    file_stem = f"query_{index:04d}"
    npz_tmp = trace_dir / f"{file_stem}.npz.tmp"
    with npz_tmp.open("wb") as stream:
        np.savez_compressed(stream, state=state, policy_h10=action)
    npz_tmp.replace(trace_dir / f"{file_stem}.npz")

    cameras = observation.get("observation", {})
    for raw_name, saved_name in CAMERAS:
        camera = cameras.get(raw_name, {})
        rgb = np.asarray(camera.get("rgb"))
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"invalid RGB for {raw_name}: {rgb.dtype} {rgb.shape}")
        image_tmp = trace_dir / f"{file_stem}_{saved_name}.png.tmp"
        Image.fromarray(rgb, mode="RGB").save(image_tmp, format="PNG")
        image_tmp.replace(trace_dir / f"{file_stem}_{saved_name}.png")
    return {"query": index, "planned_chunk_length": 10,
            "policy_steps_before_query": int(policy_steps)}


def runtime_args(cli: argparse.Namespace):
    user_args = {
        "task_name": TASK, "task_config": "demo_clean", "policy_name": "Pi_05_RobotTwin",
        "host": cli.host, "port": cli.port, "protocol": "ws",
        "xpolicylab_root": str(ROBOTWIN_ROOT / "XPolicyLab"),
        "ckpt_name": "turn_switch_proprio_5000_20261019",
        "ckpt_setting": "turn_switch_proprio_5000_20261019",
        "action_type": "joint", "evaluation_id": "turn-switch-d-outcome-20261030",
        "trial_id": "fresh-outcome-rollouts", "action_case_id": "turn-switch-d-outcome",
    }
    args, _ = load_task_args(user_args)
    user_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    user_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    args.update({"eval_instruction": "seen", "eval_mode": True,
                 "render_freq": 0, "need_plan": True, "save_data": False,
                 "save_video": False, "save_path": str(cli.output_dir), "save_freq": 15})
    args["render_max_num_materials"] = 128
    args["render_max_num_textures"] = 512
    return user_args, args


def run_episode(env, client, args, cli, episode_index: int, seed: int) -> dict:
    env.setup_demo(now_ep_num=episode_index, seed=seed, is_test=False, **args)
    env.step_lim = cli.step_limit
    env.set_instruction(PROMPT)
    if hasattr(env, "_update_render"):
        env._update_render()
    viewer = getattr(env, "viewer", None)
    if viewer is not None:
        viewer.render()
    prepare_policy_case(client, TASK, seed, PROMPT, "joint")
    reset_policy(client)

    env.save_data = False
    env.save_video = False
    env.save_dir = str(cli.output_dir)
    env.FRAME_IDX = 0
    env.folder_path = {}
    env.ep_num = int(episode_index)
    env.control_mask = []
    env.current_control_source = "policy"

    trace_dir = cli.output_dir / "evaluation" / "trace" / f"episode_{episode_index:03d}"
    if trace_dir.exists() and any(trace_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing trace: {trace_dir}")
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace = {"schema": "robotwin_policy_h10_outcome_trace_v1", "task": TASK,
             "episode": episode_index, "scene_seed": seed, "success": None,
             "complete": False, "diagnostic_only": True,
             "accepted_for_training": False, "planned_chunk_length": 10,
             "policy_checkpoint": "turn_switch state=True 5k, manifest 503dcd0e...96c0",
             "task_config": "demo_clean", "frequency_hz": cli.frequency,
             "prompt": PROMPT, "queries": []}
    atomic_json(trace_dir / "trace.json", trace)
    policy_steps = 0
    query_index = 0
    while not is_episode_end(env):
        observation = env.get_obs()
        request = robotwin_obs_to_xpolicylab(
            observation, instruction=PROMPT, env_idx=0,
            frequency=cli.frequency, task_env=env,
        )
        client.call(func_name="update_obs", obs=request)
        response = client.call(func_name="get_action")
        query = capture_query(trace_dir, query_index, observation, response, policy_steps)
        trace["queries"].append(query)
        action_chunk = normalize_action_chunk(response)
        if len(action_chunk) != 10:
            raise ValueError(f"policy chunk changed after capture: {len(action_chunk)}")
        query_index += 1
        atomic_json(trace_dir / "trace.json", trace)

        for action in action_chunk:
            flat_action, action_type = xpolicylab_action_to_robotwin(
                action, action_type="joint", current_observation=observation,
            )
            if action_type != "qpos" or flat_action.shape != (14,):
                raise ValueError(f"invalid execution action {action_type}/{flat_action.shape}")
            env.current_control_source = "policy"
            env.take_action(flat_action, action_type=action_type)
            policy_steps += 1
            if is_episode_end(env):
                break
            observation = env.get_obs()
            request = robotwin_obs_to_xpolicylab(
                observation, instruction=PROMPT, env_idx=0,
                frequency=cli.frequency, task_env=env,
            )
            client.call(func_name="update_obs", obs=request)

    success = bool(env.eval_success)
    result = {"episode": episode_index, "scene_seed": seed, "seed": seed,
              "success": success, "steps": int(policy_steps),
              "policy_steps": int(policy_steps), "queries": query_index,
              "infra_error": False, "diagnostic_only": True,
              "accepted_for_training": False,
              "final_check_success": bool(env.check_success()),
              "current_stage_id": int(getattr(env, "current_stage_id", -1))}
    trace.update({"success": success, "steps": int(policy_steps), "complete": True,
                 "final_check_success": result["final_check_success"]})
    atomic_json(trace_dir / "trace.json", trace)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18492)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--step-limit", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, required=True)
    cli = parser.parse_args()
    if cli.episodes < 1 or cli.frequency != 30 or cli.step_limit != 400:
        parser.error("locked protocol requires episodes>=1, frequency=30, step-limit=400")
    cli.output_dir = cli.output_dir.expanduser().resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = cli.output_dir / "eval_records.jsonl"
    if records_path.exists() and records_path.stat().st_size:
        raise FileExistsError(f"refusing to append/overwrite existing results: {records_path}")

    user_args, args = runtime_args(cli)
    env = class_decorator(TASK)
    client = build_policy_client(user_args)
    records = []
    start = datetime.now(timezone.utc).isoformat()
    atomic_json(cli.output_dir / "evaluation" / "metadata.json", {
        "status": "running", "task": TASK, "method": "own_pi05_h10_trace",
        "seed_start": cli.seed_start, "episodes_requested": cli.episodes,
        "frequency_hz": cli.frequency, "step_limit": cli.step_limit,
        "diagnostic_only": True, "accepted_for_training": False,
        "started_at": start,
    })
    try:
        for index in range(cli.episodes):
            seed = cli.seed_start + index
            record = run_episode(env, client, args, cli, index, seed)
            records.append(record)
            append_jsonl(cli.output_dir / "evaluation" / "episodes.jsonl", record)
            append_jsonl(records_path, record)
            atomic_json(cli.output_dir / "summary.json", {
                "task": TASK, "complete": False, "episodes": len(records),
                "successes": sum(bool(row["success"]) for row in records),
                "records": records,
            })
            print(f"[TRACE-EVAL] episode={index} seed={seed} success={record['success']} "
                  f"steps={record['steps']} queries={record['queries']}", flush=True)
            safe_close_env(env)
    except Exception:
        import traceback
        traceback.print_exc()
        safe_close_env(env, clear_cache=True)
        return 3
    finally:
        close_policy_client(client)

    success_count = sum(bool(row["success"]) for row in records)
    summary = {"task": TASK, "complete": True, "episodes": len(records),
               "successes": success_count, "success_rate": success_count / len(records),
               "seed_start": cli.seed_start, "seed_end": cli.seed_start + cli.episodes - 1,
               "records": records, "diagnostic_only": True,
               "accepted_for_training": False}
    atomic_json(cli.output_dir / "summary.json", summary)
    atomic_json(cli.output_dir / "evaluation" / "metadata.json", {
        "status": "complete", "task": TASK, "method": "own_pi05_h10_trace",
        "seed_start": cli.seed_start, "seed_end": cli.seed_start + cli.episodes - 1,
        "episodes": len(records), "successes": success_count,
        "success_rate": success_count / len(records), "frequency_hz": cli.frequency,
        "step_limit": cli.step_limit, "diagnostic_only": True,
        "accepted_for_training": False, "started_at": start,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"COMPLETE {summary['successes']}/{summary['episodes']} -> {cli.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
