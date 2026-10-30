#!/usr/bin/env python3
"""Collect expert/policy action chunks paired at identical expert observations.

The collector rolls out RoboTwin's scripted expert, samples the simulator at
50 Hz, and asks the task-specific policy for a chunk at a subset of those exact
states/images.  Expert labels are the next ``horizon`` joint-position samples;
policy labels are the first ``horizon`` actions returned for the same request.
This is deliberately a data-collection utility, not a DAgger intervention
runner and not a claim that policy-visited states are covered.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_MISSING_GRASP_POSE_ASSERTION = "target_pose cannot be None for move action."


def _is_rejectable_expert_pose_error(error: AssertionError) -> bool:
    """Recognize only the task expert's explicit missing-grasp-pose failure."""
    if str(error) != _MISSING_GRASP_POSE_ASSERTION:
        return False
    frames = traceback.extract_tb(error.__traceback__)
    return any(frame.name == "grasp_actor" and frame.filename.endswith("_base_task.py") for frame in frames)


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _check_port(port: int) -> None:
    with socket.socket() as sock:
        # Permit immediate sequential reuse after the previous policy server
        # has exited (TIME_WAIT); an active listener still makes bind fail.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def _wait_server(proc: subprocess.Popen, port: int, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.25)
    raise TimeoutError(f"policy server did not become ready on port {port}")


class _StepTap:
    """Proxy a SAPIEN scene and observe its state after each physics step."""

    def __init__(self, scene: Any, callback):
        self._scene = scene
        self._callback = callback

    def step(self, *args, **kwargs):
        result = self._scene.step(*args, **kwargs)
        self._callback()
        return result

    def __getattr__(self, name):
        return getattr(self._scene, name)


def _qpos(env) -> np.ndarray:
    left = np.asarray(env.robot.get_left_arm_jointState(), dtype=np.float32)
    right = np.asarray(env.robot.get_right_arm_jointState(), dtype=np.float32)
    state = np.concatenate([left, right])
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"expected finite 14D joint state, got {state.shape}")
    return state


def _collect_expert_trace(env, scene: dict, env_args: dict, args, episode_index: int):
    from pace.robotwin import restore_cached_scene_state, sample_instruction

    states: list[np.ndarray] = []
    anchors: list[dict[str, Any]] = []
    counters = {"physics_steps": 0, "sampled_steps": 0}
    capture_images = bool(getattr(args, "capture_images", True))

    def on_physics_step():
        counters["physics_steps"] += 1
        if counters["physics_steps"] % args.physics_stride:
            return
        state = _qpos(env)
        states.append(state)
        counters["sampled_steps"] += 1
        if (counters["sampled_steps"] - 1) % args.anchor_stride == 0:
            if capture_images:
                observation = env.get_obs()
                vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
                if not np.allclose(vector, state, atol=1e-5, rtol=1e-5):
                    raise ValueError("captured RGB observation and tapped joint state disagree")
                cameras = observation["observation"]
                images = []
                for camera_name in ("head_camera", "left_camera", "right_camera"):
                    rgb = np.asarray(cameras[camera_name]["rgb"])
                    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                        raise ValueError(f"invalid {camera_name} RGB: {rgb.dtype} {rgb.shape}")
                    images.append(rgb.copy())
                # Prefer the stage instrumentation around the expert script's
                # env.move call sites.  RoboTwin task_metadata.stage_id often
                # remains 0 even though the wrapper has advanced the stage;
                # reading it first silently discarded the call-site ID while
                # stage_name (below) was already correctly populated.
                stage_id = int(getattr(
                    env, "current_stage_id",
                    observation.get("task_metadata", {}).get("stage_id", 0),
                ))
            else:
                images = None
                stage_id = int(getattr(env, "current_stage_id", 0))
            anchors.append({
                "state_index": len(states) - 1,
                "state": state.copy(),
                "images_hwc": images,
                "sim_step": counters["physics_steps"],
                "stage_id": stage_id,
                "stage_name": getattr(env, "current_stage_name", "unlabeled"),
            })

    env.scene = _StepTap(env.scene, on_physics_step)
    restore_cached_scene_state(scene["task"], env, scene["episode_info"])
    # RoboTwin's base get_obs exposes current_stage_id, but many native expert
    # scripts never update it.  Tag each explicit expert subplan call site so
    # stage-wise metrics are not silently collapsed into stage 0.  Line-based
    # IDs are stable across scenes of one task and distinguish repeated loop
    # calls from distinct scripted subplans.
    original_move = env.move
    stage_labels: dict[str, str] = {}

    def move_with_stage(*move_args, **move_kwargs):
        caller = sys._getframe(1)
        line = int(caller.f_lineno)
        source = Path(caller.f_code.co_filename).name
        stage_id = line
        stage_labels[str(stage_id)] = f"{source}:{line}"
        env.current_stage_id = stage_id
        env.current_stage_name = stage_labels[str(stage_id)]
        return original_move(*move_args, **move_kwargs)

    env.current_stage_id = 0
    env.current_stage_name = "pre_subplan"
    env.move = move_with_stage
    try:
        info = env.play_once()
    finally:
        env.move = original_move
    prompt = sample_instruction(scene["task"], info, 1, "unseen")
    env.set_instruction(prompt)
    success = bool(env.plan_success and env.check_success())
    return states, anchors, counters, prompt, info, success, stage_labels


def collect(args) -> None:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from pace.robotwin import load_environment, load_cached_scenes

    task = args.task
    checkpoint = Path(args.checkpoint_root) / task
    if not (checkpoint / "params").is_dir() or not (
        checkpoint / "assets/robotwin/norm_stats.json"
    ).is_file():
        raise FileNotFoundError(f"incomplete task checkpoint: {checkpoint}")

    scenes, cache_meta = load_cached_scenes(
        task, args.scene_seed, args.scene_offset + args.episodes
    )
    scenes = scenes[args.scene_offset : args.scene_offset + args.episodes]
    out = Path(args.output).resolve()
    state_path = out / "state.json"
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"cannot resume without state file: {state_path}")
        run_meta = json.loads(state_path.read_text())
        expected_resume = {
            "task": task,
            "task_config": "demo_clean",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_id": args.checkpoint_id,
            "scene_cache": cache_meta,
            "scene_seed_split": args.scene_seed,
            "scene_offset": args.scene_offset,
            "expected_episodes": args.target_episodes,
            "candidate_scenes": args.episodes,
            "physics_stride": args.physics_stride,
            "anchor_stride": args.anchor_stride,
            "expert_chunk_horizon": args.horizon,
            "strictly_same_observation": True,
        }
        mismatches = {
            key: (run_meta.get(key), value)
            for key, value in expected_resume.items()
            if run_meta.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume metadata mismatch: {mismatches}")
        if run_meta.get("status") not in {"failed", "insufficient_successful_expert_episodes"}:
            raise ValueError(f"refusing resume from status={run_meta.get('status')!r}")
        start_episode_index = int(run_meta.get("attempted_scenes", 0))
        if not 0 <= start_episode_index <= len(scenes):
            raise ValueError(f"invalid resume scene index {start_episode_index}/{len(scenes)}")
        summaries = list(run_meta.get("episode_summaries", []))
        rejected_scenes = list(run_meta.get("rejected_scenes", []))
        prior_error = run_meta.pop("error", None)
        if prior_error:
            run_meta.setdefault("prior_errors", []).append(prior_error)
        run_meta["resume_count"] = int(run_meta.get("resume_count", 0)) + 1
        run_meta["last_resumed_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["status"] = "resuming"
        _json(state_path, run_meta)
    else:
        out.mkdir(parents=True, exist_ok=False)
        run_meta = {
            "status": "starting",
            "task": task,
            "task_config": "demo_clean",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_id": args.checkpoint_id,
            "scene_cache": cache_meta,
            "scene_seed_split": args.scene_seed,
            "scene_offset": args.scene_offset,
            "expected_episodes": args.target_episodes,
            "candidate_scenes": args.episodes,
            "physics_hz": 250,
            "sample_hz": 50,
            "physics_stride": args.physics_stride,
            "anchor_stride": args.anchor_stride,
            "expert_chunk_horizon": args.horizon,
            "policy_chunk_source": "first_horizon_rows_of_same-observation_pi05_chunk",
            "pair_semantics": "expert_next_50Hz_joint_samples_vs_policy_first_H_absolute_joint_actions",
            "stage_label_scheme": "expert_play_once_move_callsite_source_line_v1",
            "strictly_same_observation": True,
            "policy_visited_states_included": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status_note": "Expert-state support pairs only; not a DAgger-visited-state sample.",
        }
        start_episode_index = 0
        summaries = []
        rejected_scenes = []
        _json(state_path, run_meta)

    _check_port(args.port)
    env_vars = dict(os.environ)
    env_vars["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env_vars.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")
    py_path = os.pathsep.join([
        str(Path(args.openpi_root) / "src"),
        str(Path(args.pace_root) / "src"),
        str(Path(args.openpi_root) / "packages/openpi-client/src"),
        env_vars.get("PYTHONPATH", ""),
    ])
    env_vars["PYTHONPATH"] = py_path
    server_command = [
        # Preserve the venv's ``bin/python`` path.  Resolving its symlink to the
        # bare interpreter would discard the venv site-packages (and broke tyro).
        str(args.server_python), "-u",
        str(Path(args.openpi_root) / "scripts/serve_policy.py"),
        "--port", str(args.port), "policy:checkpoint",
        "--policy.config", f"pi05_robotwin_{task}",
        "--policy.dir", str(checkpoint.resolve()),
    ]
    server_log_path = out / "server.log"
    server_log = server_log_path.open("a" if args.resume else "w")
    server = subprocess.Popen(
        server_command, cwd=args.openpi_root, env=env_vars,
        stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    run_meta.update(status="server_starting", server_pid=server.pid, server_command=server_command)
    _json(state_path, run_meta)
    try:
        _wait_server(server, args.port, args.startup_timeout)
        policy = WebsocketClientPolicy("127.0.0.1", args.port)
        server_meta = policy.get_server_metadata()
        expected = {
            "task": task,
            "config": f"pi05_robotwin_{task}",
            "action_horizon": 50,
            "action_dim": 14,
            "action_space": "absolute_joint",
        }
        if any(server_meta.get(k) != v for k, v in expected.items()):
            raise ValueError(f"server metadata mismatch: got {server_meta}, expected {expected}")
        run_meta.update(status="collecting", server_metadata=server_meta)
        _json(state_path, run_meta)

        for episode_index, scene in enumerate(scenes[start_episode_index:], start=start_episode_index):
            env, env_args = load_environment(task)
            try:
                env.setup_demo(
                    now_ep_num=episode_index,
                    seed=int(scene["scene_seed"]),
                    is_test=True,
                    **env_args,
                )
                # The tap is installed only after scene construction.  Its proxy
                # forwards all read/render/contact APIs to the original SAPIEN scene.
                scene_for_trace = {**scene, "task": task}
                states, anchors, counters, prompt, info, success, stage_labels = _collect_expert_trace(
                    env, scene_for_trace, env_args, args, episode_index
                )
                if not success:
                    rejected_scenes.append({
                        "scene_seed": int(scene["scene_seed"]),
                        "reason": "expert_rollout_did_not_pass_original_success_check",
                        "plan_success": bool(env.plan_success),
                        "check_success": bool(env.check_success()),
                        "sampled_states": len(states),
                    })
                    print(
                        f"REJECT_EXPERT_SCENE {len(rejected_scenes)} seed={scene['scene_seed']} "
                        f"plan_success={env.plan_success} check_success={env.check_success()}",
                        flush=True,
                    )
                    run_meta.update(
                        attempted_scenes=episode_index + 1,
                        rejected_scenes=rejected_scenes,
                        completed_episodes=len(summaries),
                        total_pairs=sum(row["pairs"] for row in summaries),
                        episode_summaries=summaries,
                    )
                    _json(state_path, run_meta)
                    continue
                pairs = []
                for anchor_index, anchor in enumerate(anchors):
                    start = anchor["state_index"] + 1
                    stop = start + args.horizon
                    if stop > len(states):
                        continue
                    expert_chunk = np.stack(states[start:stop]).astype(np.float32)
                    if expert_chunk.shape != (args.horizon, 14):
                        raise ValueError(f"bad expert action chunk shape {expert_chunk.shape}")
                    images_hwc = anchor["images_hwc"]
                    obs = {
                        "state": anchor["state"],
                        "images": {
                            "cam_high": images_hwc[0].transpose(2, 0, 1),
                            "cam_left_wrist": images_hwc[1].transpose(2, 0, 1),
                            "cam_right_wrist": images_hwc[2].transpose(2, 0, 1),
                        },
                        "prompt": prompt,
                    }
                    response = policy.infer(obs)
                    policy_chunk = np.asarray(response["actions"], dtype=np.float32)
                    if policy_chunk.ndim != 2 or policy_chunk.shape[1] != 14 or len(policy_chunk) < args.horizon:
                        raise ValueError(f"bad policy chunk shape {policy_chunk.shape}")
                    policy_chunk = policy_chunk[: args.horizon]
                    if not np.isfinite(policy_chunk).all():
                        raise ValueError("policy chunk contains NaN/Inf")
                    pairs.append({
                        "anchor_index": anchor_index,
                        "sim_step": anchor["sim_step"],
                        "sampled_state_index": anchor["state_index"],
                        "stage_id": anchor["stage_id"],
                        "stage_name": anchor["stage_name"],
                        "images_hwc": images_hwc,
                        "state": anchor["state"],
                        "expert_chunk": expert_chunk,
                        "policy_chunk": policy_chunk,
                        "prompt": prompt,
                    })

                if not pairs:
                    raise RuntimeError("expert rollout produced no complete paired action chunks")

                episode_dir = out / f"episode_{episode_index:03d}_scene_{int(scene['scene_seed'])}"
                episode_dir.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(
                    episode_dir / "pairs.npz",
                    state=np.stack([p["state"] for p in pairs]).astype(np.float32),
                    images_head=np.stack([p["images_hwc"][0] for p in pairs]).astype(np.uint8),
                    images_left=np.stack([p["images_hwc"][1] for p in pairs]).astype(np.uint8),
                    images_right=np.stack([p["images_hwc"][2] for p in pairs]).astype(np.uint8),
                    expert_chunks=np.stack([p["expert_chunk"] for p in pairs]).astype(np.float32),
                    policy_chunks=np.stack([p["policy_chunk"] for p in pairs]).astype(np.float32),
                    sim_step=np.asarray([p["sim_step"] for p in pairs], dtype=np.int64),
                    sampled_state_index=np.asarray([p["sampled_state_index"] for p in pairs], dtype=np.int64),
                    stage_id=np.asarray([p["stage_id"] for p in pairs], dtype=np.int64),
                    stage_name=np.asarray([p["stage_name"] for p in pairs], dtype=np.str_),
                )
                episode_meta = {
                    "task": task,
                    "scene_seed": int(scene["scene_seed"]),
                    "expert_success": success,
                    "expert_plan_success": bool(env.plan_success),
                    "expert_anchor_observations": len(anchors),
                    "complete_pairs": len(pairs),
                    "sampled_50hz_states": len(states),
                    "physics_steps": counters["physics_steps"],
                    "instruction": prompt,
                    "expert_scene_info": info.get("info", {}),
                    "chunk_horizon": args.horizon,
                    "action_dim": 14,
                    "action_semantics": "absolute_joint qpos",
                    "stage_label_scheme": "expert_play_once_move_callsite_source_line_v1",
                    "stage_labels": stage_labels,
                    "pair_observation_check": "captured state allclose to exact sampled joint vector; policy queried on stored RGB/state",
                }
                _json(episode_dir / "metadata.json", episode_meta)
                summaries.append({
                    "scene_seed": int(scene["scene_seed"]),
                    "pairs": len(pairs),
                    "expert_success": success,
                    "episode_dir": str(episode_dir),
                })
                print(
                    f"PAIR_EPISODE {episode_index + 1}/{len(scenes)} seed={scene['scene_seed']} "
                    f"states50hz={len(states)} anchors={len(anchors)} pairs={len(pairs)} success={success}",
                    flush=True,
                )
                run_meta.update(
                    attempted_scenes=episode_index + 1,
                    rejected_scenes=rejected_scenes,
                    completed_episodes=len(summaries),
                    total_pairs=sum(row["pairs"] for row in summaries),
                    episode_summaries=summaries,
                )
                _json(state_path, run_meta)
                if len(summaries) >= args.target_episodes:
                    break
            except AssertionError as error:
                if not _is_rejectable_expert_pose_error(error):
                    raise
                rejected_scenes.append({
                    "scene_seed": int(scene["scene_seed"]),
                    "episode_info": scene.get("episode_info", {}),
                    "reason": "expert_grasp_pose_missing",
                    "exception": f"{type(error).__name__}: {error}",
                    "traceback": "".join(traceback.format_exception(error)),
                    "counted_as_policy_failure": False,
                })
                print(
                    f"REJECT_EXPERT_EXCEPTION seed={scene['scene_seed']} "
                    "reason=expert_grasp_pose_missing counted_as_policy_failure=False",
                    flush=True,
                )
                run_meta.update(
                    attempted_scenes=episode_index + 1,
                    rejected_scenes=rejected_scenes,
                    completed_episodes=len(summaries),
                    total_pairs=sum(row["pairs"] for row in summaries),
                    episode_summaries=summaries,
                )
                _json(state_path, run_meta)
                continue
            finally:
                with contextlib.suppress(Exception):
                    env.close_env()

        total_pairs = sum(row["pairs"] for row in summaries)
        if len(summaries) < args.target_episodes:
            run_meta.update(status="insufficient_successful_expert_episodes",
                            rejected_scenes=rejected_scenes,
                            completed_episodes=len(summaries), total_pairs=total_pairs)
            _json(state_path, run_meta)
            raise RuntimeError(
                f"Only {len(summaries)}/{args.target_episodes} successful expert episodes "
                f"from {len(scenes)} candidate scenes"
            )
        run_meta.update(
            status="complete",
            completed_episodes=len(summaries),
            attempted_scenes=episode_index + 1,
            rejected_scenes=rejected_scenes,
            total_pairs=total_pairs,
            episode_summaries=summaries,
            complete_utc=datetime.now(timezone.utc).isoformat(),
        )
        _json(state_path, run_meta)
        _json(out / "summary.json", {
            "complete": True,
            "task": task,
            "checkpoint_id": args.checkpoint_id,
            "episodes": len(summaries),
            "candidate_scenes": run_meta["candidate_scenes"],
            "attempted_scenes": run_meta["attempted_scenes"],
            "rejected_scenes": rejected_scenes,
            "pairs": total_pairs,
            "scene_seeds": [row["scene_seed"] for row in summaries],
            "expert_successes": sum(row["expert_success"] for row in summaries),
            "pair_semantics": run_meta["pair_semantics"],
            "server_metadata": server_meta,
            "state_file": str(state_path),
        })
    except BaseException as error:
        run_meta.update(status="failed", error=f"{type(error).__name__}: {error}")
        _json(state_path, run_meta)
        raise
    finally:
        with contextlib.suppress(Exception):
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=20)
        if server.poll() is None:
            with contextlib.suppress(Exception):
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        server_log.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint-root", required=True, type=Path)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pace-root", required=True, type=Path)
    p.add_argument("--robotwin-root", required=True, type=Path)
    p.add_argument("--openpi-root", required=True, type=Path)
    p.add_argument("--server-python", required=True, type=Path)
    p.add_argument("--scene-seed", type=int, default=0)
    p.add_argument("--scene-offset", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--target-episodes", type=int, default=None,
                    help="number of successful expert episodes to keep; unsuccessful expert scenes are logged and skipped")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--port", type=int, default=18489)
    p.add_argument("--startup-timeout", type=int, default=600)
    p.add_argument("--physics-stride", type=int, default=5)
    p.add_argument("--anchor-stride", type=int, default=5)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--resume", action="store_true",
                   help="resume an existing failed/insufficient run from its first unattempted cached scene")
    args = p.parse_args()
    if args.target_episodes is None:
        args.target_episodes = args.episodes
    if (args.episodes <= 0 or args.target_episodes <= 0 or args.target_episodes > args.episodes
            or args.scene_offset < 0 or args.scene_offset + args.episodes > 100
            or args.horizon != 50 or args.physics_stride != 5 or args.anchor_stride <= 0):
        p.error("H50 collection requires positive episodes, H=50, physics_stride=5, and positive anchor_stride")
    args.pace_root = args.pace_root.resolve()
    args.robotwin_root = args.robotwin_root.resolve()
    args.openpi_root = args.openpi_root.resolve()
    for root in (args.pace_root, args.robotwin_root, args.openpi_root):
        if not root.is_dir():
            p.error(f"missing directory: {root}")
    sys.path.insert(0, str(args.pace_root / "src"))
    sys.path.insert(0, str(args.robotwin_root))
    sys.path.insert(0, str(args.robotwin_root / "description/utils"))
    sys.path.insert(0, str(args.openpi_root / "packages/openpi-client/src"))
    return args


def main():
    args = parse_args()
    os.chdir(args.robotwin_root)
    os.environ.setdefault("MUJOCO_GL", "egl")
    collect(args)


if __name__ == "__main__":
    main()
