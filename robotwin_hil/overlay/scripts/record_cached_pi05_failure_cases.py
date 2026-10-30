#!/usr/bin/env python3
"""Rerun recorded failed PACE seed-0 scenes and save viewable failure videos.

These are fresh stochastic rollouts, not recovered recordings of the old actions.
The released scene cache, original instruction, task checkpoint, and fixed H=50
are reused. A rerun that succeeds is kept separately and never called a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--case-outcome", choices=("failure", "success"), default="failure")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=18485)
    parser.add_argument("--sample-every", type=int, default=5)
    return parser.parse_args()


def wait_ready(process, port, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("policy server exited before becoming ready")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, TimeoutError):
            time.sleep(0.5)
    raise TimeoutError("policy server readiness timeout")


def main():
    cli = arguments()
    target = cli.max_cases if cli.max_cases is not None else cli.max_failures
    want_success = cli.case_outcome == "success"
    if target < 1 or cli.sample_every < 1:
        raise ValueError("max-cases and sample-every must be positive")
    root = cli.root.resolve()
    cli.output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(root), str(root / "pace/src")]
    openpi = Path("/media/ruio/hdd/Openpi-PACE-eval-20261022")
    sys.path.insert(0, str(openpi / "packages/openpi-client/src"))
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from pace.robotwin import load_environment, restore_cached_scene_state
    from pace.rollout import run_episode

    def video_frame(observation):
        cameras = observation["observation"]
        return np.concatenate([
            np.asarray(cameras[name]["rgb"], dtype=np.uint8)
            for name in ("head_camera", "left_camera", "right_camera")
        ], axis=1)

    scenes = json.loads((root / "pace/configs/expert_scenes.json").read_text())["tasks"][cli.task]["0"]
    scenes_by_seed = {int(row["scene_seed"]): row for row in scenes}
    old_rows = [json.loads(line) for line in cli.records.read_text().splitlines() if line.strip()]
    old_cases = [row for row in old_rows if row.get("success") is want_success]
    if not old_cases:
        raise RuntimeError(f"no {cli.case_outcome} rows in original records")
    checkpoint = root / "checkpoints" / cli.task
    if not (checkpoint / "assets/robotwin/norm_stats.json").is_file():
        raise FileNotFoundError(f"checkpoint is incomplete: {checkpoint}")
    envvars = dict(os.environ)
    envvars.update(
        CUDA_VISIBLE_DEVICES=str(cli.gpu),
        OPENPI_DATA_HOME=str(root / "openpi_data"),
        HF_HOME=str(root / ".hf-home"),
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.65",
        PYTHONPATH=os.pathsep.join([
            str(openpi / "src"), str(openpi / "pace/src"),
            str(openpi / "packages/openpi-client/src"), envvars.get("PYTHONPATH", ""),
        ]),
    )
    command = [
        str(openpi / ".venv/bin/python"), "-u", str(openpi / "scripts/serve_policy.py"),
        "--port", str(cli.port), "policy:checkpoint", "--policy.config",
        f"pi05_robotwin_{cli.task}", "--policy.dir", str(checkpoint),
    ]
    (cli.output / "protocol.json").write_text(json.dumps({
        "task": cli.task, "original_records": str(cli.records),
        "checkpoint": str(checkpoint), "scene_source": "released seed0 cached scenes",
        "policy_horizon": 50, "execution_horizon": 50,
        "video_sample_every_steps": cli.sample_every,
        "video_views": ["head_camera", "left_camera", "right_camera"],
        "case_outcome": cli.case_outcome,
        "note": "Fresh stochastic rerun on the same scene and instruction; not the original action trace.",
        "server_command": command,
    }, indent=2) + "\n")
    rows_path = cli.output / "replay_records.jsonl"
    seen = set()
    if rows_path.exists():
        seen = {json.loads(line)["scene_seed"] for line in rows_path.read_text().splitlines() if line.strip()}
    confirmed = sum(json.loads(line)["rerun_success"] is want_success for line in rows_path.read_text().splitlines() if line.strip()) if rows_path.exists() else 0
    server_log = (cli.output / "server.log").open("a")
    server = subprocess.Popen(command, cwd=openpi, env=envvars, stdout=server_log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_ready(server, cli.port)
        policy = WebsocketClientPolicy("127.0.0.1", cli.port)
        metadata = policy.get_server_metadata()
        if metadata.get("task") != cli.task or metadata.get("action_horizon") != 50:
            raise RuntimeError(f"unexpected policy metadata: {metadata}")
        for source in old_cases:
            if confirmed >= target:
                break
            seed = int(source["scene_seed"])
            if seed in seen:
                continue
            if seed not in scenes_by_seed:
                raise RuntimeError(f"seed {seed} missing from released cache")
            scene = scenes_by_seed[seed]
            env, env_args = load_environment(cli.task)
            encoder = None
            video_file = cli.output / f"scene_{seed}_rerun.mp4"
            ffmpeg_log = (cli.output / f"scene_{seed}_ffmpeg.log").open("w")
            try:
                env.setup_demo(now_ep_num=int(source["episode"]), seed=seed,
                               is_test=True, **env_args)
                restore_cached_scene_state(cli.task, env, scene["episode_info"])
                env.set_instruction(source["instruction"])
                rgb = video_frame(env.get_obs())
                height, width = rgb.shape[:2]
                encoder = subprocess.Popen([
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                    "-r", "10", "-i", "-", "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "25", "-f", "mp4", str(video_file),
                ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=ffmpeg_log)
                encoder.stdin.write(np.ascontiguousarray(rgb).tobytes())
                original_take_action = env.take_action

                def take_and_record(action):
                    original_take_action(action)
                    if env.take_action_cnt % cli.sample_every == 0:
                        frame = video_frame(env.now_obs)
                        encoder.stdin.write(np.ascontiguousarray(frame).tobytes())

                env.take_action = take_and_record
                result = run_episode(env, policy, lambda actions: min(50, len(actions)))
            finally:
                if encoder is not None:
                    encoder.stdin.close()
                    if encoder.wait(timeout=30) != 0:
                        raise RuntimeError(f"ffmpeg failed for seed {seed}")
                ffmpeg_log.close()
                env.close_env(clear_cache=False)
            row = {
                "task": cli.task, "scene_seed": seed,
                "original_success": want_success, "rerun_success": bool(result["success"]),
                "steps": result["steps"], "queries": result["queries"],
                "instruction": source["instruction"],
                "video": video_file.name, "protocol": "cached_seed0_fixed_h50",
            }
            with rows_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            seen.add(seed)
            confirmed += bool(result["success"]) is want_success
            print(f"{cli.task} seed={seed} rerun_success={result['success']} "
                  f"confirmed_{cli.case_outcome}={confirmed}/{target}", flush=True)
        (cli.output / "summary.json").write_text(json.dumps({
            "task": cli.task, f"confirmed_{cli.case_outcome}_videos": confirmed,
            "target": target, "attempted": len(seen),
            "complete": confirmed >= target,
        }, indent=2) + "\n")
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        server_log.close()


if __name__ == "__main__":
    main()
