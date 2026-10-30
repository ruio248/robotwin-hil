#!/usr/bin/env python3
"""Add K stochastic policy chunks queried at each stored expert observation."""
from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def add_openpi_client_path(openpi_root: Path) -> None:
    client_path = openpi_root / "packages/openpi-client/src"
    sys.path.insert(0, str(client_path))


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def wait_server(proc, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(.25)
    raise TimeoutError(f"policy server did not start on {port}")


def atomic_npz(path, **arrays):
    import tempfile

    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


def ensure_stage_annotations(args) -> None:
    pair_root = args.input_root.resolve()
    experiment_root = next(
        (parent for parent in (pair_root, *pair_root.parents)
         if (parent / "pace/configs/expert_scenes.json").is_file()),
        None,
    )
    if experiment_root is None:
        raise FileNotFoundError(f"cannot locate PACE experiment root from {pair_root}")
    control = experiment_root / "control/goal_d_pair_pipeline_20261026"
    helper = control / "coverage_benchmark/annotate_pair_stages.py"
    collector = control / "collect_matched_pairs_v2.py"
    if not helper.is_file() or not collector.is_file():
        raise FileNotFoundError(f"stage annotation tools missing under {control}")
    env = dict(os.environ)
    paths = [experiment_root / "pace/src", experiment_root,
             experiment_root / "description/utils", control]
    env["PYTHONPATH"] = os.pathsep.join([*(str(path) for path in paths), env.get("PYTHONPATH", "")])
    command = [sys.executable, "-u", str(helper), "--task", args.task,
               "--task-root", str(pair_root), "--pace-root", str(experiment_root / "pace"),
               "--robotwin-root", str(experiment_root), "--collector", str(collector)]
    print(json.dumps({"event": "verify_expert_subplan_stages", "task": args.task,
                      "task_root": str(pair_root)}), flush=True)
    subprocess.run(command, cwd=experiment_root, env=env, check=True)


def run(args):
    # This runs before starting the policy server, while the postprocess queue
    # has an idle-GPU slot.  Older collections had only stage_id=0; replay
    # verifies those labels against the original qpos/chunks before D training.
    ensure_stage_annotations(args)
    add_openpi_client_path(args.openpi_root)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    root = args.input_root.resolve()
    checkpoint = (args.checkpoint_root / args.task).resolve()
    episodes = sorted(root.glob("episode_*_scene_*/metadata.json"))
    if not episodes:
        raise FileNotFoundError(f"no paired episodes in {root}")
    # The prior task's websocket server may have just closed on the same port.
    # SO_REUSEADDR handles its TCP TIME_WAIT sockets; a genuinely live listener
    # remains an error.  Bound the retry so we never silently share a port.
    deadline = time.monotonic() + 120
    while True:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", args.port))
            except OSError as error:
                if error.errno != errno.EADDRINUSE or time.monotonic() >= deadline:
                    raise
            else:
                break
        time.sleep(5)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    command = [
        str(args.server_python), "-u", str(args.openpi_root / "scripts/serve_policy.py"),
        "--port", str(args.port), "policy:checkpoint", "--policy.config",
        f"pi05_robotwin_{args.task}", "--policy.dir", str(checkpoint),
    ]
    server_log = (root / "candidate_server.log").open("w")
    server = subprocess.Popen(
        command, cwd=args.openpi_root, env=env, stdout=server_log,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    state_path = root / "candidate_augmentation_state.json"
    state = {"status": "server_starting", "task": args.task, "candidate_count": args.candidates,
             "server_pid": server.pid, "server_command": command, "episodes": len(episodes),
             "started_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(state_path, state)
    try:
        wait_server(server, args.port, args.startup_timeout)
        policy = WebsocketClientPolicy("127.0.0.1", args.port)
        server_meta = policy.get_server_metadata()
        expected = {"task": args.task, "config": f"pi05_robotwin_{args.task}",
                    "action_horizon": 50, "action_dim": 14, "action_space": "absolute_joint"}
        if any(server_meta.get(k) != v for k, v in expected.items()):
            raise ValueError(f"server metadata mismatch: {server_meta}")
        state.update(status="augmenting", server_metadata=server_meta)
        atomic_json(state_path, state)

        completed = []
        for index, meta_path in enumerate(episodes, 1):
            episode_dir = meta_path.parent
            source_path = episode_dir / "pairs.npz"
            out_path = episode_dir / f"policy_candidates_k{args.candidates}.npz"
            with np.load(source_path, allow_pickle=False) as archive:
                src = {key: archive[key] for key in archive.files}
            if src["policy_chunks"].shape != (len(src["state"]), 10, 14):
                raise ValueError(f"unexpected original policy chunk shape in {source_path}")
            meta = json.loads(meta_path.read_text())
            if meta["task"] != args.task or meta["chunk_horizon"] != 10:
                raise ValueError(f"episode metadata mismatch: {meta_path}")
            if out_path.exists():
                with np.load(out_path, allow_pickle=False) as archive:
                    cached = archive["policy_chunks"]
                if cached.shape != (len(src["state"]), args.candidates, 10, 14):
                    raise ValueError(f"existing candidate file has wrong shape: {out_path}")
                completed.append(str(out_path))
                continue

            all_candidates = np.empty((len(src["state"]), args.candidates, 10, 14), dtype=np.float32)
            all_candidates[:, 0] = src["policy_chunks"]
            for row in range(len(src["state"])):
                obs = {
                    "state": src["state"][row],
                    "images": {
                        "cam_high": src["images_head"][row].transpose(2, 0, 1),
                        "cam_left_wrist": src["images_left"][row].transpose(2, 0, 1),
                        "cam_right_wrist": src["images_right"][row].transpose(2, 0, 1),
                    },
                    "prompt": meta["instruction"],
                }
                for candidate in range(1, args.candidates):
                    response = policy.infer(obs)
                    actions = np.asarray(response["actions"], dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 10:
                        raise ValueError(f"bad policy output shape {actions.shape}")
                    chunk = actions[:10]
                    if not np.isfinite(chunk).all():
                        raise ValueError("nonfinite policy chunk")
                    all_candidates[row, candidate] = chunk
            if not np.isfinite(all_candidates).all():
                raise ValueError(f"nonfinite candidate array for {episode_dir}")
            atomic_npz(out_path, policy_chunks=all_candidates)
            augmentation_meta = {
                "task": args.task,
                "scene_seed": meta["scene_seed"],
                "observations": len(src["state"]),
                "candidate_count": args.candidates,
                "candidate_0": "first ten actions from the original exact-observation policy request",
                "candidates_1_to_K": "fresh stochastic policy requests on the identical stored RGB/state/prompt",
                "same_observation": True,
                "action_shape": [args.candidates, 10, 14],
                "server_metadata": server_meta,
            }
            atomic_json(episode_dir / f"policy_candidates_k{args.candidates}.json", augmentation_meta)
            completed.append(str(out_path))
            print(f"AUGMENTED {index}/{len(episodes)} seed={meta['scene_seed']} obs={len(src['state'])}", flush=True)
            state.update(completed_episodes=len(completed), completed_files=completed)
            atomic_json(state_path, state)

        state.update(status="complete", completed_episodes=len(completed), completed_files=completed,
                     complete_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(state_path, state)
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--input-root", required=True, type=Path)
    p.add_argument("--checkpoint-root", required=True, type=Path)
    p.add_argument("--openpi-root", required=True, type=Path)
    p.add_argument("--server-python", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--port", type=int, default=18489)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--startup-timeout", type=int, default=600)
    args = p.parse_args()
    if args.candidates < 2:
        p.error("candidate_count must be at least two")
    run(args)


if __name__ == "__main__":
    main()
