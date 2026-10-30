#!/usr/bin/env python3
"""Query the task's own H10 SFT checkpoint for K=4 exact-observation chunks."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def wait_server(proc: subprocess.Popen, port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(.5)
    raise TimeoutError(f"policy server did not become healthy on port {port}")


def wait_port_free(port: int, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(2)
    raise TimeoutError(f"port {port} remained occupied")


def prepare_client_path(openpi_root: Path, overlay: Path) -> None:
    sys.path[:0] = [
        str(overlay),
        str(openpi_root / "src"),
        str(openpi_root / "packages/openpi-client/src"),
    ]


def run(args) -> None:
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "params").is_dir() or not (checkpoint / "assets").is_dir():
        raise FileNotFoundError(f"incomplete SFT checkpoint: {checkpoint}")
    if not (args.config_overlay / "openpi/training/config.py").is_file():
        raise FileNotFoundError("SFT config overlay is incomplete")
    if args.candidates != 4 or args.horizon != 10:
        raise ValueError("fixed protocol requires K=4 and native H10")

    prepare_client_path(args.openpi_root, args.config_overlay)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    roots = [path.resolve() for path in args.episode_root]
    metas = []
    for root in roots:
        found = sorted(root.glob("episode_*_scene_*/metadata.json"))
        if not found:
            raise FileNotFoundError(f"no pair episodes under {root}")
        metas.extend(found)
    if len({str(path.parent) for path in metas}) != len(metas):
        raise ValueError("duplicate episode directory in candidate inputs")

    wait_port_free(args.port)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.xla_mem_fraction
    env["PYTHONPATH"] = os.pathsep.join([
        str(args.config_overlay), str(args.openpi_root / "src"),
        str(args.openpi_root / "packages/openpi-client/src"),
        str(args.pace_root / "src"), str(args.robotwin_root),
        env.get("PYTHONPATH", ""),
    ])
    command = [
        str(args.server_python), "-u", str(args.openpi_root / "scripts/serve_policy.py"),
        "--port", str(args.port), "policy:checkpoint",
        "--policy.config", args.config_name,
        "--policy.dir", str(checkpoint),
    ]
    state_path = args.state_file.resolve()
    server_log_path = args.server_log.resolve()
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = server_log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(command, cwd=args.openpi_root, env=env,
                              stdout=log_stream, stderr=subprocess.STDOUT,
                              start_new_session=True)
    state = {
        "status": "server_starting", "task": args.task,
        "checkpoint": str(checkpoint), "checkpoint_params_manifest_sha256":
            sha256(checkpoint / "params/manifest.ocdbt"),
        "config_name": args.config_name, "horizon": 10, "candidate_count": 4,
        "episode_roots": [str(root) for root in roots],
        "server_pid": server.pid, "server_command": command,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "training_checkpoint_source": "own SFT 5000-step checkpoint",
    }
    atomic_json(state_path, state)
    try:
        wait_server(server, args.port, args.startup_timeout)
        policy = WebsocketClientPolicy("127.0.0.1", args.port)
        server_metadata = policy.get_server_metadata()
        # Some deployed OpenPI servers do not expose optional model metadata
        # and return {} here. Do not mistake that API-version difference for a
        # checkpoint/config mismatch: every actual inference below is still
        # required to return the exact native H10x14 shape.
        if server_metadata and (
            server_metadata.get("action_horizon") not in (None, 10)
            or server_metadata.get("action_dim") not in (None, 14)
        ):
            raise ValueError(f"own SFT server metadata conflicts with H10x14: {server_metadata}")
        if server_metadata.get("config") not in (None, args.config_name):
            raise ValueError(f"server loaded wrong config: {server_metadata}")
        state.update(status="augmenting", server_metadata=server_metadata,
                     server_metadata_available=bool(server_metadata),
                     action_shape_verified_per_inference=True)
        atomic_json(state_path, state)

        completed = []
        total_anchors = 0
        for episode_number, meta_path in enumerate(metas, 1):
            folder = meta_path.parent
            meta = json.loads(meta_path.read_text())
            pair_path = folder / "pairs.npz"
            candidate_path = folder / "policy_candidates_k4.npz"
            with np.load(pair_path, allow_pickle=False) as archive:
                source = {key: archive[key] for key in archive.files}
            n = len(source["state"])
            if source["state"].shape != (n, 14) or source["expert_chunks"].shape != (n, 10, 14):
                raise ValueError(f"invalid state/expert shapes at {folder}")
            if any(source[key].shape[0] != n or source[key].dtype != np.uint8
                   for key in ("images_head", "images_left", "images_right")):
                raise ValueError(f"image/anchor mismatch at {folder}")
            if meta.get("expert_success") is not True or meta.get("complete_pairs") != n:
                raise ValueError(f"incomplete expert source episode {meta_path}")
            if meta.get("instruction") is None:
                raise ValueError(f"missing fixed prompt in {meta_path}")

            all_candidates = np.empty((n, 4, 10, 14), dtype=np.float32)
            for anchor in range(n):
                observation = {
                    "state": source["state"][anchor],
                    "images": {
                        "cam_high": source["images_head"][anchor].transpose(2, 0, 1),
                        "cam_left_wrist": source["images_left"][anchor].transpose(2, 0, 1),
                        "cam_right_wrist": source["images_right"][anchor].transpose(2, 0, 1),
                    },
                    "prompt": meta["instruction"],
                }
                for candidate in range(4):
                    response = policy.infer(observation)
                    actions = np.asarray(response["actions"], dtype=np.float32)
                    if actions.shape != (10, 14):
                        raise ValueError(f"expected absolute H10x14 response, got {actions.shape}")
                    if not np.isfinite(actions).all():
                        raise ValueError(f"nonfinite action candidate at {folder} anchor {anchor}")
                    all_candidates[anchor, candidate] = actions
            atomic_npz(candidate_path, policy_chunks=all_candidates)
            atomic_json(folder / "policy_candidates_k4.json", {
                "task": args.task, "scene_seed": int(meta["scene_seed"]),
                "checkpoint": str(checkpoint), "checkpoint_params_manifest_sha256":
                    state["checkpoint_params_manifest_sha256"],
                "same_observation": True, "candidate_count": 4,
                "action_shape": [4, 10, 14],
                "action_semantics": "own 5k SFT model output after configured AbsoluteActions inverse transform",
                "source_frequency_hz": 15, "prompt": meta["instruction"],
                "server_metadata": server_metadata,
            })
            completed.append(str(candidate_path))
            total_anchors += n
            state.update(completed_episodes=len(completed), total_episodes=len(metas),
                         completed_anchors=total_anchors, last_episode=folder.name)
            atomic_json(state_path, state)
            print(f"AUGMENTED {args.task} {episode_number}/{len(metas)} "
                  f"seed={meta['scene_seed']} anchors={n}", flush=True)

        state.update(status="complete", completed_episodes=len(completed),
                     completed_anchors=total_anchors,
                     complete_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(state_path, state)
        raise
    finally:
        with contextlib.suppress(Exception):
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=25)
        if server.poll() is None:
            with contextlib.suppress(Exception):
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        log_stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("turn_switch", "dump_bin_bigbin"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--config-overlay", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--pace-root", required=True, type=Path)
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--server-python", required=True, type=Path)
    parser.add_argument("--episode-root", required=True, type=Path, action="append")
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--xla-mem-fraction", default="0.55")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
