#!/usr/bin/env python3
"""Add three independent H50 proposals to each saved executed-policy query.

Candidate zero is the *actual* policy proposal saved during the rollout.  The
other three are fresh calls on its exact stored RGB, state, and prompt.  This
does not create observations between H50 execution boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from collect_matched_pairs_h50 import _check_port, _wait_server


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def read_query(folder: Path, index: int, prompt: str) -> tuple[dict, np.ndarray]:
    stem = f"query_{index:04d}"
    with np.load(folder / f"{stem}.npz", allow_pickle=False) as saved:
        state = saved["state"].astype(np.float32)
        original = saved["policy_h50"].astype(np.float32)
    if state.shape != (14,) or original.shape != (50, 14):
        raise ValueError(f"invalid H50 trace {folder}/{stem}")
    images = {}
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        with Image.open(folder / f"{stem}_{camera}.png") as image:
            images[camera] = np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1).copy()
    return {"state": state, "images": images, "prompt": prompt}, original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evaluation = args.eval_root / "evaluation"
    meta = json.loads((evaluation / "metadata.json").read_text())
    records = [json.loads(line) for line in (evaluation / "episodes.jsonl").read_text().splitlines()]
    if meta.get("status") != "complete" or meta.get("task") != args.task or meta.get("method") != "fixed_h50":
        raise ValueError("wrong or incomplete rollout protocol")
    if len(records) != 50 or len({int(row["scene_seed"]) for row in records}) != 50:
        raise ValueError("expected 50 unique held-out scenes")
    if any(row.get("error") or row.get("infra_error") for row in records):
        raise ValueError("rollout contains infrastructure errors")
    if not 1 <= args.limit <= 50:
        raise ValueError("limit must be 1..50")
    _check_port(args.port)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    checkpoint = args.checkpoint_root / args.task
    command = [str(args.server_python), "-u", str(args.openpi_root / "scripts/serve_policy.py"),
               "--port", str(args.port), "policy:checkpoint",
               "--policy.config", f"pi05_robotwin_{args.task}", "--policy.dir", str(checkpoint)]
    args.output.mkdir(parents=True, exist_ok=False)
    log = (args.output / "server.log").open("w")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(args.openpi_root / "src"), str(args.openpi_root / "pace/src"),
        str(args.openpi_root / "packages/openpi-client/src"), env.get("PYTHONPATH", "")])
    server = subprocess.Popen(command, cwd=args.openpi_root, env=env,
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    files = {}
    try:
        _wait_server(server, args.port, args.startup_timeout)
        policy = WebsocketClientPolicy("127.0.0.1", args.port)
        server_meta = policy.get_server_metadata()
        expected = {"task": args.task, "config": f"pi05_robotwin_{args.task}",
                    "action_horizon": 50, "action_dim": 14, "action_space": "absolute_joint"}
        if any(server_meta.get(key) != value for key, value in expected.items()):
            raise ValueError(f"server metadata mismatch: {server_meta}")
        for item, record in enumerate(records[:args.limit], 1):
            folder = evaluation / "trace" / f"episode_{record['episode']:03d}"
            trace = json.loads((folder / "trace.json").read_text())
            if not trace.get("complete") or int(trace["scene_seed"]) != int(record["scene_seed"]):
                raise ValueError(f"trace mismatch: {folder}")
            candidates = []
            for index, query in enumerate(trace["queries"]):
                if query["query"] != index or query["planned_chunk_length"] != 50:
                    raise ValueError(f"query index/horizon mismatch: {folder}/{index}")
                observation, original = read_query(folder, index, query["prompt"])
                variants = [original]
                for _ in range(3):
                    proposal = np.asarray(policy.infer(observation)["actions"], dtype=np.float32)
                    if proposal.shape != (50, 14) or not np.isfinite(proposal).all():
                        raise ValueError(f"bad fresh candidate {folder}/{index}: {proposal.shape}")
                    variants.append(proposal)
                candidates.append(np.stack(variants))
            array = np.stack(candidates).astype(np.float32)
            if array.shape != (int(record["queries"]), 4, 50, 14):
                raise ValueError(f"bad K4 shape {array.shape}: {folder}")
            target = args.output / f"episode_{record['episode']:03d}_scene_{record['scene_seed']}.npz"
            atomic_npz(target, candidates=array)
            files[target.name] = {"sha256": digest(target), "queries": len(candidates),
                                  "scene_seed": int(record["scene_seed"]),
                                  "success": bool(record["success"])}
            print(f"K4 {item}/{args.limit} scene={record['scene_seed']} queries={len(candidates)}", flush=True)
        atomic_json(args.output / "manifest.json", {
            "complete": True, "task": args.task, "episodes": args.limit,
            "candidate_count": 4, "action_horizon": 50,
            "observation_schedule": "executed fixed-H50 policy queries, not every saved H10-era frame",
            "candidate_0": "original executed-policy H50 prediction",
            "candidates_1_to_3": "fresh independent policy calls on identical stored RGB/state/prompt",
            "server_metadata": server_meta, "trace_metadata_sha256": digest(evaluation / "metadata.json"),
            "files": files,
        })
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=20)
        log.close()


if __name__ == "__main__":
    main()
