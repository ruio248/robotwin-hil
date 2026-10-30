#!/usr/bin/env python3
"""Benchmark the real Pi0.5 adapter on one saved observation, without simulation.

Run in the policy environment with an explicit GPU memory budget. Never sends
requests to a live HIL server; the temporary model exits after writing results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import time

import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--candidate-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 1:
        raise ValueError("Require at least one repeat")
    manifest = json.loads((args.candidate_cache / "manifest.json").read_text())
    if manifest.get("complete") is not True:
        raise ValueError("Need a complete saved observation cache")
    frame = Path(manifest["observations"][0]["frame_path"])
    with frame.open("rb") as f:
        obs = pickle.load(f)  # trusted, previously audited project collection
    from XPolicyLab.policy.Pi_05_RobotTwin import model as adapter

    model = adapter.Model(yaml.safe_load(args.config.read_text()))
    # The direct RGB/state encode path is identical to online inference. Disable
    # only the final robot-dictionary serialization so the audit can inspect the
    # physical [K,H,D] output directly; no action transform is changed.
    model.robot_action_dim_info = None
    model.update_obs({
        "state": np.asarray(obs["joint_action"]["vector"], dtype=np.float32),
        "images": {key: obs["observation"][camera]["rgb"] for key, camera in zip(
            ("cam_high", "cam_left_wrist", "cam_right_wrist"),
            ("head_camera", "left_camera", "right_camera"))},
        "instruction": "Pass the red bar from the left arm to the right arm and place it in the blue tray.",
    })
    infer = model.policy.infer
    calls = []
    def counted(inputs):
        calls.append(int(np.asarray(inputs["state"]).shape[0]))
        return infer(inputs)
    model.policy.infer = counted
    result = {"complete": False, "repeats": args.repeats,
              "frame": str(frame), "frame_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
              "adapter_sha256": hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
              "capability": model.candidate_sampling_info(), "rows": []}
    for batch_size in (4, 32):
        for iteration in range(args.repeats + 1):
            calls.clear()
            start = time.perf_counter()
            chunks = np.concatenate([model.get_action_candidates(batch_size)
                                     for _ in range(32 // batch_size)], axis=0)
            elapsed = time.perf_counter() - start  # NumPy conversion synchronizes JAX
            if chunks.shape != (32, 10, 14) or not np.isfinite(chunks).all():
                raise AssertionError(f"Invalid candidate output: {chunks.shape}")
            if calls != [batch_size] * (32 // batch_size):
                raise AssertionError(f"Wrong real model calls: {calls}")
            row = {"batch_size": batch_size, "warmup": iteration == 0, "seconds": elapsed,
                   "model_batch_sizes": list(calls), "shape": list(chunks.shape),
                   "distinct_chunks": int(len(np.unique(chunks.reshape(32, -1), axis=0)))}
            result["rows"].append(row)
            print(json.dumps(row), flush=True)
    result["median_seconds"] = {str(k): statistics.median(r["seconds"] for r in result["rows"]
        if r["batch_size"] == k and not r["warmup"]) for k in (4, 32)}
    result["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"complete": True, "median_seconds": result["median_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
