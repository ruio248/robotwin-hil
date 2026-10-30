#!/usr/bin/env python3
"""Compare legacy 4x8 and shared-prefix K=32 on one saved HIL observation.

Run each mode in a separate short-lived policy process. The script does not
connect to or modify a live HIL policy server. Both modes use identical noise,
checkpoint, input transforms, and output transforms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import time

import numpy as np
import yaml


def _memory_stats() -> dict[str, int]:
    try:
        import jax

        return {key: int(value) for key, value in (jax.devices()[0].memory_stats() or {}).items()
                if isinstance(value, (int, np.integer))}
    except (AttributeError, NotImplementedError, TypeError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("legacy", "shared", "parity"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError(args.output)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    manifest = json.loads((args.candidate_cache / "manifest.json").read_text())
    if manifest.get("complete") is not True:
        raise ValueError("candidate cache is incomplete")
    frame = Path(manifest["observations"][0]["frame_path"])
    with frame.open("rb") as stream:
        observation = pickle.load(stream)  # trusted project-owned HIL cache

    import jax
    from XPolicyLab.policy.Pi_05_RobotTwin import model as adapter

    policy_adapter = adapter.Model(yaml.safe_load(args.config.read_text()))
    policy_adapter.robot_action_dim_info = None
    policy_adapter.update_obs({
        "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        "images": {key: observation["observation"][camera]["rgb"] for key, camera in zip(
            ("cam_high", "cam_left_wrist", "cam_right_wrist"),
            ("head_camera", "left_camera", "right_camera"), strict=True)},
        "instruction": "Pass the red bar from the left arm to the right arm and place it in the blue tray.",
    })
    single = adapter.slice_stacked_obs(policy_adapter.observation_window, 0)
    horizon = policy_adapter.policy._model.action_horizon
    action_dim = policy_adapter.policy._model.action_dim
    noise = np.asarray(jax.random.normal(jax.random.key(20261013), (32, horizon, action_dim)))

    if args.mode == "parity":
        comparisons = {}
        for count in (1, 8):
            legacy_input = single if count == 1 else adapter.stack_obs([single] * count)
            legacy = np.asarray(policy_adapter.policy.infer(legacy_input, noise=noise[:count])["actions"])
            if count == 1:
                legacy = legacy[None, ...]
            shared = np.asarray(policy_adapter.policy.infer_candidates(single, count, noise=noise[:count])["actions"])
            difference = np.abs(legacy - shared)
            comparisons[str(count)] = {
                "max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
                "shape": list(shared.shape),
            }
            print(json.dumps({"count": count, **comparisons[str(count)]}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump({"complete": True, "mode": "parity", "comparisons": comparisons}, stream, indent=2)
        return

    def run(count: int) -> np.ndarray:
        if args.mode == "shared":
            return np.asarray(policy_adapter.policy.infer_candidates(single, count, noise=noise[:count])["actions"])
        outputs = []
        for start in range(0, count, 8):
            size = min(8, count - start)
            batched = adapter.stack_obs([single] * size)
            outputs.append(np.asarray(policy_adapter.policy.infer(
                batched, noise=noise[start : start + size])["actions"]))
        return np.concatenate(outputs)

    rows = []
    reference = None
    for iteration in range(args.repeats + 1):
        start = time.perf_counter()
        chunks = run(32)
        duration = time.perf_counter() - start
        if chunks.shape != (32, 10, 14) or not np.isfinite(chunks).all():
            raise AssertionError(f"invalid output shape or values: {chunks.shape}")
        if reference is None:
            reference = chunks.copy()
        row = {"warmup": iteration == 0, "seconds": duration, "memory_stats": _memory_stats()}
        rows.append(row)
        print(json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), actions=reference, noise=noise)
    result = {
        "complete": True,
        "mode": args.mode,
        "shape": list(reference.shape),
        "distinct_chunks": int(len(np.unique(reference.reshape(32, -1), axis=0))),
        "median_warm_seconds": statistics.median(row["seconds"] for row in rows if not row["warmup"]),
        "rows": rows,
        "frame": str(frame),
        "frame_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "jax_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "noise_seed": 20261013,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"complete": True, "mode": args.mode, "median_warm_seconds": result["median_warm_seconds"]}),
          flush=True)


if __name__ == "__main__":
    main()
