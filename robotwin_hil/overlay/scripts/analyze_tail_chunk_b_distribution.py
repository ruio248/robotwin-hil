#!/usr/bin/env python3
"""Re-score saved HIL candidate chunks and summarize every candidate B value.

No policy call or environment rollout is made.  K=32 contains the nested
K=4/8/16 candidates, so each of the 32 scores is counted only once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_tail_chunk_hil import score_row
from tail_value.cache import atomic_json, atomic_npz, sha256
from tail_value.chunk import ChunkCoverageCritic
from tail_value.prepare import FrozenEncoder


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Empty or nonfinite B distribution")
    return {"count": int(values.size), "mean": float(values.mean()),
            "std": float(values.std()), "min": float(values.min()),
            "p01": float(np.quantile(values, .01)),
            "p05": float(np.quantile(values, .05)),
            "p25": float(np.quantile(values, .25)),
            "median": float(np.median(values)),
            "p75": float(np.quantile(values, .75)),
            "p95": float(np.quantile(values, .95)),
            "p99": float(np.quantile(values, .99)),
            "max": float(values.max())}


def run(args):
    torch.set_num_threads(args.cpu_threads)
    reference = json.loads(args.reference_metrics.read_text())
    if sha256(args.checkpoint) != reference["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the prior HIL evaluation")
    prior_rows = json.loads(args.reference_rows.read_text())
    sources = json.loads((args.cache_dir / "per_observation_metrics.json").read_text())
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    by_cache = {source["cache"]: source for source in sources}
    if len(by_cache) != len(sources) or manifest.get("complete") is not True:
        raise ValueError("Candidate cache is incomplete or contains duplicate source keys")
    if len(prior_rows) != 1233:
        raise ValueError("Expected 1233 previously evaluated HIL observations")
    if len({row["cache"] for row in prior_rows}) != len(prior_rows):
        raise ValueError("Duplicate prior observation keys")
    selected = prior_rows[:args.observation_limit] if args.observation_limit else prior_rows
    if any(row["cache"] not in by_cache for row in selected):
        raise ValueError("Prior observations not present in the candidate cache")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (checkpoint.get("representation") != "absolute_joint_chunk"
            or checkpoint["config"].get("horizon") != 10
            or checkpoint["step"] != reference["step"]):
        raise ValueError("Wrong chunk-aware B checkpoint")
    model = ChunkCoverageCritic(checkpoint["normalization"],
                                width=checkpoint["config"]["width"])
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval().requires_grad_(False)
    encoder = FrozenEncoder(device, args.encoder_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "partial_scores.json"
    completed = []
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text())
        if partial.get("checkpoint_sha256") != reference["checkpoint_sha256"]:
            raise ValueError("Partial scores use a different checkpoint")
        completed = partial["observations"]
        if len(completed) > len(selected) or any(
            row["cache"] != selected[index]["cache"] for index, row in enumerate(completed)
        ):
            raise ValueError("Partial scores do not match the selected observations")
    for index in range(len(completed), len(selected)):
        previous = selected[index]
        scored = score_row(model, encoder, by_cache[previous["cache"]],
                           args.cache_dir, manifest, device, reference["beta"],
                           include_candidate_scores=True)
        for key in ("rollout_index", "policy_step", "cache"):
            if scored[key] != previous[key]:
                raise ValueError(f"Observation mismatch at index {index}: {key}")
        for key in ("b_mean", "b_range"):
            if abs(scored["k"]["32"][key] - previous["k"]["32"][key]) > 1e-4:
                raise ValueError(f"Changed B score at observation {index}: {key}")
        completed.append({"cache": scored["cache"], "rollout_index": scored["rollout_index"],
                          "policy_step": scored["policy_step"],
                          "scores": scored["candidate_b_scores"]})
        if len(completed) % 25 == 0 or len(completed) == len(selected):
            atomic_json(partial_path, {"checkpoint_sha256": reference["checkpoint_sha256"],
                                       "observations": completed})
            print(f"scored {len(completed)}/{len(selected)}", flush=True)

    scores = np.asarray([row["scores"] for row in completed], dtype=np.float64)
    if scores.shape != (len(selected), 32) or not np.isfinite(scores).all():
        raise ValueError("Expected a finite [observations,32] score matrix")
    centered = scores - scores.mean(axis=1, keepdims=True)
    atomic_npz(args.output_dir / "candidate_b_scores.npz", {
        "scores": scores,
        "rollout_index": np.asarray([row["rollout_index"] for row in completed], dtype=np.int32),
        "policy_step": np.asarray([row["policy_step"] for row in completed], dtype=np.int32),
        "cache": np.asarray([row["cache"] for row in completed]),
    })
    atomic_json(args.output_dir / "summary.json", {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "candidate_cache": str(args.cache_dir),
        "observations": len(completed),
        "trajectories": len({row["rollout_index"] for row in completed}),
        "candidates_per_observation": 32,
        "raw_b": distribution(scores),
        "centered_b": distribution(centered),
        "absolute_centered_b": distribution(np.abs(centered)),
        "per_observation_range": distribution(np.ptp(scores, axis=1)),
        "definition": "K=32 policy candidate B scores; centered B is B minus the mean of the 32 scores at the same observation; each observation contributes 32 values exactly once",
        "limitations": ["Saved pre-first-takeover HIL observations, not all rollout frames",
                        "Offline candidate scores, not executed outcomes or coverage ground truth",
                        "Pooled candidates weight longer trajectories more heavily"],
    })
    print(f"complete: {scores.shape} -> {args.output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "cache-dir", "reference-metrics", "reference-rows",
                 "encoder-weights", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--observation-limit", type=int, default=None,
                        help="Use only for a separate smoke-test output directory")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
