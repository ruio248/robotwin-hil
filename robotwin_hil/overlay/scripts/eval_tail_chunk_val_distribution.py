#!/usr/bin/env python3
"""Score every expert and cached policy chunk in the frozen 45-episode critic val split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tail_value.cache import atomic_json, atomic_npz, json_digest, sha256
from tail_value.chunk import ChunkCoverageCritic, ChunkTransitionTable


def describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Empty or nonfinite score distribution")
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
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    entries = manifest["episodes"]
    val_entries = [entry for entry in entries if entry["split"] == "val"]
    if (manifest.get("complete") is not True or len(entries) != 450
            or len(val_entries) != 45 or sum(e["split"] == "train" for e in entries) != 405
            or manifest["config"].get("num_candidates") != 4
            or manifest["config"].get("horizon") != 10):
        raise ValueError("Expected complete 405/45 SFT chunk cache with K=4, H=10")
    fingerprint = json_digest({"config": manifest["config"], "episodes": entries})
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (checkpoint.get("cache_fingerprint") != fingerprint
            or checkpoint.get("representation") != "absolute_joint_chunk"
            or checkpoint["config"].get("horizon") != 10
            or checkpoint["step"] != 10000):
        raise ValueError("Checkpoint does not match the formal 45-episode validation cache")
    table = ChunkTransitionTable(args.cache_dir, manifest, "val")
    episode_id, frame_index = [], []
    for entry in val_entries:
        with np.load(args.cache_dir / entry["file"], allow_pickle=False) as archive:
            frames = np.asarray(archive["frame_index"][:-1], dtype=np.int32)
        if len(frames) != entry["transitions"]:
            raise ValueError(f"Transition count mismatch for {entry['id']}")
        episode_id.extend([int(entry["id"].split("_")[-1])] * len(frames))
        frame_index.extend(frames.tolist())
    if len(episode_id) != len(table):
        raise ValueError("Episode metadata does not align with validation transitions")

    model = ChunkCoverageCritic(checkpoint["normalization"],
                                width=checkpoint["config"]["width"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    device = torch.device(args.device)
    model.to(device)
    expert_parts, policy_parts = [], []
    with torch.inference_mode():
        for start in range(0, len(table), args.batch_size):
            batch = table.batch(slice(start, start + args.batch_size), device)
            expert_parts.append(model(batch["features"], batch["state"],
                                      batch["expert_chunks"]).cpu().numpy())
            policy_parts.append(model.score_many(batch["features"], batch["state"],
                                                 batch["policy_chunks"]).cpu().numpy())
    expert = np.concatenate(expert_parts).astype(np.float64)
    policy = np.concatenate(policy_parts).astype(np.float64)
    if (expert.shape != (len(table),) or policy.shape != (len(table), 4)
            or not np.isfinite(expert).all() or not np.isfinite(policy).all()):
        raise ValueError("Invalid expert/policy score arrays")
    gap = expert - policy.mean(axis=1)

    if args.training_log:
        records = [json.loads(line) for line in args.training_log.read_text().splitlines() if line]
        reported = [r for r in records if r.get("event") == "val" and r.get("step") == 10000]
        if len(reported) != 1:
            raise ValueError("Missing unique step-10000 validation record")
        for name, actual in (("q_expert", expert.mean()), ("q_policy", policy.mean()),
                             ("expert_minus_policy", gap.mean())):
            if abs(reported[0][name]["mean"] - actual) > .01:
                raise ValueError(f"Re-scored {name} does not match training validation log")

    episode_id = np.asarray(episode_id, dtype=np.int32)
    per_episode_mean = [gap[episode_id == i].mean() for i in np.unique(episode_id)]
    per_episode_median = [np.median(gap[episode_id == i]) for i in np.unique(episode_id)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(args.output_dir / "val_candidate_b_scores.npz", {
        "expert_b": expert,
        "policy_b": policy,
        "episode_id": episode_id,
        "frame_index": np.asarray(frame_index, dtype=np.int32),
    })
    atomic_json(args.output_dir / "summary.json", {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "cache_dir": str(args.cache_dir), "cache_fingerprint": fingerprint,
        "checkpoint_step": checkpoint["step"], "val_episodes": len(val_entries),
        "val_transitions": len(table), "candidates_per_observation": 4,
        "expert_b": describe(expert), "policy_candidate_b": describe(policy),
        "policy_mean_b_per_observation": describe(policy.mean(axis=1)),
        "expert_minus_policy_mean_b": describe(gap),
        "episode_mean_gap": describe(per_episode_mean),
        "episode_median_gap": describe(per_episode_median),
        "fraction_expert_above_policy_mean": float(np.mean(gap > 0)),
        "fraction_expert_above_individual_candidates": float(np.mean(expert[:, None] > policy)),
        "fraction_expert_above_all_four_candidates": float(np.mean(expert > policy.max(axis=1))),
        "fraction_expert_below_all_four_candidates": float(np.mean(expert < policy.min(axis=1))),
        "fraction_abs_gap_le_0p01": float(np.mean(np.abs(gap) <= .01)),
        "fraction_abs_gap_le_0p05": float(np.mean(np.abs(gap) <= .05)),
        "policy_candidate_range": describe(np.ptp(policy, axis=1)),
        "observations_gap_gt_1": int(np.sum(gap > 1)),
        "observations_gap_gt_10": int(np.sum(gap > 10)),
        "observations_gap_lt_minus_1": int(np.sum(gap < -1)),
        "definition": "Same-observation frozen critic scores on 45 SFT episodes withheld from critic training; one expert chunk and four cached policy chunks per valid 10-frame transition",
        "limitations": ["These 45 SFT episodes were withheld from critic training, not necessarily from the SFT policy",
                        "Expert chunks are next-recorded-joint-target approximations, not proven executed controls",
                        "B is not a calibrated success probability or coverage ground truth"],
    })
    print(f"complete: {len(val_entries)} val episodes, {len(table)} transitions, "
          f"{policy.size} policy candidate scores -> {args.output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
