#!/usr/bin/env python3
"""Train/evaluate fixed-head coverage models on encoder-ablation caches.

This is an internal SFT diagnostic, not execution success or independent
heldout testing. The 15 calibration and 30 diagnostic episodes are drawn from
the same 45 SFT validation episodes for every encoder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tail_value.cache import atomic_json, sha256
from tail_value.chunk import ChunkCoverageCritic
from coverage_benchmark.common import calibrate, weights
from coverage_benchmark.data import sft_data, subset
from coverage_benchmark.fit import export_neural, train_discriminator
from coverage_benchmark.scorers import normalization


def score_arrays(checkpoint: Path, data: dict, device: str, batch_size: int = 256):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["representation"] != "absolute_joint_chunk" or payload["config"]["horizon"] != 10:
        raise ValueError("Not a full-chunk coverage checkpoint")
    model = ChunkCoverageCritic(payload["normalization"], 10, 256)
    model.load_state_dict(payload["model"])
    model.eval().requires_grad_(False).to(device)
    expert, policy = [], []
    with torch.inference_mode():
        for start in range(0, len(data["state"]), batch_size):
            stop = start + batch_size
            f = torch.as_tensor(data["features"][start:stop], dtype=torch.float32, device=device)
            s = torch.as_tensor(data["state"][start:stop], dtype=torch.float32, device=device)
            e = torch.as_tensor(data["expert_chunks"][start:stop], dtype=torch.float32, device=device)
            p = torch.as_tensor(data["policy_chunks"][start:stop], dtype=torch.float32, device=device)
            expert.append(model(f, s, e).cpu().numpy())
            policy.append(model.score_many(f, s, p).cpu().numpy())
    return np.concatenate(expert).astype(np.float64), np.concatenate(policy).astype(np.float64)


def episode_rank(expert: np.ndarray, policy: np.ndarray, episode: np.ndarray, seed: int = 42):
    percentile = ((expert[:, None] > policy).sum(1) + .5 * (expert[:, None] == policy).sum(1)) / policy.shape[1]
    ids = np.unique(episode)
    means = np.asarray([percentile[episode == identifier].mean() for identifier in ids])
    draws = means[np.random.default_rng(seed).integers(len(ids), size=(10000, len(ids)))].mean(1)
    return {"episode_balanced_mean": float(means.mean()),
            "episode_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist(),
            "observation_weighted_mean": float(percentile.mean()),
            "strict_top_fraction": float(np.mean(np.all(expert[:, None] > policy, axis=1))),
            "episodes": {str(int(i)): float(v) for i, v in zip(ids, means)}}


def evaluate(args):
    data, provenance = sft_data(args.cache_dir)
    cal, diag = subset(data, "cal"), subset(data, "diag")
    if len(cal["state"]) != 430 or len(diag["state"]) != 844:
        raise ValueError("Internal calibration/diagnostic split changed")
    results = {"scope": "45 SFT validation episodes split 15/30; internal analysis, not independent heldout",
               "encoder_cache_manifest_sha256": sha256(args.cache_dir / "manifest.json"),
               "calibration_anchors": len(cal["state"]), "diagnostic_anchors": len(diag["state"]),
               "source_split": provenance["calibration_episodes"], "models": {}}
    for name, checkpoint in (("td20k", args.td_checkpoint), ("discriminator10k", args.disc_checkpoint)):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        ecal, pcal = score_arrays(checkpoint, cal, args.device)
        calibration = calibrate(ecal, pcal, beta_max=100.)
        e, p = score_arrays(checkpoint, diag, args.device)
        w = np.stack([weights(row, calibration) for row in p])
        ess = 1 / np.sum(w * w, axis=1) / p.shape[1]
        shifted = np.einsum("nk,nkhd->nhd", w, diag["policy_chunks"]) - diag["policy_chunks"].mean(axis=1)
        rms = np.sqrt(np.mean(shifted[:, :, np.r_[0:6, 7:13]] ** 2, axis=(1, 2)))
        results["models"][name] = {
            "checkpoint_sha256": sha256(checkpoint),
            "rank": episode_rank(e, p, diag["episode"]),
            "calibration": calibration,
            "diagnostic": {"expert_mean": float(e.mean()), "policy_mean": float(p.mean()),
                           "score_range_median": float(np.median(np.ptp(p, axis=1))),
                           "ess_fraction_median": float(np.median(ess)),
                           "tilted_mean_joint_rms_median": float(np.median(rms))},
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "metrics.json", results)
    atomic_json(args.output_dir / "manifest.json", {"complete": True,
        "cache_manifest_sha256": results["encoder_cache_manifest_sha256"],
        "metrics_sha256": sha256(args.output_dir / "metrics.json")})
    print(f"EVALUATED {args.output_dir / 'metrics.json'}", flush=True)


def train(args):
    data, _ = sft_data(args.cache_dir)
    train_data = subset(data, "train")
    if len(train_data["state"]) != 11461:
        raise ValueError("Train anchor count changed")
    norm = normalization(train_data)
    last = train_discriminator(train_data, norm, args.output_dir, args.seed, 10000)
    export_neural(last, args.output_dir / "inference.pt")
    atomic_json(args.output_dir / "manifest.json", {"complete": True, "step": 10000,
        "seed": args.seed, "source_cache_manifest_sha256": sha256(args.cache_dir / "manifest.json"),
        "inference_sha256": sha256(args.output_dir / "inference.pt")})
    print(f"TRAINED {args.output_dir / 'inference.pt'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "evaluate"))
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--td-checkpoint", type=Path)
    p.add_argument("--disc-checkpoint", type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.stage == "train":
        train(args)
    else:
        if args.td_checkpoint is None or args.disc_checkpoint is None:
            p.error("evaluation requires --td-checkpoint and --disc-checkpoint")
        evaluate(args)


if __name__ == "__main__":
    main()
