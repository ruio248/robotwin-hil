"""Offline DART noise-scale pilot on *proxy* SFT expert targets.

This does not measure success, recovery, or true DART data quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dart_covariance import ARM_INDICES, fit_covariance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--shrinkage", type=float, default=0.0)
    args = parser.parse_args()
    with np.load(args.pairs, allow_pickle=False) as source:
        expert = source["expert_intended_actions"][:, ARM_INDICES]
        learner = source["learner_actions"][:, ARM_INDICES]
        offsets = source["episode_offsets"]
    if len(offsets) - 1 != 405 or len(expert) != 11461:
        raise ValueError("unexpected proxy cohort")
    rng = np.random.default_rng(args.seed)
    episode_order = rng.permutation(len(offsets) - 1)
    train_episodes = set(episode_order[:324].tolist())
    train_indices = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in sorted(train_episodes)])
    test_indices = np.concatenate([np.arange(offsets[i], offsets[i + 1])
                                   for i in range(len(offsets) - 1) if i not in train_episodes])
    train_residual = learner[train_indices] - expert[train_indices]
    test_residual = learner[test_indices] - expert[test_indices]
    covariance = fit_covariance(learner[train_indices], expert[train_indices],
                                shrinkage=args.shrinkage)
    ref_rms = np.sqrt(np.mean(test_residual ** 2, axis=1))
    results = []
    for scale in args.scales:
        if scale <= 0:
            raise ValueError("scales must be positive")
        scaled_covariance = scale * covariance
        noise = rng.multivariate_normal(np.zeros(12), scaled_covariance,
                                        size=len(test_indices), check_valid="raise")
        rms = np.sqrt(np.mean(noise ** 2, axis=1))
        results.append({
            "covariance_scale": scale,
            "trace_rad2": float(np.trace(scaled_covariance)),
            "per_joint_rms_rad": float(np.sqrt(np.trace(scaled_covariance) / 12)),
            "simulated_noise_rms_p50_rad": float(np.median(rms)),
            "simulated_noise_rms_p95_rad": float(np.quantile(rms, .95)),
            "noise_p95_over_proxy_residual_p95": float(np.quantile(rms, .95) / np.quantile(ref_rms, .95)),
        })
    report = {
        "status": "proxy_offline_only_not_true_dart_validation",
        "train_episodes": len(train_episodes),
        "test_episodes": len(offsets) - 1 - len(train_episodes),
        "train_anchors": len(train_indices),
        "test_anchors": len(test_indices),
        "source_fps": 15,
        "proxy_residual_rms_p50_rad": float(np.median(ref_rms)),
        "proxy_residual_rms_p95_rad": float(np.quantile(ref_rms, .95)),
        "covariance_trace_rad2": float(np.trace(covariance)),
        "covariance_eigenvalues_rad2": np.linalg.eigvalsh(covariance).tolist(),
        "scales": results,
        "limitation": "next-frame qpos proxy; no noisy-supervisor rollout, clean intended command, or success measurement",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "trace_rad2": report["covariance_trace_rad2"],
                      "train_anchors": len(train_indices), "test_anchors": len(test_indices)}))


if __name__ == "__main__":
    main()
