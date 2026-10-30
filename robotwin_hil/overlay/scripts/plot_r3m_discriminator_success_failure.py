#!/usr/bin/env python3
"""Plot retrospective R3M discriminator scores on labeled policy rollouts.

The first panel uses each full trajectory's normalized progress; the second
uses the 16 saved observations available for every episode. Neither panel is
a fresh success-rate evaluation, and full progress does not align task phase.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from coverage_encoder_discriminator_bigmetrics import load_model, score
from tail_value.cache import atomic_json, read_json, sha256


def load_episodes(args):
    reference = read_json(args.bigmetrics)
    source_hashes = reference["source_sha256"]
    if sha256(args.model) != source_hashes["model"]:
        raise ValueError("R3M discriminator checkpoint differs from published metrics")
    manifest_path = args.historical_cache / "manifest.json"
    if sha256(manifest_path) != source_hashes["historical_cache_manifest"]:
        raise ValueError("R3M historical cache differs from published metrics")
    manifest = read_json(manifest_path)
    if not manifest.get("complete") or len(manifest["episodes"]) != 100:
        raise ValueError("Expected a complete 100-episode historical cache")
    prior = {row["name"]: row for row in reference["historical"]["per_episode"]}
    model = load_model(args.model, args.device)
    episodes = []
    for entry in sorted(manifest["episodes"], key=lambda row: row["name"]):
        path = args.historical_cache / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Historical episode cache changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            features = archive["features"]
            state = archive["state"]
            candidates = archive["candidates"]
        if candidates.shape[1:] != (4, 10, 14):
            raise ValueError(f"Unexpected candidate chunk shape for {entry['name']}")
        series = score(model, features, state, candidates, args.device).mean(axis=1)
        if len(series) != int(entry["frames"]):
            raise ValueError(f"Observation count changed: {entry['name']}")
        if bool(entry["success"]) != bool(prior[entry["name"]]["success"]):
            raise ValueError(f"Success label changed: {entry['name']}")
        if not np.isclose(series.mean(), prior[entry["name"]]["score_mean"], atol=1e-3, rtol=0):
            raise ValueError(f"Score reproduction failed: {entry['name']}")
        episodes.append({"name": entry["name"], "success": bool(entry["success"]),
                         "scores": series})
    if (sum(e["success"] for e in episodes), sum(not e["success"] for e in episodes)) != (31, 69):
        raise ValueError("Historical cohort changed")
    if sum(len(e["scores"]) for e in episodes) != 4834:
        raise ValueError("Historical observation total changed")
    return episodes, {"model": sha256(args.model), "historical_manifest": sha256(manifest_path),
                      "bigmetrics": sha256(args.bigmetrics)}


def bootstrap_curve(matrix, seed=42):
    # Resampling whole episodes preserves the correct unit of uncertainty.
    rng = np.random.default_rng(seed)
    means = matrix[rng.integers(0, len(matrix), size=(10000, len(matrix)))].mean(axis=1)
    return {"mean": matrix.mean(axis=0), "ci_low": np.quantile(means, .025, axis=0),
            "ci_high": np.quantile(means, .975, axis=0)}


def curves(episodes):
    progress = np.linspace(0, 100, 21)
    common_frame = np.arange(16)
    result = {}
    for name, success in (("success", True), ("failure", False)):
        selected = [e["scores"] for e in episodes if e["success"] == success]
        full = np.stack([s[np.rint((progress / 100) * (len(s) - 1)).astype(int)] for s in selected])
        common = np.stack([s[:16] for s in selected])
        result[name] = {"n_episodes": len(selected),
                        "length_median": float(np.median([len(s) for s in selected])),
                        "length_range": [min(map(len, selected)), max(map(len, selected))],
                        "full_progress": bootstrap_curve(full),
                        "common_prefix": bootstrap_curve(common)}
    return progress, common_frame, result


def figure(progress, common_frame, result, output_dir):
    fig, axs = plt.subplots(1, 2, figsize=(13.4, 5.6), dpi=165, sharey=True)
    for ax, x, key, title in (
        (axs[0], progress, "full_progress", "Entire trajectory · normalized progress"),
        (axs[1], common_frame, "common_prefix", "Same first 16 saved observations"),
    ):
        for group, color, linestyle in (("success", "#186ca8", "-"),
                                        ("failure", "#cf623b", "--")):
            data = result[group][key]
            label = f"{group.capitalize()} (n={result[group]['n_episodes']})"
            ax.plot(x, data["mean"], color=color, linestyle=linestyle, linewidth=2.4, label=label)
            ax.fill_between(x, data["ci_low"], data["ci_high"], color=color, alpha=.13,
                            linewidth=0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.23)
        ax.legend(loc="lower left", frameon=True)
    axs[0].set_xlabel("Within-trajectory progress (%)")
    axs[1].set_xlabel("Saved observation index from rollout start")
    axs[0].set_ylabel("Raw R3M discriminator score (mean over K=4 candidate chunks)")
    axs[1].set_xlim(0, 15)
    fig.suptitle("R3M ResNet18 discriminator: historical success vs failure rollouts", fontsize=15)
    fig.text(.5, .012,
             "Lines: episode-equal means; bands: 95% episode-bootstrap intervals. "
             "Success stops early (median 18 observations); all failures have 60.\n"
             "Normalized progress is not a matched task state; the right panel is the common-prefix check.",
             ha="center", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .10, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"r3m_success_failure.{suffix}", bbox_inches="tight")
    plt.close(fig)


def serialize_curve(values, x):
    return [{"x": float(t), "mean": float(a), "bootstrap_ci95": [float(b), float(c)]}
            for t, a, b, c in zip(x, values["mean"], values["ci_low"], values["ci_high"], strict=True)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "historical_cache", "bigmetrics", "output_dir"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    episodes, hashes = load_episodes(args)
    progress, common, result = curves(episodes)
    figure(progress, common, result, args.output_dir)
    atomic_json(args.output_dir / "curve_metrics.json", {
        "scope": "Historical 100 policy rollouts, not an independent heldout set or new evaluation",
        "source_sha256": hashes,
        "cohorts": {group: {"n_episodes": v["n_episodes"], "length_median": v["length_median"],
                             "length_range": v["length_range"],
                             "full_progress": serialize_curve(v["full_progress"], progress),
                             "common_prefix": serialize_curve(v["common_prefix"], common)}
                    for group, v in result.items()},
        "method": "Mean score of four saved full-chunk policy candidates per observation; full progress uses nearest real saved observation at 0,5,...100%; 10000 episode bootstrap replicates, seed 42",
        "caveats": ["Success/failure rollouts differ sharply in length and phase occupancy",
                    "Normalized progress does not align task semantics",
                    "This is retrospective association, not evidence that the scorer causes success"],
    })
    atomic_json(args.output_dir / "manifest.json", {"complete": True,
        "curve_metrics_sha256": sha256(args.output_dir / "curve_metrics.json"),
        "png_sha256": sha256(args.output_dir / "r3m_success_failure.png"),
        "pdf_sha256": sha256(args.output_dir / "r3m_success_failure.pdf")})
    print(f"COMPLETE {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
