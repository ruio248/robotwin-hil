#!/usr/bin/env python3
"""Plot the frozen R3M discriminator's saved-frame HIL takeover curve.

This is a retrospective diagnostic. It never runs a policy or simulator.
The observation-only control uses the same R3M features and expert support.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from coverage_encoder_discriminator_bigmetrics import (
    hil_summary,
    load_model,
    observation_score,
    score,
)
from coverage_encoder_knn_ablation import load_support
from tail_value.cache import atomic_json, read_json, sha256


def collect(args):
    reference = read_json(args.bigmetrics)
    sources = reference["source_sha256"]
    expected = {
        "model": args.model,
        "support": args.support,
        "hil_cache_manifest": args.hil_cache / "manifest.json",
        "hil_source_manifest": args.source_manifest,
    }
    for key, path in expected.items():
        if sha256(path) != sources[key]:
            raise ValueError(f"Source changed since big-metrics evaluation: {key}")
    manifest = read_json(args.hil_cache / "manifest.json")
    source = read_json(args.source_manifest)
    if not manifest.get("complete") or len(manifest["episodes"]) != 50:
        raise ValueError("Expected the complete 50-episode R3M HIL cache")
    if sha256(args.source_manifest) != manifest["config"]["source_candidate_manifest_sha256"]:
        raise ValueError("Original candidate-source manifest differs")
    takeovers = {int(item["rollout_index"]): int(item["takeover"])
                 for item in source["identity"]["episodes"]}
    model = load_model(args.model, args.device)
    support = load_support(args.support)
    episodes = []
    for entry in sorted(manifest["episodes"], key=lambda row: row["rollout_index"]):
        rid = int(entry["rollout_index"])
        path = args.hil_cache / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Cached HIL shard changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            features = archive["features"]
            state = archive["state"]
            candidates = archive["candidates"]
            frame = archive["frame_index"].astype(int)
        if candidates.shape[1:] != (4, 10, 14):
            raise ValueError(f"Unexpected full-chunk candidates for rollout {rid}")
        disc = score(model, features, state, candidates, args.device).mean(1)
        obs = observation_score(features, state, support, args.device)
        episodes.append({"rollout_index": rid,
                         "relative_frame": frame - takeovers[rid],
                         "score": disc,
                         "observation": obs})
    diagnostic = hil_summary(episodes)
    prior = reference["hil_takeover"]["summary"]
    for key, metric in prior.items():
        actual = diagnostic["summary"][key]["mean"]
        # CPU/GPU nearest-neighbor kernels differ at ~1e-4 percentile points.
        if not np.isclose(actual, metric["mean"], atol=1e-3, rtol=0):
            raise ValueError(f"Window-statistic reproduction failed: {key}: "
                             f"actual={actual}, expected={metric['mean']}")
    return episodes, diagnostic, {key: sha256(path) for key, path in expected.items()}


def summarize_curve(episodes, start=-15, stop=39):
    rows = []
    for relative_frame in range(start, stop + 1):
        scores, observations = [], []
        for episode in episodes:
            indices = np.flatnonzero(episode["relative_frame"] == relative_frame)
            if len(indices) != 1:
                raise ValueError(f"Frame {relative_frame} missing/duplicated in rollout {episode['rollout_index']}")
            i = int(indices[0])
            scores.append(float(episode["score_percentile"][i]))
            observations.append(float(episode["observation_percentile"][i]))
        score_array = np.asarray(scores)
        obs_array = np.asarray(observations)
        delta = score_array - obs_array
        row = {"relative_saved_frame": relative_frame, "n_episodes": len(scores)}
        for name, values in (("discriminator", score_array),
                             ("observation_only", obs_array),
                             ("discriminator_minus_observation", delta)):
            row[name] = {"mean": float(values.mean()),
                         "q25": float(np.quantile(values, .25)),
                         "median": float(np.median(values)),
                         "q75": float(np.quantile(values, .75))}
        rows.append(row)
    return rows


def plot(rows, output_dir):
    x = np.asarray([row["relative_saved_frame"] for row in rows])
    fig, ax = plt.subplots(figsize=(11.8, 6.1), dpi=160)
    for key, label, color, linestyle in (
        ("discriminator", "R3M ResNet18 discriminator", "#196cad", "-"),
        ("observation_only", "Same-encoder observation-only control", "#d47432", "--"),
    ):
        mean = np.asarray([row[key]["mean"] for row in rows])
        q25 = np.asarray([row[key]["q25"] for row in rows])
        q75 = np.asarray([row[key]["q75"] for row in rows])
        ax.fill_between(x, q25, q75, color=color, alpha=.11, linewidth=0)
        ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=2.6, label=label)
    ax.axvspan(0, 39.5, color="#f4ebe6", alpha=.4, zorder=-5)
    ax.axvline(0, color="#333333", linestyle=(0, (5, 3)), linewidth=1.25)
    ax.text(.9, 96, "Recorded human control", va="top", color="#444444", fontsize=10)
    ax.set(xlim=(-15, 39), ylim=(0, 100),
           xlabel="Saved frame relative to first recorded human-control frame",
           ylabel="Episode-equal percentile score (0–100)",
           title="R3M ResNet18: takeover-aligned score across 50 saved HIL episodes")
    ax.grid(axis="y", alpha=.22)
    ax.legend(loc="lower left", frameon=True)
    fig.text(.5, .012,
             "Lines: mean across the same 50 episodes at each frame. Shading: between-episode 25th–75th percentile.\n"
             "Each scorer is mapped to its own episode-equal empirical percentile; these are not raw score units or success rates.",
             ha="center", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .09, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"r3m_discriminator_takeover.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "support", "hil_cache", "source_manifest", "bigmetrics", "output_dir"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    episodes, diagnostic, hashes = collect(args)
    rows = summarize_curve(episodes)
    plot(rows, args.output_dir)
    atomic_json(args.output_dir / "curve_metrics.json", {
        "scope": "Retrospective 50 saved HIL episode takeover-aligned diagnostic; not a success or causal recovery evaluation",
        "source_sha256": hashes,
        "percentile_reference": diagnostic["percentile_reference"],
        "window_summary": diagnostic["summary"],
        "windows": diagnostic["windows"],
        "curve": rows,
        "caveats": ["x=0 is the first saved human-control frame, not exact keypress time",
                    "human-control frames score policy suggestions, not recorded human actions",
                    "shading shows 25–75% between-episode distribution, not confidence intervals"],
    })
    atomic_json(args.output_dir / "manifest.json", {
        "complete": True,
        "curve_metrics_sha256": sha256(args.output_dir / "curve_metrics.json"),
        "png_sha256": sha256(args.output_dir / "r3m_discriminator_takeover.png"),
        "pdf_sha256": sha256(args.output_dir / "r3m_discriminator_takeover.pdf"),
    })
    print(f"COMPLETE {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
