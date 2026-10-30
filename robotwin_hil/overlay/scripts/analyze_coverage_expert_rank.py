#!/usr/bin/env python3
"""Rank expert chunks against policy chunks at the same diagnostic observation.

This is an internal, episode-held-out diagnostic of action ordering, not a
success classifier or an independent policy evaluation. It reuses frozen
offline score files and never calls the policy or a simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coverage_benchmark.common import atomic_json, sha256


METHODS = (
    ("td_s42_1000", "TD-B 1k"),
    ("td_s42_10000", "TD-B 10k"),
    ("td_s42_20000", "TD-B 20k"),
    ("knn", "kNN + action kernel"),
    ("svm", "One-Class SVM"),
    ("disc_s42", "Source discriminator"),
    ("observation", "Observation only"),
)


def paired_percentile(expert: float, candidates: np.ndarray) -> float:
    """P(expert > random candidate), with an exact tie worth half a win."""
    values = np.asarray(candidates, dtype=np.float64)
    if values.ndim != 1 or len(values) != 4 or not np.isfinite(values).all():
        raise ValueError("Expected four finite candidate scores")
    if not np.isfinite(expert):
        raise ValueError("Expert score must be finite")
    return float((np.count_nonzero(expert > values)
                  + 0.5 * np.count_nonzero(expert == values)) / len(values))


def collect_rows(offline_dir: Path):
    manifest_path = offline_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError("Offline score manifest is incomplete")
    entries = {name: digest for name, digest in manifest["completed"].items()
               if name.startswith("validation_")}
    if len(entries) != 30:
        raise ValueError(f"Expected 30 internal diagnostic episodes; found {len(entries)}")

    episodes = []
    for name in sorted(entries, key=lambda n: int(n.split("_")[1])):
        path = offline_dir / "scores" / f"{name}.json"
        if sha256(path) != entries[name]:
            raise ValueError(f"Changed or incomplete score file: {path}")
        episode = json.loads(path.read_text())
        meta = episode["metadata"]
        if meta["cohort"] != "internal_diagnostic" or meta["fit_manifest_sha256"] != manifest["fit_manifest_sha256"]:
            raise ValueError(f"Wrong cohort or scorer assets in {name}")
        if not episode["rows"]:
            raise ValueError(f"Empty diagnostic episode: {name}")
        episodes.append(episode)
    if sum(len(ep["rows"]) for ep in episodes) != 844:
        raise ValueError("Expected 844 complete-chunk diagnostic observations")
    return manifest_path, episodes


def analyze(episodes, seed=42, bootstrap_repeats=10_000):
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    episode_ids = [int(ep["metadata"]["episode"]) for ep in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Repeated episode in diagnostic data")

    per_observation = []
    for ep in episodes:
        episode_id = int(ep["metadata"]["episode"])
        for row in ep["rows"]:
            methods = {}
            for method, _ in METHODS:
                expert = float(row["expert_scores"][method])
                policy = np.asarray(row["scores"][method], dtype=np.float64)
                percentile = paired_percentile(expert, policy)
                methods[method] = {
                    "percentile": percentile,
                    "strict_top": bool(np.all(expert > policy)),
                    "strict_bottom": bool(np.all(expert < policy)),
                    "tied_pairs": int(np.count_nonzero(expert == policy)),
                }
            per_observation.append({"episode": episode_id,
                                    "frame_index": int(row["frame_index"]),
                                    "methods": methods})

    rng = np.random.default_rng(seed)
    episode_indices = rng.integers(len(episodes), size=(bootstrap_repeats, len(episodes)))
    summary = {}
    for method, label in METHODS:
        values = np.array([r["methods"][method]["percentile"] for r in per_observation])
        by_episode = np.array([
            np.mean([r["methods"][method]["percentile"] for r in per_observation
                     if r["episode"] == episode_id])
            for episode_id in episode_ids
        ])
        draws = by_episode[episode_indices].mean(axis=1)
        summary[method] = {
            "label": label,
            "episode_balanced_mean": float(by_episode.mean()),
            "episode_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist(),
            "observation_weighted_mean": float(values.mean()),
            "episode_mean_q25_q50_q75": np.quantile(by_episode, [.25, .5, .75]).tolist(),
            "strict_top_fraction": float(np.mean([r["methods"][method]["strict_top"]
                                                  for r in per_observation])),
            "strict_bottom_fraction": float(np.mean([r["methods"][method]["strict_bottom"]
                                                     for r in per_observation])),
            "tied_pair_fraction": float(sum(r["methods"][method]["tied_pairs"]
                                              for r in per_observation) / (4 * len(per_observation))),
            "episode_means": {str(ep_id): float(value)
                              for ep_id, value in zip(episode_ids, by_episode)},
        }
    return per_observation, summary


def make_plot(summary, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.3), layout="constrained")
    labels = [label for _, label in METHODS]
    centers = np.array([summary[method]["episode_balanced_mean"] for method, _ in METHODS])
    limits = np.array([summary[method]["episode_bootstrap_ci95"] for method, _ in METHODS])
    y = np.arange(len(METHODS))[::-1]
    ax.axvline(.5, color="#666666", linestyle="--", linewidth=1,
               label="Chance / action-insensitive score = 0.5")
    colors = ["#5483a4", "#2473aa", "#5483a4", "#e18b36", "#6b9b62",
              "#8761aa", "#777777"]
    for i, ((method, _), ypos) in enumerate(zip(METHODS, y)):
        ep_values = np.array(list(summary[method]["episode_means"].values()))
        ax.scatter(ep_values, np.full(len(ep_values), ypos), s=10,
                   color=colors[i], alpha=.22, linewidths=0, zorder=1)
        ax.plot(limits[i], [ypos, ypos], color=colors[i], linewidth=3, zorder=2)
        ax.plot(centers[i], ypos, marker="o", color=colors[i], markersize=7, zorder=3)
        ax.annotate(f"{centers[i]:.3f}", (centers[i], ypos), xytext=(8, -9),
                    textcoords="offset points", fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(expert score > a sampled policy score), exact ties = 0.5")
    ax.set_title("Expert-chunk rank at the same observation | 30 internal episodes, 844 anchors")
    ax.grid(axis="x", alpha=.17)
    ax.legend(loc="lower left", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"expert_rank.{ext}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path, episodes = collect_rows(args.offline_dir)
    rows, methods = analyze(episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "per_observation.json", rows)
    atomic_json(args.output_dir / "metrics.json", {
        "scope": "30 internal diagnostic SFT episodes; not independent held-out policy evaluation",
        "observations": len(rows), "episodes": len(episodes), "candidate_count": 4,
        "quantity": "Expert chunk score percentile against four same-observation policy chunks",
        "tie_rule": "exact ties count as half a win",
        "aggregation": "mean within episode, then mean across episodes; 10000 episode bootstrap resamples",
        "offline_manifest_sha256": sha256(manifest_path), "methods": methods,
        "limitations": [
            "Expert demonstrations are not unique valid actions; high expert rank is not ground-truth success.",
            "Rank ignores score-gap magnitude and does not establish useful tilted execution.",
            "The observation-only control is exactly tied within each observation by construction.",
            "HIL pre-takeover observations have no paired expert chunk and are excluded."
        ],
    })
    make_plot(methods, args.output_dir)
    atomic_json(args.output_dir / "manifest.json", {
        "complete": True,
        "offline_manifest_sha256": sha256(manifest_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "files": {name: sha256(args.output_dir / name) for name in
                  ("metrics.json", "per_observation.json", "expert_rank.png", "expert_rank.pdf")},
    })


if __name__ == "__main__":
    main()
