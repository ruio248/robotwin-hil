#!/usr/bin/env python3
"""Pair internal expert-rank comparisons and draw encoder/HIL diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tail_value.cache import atomic_json, read_json, sha256

ENCODERS = ("imagenet", "r3m", "dinov2")
LABELS = {"imagenet": "ImageNet ResNet18", "r3m": "R3M ResNet18", "dinov2": "DINOv2-S/14"}
OBJECTIVES = ("td20k", "discriminator10k")
COLORS = {"imagenet": "#3b6fa1", "r3m": "#dc8b36", "dinov2": "#40a68a"}


def paired_difference(base: dict, other: dict, seed: int = 42) -> dict:
    x = base["rank"]["episodes"]
    y = other["rank"]["episodes"]
    if set(x) != set(y) or len(x) != 30:
        raise ValueError("Diagnostic episode pairing changed")
    keys = sorted(x, key=int)
    delta = np.asarray([y[key] - x[key] for key in keys], dtype=float)
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
    return {"mean_delta": float(delta.mean()),
            "episode_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist(),
            "paired_episodes": len(delta)}


def plot_rank(summary, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True, layout="constrained")
    for ax, objective in zip(axes, OBJECTIVES):
        for index, encoder in enumerate(ENCODERS):
            rank = summary["seed42"][encoder][objective]["rank"]
            mean = rank["episode_balanced_mean"]
            lo, hi = rank["episode_bootstrap_ci95"]
            ax.errorbar(mean, index, xerr=[[mean - lo], [hi - mean]], fmt="o",
                        color=COLORS[encoder], capsize=4, markersize=8, linewidth=2)
            ax.annotate(f"{mean:.3f}", (mean, index), xytext=(10, -4), textcoords="offset points")
            for seed, dy in ((43, -.14), (44, .14)):
                repeat = summary["repeated_seeds"][encoder].get(str(seed))
                if repeat:
                    ax.plot(repeat[objective], index + dy, marker="o", markersize=5,
                            markerfacecolor="none", markeredgecolor=COLORS[encoder], linestyle="none")
        ax.axvline(.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_yticks(range(3), [LABELS[k] for k in ENCODERS])
        ax.invert_yaxis(); ax.set_xlim(.45, .88)
        ax.set_title("TD-B (20k)" if objective == "td20k" else "Source discriminator (10k)")
        ax.grid(axis="x", alpha=.2)
    fig.supxlabel("P(expert chunk scores above a same-observation policy chunk); ties = 0.5")
    fig.suptitle("Frozen visual encoder ablation | 30 internal SFT episodes; filled = seed 42, open = 43/44")
    for ext in ("png", "pdf"):
        fig.savefig(root / f"expert_rank_encoder.{ext}", dpi=200)
    plt.close(fig)


def plot_hil(summary, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "hil" not in summary:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, objective in zip(axes, OBJECTIVES):
        for encoder in ENCODERS:
            method = summary["hil"][encoder][objective]
            curve = [row for row in method["curve"] if -25 <= row["relative_saved_frame"] <= 5]
            x = np.asarray([row["relative_saved_frame"] for row in curve])
            y = np.asarray([row["offset_score_mean"] for row in curve])
            lo = np.asarray([row["offset_score_q25"] for row in curve])
            hi = np.asarray([row["offset_score_q75"] for row in curve])
            ax.plot(x, y, label=LABELS[encoder], color=COLORS[encoder], linewidth=2)
            ax.fill_between(x, lo, hi, color=COLORS[encoder], alpha=.10)
        ax.axvline(0, linestyle="--", color="#555555", linewidth=1)
        ax.axhline(0, color="#999999", linewidth=.7)
        ax.set_title("TD-B (20k)" if objective == "td20k" else "Source discriminator (10k)")
        ax.set_xlabel("Saved frame relative to first human takeover")
        ax.grid(alpha=.15); ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Candidate mean B offset / global SFT calibration scale")
    fig.suptitle("Same 50 saved HIL episodes | first ten policy frames define each trajectory offset")
    for ext in ("png", "pdf"):
        fig.savefig(root / f"hil_takeover_encoder.{ext}", dpi=200)
    plt.close(fig)


def run(root: Path):
    summary = {"scope": "Internal SFT and saved HIL diagnostics; no independent execution evaluation",
               "seed42": {}, "paired_vs_imagenet_seed42": {}, "paired_vs_imagenet_by_seed": {},
               "repeated_seeds": {},
               "limitations": ["All visual blocks unit-L2 normalized; DINOv2-S 384D padded to 512D per view.",
                               "ImageNet baseline was retrained with this normalization, not compared directly to old weights.",
                               "Same 45 SFT validation episodes were previously monitored; 30 diagnostics are not untouched heldout.",
                               "Expert demonstrations are not unique valid actions; rank is not task success.",
                               "HIL saved subset (50 episodes) is not all 87 rollouts or an online control experiment.",
                               "Different encoder architecture and pretraining are confounded for DINOv2."]}
    paths = {}
    seed_models = {}
    for encoder in ENCODERS:
        for seed in (42, 43, 44):
            path = root / f"eval_{encoder}_s{seed}" / "metrics.json"
            if not path.is_file():
                if seed == 42:
                    raise FileNotFoundError(path)
                continue
            paths[f"{encoder}_s{seed}"] = sha256(path)
            data = read_json(path)
            seed_models[(encoder, seed)] = data["models"]
            summary["repeated_seeds"].setdefault(encoder, {})[str(seed)] = {
                objective: data["models"][objective]["rank"]["episode_balanced_mean"]
                for objective in OBJECTIVES}
            if seed == 42:
                summary["seed42"][encoder] = data["models"]
    for objective in OBJECTIVES:
        summary["paired_vs_imagenet_seed42"][objective] = {
            encoder: paired_difference(summary["seed42"]["imagenet"][objective],
                                       summary["seed42"][encoder][objective])
            for encoder in ("r3m", "dinov2")}
        summary["paired_vs_imagenet_by_seed"][objective] = {
            encoder: {str(seed): paired_difference(seed_models[("imagenet", seed)][objective],
                                                   seed_models[(encoder, seed)][objective])
                      for seed in (42, 43, 44)
                      if ("imagenet", seed) in seed_models and (encoder, seed) in seed_models}
            for encoder in ("r3m", "dinov2")}
    hil = {}
    for encoder in ENCODERS:
        path = root / f"hil_score_{encoder}" / "metrics.json"
        if path.is_file():
            paths[f"hil_{encoder}"] = sha256(path)
            hil[encoder] = read_json(path)["methods"]
    if len(hil) == 3:
        summary["hil"] = hil
    summary["input_sha256"] = paths
    atomic_json(root / "summary.json", summary)
    plot_rank(summary, root)
    plot_hil(summary, root)
    files = ["summary.json", "expert_rank_encoder.png", "expert_rank_encoder.pdf"]
    if "hil" in summary:
        files += ["hil_takeover_encoder.png", "hil_takeover_encoder.pdf"]
    atomic_json(root / "report_manifest.json", {"complete": len(hil) == 3 and all(
        len(summary["repeated_seeds"].get(enc, {})) == 3 for enc in ENCODERS),
        "files": {name: sha256(root / name) for name in files}})
    print(f"REPORT {root / 'summary.json'} complete={read_json(root / 'report_manifest.json')['complete']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
