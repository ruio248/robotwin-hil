"""Replot cached candidate diversity with mean, SD, and K-extrema views."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


KS = (4, 8, 16, 32)
METRIC = "mean_pairwise_rad"
COLORS = ("#277DA8", "#DB8A26", "#3D9567", "#9565B7")
MARKERS = ("o", "s", "^", "D")


def summarize(rows: list[dict], bins: int = 20, metric: str = METRIC) -> list[dict]:
    """Give each trajectory one vote in every progress bin it contributes to."""
    grouped: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        progress = float(row["progress"])
        if not 0 <= progress <= 1:
            raise ValueError(f"Invalid progress: {progress}")
        bin_index = min(int(progress * bins), bins - 1)
        for k in KS:
            value = float(row["metrics"][str(k)][metric])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid distance: {value}")
            grouped[(bin_index, k)][int(row["rollout_index"])].append(value)

    result = []
    for bin_index in range(bins):
        series = {}
        for k in KS:
            episodes = grouped[(bin_index, k)]
            episode_means = np.asarray([np.mean(values) for values in episodes.values()])
            series[str(k)] = {
                "mean_rad": float(episode_means.mean()) if len(episode_means) else None,
                "sd_rad": float(episode_means.std(ddof=1)) if len(episode_means) > 1 else None,
                "episodes": len(episode_means),
            }
        result.append({
            "progress_lower": bin_index / bins,
            "progress_upper": (bin_index + 1) / bins,
            "k": series,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    rows = json.loads((args.input_dir / "per_observation_metrics.json").read_text())
    if manifest.get("complete") is not True or len(rows) != 1260:
        raise ValueError("Expected the complete 1260-observation formal result")
    if len({row["rollout_index"] for row in rows}) != 51:
        raise ValueError("Expected 51 saved trajectories")
    summary = summarize(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    mac_font = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    if mac_font.is_file():
        font_manager.fontManager.addfont(str(mac_font))
        cjk_family = font_manager.FontProperties(fname=str(mac_font)).get_name()
    else:
        cjk_family = "Noto Sans CJK JP"

    plt.rcParams.update({
        "font.family": cjk_family,
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    fig.subplots_adjust(top=0.82, bottom=0.19, left=0.12, right=0.97)
    fig.suptitle("同一观测下的候选动作平均差异", x=0.12, ha="left", y=0.97, fontsize=18)
    fig.text(0.12, 0.90, "51 条已保存轨迹 · 1260 个观测 · 首动作的 12 个机械臂关节", fontsize=11)
    x = np.asarray([(item["progress_lower"] + item["progress_upper"]) * 50 for item in summary])
    for k, color, marker in zip(KS, COLORS, MARKERS):
        means = np.asarray([item["k"][str(k)]["mean_rad"] for item in summary], dtype=float) * 1000
        sd = np.asarray([item["k"][str(k)]["sd_rad"] for item in summary], dtype=float) * 1000
        ax.fill_between(x, np.maximum(0, means - sd), means + sd, color=color, alpha=0.07, linewidth=0)
        ax.plot(x, means, color=color, marker=marker, markersize=4, linewidth=2, label=f"K = {k}")

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("策略执行步数占比（%）")
    ax.set_ylabel("平均两两关节动作差异（mrad）")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=4, frameon=False, loc="upper left")
    fig.text(0.12, 0.05, "每 5% 分箱：轨迹内先平均，再算轨迹间平均；阴影为轨迹间 ±1 标准差（下界截于 0）。", fontsize=10)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "candidate_mean_distance_mean_sd"
    fig.savefig(stem.with_suffix(".png"), dpi=190)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    # The four bands overlap in the combined plot because their means are close.
    # Facets preserve a shared scale while making each trajectory SD visible.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.82, bottom=0.19,
                        hspace=0.26, wspace=0.16)
    fig.suptitle("同一观测下的候选动作平均差异", x=0.11, ha="left", y=0.98, fontsize=18)
    fig.text(0.11, 0.91, "四组分别展示阴影；全部小图使用相同的横纵坐标尺度", fontsize=11)
    upper_limits = []
    for ax, k, color, marker in zip(axes.flat, KS, COLORS, MARKERS):
        means = np.asarray([item["k"][str(k)]["mean_rad"] for item in summary], dtype=float) * 1000
        sd = np.asarray([item["k"][str(k)]["sd_rad"] for item in summary], dtype=float) * 1000
        upper_limits.append(np.nanmax(means + sd))
        ax.fill_between(x, np.maximum(0, means - sd), means + sd,
                        color=color, alpha=0.24, linewidth=0)
        ax.plot(x, means, color=color, marker=marker, markersize=3.4,
                linewidth=2, zorder=2)
        ax.set_title(f"K = {k}", loc="left", color=color, fontsize=13, pad=6)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.grid(axis="y", alpha=0.2)
    for ax in axes.flat:
        ax.set_ylim(0, np.ceil(max(upper_limits) * 2) / 2)
    fig.supxlabel("策略执行步数占比（%）", y=0.105)
    fig.supylabel("平均两两关节动作差异（mrad）", x=0.025)
    fig.text(0.11, 0.045, "每 5% 分箱：轨迹内先平均，再算轨迹间平均；阴影为轨迹间 ±1 标准差（下界截于 0）。", fontsize=10)
    facet_stem = args.output_dir / "candidate_mean_distance_facets"
    fig.savefig(facet_stem.with_suffix(".png"), dpi=190)
    fig.savefig(facet_stem.with_suffix(".pdf"))
    plt.close(fig)

    # A single curve for the four K groups, with one envelope from their extrema.
    # These extrema are across K-specific trajectory means, never across frames.
    k_envelope = []
    for item in summary:
        values = np.asarray([item["k"][str(k)]["mean_rad"] for k in KS], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Every progress bin must contain all four K groups")
        k_envelope.append({
            "progress_lower": item["progress_lower"],
            "progress_upper": item["progress_upper"],
            "mean_rad": float(values.mean()),
            "min_k_mean_rad": float(values.min()),
            "max_k_mean_rad": float(values.max()),
        })

    mean = np.asarray([item["mean_rad"] for item in k_envelope]) * 1000
    lower = np.asarray([item["min_k_mean_rad"] for item in k_envelope]) * 1000
    upper = np.asarray([item["max_k_mean_rad"] for item in k_envelope]) * 1000
    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.19)
    fig.suptitle("候选动作差异：四组均值与极值范围", x=0.12, ha="left", y=0.97, fontsize=18)
    fig.text(0.12, 0.90, "K = 4 / 8 / 16 / 32 等权合并 · 51 条轨迹 · 1260 个观测", fontsize=11)
    ax.fill_between(x, lower, upper, color="#6F9DBC", alpha=0.48, linewidth=0,
                    label="四组均值的最小–最大")
    ax.plot(x, mean, color="#175B83", marker="o", markersize=4, linewidth=2.2,
            label="四组均值的平均")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_ylim(0, np.ceil(max(upper) * 5) / 5)
    ax.set_xlabel("策略执行步数占比（%）")
    ax.set_ylabel("平均两两关节动作差异（mrad）")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.12, 0.05, "每 5% 分箱：每组先算轨迹内、再算轨迹间平均；阴影是四组均值的最小至最大。", fontsize=10)
    envelope_stem = args.output_dir / "candidate_mean_distance_k_envelope"
    fig.savefig(envelope_stem.with_suffix(".png"), dpi=190)
    fig.savefig(envelope_stem.with_suffix(".pdf"))
    plt.close(fig)
    (args.output_dir / "candidate_mean_distance_k_envelope.json").write_text(
        json.dumps({"line": "equal mean of four K-specific trajectory means",
                    "band": "min/max of four K-specific trajectory means",
                    "source_observations": len(rows), "bins": k_envelope}, indent=2) + "\n"
    )

    (args.output_dir / "candidate_mean_distance_mean_sd.json").write_text(
        json.dumps({"method": "mean of episode means ± sample SD across episodes",
                    "metric": METRIC, "source_observations": len(rows), "bins": summary}, indent=2) + "\n"
    )

    # Two separate four-line figures: one for the mean pairwise distance and
    # one for the maximum pairwise distance among candidates at each state.
    for metric_summary, title, ylabel, stem_name in (
        (summary, "平均两两动作差异", "平均两两关节动作差异（mrad）", "candidate_mean_pairwise_four_k"),
        (summarize(rows, metric="max_pairwise_rad"), "最大两两动作差异",
         "最大两两关节动作差异（mrad）", "candidate_max_pairwise_four_k"),
    ):
        fig, ax = plt.subplots(figsize=(10.8, 6.3))
        fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.19)
        fig.suptitle(title, x=0.12, ha="left", y=0.97, fontsize=18)
        fig.text(0.12, 0.90, "K = 4 / 8 / 16 / 32 · 51 条轨迹 · 1260 个观测 · 首动作的 12 个机械臂关节", fontsize=11)
        for k, color, marker in zip(KS, COLORS, MARKERS):
            values = np.asarray([item["k"][str(k)]["mean_rad"] for item in metric_summary], dtype=float) * 1000
            ax.plot(x, values, color=color, marker=marker, markersize=4.5,
                    linewidth=2.2, label=f"K = {k}")
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("策略执行步数占比（%）")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(ncol=4, frameon=False, loc="upper left")
        fig.text(0.12, 0.05, "每 5% 分箱：轨迹内先平均，再取轨迹间平均；四组使用同一批观测。", fontsize=10)
        plot_stem = args.output_dir / stem_name
        fig.savefig(plot_stem.with_suffix(".png"), dpi=190)
        fig.savefig(plot_stem.with_suffix(".pdf"))
        plt.close(fig)


if __name__ == "__main__":
    main()
