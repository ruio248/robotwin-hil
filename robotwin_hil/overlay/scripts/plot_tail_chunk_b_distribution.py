#!/usr/bin/env python3
"""Plot pooled raw and same-observation-centered chunk B distributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tail_value.cache import atomic_json


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def save(fig, directory, stem):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=180)


def run(args):
    with np.load(args.scores, allow_pickle=False) as archive:
        scores = np.asarray(archive["scores"], dtype=np.float64)
        trajectories = np.asarray(archive["rollout_index"])
    if scores.ndim != 2 or scores.shape[1] != 32 or not np.isfinite(scores).all():
        raise ValueError("Expected finite [observations,32] B scores")
    summary = json.loads(args.summary.read_text())
    if (scores.shape[0] != summary["observations"]
            or scores.size != summary["raw_b"]["count"]
            or len(np.unique(trajectories)) != summary["trajectories"]):
        raise ValueError("Score array and summary disagree")
    raw = scores.reshape(-1)
    centered = (scores - scores.mean(axis=1, keepdims=True)).reshape(-1)
    if abs(centered.mean()) > 1e-10:
        raise ValueError("Centered B scores are not zero-mean")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt = configure_plotting()

    # Every candidate is counted exactly once.  A log count axis reveals the
    # low-density bridge between the two score concentrations.
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.subplots_adjust(left=.10, right=.97, top=.82, bottom=.21)
    raw_edges = np.linspace(raw.min() - .5, raw.max() + .5, 300)
    ax.hist(raw, bins=raw_edges, color="#277DA8", alpha=.88)
    ax.set_yscale("log")
    ax.set_xlim(raw_edges[0], raw_edges[-1])
    ax.set_xlabel("完整 10×14 动作 chunk 的 B 分数")
    ax.set_ylabel("候选数（对数刻度）")
    ax.grid(axis="y", alpha=.2)
    ax.axvline(summary["raw_b"]["median"], color="#B34842", linestyle="--",
               linewidth=1.6, label=f"中位数 {summary['raw_b']['median']:.3f}")
    ax.legend(frameon=False, loc="upper center")
    fig.suptitle("全部候选 B 分数的汇总分布", x=.10, ha="left", fontsize=17)
    fig.text(.10, .09,
             f"{scores.shape[0]} 个观测 × 32 个候选 = {raw.size:,} 个 B；"
             f"{summary['trajectories']} 条已保存的首次接管前 HIL 轨迹。", fontsize=10)
    fig.text(.10, .06,
             "每个观测贡献 32 个分数，未重复统计嵌套的 K=4/8/16；纵轴为对数刻度。", fontsize=10)
    save(fig, args.output_dir, "b_all_raw_distribution")
    plt.close(fig)

    # Use an explicit central zoom plus a full-range panel; a single linear
    # x-axis would hide the dense center behind a few large outliers.
    zoom = .05
    fraction_inside = float(np.mean(np.abs(centered) <= zoom))
    fig, (central_ax, full_ax) = plt.subplots(2, 1, figsize=(11.5, 9.1))
    fig.subplots_adjust(left=.10, right=.97, top=.84, bottom=.16, hspace=.42)
    central = centered[np.abs(centered) <= zoom]
    central_ax.hist(central, bins=np.linspace(-zoom, zoom, 161),
                    color="#3D9567", alpha=.88)
    central_ax.set_xlim(-zoom, zoom)
    central_ax.set_ylabel("候选数")
    central_ax.set_xlabel("同观测内 B − 候选平均 B")
    central_ax.set_title(f"中心细节：|分差| ≤ {zoom:g}，包含 {fraction_inside:.1%} 的候选",
                         loc="left", fontsize=12)
    central_ax.axvline(0, color="#59636D", linewidth=1)
    central_ax.grid(axis="y", alpha=.2)
    full_edges = np.linspace(centered.min() - 1, centered.max() + 1, 300)
    full_ax.hist(centered, bins=full_edges, color="#9565B7", alpha=.85)
    full_ax.set_xlim(full_edges[0], full_edges[-1])
    full_ax.set_yscale("log")
    full_ax.set_ylabel("候选数（对数刻度）")
    full_ax.set_xlabel("同观测内 B − 候选平均 B")
    full_ax.set_title("完整范围：显示少量大幅偏离候选", loc="left", fontsize=12)
    full_ax.axvline(0, color="#59636D", linewidth=1)
    full_ax.grid(axis="y", alpha=.2)
    fig.suptitle("同一观测内的候选 B 相对分布", x=.10, ha="left", fontsize=17)
    fig.text(.10, .08,
             "每个观测的 32 个 B 各减去这 32 个分数的均值，再合并全部观测。", fontsize=10)
    fig.text(.10, .055,
             "中心面板只显示范围内的候选；下方面板显示全部候选，纵轴为对数刻度。", fontsize=10)
    save(fig, args.output_dir, "b_all_centered_distribution")
    plt.close(fig)

    atomic_json(args.output_dir / "plot_summary.json", {
        "observations": int(scores.shape[0]),
        "trajectories": int(len(np.unique(trajectories))),
        "candidate_scores": int(raw.size),
        "fraction_abs_centered_le_0p01": float(np.mean(np.abs(centered) <= .01)),
        "fraction_abs_centered_le_0p05": fraction_inside,
        "fraction_abs_centered_le_1": float(np.mean(np.abs(centered) <= 1)),
        "negative_scores": int(np.sum(raw < 0)),
        "observations_all_negative": int(np.sum(scores.max(axis=1) < 0)),
        "observations_all_above_80": int(np.sum(scores.min(axis=1) > 80)),
        "observations_cross_negative_and_above_80": int(np.sum(
            (scores.min(axis=1) < 0) & (scores.max(axis=1) > 80))),
        "interpretation_note": "B is a learned score, not a calibrated probability or true coverage label",
    })
    print(f"saved two B-distribution figures -> {args.output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
