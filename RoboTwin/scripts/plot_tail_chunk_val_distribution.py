#!/usr/bin/env python3
"""Plot same-observation expert-versus-policy B scores on the critic val split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def setup_plotting():
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


def save(fig, output_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{ext}", dpi=180)


def plot(args):
    with np.load(args.scores, allow_pickle=False) as archive:
        expert = np.asarray(archive["expert_b"], dtype=np.float64)
        policy = np.asarray(archive["policy_b"], dtype=np.float64)
        episodes = np.asarray(archive["episode_id"], dtype=np.int32)
        frame_index = np.asarray(archive["frame_index"], dtype=np.int32)
    summary = json.loads(args.summary.read_text())
    if (expert.shape != (summary["val_transitions"],)
            or policy.shape != (summary["val_transitions"], 4)
            or frame_index.shape != expert.shape
            or len(np.unique(episodes)) != 45
            or not np.isfinite(expert).all() or not np.isfinite(policy).all()):
        raise ValueError("Invalid frozen validation score arrays")
    policy_flat = policy.reshape(-1)
    gap = expert - policy.mean(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_plotting()

    # Both histograms integrate to one separately, since there are four times
    # as many policy candidate scores as expert scores.
    fig, (full, detail) = plt.subplots(2, 1, figsize=(11.5, 9.2))
    fig.subplots_adjust(left=.10, right=.97, top=.85, bottom=.16, hspace=.42)
    raw_bins = np.linspace(min(expert.min(), policy_flat.min()) - 1,
                           max(expert.max(), policy_flat.max()) + 1, 270)
    for ax, bins in ((full, raw_bins), (detail, np.linspace(96.5, 99.0, 160))):
        for values, color, label in ((expert, "#277DA8", "专家 chunk"),
                                     (policy_flat, "#DB8A26", "策略候选 chunk")):
            ax.hist(values, bins=bins, weights=np.full(len(values), 1 / len(values)),
                    histtype="step", linewidth=2, color=color, label=label)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylabel("各组内部的候选比例／分箱")
    full.set_yscale("log")
    full.set_xlim(raw_bins[0], raw_bins[-1])
    full.set_title("完整评分范围（纵轴对数）", loc="left", fontsize=12)
    full.legend(frameon=False, ncol=2, loc="upper center")
    detail.set_xlim(96.5, 99.0)
    detail.set_xlabel("B(同一观测, 完整 10×14 动作 chunk)")
    detail.set_title("主要评分区域放大", loc="left", fontsize=12)
    fig.suptitle("45 条专家验证轨迹：专家与策略候选 B 分数", x=.10, ha="left", fontsize=17)
    fig.text(.10, .08,
             f"{len(expert)} 个有效观测：每个有 1 个专家 chunk 和 4 个策略候选 chunk。"
             "两组各自归一化，不能按原始柱高比样本数。", fontsize=10)
    fig.text(.10, .055,
             f"中位数：专家 {np.median(expert):.4f}，策略候选 {np.median(policy_flat):.4f}。"
             "验证轨迹未参与 critic 更新。", fontsize=10)
    save(fig, args.output_dir, "val_expert_vs_candidate_b")
    plt.close(fig)

    zoom = .05
    fig, (central, full) = plt.subplots(2, 1, figsize=(11.5, 9.2))
    fig.subplots_adjust(left=.10, right=.97, top=.85, bottom=.16, hspace=.42)
    central.hist(gap[np.abs(gap) <= zoom], bins=np.linspace(-zoom, zoom, 160),
                 color="#3D9567", alpha=.88)
    central.set_xlim(-zoom, zoom)
    central.set_ylabel("观测数")
    central.set_xlabel("专家 B − 同观测 4 个候选的平均 B")
    central.set_title(f"零附近放大：{np.mean(np.abs(gap) <= zoom):.1%} 的观测落在 ±{zoom:g} 内",
                      loc="left", fontsize=12)
    central.axvline(0, color="#59636D", linewidth=1)
    central.grid(axis="y", alpha=.2)
    full_edges = np.linspace(min(-.25, gap.min() - .02), gap.max() + 2, 250)
    full.hist(gap, bins=full_edges, color="#9565B7", alpha=.88)
    full.set_yscale("log")
    full.set_xlim(full_edges[0], full_edges[-1])
    full.set_ylabel("观测数（对数刻度）")
    full.set_xlabel("专家 B − 同观测 4 个候选的平均 B")
    full.set_title("完整分差范围：少数大正分差会拉高均值", loc="left", fontsize=12)
    full.axvline(0, color="#59636D", linewidth=1)
    full.grid(axis="y", alpha=.2)
    fig.suptitle("同一观测的专家−策略候选配对 B 分差", x=.10, ha="left", fontsize=17)
    fig.text(.10, .08,
             f"中位数 {np.median(gap):.6f}；平均值 {gap.mean():.3f}；"
             f"{np.sum(gap > 10)} / {len(gap)} 个观测的专家分数高出候选均值超过 10。", fontsize=10)
    fig.text(.10, .055,
             "正值表示专家分数更高；本图不将候选自动当成失败动作，也不测量任务成功率。", fontsize=10)
    save(fig, args.output_dir, "val_expert_minus_candidate_b")
    plt.close(fig)

    frames = np.unique(frame_index)
    frame_median = [float(np.median(gap[frame_index == frame])) for frame in frames]
    frame_p95 = [float(np.quantile(gap[frame_index == frame], .95)) for frame in frames]
    large_count = [int(np.sum(gap[frame_index == frame] > 10)) for frame in frames]
    fig, (ax, count_ax) = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True,
                                      gridspec_kw={"height_ratios": [4, 1]})
    fig.subplots_adjust(left=.10, right=.97, top=.83, bottom=.17, hspace=.13)
    ax.scatter(frame_index, gap, s=16, color="#7A9FB5", alpha=.45,
               linewidths=0, label="逐观测分差")
    ax.plot(frames, frame_median, color="#277DA8", linewidth=2,
            marker="o", markersize=3, label="同帧跨轨迹中位数")
    ax.plot(frames, frame_p95, color="#DB8A26", linewidth=2,
            marker="o", markersize=3, label="同帧跨轨迹第95百分位")
    ax.set_yscale("symlog", linthresh=.01)
    ax.set_ylim(-.25, max(130, gap.max() * 1.05))
    ax.set_ylabel("专家 B − 候选平均 B（对称对数）")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    count_ax.bar(frames, large_count, width=8, color="#B34842", alpha=.75)
    count_ax.set_ylabel("分差 > 10\n的观测数")
    count_ax.set_xlabel("专家轨迹中记录的帧索引（SFT 15 Hz）")
    count_ax.set_xlim(frames[0] - 5, frames[-1] + 5)
    count_ax.grid(axis="y", alpha=.2)
    fig.suptitle("专家验证轨迹：评分分差集中在哪些记录帧？", x=.10, ha="left", fontsize=17)
    fig.text(.10, .08,
             "每个散点是一个专家观测与同观测 4 个策略候选的配对评分；不是实际执行效果。",
             fontsize=10)
    fig.text(.10, .055,
             f"大于 10 的观测共 {sum(large_count)} 个；同帧参与轨迹数可能不同。", fontsize=10)
    save(fig, args.output_dir, "val_gap_by_expert_frame")
    plt.close(fig)
    print(f"saved frozen validation B figures -> {args.output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    plot(parse_args())
