#!/usr/bin/env python3
"""Frozen chunk-critic diagnosis on cached HIL candidates; never executes actions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from tail_value.cache import CAMERAS, atomic_json, sha256
from tail_value.chunk import ChunkCoverageCritic
from tail_value.prepare import FrozenEncoder
from plot_candidate_takeover_boundary import first_takeovers

KS = (4, 8, 16, 32)
BIN_STEPS = 30
JOINTS = np.r_[0:6, 7:13]


def score_row(scorer, encoder, source, cache_dir, manifest, device, beta,
              *, include_candidate_scores=False):
    path = cache_dir / source["cache"]
    if sha256(path) != manifest["completed"].get(source["cache"]):
        raise ValueError(f"Changed candidate cache {path}")
    with np.load(path, allow_pickle=False) as archive:
        chunks = np.asarray(archive["candidates"], dtype=np.float32)
        order = np.asarray(archive["subset_order"], dtype=np.int64)
        state = np.asarray(archive["state"], dtype=np.float32)
    if chunks.shape != (32, 10, 14) or sorted(order.tolist()) != list(range(32)):
        raise ValueError("Invalid full candidate chunks/order")
    frame_path = Path(source["frame_path"])
    if not frame_path.is_file() or frame_path.stat().st_size != source["source_size"]:
        raise ValueError(f"Changed HIL observation {frame_path}")
    with frame_path.open("rb") as stream:
        obs = pickle.load(stream)  # trusted local collection only
    raw_state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
    np.testing.assert_allclose(raw_state, state, atol=1e-6, rtol=0)
    images = {key: np.asarray(obs["observation"][camera]["rgb"]) for key, camera in zip(
        CAMERAS, ("head_camera", "left_camera", "right_camera"), strict=True)}
    features = encoder(images)
    with torch.inference_mode():
        scores = scorer.score_many(torch.from_numpy(features)[None].to(device),
                                  torch.from_numpy(state)[None].to(device),
                                  torch.from_numpy(chunks)[None].to(device))[0].cpu().numpy().astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite chunk B score")
    by_k = {}
    for k in KS:
        indices = order[:k]
        selected = scores[indices]
        logits = -beta * (selected - selected.min())
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        uniform = chunks[indices].mean(axis=0)
        tilted = np.tensordot(weights, chunks[indices].astype(np.float64), axes=(0, 0))
        by_k[str(k)] = {"b_mean": float(selected.mean()), "b_range": float(np.ptp(selected)),
                        "beta_b_range": float(beta * np.ptp(selected)),
                        "ess_fraction": float(1 / np.square(weights).sum() / k),
                        "whole_joint_tilt_shift_mrad": float(np.sqrt(np.sum(
                            (tilted[:, JOINTS] - uniform[:, JOINTS]) ** 2)) * 1000)}
    result = {"rollout_index": int(source["rollout_index"]), "policy_step": int(source["policy_step"]),
              "progress": float(source["progress"]), "cache": source["cache"], "k": by_k}
    if include_candidate_scores:
        result["candidate_b_scores"] = scores[order].tolist()
    return result


def bins(rows, metric):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        step_bin = row["policy_step"] // BIN_STEPS
        for k in KS:
            grouped[step_bin][k][row["rollout_index"]].append(row["k"][str(k)][metric])
    out = []
    for index, groups in sorted(grouped.items()):
        counts = {len(groups[k]) for k in KS}
        if len(counts) != 1:
            raise ValueError("K cohorts differ")
        count = counts.pop()
        if count < 5:
            continue
        out.append({"step_lower": index * BIN_STEPS, "step_upper": (index + 1) * BIN_STEPS,
                    "trajectories": count,
                    "mean_by_k": {str(k): float(np.mean([np.mean(values) for values in groups[k].values()]))
                                  for k in KS}})
    return out


def exact_step_ranges(rows, reduction):
    """Reduce per-observation candidate ranges across trajectories at the exact step."""
    if reduction not in ("mean", "median", "max"):
        raise ValueError(f"Unsupported reduction: {reduction}")
    reducer = {"mean": np.mean, "median": np.median, "max": np.max}[reduction]
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["rollout_index"], row["policy_step"])
        if key in seen:
            raise ValueError(f"Duplicate trajectory/step observation: {key}")
        seen.add(key)
        grouped[row["policy_step"]].append(row)
    return [{"policy_step": step, "trajectories": len(at_step),
             f"{reduction}_by_k": {str(k): float(reducer([
                 row["k"][str(k)]["b_range"] for row in at_step])) for k in KS}}
            for step, at_step in sorted(grouped.items())]


def exact_step_mean_ranges(rows):
    return exact_step_ranges(rows, "mean")


def exact_step_median_ranges(rows):
    return exact_step_ranges(rows, "median")


def exact_step_max_ranges(rows):
    return exact_step_ranges(rows, "max")


def exact_step_min_median_max_ranges(rows):
    """Keep the full cross-trajectory envelope at every exact policy step."""
    points = exact_step_median_ranges(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy_step"]].append(row)
    for point in points:
        at_step = grouped[point["policy_step"]]
        point["min_by_k"] = {str(k): float(min(
            row["k"][str(k)]["b_range"] for row in at_step)) for k in KS}
        point["max_by_k"] = {str(k): float(max(
            row["k"][str(k)]["b_range"] for row in at_step)) for k in KS}
    return points


def plot_exact_step_min_median_max_ranges(rows, events, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    points = exact_step_min_median_max_ranges(rows)
    x = [point["policy_step"] for point in points]
    fig = plt.figure(figsize=(12, 9.5))
    grid = fig.add_gridspec(3, 2, height_ratios=(3, 3, 1))
    axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    count_ax = fig.add_subplot(grid[2, :])
    fig.subplots_adjust(left=.10, right=.97, top=.82, bottom=.12, hspace=.18, wspace=.18)
    colors = ("#277DA8", "#DB8A26", "#3D9567", "#9565B7")
    upper = max(point["max_by_k"][str(k)] for point in points for k in KS)
    for ax, k, color in zip(axes, KS, colors, strict=True):
        for name, style, weight, opacity in (("min", ":", 1.2, .65),
                                             ("median", "-", 2.2, 1.),
                                             ("max", "--", 1.2, .75)):
            ax.plot(x, [point[f"{name}_by_k"][str(k)] for point in points],
                    linestyle=style, linewidth=weight, color=color, alpha=opacity,
                    label={"min": "轨迹间最小值", "median": "轨迹间中位数",
                           "max": "轨迹间最大值"}[name])
        ax.axvline(events["enhancement_step"], color="#59636D", linestyle=":", linewidth=1)
        ax.axvline(events["takeover_step"], color="#B34842", linestyle="--", linewidth=1)
        ax.set_title(f"K={k}", loc="left")
        ax.set_yscale("symlog", linthresh=.01)
        ax.set_yticks([0, .01, .1, 1, 10, 100])
        ax.set_ylim(0, upper * 1.12)
        ax.set_xlim(-5, max(x) + 15)
        ax.grid(axis="y", alpha=.2)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("max B − min B")
    count_ax.bar(x, [point["trajectories"] for point in points], width=10,
                 color="#7A9FB5", alpha=.65)
    count_ax.set_ylabel("轨迹数")
    count_ax.set_ylim(0, 55)
    count_ax.set_xlim(-5, max(x) + 15)
    count_ax.set_xlabel("已执行策略动作数（步）")
    fig.suptitle("同一时刻候选 B 分差的跨轨迹范围与中位数", x=.10, ha="left", fontsize=17)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, frameon=False,
               loc="upper center", bbox_to_anchor=(.54, .91))
    fig.text(.10, .065, "每个时刻先在各轨迹内算 max B − min B，再跨轨迹取最小/中位/最大；纵轴为对称对数刻度。", fontsize=10)
    fig.text(.10, .04,
             f"灰线：平均首次增强 {events['enhancement_step']:.1f} 步；"
             f"红线：平均首次接管 {events['takeover_step']:.1f} 步。后期轨迹数减少。", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"b_range_same_step_envelope.{extension}", dpi=180)
    plt.close(fig)
    atomic_json(output_dir / "b_range_same_step_envelope.json", {"points": points,
        "definition": "At each exact policy_step: per-trajectory candidate max B - min B, then min/median/max across trajectories"})
    return points


def plot_exact_step_ranges(rows, events, output_dir, reduction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    values = exact_step_ranges(rows, reduction)
    label = {"mean": "均值", "median": "中位数", "max": "最大值"}[reduction]
    x = [point["policy_step"] for point in values]
    fig, (ax, count_ax) = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True,
                                      gridspec_kw={"height_ratios": [4, 1]})
    fig.subplots_adjust(left=.10, right=.97, top=.80, bottom=.16, hspace=.10)
    for k, color in zip(KS, ("#277DA8", "#DB8A26", "#3D9567", "#9565B7"), strict=True):
        ax.plot(x, [point[f"{reduction}_by_k"][str(k)] for point in values],
                color=color, marker="o", markersize=3.5, linewidth=1.7, label=f"K={k}")
    for current_ax in (ax, count_ax):
        current_ax.axvline(events["enhancement_step"], color="#59636D", linestyle=":", linewidth=1.5)
        current_ax.axvline(events["takeover_step"], color="#B34842", linestyle="--", linewidth=1.5)
        current_ax.grid(axis="y", alpha=.2)
    ax.set_ylabel(f"同一时刻的{label} (max B − min B)")
    ax.set_ylim(bottom=0)
    count_ax.bar(x, [point["trajectories"] for point in values], width=10,
                 color="#7A9FB5", alpha=.65)
    count_ax.set_ylabel("轨迹数")
    count_ax.set_ylim(0, 55)
    count_ax.set_xlim(-5, max(x) + 15)
    count_ax.set_xlabel("已执行策略动作数（步）")
    fig.suptitle(f"逐观测最大分差，在相同策略步数的轨迹间取{label}",
                 x=.10, ha="left", fontsize=17)
    fig.legend(*ax.get_legend_handles_labels(), ncol=4, frameon=False,
               loc="upper center", bbox_to_anchor=(.54, .92))
    fig.text(.10, .055,
             f"不合并相邻时刻；每 15 个策略步有一个缓存观测。后期参与轨迹数减少，{label}更易波动。",
             fontsize=10)
    fig.text(.10, .03,
             f"灰线：平均首次增强 {events['enhancement_step']:.1f} 步；"
             f"红线：平均首次接管 {events['takeover_step']:.1f} 步。", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"b_range_same_step_{reduction}.{extension}", dpi=180)
    plt.close(fig)
    atomic_json(output_dir / f"b_range_same_step_{reduction}.json", {"points": values,
        "definition": f"At each exact policy_step, {reduction} across trajectories of per-observation candidate max B - min B"})
    return values


def plot_exact_step_mean_ranges(rows, events, output_dir):
    return plot_exact_step_ranges(rows, events, output_dir, "mean")


def plot_exact_step_median_ranges(rows, events, output_dir):
    return plot_exact_step_ranges(rows, events, output_dir, "median")


def plot_exact_step_max_ranges(rows, events, output_dir):
    return plot_exact_step_ranges(rows, events, output_dir, "max")


def plot_curve(bin_rows, metric, title, ylabel, events, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    fig.subplots_adjust(left=.12, right=.96, top=.75, bottom=.18)
    x = [(row["step_lower"] + row["step_upper"]) / 2 for row in bin_rows]
    for k, color in zip(KS, ("#277DA8", "#DB8A26", "#3D9567", "#9565B7")):
        ax.plot(x, [row["mean_by_k"][str(k)] for row in bin_rows], marker="o",
                markersize=4, linewidth=2, color=color, label=f"K={k}")
    ax.axvline(events["enhancement_step"], color="#59636D", linestyle=":", linewidth=2,
               label=f"平均首次增强 {events['enhancement_step']:.1f} 步")
    ax.axvline(events["takeover_step"], color="#B34842", linestyle="--", linewidth=2,
               label=f"平均首次接管 {events['takeover_step']:.1f} 步")
    ax.set_title(title, fontsize=17, loc="left")
    ax.set_xlabel("已执行策略动作数（步）")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(650, x[-1] + BIN_STEPS))
    if metric == "ess_fraction":
        minimum = min(row["mean_by_k"][str(k)] for row in bin_rows for k in KS)
        ax.set_ylim(max(0, minimum - 1.25 * (1 - minimum)), 1.00005)
    elif metric != "b_mean":
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center",
               bbox_to_anchor=(.54, .99))
    fig.text(.12, .09, "50 条已保存 HIL 轨迹；1233 个接管前观测；相同观测的 K 条完整 10×14 chunk。", fontsize=10)
    fig.text(.12, .06, "每 30 步：轨迹内平均，再对轨迹平均；少于 5 条轨迹的区间不绘制。", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{metric}_by_policy_step.{extension}", dpi=180)
    plt.close(fig)


def plot_observation_ranges(rows, events, output_dir):
    """One mark per saved observation; no averaging over time or trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(len(KS), 1, figsize=(11.5, 9.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.97, top=.89, bottom=.11, hspace=.14)
    x = [row["policy_step"] for row in rows]
    upper = max(row["k"][str(k)]["b_range"] for row in rows for k in KS)
    colors = ("#277DA8", "#DB8A26", "#3D9567", "#9565B7")
    for ax, k, color in zip(axes, KS, colors, strict=True):
        y = [row["k"][str(k)]["b_range"] for row in rows]
        ax.scatter(x, y, s=11, color=color, alpha=.5, linewidths=0, rasterized=True)
        ax.axvline(events["enhancement_step"], color="#59636D", linestyle=":", linewidth=1)
        ax.axvline(events["takeover_step"], color="#B34842", linestyle="--", linewidth=1)
        ax.set_yscale("symlog", linthresh=.01)
        ax.set_yticks([0, .01, .1, 1, 10, 100])
        ax.set_ylim(0, upper * 1.1)
        ax.set_xlim(0, max(x) + 15)
        ax.set_ylabel(f"K={k}\nmax B − min B")
        ax.grid(axis="y", alpha=.2)
    axes[-1].set_xlabel("该观测的已执行策略动作数（步）")
    fig.suptitle("每个观测时刻的候选 B 最大分差", x=.10, ha="left", fontsize=17)
    fig.text(.10, .04, "每个点是一条轨迹中的一次观测；纵轴为对称对数刻度，0—0.01 为线性段。", fontsize=10)
    fig.text(.10, .02,
             f"灰线：平均首次增强 {events['enhancement_step']:.1f} 步；"
             f"红线：平均首次接管 {events['takeover_step']:.1f} 步。", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"b_range_per_observation.{extension}", dpi=180)
    plt.close(fig)


def run(args):
    if args.beta < 0 or not np.isfinite(args.beta):
        raise ValueError("beta must be finite/nonnegative")
    torch.set_num_threads(args.cpu_threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (checkpoint.get("representation") != "absolute_joint_chunk"
            or checkpoint["config"].get("horizon") != 10):
        raise ValueError("Checkpoint does not contain a 10-step chunk critic")
    model = ChunkCoverageCritic(checkpoint["normalization"], width=checkpoint["config"]["width"])
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval().requires_grad_(False)
    encoder = FrozenEncoder(device, args.encoder_weights)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    sources = json.loads((args.cache_dir / "per_observation_metrics.json").read_text())
    audit = json.loads(args.audit_records.read_text())
    records = first_takeovers(audit)
    selected = [s for s in sources if int(s["rollout_index"]) in records]
    if (manifest.get("complete") is not True or len(manifest.get("completed", {})) != 1260
            or len(selected) != 1233):
        raise ValueError("Expected complete formal HIL candidate cache and 50 HIL trajectories")
    events = {"takeover_step": float(np.mean([int(r["intervention_details"][0]["policy_steps_before_intervention"])
                                               for r in records.values()])),
              "enhancement_step": float(np.mean([int(r["active_first_step"])
                                                   for r in records.values() if r["active_first_step"] is not None]))}
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in selected:
        rows.append(score_row(model, encoder, source, args.cache_dir, manifest, device, args.beta))
        if len(rows) % 50 == 0:
            print(f"scored {len(rows)}/{len(selected)}", flush=True)
    metrics = {}
    names = ("b_mean", "b_range", "beta_b_range", "ess_fraction", "whole_joint_tilt_shift_mrad")
    for k in KS:
        metrics[str(k)] = {}
        for name in names:
            values = [row["k"][str(k)][name] for row in rows]
            metrics[str(k)][name] = {"median": float(np.median(values)),
                                       "p90": float(np.quantile(values, .9)),
                                       "mean": float(np.mean(values))}
    curves = {name: bins(rows, name) for name in names}
    plot_observation_ranges(rows, events, args.output_dir)
    plot_exact_step_mean_ranges(rows, events, args.output_dir)
    plot_exact_step_median_ranges(rows, events, args.output_dir)
    plot_exact_step_max_ranges(rows, events, args.output_dir)
    plot_exact_step_min_median_max_ranges(rows, events, args.output_dir)
    for name, title, ylabel in (("b_range", "同观测 B 分差：30 步区间轨迹均值", "max B − min B"),
                                ("b_mean", "完整 chunk 的 B 评分", "候选平均 B"),
                                ("ess_fraction", "完整 chunk 倾斜权重有效样本比例", "ESS / K")):
        plot_curve(curves[name], name, title, ylabel, events, args.output_dir)
    atomic_json(args.output_dir / "metrics.json", {"checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint), "step": checkpoint["step"], "beta": args.beta,
        "trajectories": 50, "observations": len(rows), "events": events,
        "summary_by_k": metrics, "step_curves": curves,
        "limitations": ["Offline policy candidates at saved HIL observations, not executed branch outcomes",
                        "critic trained on 15 Hz SFT windows; HIL execution is 30 Hz",
                        "no HIL coverage ground truth"]})
    atomic_json(args.output_dir / "per_observation.json", rows)
    print(f"complete: {len(rows)} observations -> {args.output_dir}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--audit-records", type=Path, required=True)
    p.add_argument("--encoder-weights", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--beta", type=float, default=10.)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cpu-threads", type=int, default=4)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
