"""Diagnose same-observation action distance versus frozen coverage-score difference.

Reads the completed candidate-progress cache only; no policy or critic inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


JOINTS = np.r_[0:6, 7:13]
K = 32
PAIR_I, PAIR_J = np.triu_indices(K, 1)
BIN_STEPS = 30


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite values")
    return {key: float(value) for key, value in zip(
        ("min", "p10", "median", "p90", "p99", "max", "mean"),
        (values.min(), *np.quantile(values, [.1, .5, .9, .99]), values.max(), values.mean()),
    )}


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0) + 1, len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:end]] = (start + end - 1) / 2
    return ranks


def within_state_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    xr, yr = average_tie_ranks(x), average_tie_ranks(y)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def step_bins(observations: list[dict]) -> list[dict]:
    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in observations:
        grouped[row["policy_step"] // BIN_STEPS][row["rollout_index"]].append(row["score_range"])
    result = []
    for index, episodes in sorted(grouped.items()):
        if len(episodes) < 5:
            continue
        values = np.asarray([np.median(v) for v in episodes.values()])
        result.append({"step_lower": index * BIN_STEPS,
                       "step_upper": (index + 1) * BIN_STEPS,
                       "episodes": len(episodes),
                       "median": float(np.median(values)),
                       "q25": float(np.quantile(values, .25)),
                       "q75": float(np.quantile(values, .75))})
    return result


def progress_bins(observations: list[dict], count: int = 20) -> list[dict]:
    """Match the original progress report: episode mean, then episode median."""
    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in observations:
        progress = row["progress"]
        if not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError(f"Invalid relative progress: {progress}")
        index = min(int(progress * count), count - 1)
        grouped[index][row["rollout_index"]].append(row["score_range"])
    result = []
    for index in range(count):
        episodes = grouped[index]
        values = np.asarray([np.mean(scores) for scores in episodes.values()])
        result.append({"progress_lower": index / count,
                       "progress_upper": (index + 1) / count,
                       "episodes": len(episodes),
                       "observations": sum(map(len, episodes.values())),
                       "median": float(np.median(values)) if len(values) else None})
    return result


def event_steps(audit: list[dict]) -> tuple[dict, dict]:
    saved = [r for r in audit if r.get("saved")]
    hil = [r for r in saved if r["interventions"]]
    active = [r for r in hil if r["active_first_step"] is not None]
    if len(saved) != 51 or len(hil) != 50 or len(active) != 46:
        raise ValueError("Audit cohort does not match the formal 51/50/46 records")
    enhancement = {"n": len(active), "mean_step": float(np.mean(
        [int(r["active_first_step"]) for r in active]))}
    takeover = {"n": len(hil), "mean_step": float(np.mean(
        [int(r["intervention_details"][0]["policy_steps_before_intervention"])
         for r in hil]))}
    return enhancement, takeover


def load_and_analyze(root: Path) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads((root / "per_observation_metrics.json").read_text())
    if manifest.get("complete") is not True or len(rows) != 1260 or len(manifest["completed"]) != 1260:
        raise ValueError("Expected complete formal 1260-observation cache")

    all_distances, all_differences, observations = [], [], []
    for row in rows:
        relative_path = row["cache"]
        expected_hash = manifest["completed"].get(relative_path)
        path = root / relative_path
        if not expected_hash or not path.is_file():
            raise ValueError(f"Missing or unmanifested cache: {relative_path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Cache hash mismatch: {relative_path}")
        with np.load(path, allow_pickle=False) as archive:
            chunks = np.asarray(archive["candidates"], dtype=np.float64)
            scores = np.asarray(archive["coverage_scores"], dtype=np.float64)
            order = np.asarray(archive["subset_order"])
        if (chunks.shape != (K, 10, 14) or scores.shape != (K,)
                or sorted(order.tolist()) != list(range(K))
                or not np.isfinite(chunks).all() or not np.isfinite(scores).all()):
            raise ValueError(f"Invalid candidate or score array: {relative_path}")
        first = chunks[:, 0, JOINTS]
        distance = np.sqrt(np.mean((first[PAIR_I] - first[PAIR_J]) ** 2, axis=1))
        difference = np.abs(scores[PAIR_I] - scores[PAIR_J])
        score_range = float(np.ptp(scores))
        if (not np.isclose(distance.mean(), row["metrics"]["32"]["mean_pairwise_rad"], atol=1e-10)
                or not np.isclose(score_range, row["metrics"]["32"]["coverage_range"], atol=1e-10)):
            raise ValueError(f"Recomputed metrics disagree with formal report: {relative_path}")
        observations.append({"rollout_index": int(row["rollout_index"]),
                             "policy_step": int(row["policy_step"]),
                             "progress": float(row["progress"]),
                             "score_range": score_range,
                             "mean_pairwise_mrad": float(distance.mean() * 1000),
                             "spearman_action_vs_score_difference": within_state_spearman(distance, difference),
                             "median_pairwise_score_per_rad": float(np.median(difference / distance))})
        all_distances.append(distance * 1000)
        all_differences.append(difference)
    return np.concatenate(all_distances), np.concatenate(all_differences), observations, manifest


def plot(distance_mrad: np.ndarray, delta_b: np.ndarray, bins: list[dict],
         relative_bins: list[dict],
         enhancement: dict, takeover: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = "Noto Sans CJK JP"
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False,
                         "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False})

    positive = delta_b > 0
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=.13, right=.94, top=.83, bottom=.17)
    fig.suptitle("同观测候选：动作距离与 B 分差", x=.13, y=.97, ha="left", fontsize=18)
    fig.text(.13, .90, "51 条已保存轨迹 · 1260 个观测 · 每个观测 32 候选 / 496 对", fontsize=11)
    density = ax.hexbin(distance_mrad[positive], delta_b[positive],
                        xscale="log", yscale="log", gridsize=62,
                        bins="log", mincnt=1, cmap="viridis")
    bar = fig.colorbar(density, ax=ax, pad=.015)
    bar.set_label("同观测候选对数（颜色对数刻度）")
    ax.set_xlabel("首动作的双臂 12 关节 RMS 距离（mrad，对数轴）")
    ax.set_ylabel("同观测 |ΔB|（对数轴）")
    ax.grid(alpha=.14)
    fig.text(.13, .055, f"颜色表示候选对密度；{int((~positive).sum()):,} 对 ΔB=0 无法置于对数轴（占 {(~positive).mean():.2%}）。", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"same_state_action_vs_b_difference.{suffix}", dpi=190)
    plt.close(fig)

    x = np.asarray([(b["step_lower"] + b["step_upper"]) / 2 for b in bins])
    median = np.asarray([b["median"] for b in bins]) * 1000
    q25 = np.asarray([b["q25"] for b in bins]) * 1000
    q75 = np.asarray([b["q75"] for b in bins]) * 1000
    counts = np.asarray([b["episodes"] for b in bins])
    fig, (ax, support) = plt.subplots(2, 1, sharex=True, figsize=(10, 6.8),
                                      gridspec_kw={"height_ratios": [4, 1], "hspace": .12})
    fig.subplots_adjust(left=.12, right=.96, top=.79, bottom=.16)
    fig.suptitle("B 的同观测候选区分度", x=.12, y=.97, ha="left", fontsize=18)
    fig.text(.12, .90, "每个观测取 32 候选的 max(B) − min(B)；每 30 个策略动作分箱", fontsize=11)
    ax.fill_between(x, q25, q75, color="#387EA0", alpha=.18, label="轨迹间四分位区间")
    ax.plot(x, median, color="#276F92", marker="o", markersize=4.5,
            linewidth=2.2, label="轨迹间中位数")
    for axis in (ax, support):
        axis.axvline(enhancement["mean_step"], color="#5A6170", linewidth=1.8,
                     linestyle=":")
        axis.axvline(takeover["mean_step"], color="#B34842", linewidth=1.8,
                     linestyle="--")
    ax.set_ylabel("候选 B 分数范围（千分之一单位）")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=2)
    support.bar(x, counts, width=24, color="#A8B6C0")
    support.set_ylabel("轨迹数")
    support.set_xlabel("已执行策略动作数（步）")
    support.set_xlim(0, max(650, x[-1] + BIN_STEPS))
    support.set_xticks(np.arange(0, support.get_xlim()[1] + 1, 100))
    fig.text(.12, .075, f"灰点线：平均首次增强采样 {enhancement['mean_step']:.1f} 步（{enhancement['n']} 条）；"
             f"红虚线：平均首次接管 {takeover['mean_step']:.1f} 步（{takeover['n']} 条）。", fontsize=10)
    fig.text(.12, .045, "轨迹内先取中位数、再取轨迹间中位数；少于 5 条轨迹的区间不绘制。", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"b_range_by_policy_step.{suffix}", dpi=190)
    plt.close(fig)

    progress_x = np.asarray([(b["progress_lower"] + b["progress_upper"]) * 50
                             for b in relative_bins])
    progress_y = np.asarray([b["median"] * 1000 if b["median"] is not None else np.nan
                             for b in relative_bins])
    progress_support = np.asarray([b["episodes"] for b in relative_bins])
    fig, (ax, support) = plt.subplots(2, 1, sharex=True, figsize=(10, 6.5),
                                      gridspec_kw={"height_ratios": [4, 1], "hspace": .10})
    fig.subplots_adjust(left=.12, right=.96, top=.83, bottom=.18)
    fig.suptitle("B 的最大候选分差随相对进度变化", x=.12, y=.97, ha="left", fontsize=18)
    fig.text(.12, .90, "K = 32 · 每个观测取 Δmax = max(B) − min(B) · 51 条轨迹 / 1260 个观测", fontsize=11)
    ax.plot(progress_x, progress_y, color="#276F92", marker="o", markersize=5,
            linewidth=2.3)
    ax.set_ylabel("Δmax（千分之一单位）")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.2)
    support.bar(progress_x, progress_support, width=3.9, color="#A8B6C0")
    support.set_xlim(0, 100)
    support.set_xticks(np.arange(0, 101, 10))
    support.set_ylabel("轨迹数")
    support.set_xlabel("接管前或自主结束前的策略执行进度（%）")
    fig.text(.12, .065, "每 5% 分箱：轨迹内先平均，再取轨迹间中位数；不补零、不外推。", fontsize=10)
    fig.text(.12, .04, "100% 是首次接管或自主结束的边界，不代表任务完成度；曲线不包含接管后观测。", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"b_delta_max_vs_progress.{suffix}", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--audit-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    distance, difference, observations, manifest = load_and_analyze(args.cache_dir)
    audit = json.loads(args.audit_records.read_text())
    enhancement, takeover = event_steps(audit)
    bins = step_bins(observations)
    relative_bins = progress_bins(observations)
    correlations = np.asarray([row["spearman_action_vs_score_difference"]
                               for row in observations if row["spearman_action_vs_score_difference"] is not None])
    episode_correlations: dict[int, list[float]] = defaultdict(list)
    for row in observations:
        correlation = row["spearman_action_vs_score_difference"]
        if correlation is not None:
            episode_correlations[row["rollout_index"]].append(correlation)
    episode_medians = np.asarray([np.median(values) for values in episode_correlations.values()])
    report = {
        "protocol": "Same saved pre-first-HIL/autonomous-end observation; chunk first action; 12 arm joints RMS; 32 independent candidates; 496 unordered pairs per observation; no new policy/critic inference",
        "manifest_complete": manifest["complete"],
        "observations": len(observations),
        "trajectories": len({row["rollout_index"] for row in observations}),
        "candidate_pairs": len(distance),
        "zero_score_difference_pairs": int(np.count_nonzero(difference == 0)),
        "joint_distance_mrad": quantiles(distance),
        "absolute_score_difference": quantiles(difference),
        "per_observation_score_range": quantiles(np.asarray([row["score_range"] for row in observations])),
        "per_observation_spearman": quantiles(correlations),
        "per_observation_positive_spearman_fraction": float(np.mean(correlations > 0)),
        "per_trajectory_median_spearman": quantiles(episode_medians),
        "positive_per_trajectory_median_spearman_fraction": float(np.mean(episode_medians > 0)),
        "per_observation_median_score_per_rad": quantiles(np.asarray(
            [row["median_pairwise_score_per_rad"] for row in observations])),
        "enhancement_event": enhancement,
        "takeover_event": takeover,
        "step_bins": bins,
        "progress_bins": relative_bins,
        "interpretation_limit": "Correlation is descriptive only; score differences do not validate long-tail/risk ranking or intervention causality.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sensitivity_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "per_observation_sensitivity.json").write_text(
        json.dumps(observations, indent=2) + "\n")
    plot(distance, difference, bins, relative_bins, enhancement, takeover, args.output_dir)
    print(f"Analyzed {len(observations)} observations and {len(distance)} pairs; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
