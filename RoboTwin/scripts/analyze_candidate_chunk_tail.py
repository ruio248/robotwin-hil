"""Compare first and final action diversity in the frozen policy's 10-step chunks.

Uses the existing formal 32-candidate cache.  Coverage scores are *only* the
original first-action scores; no final-action score is inferred or fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


KS = (4, 8, 16, 32)
JOINTS = np.r_[0:6, 7:13]
GRIPPERS = (6, 13)
ACTION_INDEX = (0, 9)
PROGRESS_BINS = 20


def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite values")
    return {name: float(value) for name, value in zip(
        ("median", "p10", "p90", "mean"),
        (np.median(values), *np.quantile(values, [.1, .9]), np.mean(values)),
    )}


def diversity(actions: np.ndarray) -> dict[str, float]:
    k = actions.shape[0]
    if actions.shape != (k, 14) or k < 2:
        raise ValueError("Expected K>=2 physical 14D action targets")
    i, j = np.triu_indices(k, 1)
    joint_distance = np.sqrt(np.mean((actions[i][:, JOINTS] - actions[j][:, JOINTS]) ** 2, axis=1))
    return {"mean_pairwise_mrad": float(joint_distance.mean() * 1000),
            "max_pairwise_mrad": float(joint_distance.max() * 1000),
            "left_gripper_mean_abs": float(np.mean(np.abs(actions[i, GRIPPERS[0]] - actions[j, GRIPPERS[0]]))),
            "right_gripper_mean_abs": float(np.mean(np.abs(actions[i, GRIPPERS[1]] - actions[j, GRIPPERS[1]])))}


def progress_summary(rows: list[dict], metric: str, k: int = 32) -> list[dict]:
    grouped: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        progress = row["progress"]
        if not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError(f"Invalid relative progress: {progress}")
        index = min(int(progress * PROGRESS_BINS), PROGRESS_BINS - 1)
        grouped[index][row["rollout_index"]].append(row["k"][str(k)])
    summary = []
    for index in range(PROGRESS_BINS):
        episodes = grouped[index]
        first = [np.mean([item["first"][metric] for item in items]) for items in episodes.values()]
        last = [np.mean([item["last"][metric] for item in items]) for items in episodes.values()]
        summary.append({"progress_lower": index / PROGRESS_BINS,
                        "progress_upper": (index + 1) / PROGRESS_BINS,
                        "episodes": len(episodes),
                        "observations": sum(map(len, episodes.values())),
                        "first_episode_median": float(np.median(first)) if first else None,
                        "last_episode_median": float(np.median(last)) if last else None})
    return summary


def analyze(cache_dir: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    source_rows = json.loads((cache_dir / "per_observation_metrics.json").read_text())
    if manifest.get("complete") is not True or len(source_rows) != 1260 or len(manifest["completed"]) != 1260:
        raise ValueError("Expected completed formal 1260-observation candidate cache")
    beta = float(manifest["config"]["beta"])
    result = []
    for source in source_rows:
        relative_path = source["cache"]
        path = cache_dir / relative_path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["completed"][relative_path]:
            raise ValueError(f"Missing or corrupted candidate cache: {relative_path}")
        with np.load(path, allow_pickle=False) as archive:
            chunks = np.asarray(archive["candidates"], dtype=np.float64)
            order = np.asarray(archive["subset_order"], dtype=np.int64)
            scores = np.asarray(archive["coverage_scores"], dtype=np.float64)
        if (chunks.shape != (32, 10, 14) or scores.shape != (32,)
                or sorted(order.tolist()) != list(range(32))
                or not np.isfinite(chunks).all() or not np.isfinite(scores).all()):
            raise ValueError(f"Invalid candidates/scores: {relative_path}")
        by_k = {}
        for k in KS:
            selected = order[:k]
            pool = chunks[selected]
            first = diversity(pool[:, ACTION_INDEX[0], :])
            last = diversity(pool[:, ACTION_INDEX[1], :])
            expected = source["metrics"][str(k)]
            if (not np.isclose(first["mean_pairwise_mrad"] / 1000, expected["mean_pairwise_rad"], atol=1e-10)
                    or not np.isclose(first["max_pairwise_mrad"] / 1000, expected["max_pairwise_rad"], atol=1e-10)):
                raise ValueError(f"First-action diversity differs from formal report: {relative_path}")
            b = scores[selected]
            weights = np.exp(-beta * (b - b.min()))
            weights /= weights.sum()
            shifts = {}
            for name, action_index in zip(("first", "last"), ACTION_INDEX):
                action_pool = pool[:, action_index, :]
                weighted = np.sum(weights[:, None] * action_pool, axis=0)
                uniform = action_pool.mean(axis=0)
                shifts[name] = float(np.sqrt(np.mean((weighted[JOINTS] - uniform[JOINTS]) ** 2)) * 1000)
            by_k[str(k)] = {"first": first, "last": last,
                            "tail_over_first_mean_pairwise": last["mean_pairwise_mrad"] / first["mean_pairwise_mrad"],
                            "tilted_vs_uniform_mrad": shifts}
        result.append({"rollout_index": int(source["rollout_index"]),
                       "frame_index": int(source["frame_index"]),
                       "policy_step": int(source["policy_step"]),
                       "progress": float(source["progress"]),
                       "k": by_k})
    return result, manifest


def plot_progress(bins: list[dict], metric: str, title: str, stem: Path) -> None:
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
    x = np.asarray([(row["progress_lower"] + row["progress_upper"]) * 50 for row in bins])
    first = np.asarray([row["first_episode_median"] if row["episodes"] else np.nan for row in bins])
    last = np.asarray([row["last_episode_median"] if row["episodes"] else np.nan for row in bins])
    support = np.asarray([row["episodes"] for row in bins])
    fig, (ax, counts) = plt.subplots(2, 1, sharex=True, figsize=(10, 6.6),
                                     gridspec_kw={"height_ratios": [4, 1], "hspace": .10})
    fig.subplots_adjust(left=.12, right=.96, top=.80, bottom=.18)
    fig.suptitle(title, x=.12, y=.97, ha="left", fontsize=18)
    fig.text(.12, .90, "K=32 · 同观测独立采样 · 51 条已保存轨迹 / 1260 个观测 · 双臂 12 关节 RMS", fontsize=11)
    ax.plot(x, first, marker="o", markersize=4.5, linewidth=2.2,
            color="#277DA8", label="chunk 第 1 步")
    ax.plot(x, last, marker="s", markersize=4.5, linewidth=2.2,
            color="#DB8A26", label="chunk 第 10 步")
    ax.set_ylabel(("平均" if metric == "mean_pairwise_mrad" else "最大") + "两两关节差异（mrad）")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    counts.bar(x, support, width=3.9, color="#A8B6C0")
    counts.set_xlim(0, 100)
    counts.set_xticks(np.arange(0, 101, 10))
    counts.set_ylabel("轨迹数")
    counts.set_xlabel("接管前或自主结束前的策略执行进度（%）")
    fig.text(.12, .06, "每 5% 分箱：轨迹内先平均，再取轨迹间中位数；两条线使用完全相同的观测和候选。", fontsize=10)
    fig.text(.12, .035, "末动作仅用于动作差异诊断；B 没有在未来观测下重新评分。", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = analyze(args.cache_dir)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = {"protocol": "Compare candidate chunks' first (index 0) and final (index 9) physical actions at the same saved observation; RMS over 12 arm joints, grippers reported separately. No future-observation B score.",
               "observations": len(rows),
               "trajectories": len({row["rollout_index"] for row in rows}),
               "candidate_count_per_observation": 32,
               "chunk_length": 10,
               "beta_for_cached_first_action_scores": float(manifest["config"]["beta"]),
               "groups": {}, "progress": {}}
    for k in KS:
        group = [row["k"][str(k)] for row in rows]
        summary["groups"][str(k)] = {
            "first": {metric: quantiles([entry["first"][metric] for entry in group])
                      for metric in ("mean_pairwise_mrad", "max_pairwise_mrad",
                                     "left_gripper_mean_abs", "right_gripper_mean_abs")},
            "last": {metric: quantiles([entry["last"][metric] for entry in group])
                     for metric in ("mean_pairwise_mrad", "max_pairwise_mrad",
                                    "left_gripper_mean_abs", "right_gripper_mean_abs")},
            "tail_over_first_mean_pairwise": quantiles(
                [entry["tail_over_first_mean_pairwise"] for entry in group]),
            "fraction_tail_mean_pairwise_larger": float(np.mean([
                entry["last"]["mean_pairwise_mrad"] > entry["first"]["mean_pairwise_mrad"]
                for entry in group])),
            "first_action_score_tilt_shift_mrad": {
                action: quantiles([entry["tilted_vs_uniform_mrad"][action] for entry in group])
                for action in ("first", "last")},
        }
    for metric, title, filename in (
        ("mean_pairwise_mrad", "首动作与末动作：平均两两候选差异", "mean_pairwise_first_vs_last"),
        ("max_pairwise_mrad", "首动作与末动作：最大两两候选差异", "max_pairwise_first_vs_last"),
    ):
        bins = progress_summary(rows, metric)
        summary["progress"][metric] = bins
        plot_progress(bins, metric, title, output / filename)
    (output / "chunk_tail_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "per_observation_chunk_tail.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Analyzed {len(rows)} observations; wrote {output}")


if __name__ == "__main__":
    main()
