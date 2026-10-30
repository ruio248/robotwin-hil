#!/usr/bin/env python3
"""Read-only analysis of complete action chunks in the formal candidate cache.

The cached critic score is for the first action at the current observation.
Distance or a changed score ranking is not a recovery/failure label.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


KS = (4, 8, 16, 32)
JOINTS = np.r_[0:6, 7:13]
GRIPPERS = (6, 13)
N_BINS = 20


def describe(values: list[float] | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite values")
    return {"median": float(np.median(x)), "mean": float(x.mean()),
            "p10": float(np.quantile(x, .1)), "p90": float(np.quantile(x, .9)),
            "max": float(x.max())}


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1, len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(bounds[:-1], bounds[1:]):
        ranks[order[start:end]] = (start + end - 1) / 2
    return ranks


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    a, b = average_ranks(left), average_ranks(right)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def pool_metrics(pool: np.ndarray, scores: np.ndarray) -> dict:
    k = len(pool)
    if pool.shape != (k, 10, 14) or scores.shape != (k,) or k < 2:
        raise ValueError("Expected K>=2 chunks of shape [10,14] with first-action scores")
    i, j = np.triu_indices(k, 1)
    joint_delta = pool[i][:, :, JOINTS] - pool[j][:, :, JOINTS]
    by_step = np.sqrt(np.mean(joint_delta * joint_delta, axis=2)) * 1000
    whole = np.sqrt(np.mean(joint_delta * joint_delta, axis=(1, 2))) * 1000
    mean_by_step = np.mean(by_step, axis=0)
    max_by_step = np.max(by_step, axis=0)
    score_gap = np.abs(scores[i] - scores[j])
    grips = np.abs(pool[i][:, :, GRIPPERS] - pool[j][:, :, GRIPPERS])
    return {
        "mean_pairwise_whole_mrad": float(whole.mean()),
        "max_pairwise_whole_mrad": float(whole.max()),
        "mean_pairwise_first_mrad": float(mean_by_step[0]),
        "mean_pairwise_last_mrad": float(mean_by_step[-1]),
        "mean_pairwise_by_step_mrad": mean_by_step.tolist(),
        "max_pairwise_by_step_mrad": max_by_step.tolist(),
        "left_gripper_mean_abs_whole": float(grips[:, :, 0].mean()),
        "right_gripper_mean_abs_whole": float(grips[:, :, 1].mean()),
        "score_range_first": float(np.ptp(scores)),
        "pair_rank_rho_first_distance_vs_score_gap": rank_correlation(by_step[:, 0], score_gap),
        "pair_rank_rho_whole_distance_vs_score_gap": rank_correlation(whole, score_gap),
        "fraction_pairs_last_gt_twice_first": float(np.mean(by_step[:, -1] > 2 * by_step[:, 0])),
        "fraction_pairs_whole_gt_1p5_first": float(np.mean(whole > 1.5 * by_step[:, 0])),
    }


def load_rows(cache_dir: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    sources = json.loads((cache_dir / "per_observation_metrics.json").read_text())
    if (manifest.get("complete") is not True or len(sources) != 1260
            or len(manifest.get("completed", {})) != 1260):
        raise ValueError("Formal cache must contain all 1260 completed observations")
    rows = []
    for source in sources:
        relative = source["cache"]
        path = cache_dir / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["completed"][relative]:
            raise ValueError(f"Missing or corrupted cache: {relative}")
        with np.load(path, allow_pickle=False) as archive:
            chunks = np.asarray(archive["candidates"], dtype=np.float64)
            order = np.asarray(archive["subset_order"], dtype=np.int64)
            scores = np.asarray(archive["coverage_scores"], dtype=np.float64)
        if (chunks.shape != (32, 10, 14) or scores.shape != (32,)
                or sorted(order.tolist()) != list(range(32))
                or not np.isfinite(chunks).all() or not np.isfinite(scores).all()):
            raise ValueError(f"Bad candidate array: {relative}")
        metrics = {}
        for k in KS:
            selected = order[:k]
            metrics[str(k)] = pool_metrics(chunks[selected], scores[selected])
            expected = source["metrics"][str(k)]
            actual = metrics[str(k)]
            if (not np.isclose(actual["mean_pairwise_first_mrad"] / 1000,
                               expected["mean_pairwise_rad"], atol=1e-10)
                    or not np.isclose(np.max(actual["max_pairwise_by_step_mrad"][:1]) / 1000,
                                      expected["max_pairwise_rad"], atol=1e-10)):
                raise ValueError(f"First-action result disagrees with formal report: {relative}")
        rows.append({"cache": relative, "rollout_index": int(source["rollout_index"]),
                     "episode_index": int(source["episode_index"]),
                     "frame_index": int(source["frame_index"]),
                     "policy_step": int(source["policy_step"]),
                     "progress": float(source["progress"]), "k": metrics})
    return rows, manifest


def episode_balanced_bins(rows: list[dict], k: int, metric: str) -> list[dict]:
    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        p = row["progress"]
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"Bad progress {p}")
        index = min(int(p * N_BINS), N_BINS - 1)
        grouped[index][row["rollout_index"]].append(row["k"][str(k)][metric])
    bins = []
    for index in range(N_BINS):
        groups = grouped[index]
        values = [float(np.mean(v)) for v in groups.values()]
        bins.append({"progress_lower": index / N_BINS,
                     "progress_upper": (index + 1) / N_BINS,
                     "episodes": len(groups), "observations": sum(map(len, groups.values())),
                     "episode_mean_distribution": describe(values) if values else None})
    return bins


def build_summary(rows: list[dict], manifest: dict) -> dict:
    metrics = ("mean_pairwise_whole_mrad", "max_pairwise_whole_mrad",
               "mean_pairwise_first_mrad", "mean_pairwise_last_mrad",
               "left_gripper_mean_abs_whole", "right_gripper_mean_abs_whole",
               "score_range_first", "fraction_pairs_last_gt_twice_first",
               "fraction_pairs_whole_gt_1p5_first")
    result = {
        "protocol": {
            "source_manifest_identity": manifest["identity"],
            "observations": len(rows),
            "trajectories": len({r["rollout_index"] for r in rows}),
            "candidate_ks": list(KS), "chunk_steps": 10,
            "joint_distance": "RMS over 12 arm joints, units mrad; all 10 aligned steps for whole-chunk RMS",
            "gripper_distance": "Absolute physical target difference, averaged over pairs and all 10 steps; original gripper units",
            "score": "Frozen original B(o_t, first action), not B of a chunk",
            "progress": "policy command count / command count before first takeover or autonomous end; not semantic stage",
            "aggregation": "Overall: per-observation distribution. Progress: mean within rollout/bin, median across rollouts",
            "stage": "Not inferred from progress or eventual takeover stage; recorded policy stage IDs must be audited separately",
        }, "overall": {}, "progress": {}, "step_profile": {}, "score_vs_distance": {},
    }
    for k in KS:
        subset = [r["k"][str(k)] for r in rows]
        result["overall"][str(k)] = {metric: describe([x[metric] for x in subset]) for metric in metrics}
        result["step_profile"][str(k)] = {
            "mean_pairwise_mrad": [describe([x["mean_pairwise_by_step_mrad"][h] for x in subset])
                                   for h in range(10)],
            "max_pairwise_mrad": [describe([x["max_pairwise_by_step_mrad"][h] for x in subset])
                                  for h in range(10)],
        }
        result["progress"][str(k)] = {
            metric: episode_balanced_bins(rows, k, metric)
            for metric in ("mean_pairwise_whole_mrad", "max_pairwise_whole_mrad",
                           "left_gripper_mean_abs_whole", "right_gripper_mean_abs_whole")
        }
        result["score_vs_distance"][str(k)] = {
            metric: describe([x[metric] for x in subset if x[metric] is not None])
            for metric in ("pair_rank_rho_first_distance_vs_score_gap",
                           "pair_rank_rho_whole_distance_vs_score_gap")
        }
    return result


def crosscheck_score_windows(rows: list[dict], manifest: dict, score_dir: Path) -> dict:
    """Join the earlier frozen-current-observation B-window ablation by cache key."""
    protocol = json.loads((score_dir / "protocol.json").read_text())
    if (protocol["source_identity"] != manifest["identity"]
            or protocol["critic_sha256"] != manifest["config"]["critic_sha256"]
            or protocol["source_observations"] != len(rows)):
        raise ValueError("Score-window ablation does not match formal candidate cache")
    scored = {}
    for line in (score_dir / "per_observation.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["cache"] in scored:
            raise ValueError(f"Duplicate score-window entry: {entry['cache']}")
        scored[entry["cache"]] = entry
    if set(scored) != {row["cache"] for row in rows}:
        raise ValueError("Score-window ablation and chunk analysis have different observations")

    result = {"protocol": protocol, "by_k": {}, "k4_diversity_deciles": []}
    for k in KS:
        entries = [scored[row["cache"]]["k"][str(k)] for row in rows]
        first_range = np.asarray([e["first"]["b_range"] for e in entries])
        mean10_range = np.asarray([e["mean10"]["b_range"] for e in entries])
        whole_spread = np.asarray([row["k"][str(k)]["mean_pairwise_whole_mrad"] for row in rows])
        result["by_k"][str(k)] = {
            "first_score_range": describe(first_range),
            "mean10_score_range": describe(mean10_range),
            "mean10_winner_changed_fraction": float(np.mean([
                e["mean10"]["winner_changed_from_first"] for e in entries])),
            "mean10_weight_tv_from_first": describe([
                e["mean10"]["weight_tv_from_first"] for e in entries]),
            "mean10_ess_fraction": describe([e["mean10"]["ess_fraction"] for e in entries]),
            "mean10_blended_chunk_shift_from_first_mrad": describe([
                e["mean10"]["whole_chunk_shift_from_first_mrad"] for e in entries]),
            "rho_whole_spread_vs_first_score_range": rank_correlation(whole_spread, first_range),
            "rho_whole_spread_vs_mean10_score_range": rank_correlation(whole_spread, mean10_range),
        }

    spread = np.asarray([row["k"]["4"]["mean_pairwise_whole_mrad"] for row in rows])
    order = np.argsort(spread, kind="stable")
    for index, selected in enumerate(np.array_split(order, 10)):
        data = [scored[rows[i]["cache"]]["k"]["4"] for i in selected]
        result["k4_diversity_deciles"].append({
            "decile": index + 1, "observations": len(selected),
            "whole_chunk_spread_mrad": describe(spread[selected]),
            "first_score_range": describe([e["first"]["b_range"] for e in data]),
            "mean10_score_range": describe([e["mean10"]["b_range"] for e in data]),
            "mean10_winner_changed_fraction": float(np.mean([
                e["mean10"]["winner_changed_from_first"] for e in data])),
            "mean10_blended_chunk_shift_from_first_mrad": describe([
                e["mean10"]["whole_chunk_shift_from_first_mrad"] for e in data]),
        })
    return result


def draw(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = {4: "#287BA4", 8: "#C58024", 16: "#499A73", 32: "#8B65A5"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, kind, label in zip(axes, ("mean_pairwise_mrad", "max_pairwise_mrad"),
                               ("Mean pairwise distance", "Maximum pairwise distance")):
        for k in KS:
            values = summary["step_profile"][str(k)][kind]
            ax.plot(range(1, 11), [v["median"] for v in values], marker="o",
                    label=f"K={k}", color=colors[k], linewidth=2)
        ax.set_ylabel(label + " (mrad)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(ncol=4, frameon=False)
    axes[-1].set_xticks(range(1, 11))
    axes[-1].set_xlabel("Action position within 10-step chunk")
    fig.suptitle("Candidate diversity across the complete action chunk | 1260 observations")
    fig.text(.5, .015, "Same-observation candidates; 12 arm joints; median across observations. Grippers analyzed separately.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .03, 1, .95])
    for ext in ("png", "pdf"):
        fig.savefig(output / f"chunk_diversity_over_horizon.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, .8]})
    for ax, metric, label in zip(axes[:2], ("mean_pairwise_whole_mrad", "max_pairwise_whole_mrad"),
                                  ("Mean pairwise whole-chunk RMS", "Max pairwise whole-chunk RMS")):
        for k in KS:
            bins = summary["progress"][str(k)][metric]
            x = [(b["progress_lower"] + b["progress_upper"]) * 50 for b in bins]
            y = [b["episode_mean_distribution"]["median"] if b["episodes"] else np.nan for b in bins]
            ax.plot(x, y, marker="o", markersize=3.5, linewidth=2, color=colors[k], label=f"K={k}")
        ax.set_ylabel(label + " (mrad)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(ncol=4, frameon=False)
    bins = summary["progress"]["32"]["mean_pairwise_whole_mrad"]
    axes[2].bar([(b["progress_lower"] + b["progress_upper"]) * 50 for b in bins],
                [b["episodes"] for b in bins], width=3.8, color="#AFC2CC")
    axes[2].set_ylabel("Rollouts")
    axes[2].set_xlim(0, 100)
    axes[2].set_xticks(np.arange(0, 101, 10))
    axes[2].set_xlabel("Pre-takeover/autonomous-end policy progress (%)")
    fig.suptitle("Whole-chunk candidate diversity by policy progress | 51 saved rollouts")
    fig.text(.5, .012, "5% bins: average observations within rollout, then median across rollouts; progress is not task stage.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .03, 1, .95])
    for ext in ("png", "pdf"):
        fig.savefig(output / f"chunk_diversity_vs_progress.{ext}", dpi=180)
    plt.close(fig)


def draw_score_crosscheck(crosscheck: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = crosscheck["k4_diversity_deciles"]
    x = [row["decile"] for row in data]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(x, [row["first_score_range"]["median"] * 1000 for row in data],
                 marker="o", color="#287BA4", label="First-action B range")
    axes[0].plot(x, [row["mean10_score_range"]["median"] * 1000 for row in data],
                 marker="s", color="#C58024", label="Mean-10 B range")
    axes[0].set_ylabel("Candidate score range (×10⁻³)")
    axes[0].legend(frameon=False)
    axes[1].plot(x, [row["mean10_blended_chunk_shift_from_first_mrad"]["median"] for row in data],
                 marker="o", color="#499A73")
    axes[1].set_ylabel("Mean-10 vs first weighted\nchunk RMS shift (mrad)")
    axes[1].set_xlabel("Decile of K=4 whole-chunk candidate diversity")
    axes[1].set_xticks(x)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(bottom=0)
    fig.suptitle("Does first-action scoring miss later candidate divergence? | 1260 observations")
    fig.text(.5, .01, "Same frozen current observation for every B evaluation; mean-10 is a diagnostic, not trained chunk value.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, .03, 1, .95])
    for ext in ("png", "pdf"):
        fig.savefig(output / f"score_window_vs_chunk_diversity.{ext}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-ablation-dir", type=Path)
    args = parser.parse_args()
    rows, manifest = load_rows(args.cache_dir)
    summary = build_summary(rows, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "full_chunk_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "per_observation_full_chunks.json").write_text(json.dumps(rows, indent=2) + "\n")
    draw(summary, args.output_dir)
    if args.score_ablation_dir is not None:
        crosscheck = crosscheck_score_windows(rows, manifest, args.score_ablation_dir)
        (args.output_dir / "score_window_crosscheck.json").write_text(json.dumps(crosscheck, indent=2) + "\n")
        draw_score_crosscheck(crosscheck, args.output_dir)
    print(f"Analyzed {len(rows)} complete candidate pools from {summary['protocol']['trajectories']} rollouts")


if __name__ == "__main__":
    main()
