#!/usr/bin/env python3
"""Plot full 10-step candidate-chunk Frobenius distances against policy steps.

Reuses the formal 32-candidate cache and the saved-HIL cohort of the original
first-action figure. Cached B scores are deliberately not used here.
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
BIN_STEPS = 30
COLORS = ("#277DA8", "#DB8A26", "#3D9567", "#9565B7")
MARKERS = ("o", "s", "^", "D")
METRICS = (
    ("mean_joint_f_mrad", "平均两两完整 chunk 关节 Frobenius 距离", "平均两两 $\\|A-B\\|_F$（mrad；10×12 关节）", "mean_joint_frobenius"),
    ("max_joint_f_mrad", "最大两两完整 chunk 关节 Frobenius 距离", "最大两两 $\\|A-B\\|_F$（mrad；10×12 关节）", "max_joint_frobenius"),
    ("mean_raw14_f", "平均两两完整 chunk 14 维 Frobenius 距离", "平均两两 $\\|A-B\\|_F$（原始混合单位；10×14）", "mean_raw14_frobenius"),
    ("max_raw14_f", "最大两两完整 chunk 14 维 Frobenius 距离", "最大两两 $\\|A-B\\|_F$（原始混合单位；10×14）", "max_raw14_frobenius"),
)


def pool_distances(chunks: np.ndarray, order: np.ndarray, k: int) -> dict[str, float]:
    """Pairwise Frobenius norms across aligned horizon positions and dimensions."""
    if chunks.shape != (32, 10, 14) or order.shape != (32,):
        raise ValueError("Expected 32 candidate [10,14] chunks and an order permutation")
    pool = np.asarray(chunks[order[:k]], dtype=np.float64)
    i, j = np.triu_indices(k, 1)
    difference = pool[i] - pool[j]
    joint = np.sqrt(np.sum(difference[:, :, JOINTS] ** 2, axis=(1, 2)))
    all14 = np.sqrt(np.sum(difference ** 2, axis=(1, 2)))
    gripper = np.sqrt(np.sum(difference[:, :, GRIPPERS] ** 2, axis=(1, 2)))
    return {
        "mean_joint_f_mrad": float(joint.mean() * 1000),
        "max_joint_f_mrad": float(joint.max() * 1000),
        "mean_raw14_f": float(all14.mean()),
        "max_raw14_f": float(all14.max()),
        "mean_gripper_f_raw": float(gripper.mean()),
    }


def summarize_bins(rows: list[dict], metric: str) -> list[dict]:
    grouped: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        index = row["policy_step"] // BIN_STEPS
        for k in KS:
            value = row["k"][str(k)][metric]
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid metric {metric}: {value}")
            grouped[index][k][row["rollout_index"]].append(value)
    bins = []
    for index in sorted(grouped):
        episodes = grouped[index]
        counts = {len(episodes[k]) for k in KS}
        if len(counts) != 1:
            raise ValueError("K groups have different rollout support")
        count = counts.pop()
        if count < 5:
            continue
        bins.append({"policy_step_lower": index * BIN_STEPS,
                     "policy_step_upper": (index + 1) * BIN_STEPS,
                     "episodes": count,
                     "mean_by_k": {str(k): float(np.mean([
                         np.mean(values) for values in episodes[k].values()])) for k in KS}})
    if not bins:
        raise ValueError("No bins with at least five trajectories")
    return bins


def report_quantiles(rows: list[dict]) -> dict:
    out = {}
    for k in KS:
        out[str(k)] = {}
        for metric in (name for name, *_ in METRICS):
            values = np.asarray([row["k"][str(k)][metric] for row in rows])
            out[str(k)][metric] = {"median": float(np.median(values)),
                                   "p10": float(np.quantile(values, .1)),
                                   "p90": float(np.quantile(values, .9))}
        values = np.asarray([row["k"][str(k)]["mean_gripper_f_raw"] for row in rows])
        out[str(k)]["mean_gripper_f_raw"] = {
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9))}
    return out


def draw(bins: list[dict], title: str, ylabel: str, output: Path,
         mean_takeover: float, mean_enhancement: float, enhancement_n: int,
         note: str) -> None:
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
    x = [(b["policy_step_lower"] + b["policy_step_upper"]) / 2 for b in bins]
    fig, ax = plt.subplots(figsize=(11, 6.7))
    fig.subplots_adjust(left=.13, right=.97, top=.78, bottom=.18)
    fig.suptitle(title, x=.13, ha="left", y=.97, fontsize=17)
    fig.text(.13, .905, "K = 4 / 8 / 16 / 32 · 50 条已保存 HIL 轨迹 · 1233 个接管前观测 · 完整 10 步 chunk", fontsize=10.5)
    for k, color, marker in zip(KS, COLORS, MARKERS):
        ax.plot(x, [b["mean_by_k"][str(k)] for b in bins], color=color,
                marker=marker, markersize=4.5, linewidth=2.2, label=f"K = {k}")
    ax.axvline(mean_takeover, color="#B34842", linewidth=2, linestyle="--",
               label=f"平均首次接管：{mean_takeover:.1f} 步")
    ax.axvline(mean_enhancement, color="#5A6170", linewidth=2, linestyle=":",
               label=f"平均首次增强采样：{mean_enhancement:.1f} 步（{enhancement_n}/50）")
    ax.set_xlim(0, max(650, x[-1] + BIN_STEPS))
    ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 100))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("已执行策略动作数（步）")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.015))
    fig.text(.13, .075, "每 30 步分箱：轨迹内先平均，再取轨迹间平均；少于 5 条轨迹的区间不绘制。", fontsize=9.5)
    fig.text(.13, .048, note, fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{ext}"), dpi=190)
    plt.close(fig)


def run(cache_dir: Path, audit_path: Path, output_dir: Path) -> None:
    from plot_candidate_takeover_boundary import first_takeovers

    manifest = json.loads((cache_dir / "manifest.json").read_text())
    sources = json.loads((cache_dir / "per_observation_metrics.json").read_text())
    audit = json.loads(audit_path.read_text())
    if (manifest.get("complete") is not True or len(sources) != 1260
            or len(manifest.get("completed", {})) != 1260):
        raise ValueError("Expected complete formal 1260-observation cache")
    records = first_takeovers(audit)
    selected = [source for source in sources if int(source["rollout_index"]) in records]
    if len(selected) != 1233 or len({r["rollout_index"] for r in selected}) != 50:
        raise ValueError("Expected 1233 pre-takeover observations in 50 saved HIL trajectories")

    takeover_steps = [int(r["intervention_details"][0]["policy_steps_before_intervention"])
                      for r in records.values()]
    active_steps = []
    for record in records.values():
        step = record["active_first_step"]
        if step is None:
            continue
        first = record["activation_events"][0]
        if (first["event"] != "sampling_activation" or first["enabled"] is not True
                or int(first["policy_steps"]) != int(step)):
            raise ValueError("Invalid first enhanced-sampling activation event")
        active_steps.append(int(step))
    if len(active_steps) != 46:
        raise ValueError("Expected 46 activation records")
    mean_takeover = float(np.mean(takeover_steps))
    mean_enhancement = float(np.mean(active_steps))

    rows = []
    for source in selected:
        rollout = int(source["rollout_index"])
        first_takeover = int(records[rollout]["intervention_details"][0]["policy_steps_before_intervention"])
        step = int(source["policy_step"])
        if (int(source["seed"]) != int(records[rollout]["seed"])
                or int(source["policy_end_step"]) != first_takeover
                or step < 0 or step > first_takeover or step % 15):
            raise ValueError(f"Candidate observation does not match HIL record: {rollout=}, {step=}")
        relative = source["cache"]
        path = cache_dir / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["completed"][relative]:
            raise ValueError(f"Missing or changed formal candidate cache: {relative}")
        with np.load(path, allow_pickle=False) as archive:
            chunks = np.asarray(archive["candidates"], dtype=np.float64)
            order = np.asarray(archive["subset_order"], dtype=np.int64)
        if (sorted(order.tolist()) != list(range(32)) or not np.isfinite(chunks).all()):
            raise ValueError(f"Invalid candidate pool: {relative}")
        by_k = {str(k): pool_distances(chunks, order, k) for k in KS}
        rows.append({"cache": relative, "rollout_index": rollout, "policy_step": step,
                     "k": by_k})

    output_dir.mkdir(parents=True, exist_ok=True)
    bins = {}
    for metric, title, ylabel, filename in METRICS:
        result = summarize_bins(rows, metric)
        bins[metric] = result
        note = ("关节图为真正 Frobenius 范数，未除以 10×12 的维度数；与旧图单步 RMS 不可直接比较。"
                if "joint" in metric else
                "14 维含关节弧度与夹爪原始目标值，单位混合；仅供核对，不宜解释为关节角度。")
        draw(result, title, ylabel, output_dir / filename, mean_takeover,
             mean_enhancement, len(active_steps), note)
    report = {
        "protocol": {
            "cache_identity": manifest["identity"],
            "candidate_shape": [32, 10, 14], "ks": list(KS),
            "joint_columns": JOINTS.tolist(), "gripper_columns": list(GRIPPERS),
            "metrics": "True pairwise Frobenius norm sqrt(sum_h,sum_j (A_hj-B_hj)^2), with no dimension normalization",
            "joint_unit": "mrad; 10x12 joint submatrix",
            "raw14_unit": "mixed raw units; 10x14 including two grippers",
            "selection": "50 saved HIL trajectories, 1233 observations before first takeover",
            "aggregation": "30-policy-step bins; mean within rollout, mean across rollouts; minimum five rollouts",
            "coverage_scores_used": False,
            "warning": "Does not measure B, task stage, executed candidate outcome, or recovery success",
        },
        "events": {"mean_first_takeover_step": mean_takeover,
                   "mean_first_enhancement_step": mean_enhancement,
                   "enhancement_trajectories": len(active_steps)},
        "overall_per_observation": report_quantiles(rows),
        "bins": bins,
    }
    (output_dir / "frobenius_takeover_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Analyzed {len(rows)} observations from {len(records)} saved HIL trajectories; output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--audit-records", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.cache_dir, args.audit_records, args.output_dir)


if __name__ == "__main__":
    main()
