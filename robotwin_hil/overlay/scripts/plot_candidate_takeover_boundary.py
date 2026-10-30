"""Plot candidate diversity with mean enhancement and first-takeover steps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


KS = (4, 8, 16, 32)
BIN_STEPS = 30
COLORS = ("#277DA8", "#DB8A26", "#3D9567", "#9565B7")
MARKERS = ("o", "s", "^", "D")
FIGURES = (
    ("mean_pairwise_rad", "平均两两动作差异", "平均两两关节动作差异（mrad）",
     "candidate_mean_pairwise_takeover_steps"),
    ("max_pairwise_rad", "最大两两动作差异", "最大两两关节动作差异（mrad）",
     "candidate_max_pairwise_takeover_steps"),
)


def first_takeovers(audit: list[dict]) -> dict[int, dict]:
    records = {int(r["rollout_index"]): r for r in audit if r.get("saved") and r["interventions"]}
    if len(records) != 50 or any(len(r["intervention_details"]) != 1 for r in records.values()):
        raise ValueError("Expected 50 saved trajectories with one first takeover each")
    for record in records.values():
        first = record["intervention_details"][0]
        step = int(first["policy_steps_before_intervention"])
        if step <= 0 or step > int(record["policy_steps"]):
            raise ValueError(f"Invalid first-takeover step in rollout {record['rollout_index']}")
    return records


def summarize(rows: list[dict], records: dict[int, dict], metric: str) -> list[dict]:
    # Take one mean per trajectory per bin so long trajectories do not dominate.
    grouped: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        rollout = int(row["rollout_index"])
        if rollout not in records:
            continue
        first = int(records[rollout]["intervention_details"][0]["policy_steps_before_intervention"])
        step = int(row["policy_step"])
        if (row["seed"] != records[rollout]["seed"]
                or int(row["policy_end_step"]) != first
                or step < 0 or step > first or step % 15):
            raise ValueError(f"Candidate observation does not match takeover record: {rollout=}, {step=}")
        for k in KS:
            value = float(row["metrics"][str(k)][metric])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {metric}: {value}")
            grouped[step // BIN_STEPS][k][rollout].append(value)

    result = []
    for bin_index in sorted(grouped):
        episodes = grouped[bin_index]
        counts = {len(episodes[k]) for k in KS}
        if len(counts) != 1:
            raise ValueError("Four K groups have different episode coverage")
        n_episodes = counts.pop()
        if n_episodes < 5:
            continue  # Do not extrapolate the far tail from 1–4 trajectories.
        result.append({
            "policy_step_lower": bin_index * BIN_STEPS,
            "policy_step_upper": (bin_index + 1) * BIN_STEPS,
            "episodes": n_episodes,
            "mean_rad_by_k": {
                str(k): float(np.mean([np.mean(values) for values in episodes[k].values()]))
                for k in KS
            },
        })
    if not result:
        raise ValueError("No progress bins have at least five trajectories")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--audit-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    rows = json.loads((args.input_dir / "per_observation_metrics.json").read_text())
    audit = json.loads(args.audit_records.read_text())
    if manifest.get("complete") is not True or len(rows) != 1260:
        raise ValueError("Expected the complete formal candidate cache")
    records = first_takeovers(audit)
    selected = [r for r in rows if int(r["rollout_index"]) in records]
    if len(selected) != 1233 or len({r["rollout_index"] for r in selected}) != 50:
        raise ValueError("Expected 1233 pre-first-takeover observations from 50 saved HIL episodes")
    takeover_steps = [int(r["intervention_details"][0]["policy_steps_before_intervention"])
                      for r in records.values()]
    mean_takeover = float(np.mean(takeover_steps))
    activation_steps = []
    for record in records.values():
        first_active = record["active_first_step"]
        if first_active is None:
            continue
        first_event = record["activation_events"][0]
        first_takeover = int(record["intervention_details"][0]["policy_steps_before_intervention"])
        if (first_event["event"] != "sampling_activation"
                or first_event["enabled"] is not True
                or int(first_event["policy_steps"]) != int(first_active)
                or not 0 < int(first_active) < first_takeover):
            raise ValueError(f"Invalid enhancement start in rollout {record['rollout_index']}")
        activation_steps.append(int(first_active))
    if len(activation_steps) != 46:
        raise ValueError("Expected 46 of the 50 HIL trajectories to activate enhancement")
    mean_activation = float(np.mean(activation_steps))

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
    plt.rcParams.update({"font.family": cjk_family, "axes.unicode_minus": False,
                         "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for metric, title, ylabel, filename in FIGURES:
        bins = summarize(rows, records, metric)
        x = [(b["policy_step_lower"] + b["policy_step_upper"]) / 2 for b in bins]
        fig, ax = plt.subplots(figsize=(10.8, 6.3))
        fig.subplots_adjust(left=0.12, right=0.97, top=0.78, bottom=0.19)
        fig.suptitle(title, x=0.12, ha="left", y=0.97, fontsize=18)
        fig.text(0.12, 0.90, "K = 4 / 8 / 16 / 32 · 50 条已保存 HIL 轨迹 · 1233 个接管前观测", fontsize=11)
        for k, color, marker in zip(KS, COLORS, MARKERS):
            y = [b["mean_rad_by_k"][str(k)] * 1000 for b in bins]
            ax.plot(x, y, color=color, marker=marker, markersize=4.5,
                    linewidth=2.2, label=f"K = {k}")
        ax.axvline(mean_takeover, color="#B34842", linewidth=2,
                   linestyle="--", label=f"平均首次接管：{mean_takeover:.1f} 步")
        ax.axvline(mean_activation, color="#5A6170", linewidth=2,
                   linestyle=":", label=f"平均首次增强采样：{mean_activation:.1f} 步（{len(activation_steps)}/50）")
        ax.set_xlim(0, max(650, x[-1] + BIN_STEPS))
        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 100))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("已执行策略动作数（步）")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.015))
        fig.text(0.12, 0.08, "每 30 步分箱：轨迹内先平均，再取轨迹间平均；少于 5 条轨迹的区间不绘制。", fontsize=10)
        fig.text(0.12, 0.05, "两条线是不同轨迹子集的平均事件位置；接管线右侧仍是接管前观测。", fontsize=10)
        stem = args.output_dir / filename
        fig.savefig(stem.with_suffix(".png"), dpi=190)
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
        reports[metric] = bins

    (args.output_dir / "candidate_takeover_step_summary.json").write_text(
        json.dumps({"x_axis": "executed policy action count before first HIL takeover",
                    "mean_first_takeover_step": mean_takeover,
                    "mean_first_enhancement_step": mean_activation,
                    "enhancement_trajectories": len(activation_steps),
                    "takeover_trajectories": len(records),
                    "selected_observations": len(selected),
                    "bin_width_policy_steps": BIN_STEPS,
                    "min_trajectories_per_bin": 5,
                    "metrics": reports}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
