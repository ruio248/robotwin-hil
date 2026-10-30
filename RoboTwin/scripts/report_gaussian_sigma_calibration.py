#!/usr/bin/env python3
"""Compare Gaussian sigma arms to an enhanced sampler on identical pilot seeds."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--enhanced-dir", default="enhanced_same_seeds")
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--output", default="paired_calibration_report.json")
    return parser.parse_args()


def read_records(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(row["seed"]): row for row in rows if row["status"] == "completed"}


def wilson(successes: int, trials: int, z: float = 1.959963984540054):
    if not trials:
        return None
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - radius, center + radius]


def rate_row(records):
    successes = sum(bool(row.get("success")) for row in records.values())
    trials = len(records)
    rate = successes / trials if trials else None
    return {
        "trials": trials,
        "successes": successes,
        "success_rate": rate,
        "success_rate_wilson95": wilson(successes, trials),
        "all_planned_joints_legal": all(bool(row.get("joints_legal")) for row in records.values()),
        "failure_proxy_rate": 1 - rate if rate is not None else None,
        "optimistic_rollouts_per_30_failure_events": (
            30 / (1 - rate) if rate is not None and rate < 1 else None
        ),
    }


def paired_bootstrap_interval(differences, replicates=10000):
    """Scene-level percentile interval for Gaussian minus enhanced success."""
    if not differences:
        return None
    rng = random.Random(42)
    count = len(differences)
    draws = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    )
    return [draws[int(0.025 * replicates)], draws[int(0.975 * replicates)]]


def paired_metrics(enhanced, gaussian, seeds):
    if not seeds:
        return None
    g = sum(bool(gaussian[seed].get("success")) for seed in seeds)
    e = sum(bool(enhanced[seed].get("success")) for seed in seeds)
    return {
        "seeds": len(seeds), "gaussian_successes": g, "enhanced_successes": e,
        "gaussian_success_rate": g / len(seeds),
        "enhanced_success_rate": e / len(seeds),
        "absolute_rate_gap": abs(g - e) / len(seeds),
        "gaussian_minus_enhanced_success_rate": (g - e) / len(seeds),
        "paired_bootstrap95_gaussian_minus_enhanced": paired_bootstrap_interval([
            int(bool(gaussian[seed].get("success"))) - int(bool(enhanced[seed].get("success")))
            for seed in seeds
        ]),
        "gaussian_success_enhanced_failure": sum(
            bool(gaussian[seed].get("success")) and not bool(enhanced[seed].get("success"))
            for seed in seeds
        ),
        "enhanced_success_gaussian_failure": sum(
            bool(enhanced[seed].get("success")) and not bool(gaussian[seed].get("success"))
            for seed in seeds
        ),
    }


def main():
    cli = parse_args()
    root = cli.root.expanduser().resolve()
    if cli.expected_episodes < 1:
        raise ValueError("--expected-episodes must be positive")
    enhanced_path = root / cli.enhanced_dir / "records.jsonl"
    enhanced = read_records(enhanced_path)
    enhanced_raw = [json.loads(line) for line in enhanced_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    complete_arms = {}
    for directory in sorted(root.glob("gaussian_sigma_*")):
        summary_path = directory / "summary.json"
        records_path = directory / "records.jsonl"
        if not summary_path.exists() or not records_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sigma = float(summary["sampling_config"]["gaussian_sigma"])
        gaussian = read_records(records_path)
        raw_records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        full_grid = len(raw_records) == cli.expected_episodes
        if full_grid:
            complete_arms[sigma] = gaussian
        common = sorted(set(enhanced) & set(gaussian))
        enhanced_successes = sum(bool(enhanced[seed].get("success")) for seed in common)
        gaussian_successes = sum(bool(gaussian[seed].get("success")) for seed in common)
        enhanced_only = sum(
            bool(enhanced[seed].get("success")) and not bool(gaussian[seed].get("success"))
            for seed in common
        )
        gaussian_only = sum(
            bool(gaussian[seed].get("success")) and not bool(enhanced[seed].get("success"))
            for seed in common
        )
        rows.append({
            "sigma": sigma,
            "episodes_recorded": len(raw_records),
            "full_grid": full_grid,
            "mean_realized_gaussian_noise_rms": summary.get("mean_realized_gaussian_noise_rms"),
            "gaussian_overall": rate_row(gaussian),
            "paired": {
                "common_seeds": len(common),
                "seed_ids": common,
                "enhanced_successes": enhanced_successes,
                "gaussian_successes": gaussian_successes,
                "enhanced_success_rate": enhanced_successes / len(common) if common else None,
                "gaussian_success_rate": gaussian_successes / len(common) if common else None,
                "absolute_rate_gap": (
                    abs(gaussian_successes - enhanced_successes) / len(common) if common else None
                ),
                "enhanced_success_gaussian_failure": enhanced_only,
                "gaussian_success_enhanced_failure": gaussian_only,
            },
        })
    positive_arms = {sigma: arm for sigma, arm in complete_arms.items() if sigma > 0}
    fixed_seeds = sorted(set(enhanced).intersection(*(set(arm) for arm in positive_arms.values()))) if positive_arms else []
    all_arm_seeds = sorted(set(enhanced).intersection(*(set(arm) for arm in complete_arms.values()))) if complete_arms else []
    for row in rows:
        sigma = row["sigma"]
        row["positive_arm_common"] = (
            paired_metrics(enhanced, positive_arms[sigma], fixed_seeds)
            if sigma in positive_arms else None
        )
        row["all_arm_common"] = (
            paired_metrics(enhanced, complete_arms[sigma], all_arm_seeds)
            if sigma in complete_arms else None
        )
    eligible = [row for row in rows if row["positive_arm_common"] is not None]
    selected = min(
        eligible,
        key=lambda row: (row["positive_arm_common"]["absolute_rate_gap"], row["sigma"]),
    ) if eligible else None
    all_arm_selected = min(
        eligible,
        key=lambda row: (row["all_arm_common"]["absolute_rate_gap"], row["sigma"]),
    ) if eligible and all_arm_seeds else None
    report = {
        "complete": bool(eligible) and len(enhanced_raw) == cli.expected_episodes,
        "expected_episodes_per_arm": cli.expected_episodes,
        "enhanced_episodes_recorded": len(enhanced_raw),
        "pilot_seed_ids": sorted(int(row["seed"]) for row in enhanced_raw),
        "enhanced_overall": rate_row(enhanced),
        "selection_rule": (
            "Exploratory only: among fully recorded positive-sigma arms, minimize the absolute "
            "enhanced-vs-Gaussian success-rate gap on their identical valid-seed intersection; "
            "smaller sigma breaks ties. Also repeat on the intersection including sigma=0 "
            "to expose selection instability. Neither is a calibrated HIL rate."
        ),
        "pilot_selected_sigma_positive_arms": selected["sigma"] if selected else None,
        "pilot_selected_sigma_all_arms": all_arm_selected["sigma"] if all_arm_selected else None,
        "selection_stable_across_intersections": (
            selected is not None and all_arm_selected is not None
            and selected["sigma"] == all_arm_selected["sigma"]
        ),
        "positive_arm_common_seed_ids": fixed_seeds,
        "all_arm_common_seed_ids": all_arm_seeds,
        "rows": rows,
        "interpretation_limit": (
            "This 20-seed autonomous-success pilot cannot estimate HIL takeover probability: "
            "failure, operator detection, intervention, expert recovery, and valid save are distinct. "
            "The chosen sigma is provisional, and failure-to-success discordance and binomial "
            "uncertainty must be reported. K=32 uniform candidate mean plus Gaussian noise also "
            "changes aggregation relative to the K=1 baseline."
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    output = root / cli.output
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
