#!/usr/bin/env python3
"""Combine task-wise episode-held-out D(o,a) reports without inventing a scalar winner."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def episode_bootstrap(values: list[float], seed: int, repeats: int = 10000) -> dict:
    """Bootstrap independent episode-level values, not correlated anchors."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Need finite values from at least one episode")
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(len(x), size=(repeats, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "episodes": int(len(x))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    reports = {}
    for task in args.tasks:
        path = args.results_root / task / "metrics.json"
        payload = json.loads(path.read_text())
        manifest = json.loads((args.results_root / task / "manifest.json").read_text())
        if payload.get("complete") is not True or manifest.get("complete") is not True:
            raise ValueError(f"Incomplete D report: {task}")
        if len(payload.get("per_episode_out_of_fold", [])) != 30:
            raise ValueError(f"Expected 30 unique out-of-fold test episodes: {task}")
        reports[task] = payload

    keys = ("expert_rank", "d_b_mean", "d_b_median", "normalized_range_median",
            "normalized_std_median", "ess_fraction_median", "max_weight_median",
            "candidate_pair_joint_rms_mrad_median", "candidate_pair_gripper_rms_median",
            "candidate_exact_duplicate_pair_fraction",
            "weighted_minus_uniform_joint_rms_mrad", "weighted_minus_uniform_gripper_rms",
            "action_gap_spearman_median")
    # Only rank metrics with an explicitly monotonic interpretation. D_B,
    # ESS/K and action displacement are diagnostics, not "higher is better".
    ranking_directions = {"expert_rank": "higher_is_better_for_expert_action_separation",
                          "candidate_exact_duplicate_pair_fraction": "lower_is_better_for_candidate_diversity",
                          "action_gap_spearman_median": "higher_is_better_for_score_action_gap_alignment"}
    rankings = {}
    for key, preference in ranking_directions.items():
        defined = [(task, row["aggregate_out_of_fold"][key]["mean"])
                   for task, row in reports.items()
                   if row["aggregate_out_of_fold"][key]["mean"] is not None]
        reverse = preference != "lower_is_better_for_candidate_diversity"
        rankings[key] = {"preference": preference,
            "ordered": [{"rank": i + 1, "task": task, "value": value}
                        for i, (task, value) in enumerate(sorted(defined, key=lambda x: x[1], reverse=reverse))]}

    table = []
    for task, row in reports.items():
        aggregate = row["aggregate_out_of_fold"]
        table.append({"task": task,
            "expert_rank_mean_ci95": aggregate["expert_rank"],
            "d_b_mean": aggregate["d_b_mean"],
            "d_b_median": aggregate["d_b_median"],
            "normalized_range_median": aggregate["normalized_range_median"],
            "normalized_std_median": aggregate["normalized_std_median"],
            "ess_fraction_median": aggregate["ess_fraction_median"],
            "max_weight_median": aggregate["max_weight_median"],
            "candidate_pair_joint_rms_mrad_median": aggregate["candidate_pair_joint_rms_mrad_median"],
            "candidate_pair_gripper_rms_median": aggregate["candidate_pair_gripper_rms_median"],
            "candidate_exact_duplicate_pair_fraction": aggregate["candidate_exact_duplicate_pair_fraction"],
            "weighted_minus_uniform_joint_rms_mrad": aggregate["weighted_minus_uniform_joint_rms_mrad"],
            "weighted_minus_uniform_gripper_rms": aggregate["weighted_minus_uniform_gripper_rms"],
            "action_gap_spearman_median": aggregate["action_gap_spearman_median"],
            "anchors": row["data"]["anchors"], "episodes": row["data"]["episodes"]})

    stages = {}
    for task, payload in reports.items():
        stage_rows = {}
        for episode in payload["per_episode_out_of_fold"]:
            for stage_id, values in episode["stage_summaries"].items():
                stage_rows.setdefault(stage_id, []).append({"scene_seed": episode["scene_seed"], **values})
        stages[task] = {}
        for stage_id, values in stage_rows.items():
            metrics = ("expert_rank", "d_b_mean", "d_b_median", "ess_fraction_median")
            stage_names = sorted({row["stage_name"] for row in values})
            if len(stage_names) != 1:
                raise ValueError(f"stage id {stage_id} has inconsistent call-site labels in {task}: {stage_names}")
            stages[task][stage_id] = {
                "stage_name": stage_names[0],
                "episode_count": len(values),
                "total_anchors": sum(x["anchors"] for x in values),
                "episode_bootstrap": {
                    key: episode_bootstrap([x[key] for x in values],
                                           seed=42 + int(stage_id) * 101 + metric_index)
                    for metric_index, key in enumerate(metrics)
                },
                "per_episode": values}

    output = {
        "complete": True,
        "scope": "Ten candidate tasks; D is an expert-vs-policy action-source support score on expert-visited observations, not a success or rescue predictor.",
        "task_count": len(reports),
        "episode_count_per_task": 30,
        "candidate_count": 4,
        "table": sorted(table, key=lambda x: x["expert_rank_mean_ci95"]["mean"], reverse=True),
        "primary_task_order": "expert_rank descending; use its episode-bootstrap interval and the diagnostic metrics jointly, not as a composite score",
        "monotonic_metric_rankings": rankings,
        "descriptive_diagnostics": {task: {key: reports[task]["aggregate_out_of_fold"][key]
                                            for key in keys if key not in ranking_directions}
                                    for task in reports},
        "numeric_stage_summaries": stages,
        "screening_interpretation": [
            "Primary evidence is episode-held-out expert rank relative to the 0.5 chance line and its episode-bootstrap interval.",
            "D_B, score range, and score std describe within-observation policy-candidate resolution; large separation can also mean saturation and is not automatically better.",
            "ESS/K and weighted action displacement describe how the existing low-score sampler would concentrate/change the candidate mean; they do not prove the selected action succeeds.",
            "There is no composite scalar rank: tasks should be selected from the joint pattern of reliable expert ranking, nondegenerate resolution, nonpathological ESS/K, and nontrivial but physically plausible action changes.",
            "The 30 episodes per task are successful expert demonstrations and the observations lie on expert trajectories. Policy-visited failures and takeover outcomes are absent, so this screening cannot establish failure detection or rescue rate."
        ],
        "tasks": reports,
    }
    path = args.results_root / "aggregate_d_task_screening.json"
    atomic_json(path, output)
    print(json.dumps({"complete": True, "report": str(path), "task_count": len(reports),
                      "expert_rank_order": [x["task"] for x in output["table"]]}, indent=2))


if __name__ == "__main__":
    main()
