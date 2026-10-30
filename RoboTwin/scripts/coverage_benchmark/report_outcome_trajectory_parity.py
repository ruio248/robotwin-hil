#!/usr/bin/env python3
"""Historical handover-style, episode-equal outcome summary for task D traces.

The older handover experiment averaged K4 D scores over every saved frame.
Current task traces instead contain one executed-policy H10 suggestion at each
fixed-H50 query. The same episode-level summary is useful descriptively, but
these two observation schedules and candidate counts must not be conflated.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path
import random


def auc(success: list[float], failure: list[float]) -> float:
    if not success or not failure:
        raise ValueError("AUC needs both outcomes")
    return sum(1.0 if a > b else 0.5 if a == b else 0.0
               for a in success for b in failure) / (len(success) * len(failure))


def ci(values: list[float], low: float = .025, high: float = .975) -> list[float]:
    ordered = sorted(values)

    def point(p: float) -> float:
        position = p * (len(ordered) - 1)
        left = int(position)
        weight = position - left
        return ordered[left] * (1 - weight) + ordered[min(left + 1, len(ordered) - 1)] * weight

    return [point(low), point(high)]


def boot_auc(success: list[float], failure: list[float], seed: int, draws: int) -> dict:
    rng = random.Random(seed)
    values = [auc(rng.choices(success, k=len(success)), rng.choices(failure, k=len(failure)))
              for _ in range(draws)]
    return {"auc": auc(success, failure), "ci95": ci(values),
            "successful_episodes": len(success), "failed_episodes": len(failure)}


def summarize(path: Path, seed: int, draws: int) -> dict:
    data = json.loads(path.read_text())
    task = data.get("task")
    rows = data.get("per_episode", [])
    if not data.get("complete") or not isinstance(task, str) or len(rows) != 50:
        raise ValueError(f"incomplete task result: {path}")
    if len({row["episode"] for row in rows}) != 50 or len({row["scene_seed"] for row in rows}) != 50:
        raise ValueError(f"duplicate episode/scene: {path}")
    if any(type(row.get("success")) is not bool or row.get("queries", 0) < 1 or
           not math.isfinite(row.get("full", float("nan"))) for row in rows):
        raise ValueError(f"invalid outcome/query/score: {path}")
    scores = sorted(float(row["full"]) for row in rows)
    trajectory = []
    for row in rows:
        value = float(row["full"])
        percentile = 100 * (bisect_left(scores, value) + bisect_right(scores, value)) / (2 * len(scores))
        trajectory.append({"episode": row["episode"], "scene_seed": row["scene_seed"],
                           "success": row["success"], "queries": row["queries"],
                           "score_mean": value, "score_percentile": percentile})
    success = [row for row in trajectory if row["success"]]
    failure = [row for row in trajectory if not row["success"]]
    if not success or not failure:
        raise ValueError(f"only one outcome class: {path}")
    positive = [row["score_mean"] for row in success]
    negative = [row["score_mean"] for row in failure]
    outcome_auc = boot_auc(positive, negative, seed, draws)
    old = data["outcome_auc"]["full"]
    if abs(outcome_auc["auc"] - old["auc"]) > 1e-12:
        raise ValueError(f"trajectory AUC does not match prior per-episode result: {path}")
    length_auc = boot_auc([row["queries"] for row in success],
                          [row["queries"] for row in failure], seed + 1, draws)
    return {"task": task, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_successful_episodes": len(success), "source_failed_episodes": len(failure),
            "trajectory_score": {
                "success_mean": sum(positive) / len(positive),
                "failure_mean": sum(negative) / len(negative),
                "success_minus_failure_raw": sum(positive) / len(positive) - sum(negative) / len(negative),
                "success_minus_failure_percentile_pp": (
                    sum(row["score_percentile"] for row in success) / len(success) -
                    sum(row["score_percentile"] for row in failure) / len(failure)),
                **outcome_auc,
            },
            "query_count_control": {
                "success_mean": sum(row["queries"] for row in success) / len(success),
                "failure_mean": sum(row["queries"] for row in failure) / len(failure),
                **length_auc,
            },
            "first_query_auc": data["outcome_auc"]["first1"],
            "per_episode": trajectory}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    paths = sorted(path for path in args.score_root.glob("*.json")
                   if path.resolve() != args.output.resolve())
    if len(paths) != 10 or args.bootstrap_draws < 1:
        raise ValueError("expected exactly ten complete task score JSON files")
    tasks = [summarize(path, 42 + index, args.bootstrap_draws)
             for index, path in enumerate(paths)]
    if len({task["task"] for task in tasks}) != 10:
        raise ValueError("duplicate task")
    output = {
        "complete": True,
        "scope": "Held-out seed2 published-policy autonomous rollouts, 50 episodes per task",
        "historical_handover_parity": "Episode-equal full-saved-trajectory mean, success/failure gap, empirical percentile gap, AUC, and episode-stratified bootstrap",
        "nonparity": "New traces save only one executed-policy H10 suggestion at each fixed-H50 query; historical handover saved K4 suggestions at every recorded frame",
        "score_direction": "higher D means more expert-source-like, not a calibrated success probability",
        "percentile_reference": "50 equal-weight episode mean D scores within each task",
        "tasks": tasks,
        "limitations": [
            "Full-trajectory averages mix different numbers of policy queries and task stages.",
            "Query-count control is reported; it is not an observation-only support model.",
            "Do not compare raw calibrated D score gaps across independently trained tasks.",
            "This is outcome association, not HIL-event detection or rescue probability.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    for task in tasks:
        score = task["trajectory_score"]
        print(task["task"], f"{task['source_successful_episodes']}/50",
              f"gap={score['success_minus_failure_percentile_pp']:.1f}pp",
              f"AUC={score['auc']:.3f}")


if __name__ == "__main__":
    main()
