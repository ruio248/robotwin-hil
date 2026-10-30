#!/usr/bin/env python3
"""Independently audit a completed public-pi0.5 candidate's 50-seed JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text())


def audit(root: Path) -> dict:
    aggregate = read_json(root / "aggregate_summary.json")
    protocol = read_json(root / "control" / "protocol.json")
    task = protocol["task"]
    assert aggregate["complete"] is True and aggregate["task"] == task
    records = aggregate["records"]
    expected_count = len(protocol["seeds"])
    assert len(records) == aggregate["episodes"] == expected_count
    seeds = [record["seed"] for record in records]
    assert seeds == aggregate["seeds"] and len(set(seeds)) == expected_count
    excluded = aggregate.get("excluded_infra_seeds", [])
    excluded_seeds = [item["seed"] for item in excluded]
    assert len(set(excluded_seeds)) == len(excluded_seeds)
    assert not set(seeds) & set(excluded_seeds)
    if protocol.get("replace_unstable_setup_seeds"):
        assert aggregate["seed_rule"] == "first N valid scene initializations from consecutive seeds"
        assert all("pre-policy unstable scene" in item["reason"] for item in excluded)
        assert seeds + excluded_seeds
        assert sorted(seeds + excluded_seeds) == list(range(
            protocol["seeds"][0], max(seeds + excluded_seeds) + 1
        ))
        assert len(excluded) <= protocol["max_unstable_replacements"]
    else:
        assert not excluded and seeds == protocol["seeds"]

    success_count = 0
    for record in records:
        seed = record["seed"]
        assert record["task"] == task
        assert record["expert_prefilter"] is False
        assert record["generated_horizon"] == protocol["generated_horizon"]
        assert record["execute_horizon"] == protocol["execute_horizon"]
        assert record["frequency_field"] == protocol["frequency_field"]
        assert record["step_limit"] == protocol["step_limit"]
        assert record["success"] == record["final_check_success"], (
            f"sticky/evaluation success disagrees with final task check at seed {seed}"
        )
        path = root / "formal" / f"seed_{seed}" / "eval_records.jsonl"
        lines = path.read_text().splitlines()
        assert len(lines) == 1 and json.loads(lines[0]) == record, path
        success_count += bool(record["success"])
    assert success_count == aggregate["successes"]
    assert abs(aggregate["success_rate"] - success_count / expected_count) < 1e-12
    assert len(list((root / "formal").glob("seed_*/eval_records.jsonl"))) == expected_count
    return {
        "task": task, "complete": True, "episodes": expected_count,
        "successes": success_count, "success_rate": success_count / expected_count,
        "seeds": seeds, "excluded_infra_seeds": excluded,
        "same_initial_fixed_cohort": not excluded and seeds == protocol["seeds"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    cli = parser.parse_args()
    print(json.dumps(audit(cli.root), ensure_ascii=False, indent=2))
