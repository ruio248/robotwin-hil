"""Select fixed expert-valid development and test seeds for handover_to_tray.

Candidate seeds are never used for demonstration collection. The selected lists are
written once and reused for every checkpoint comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from smoke_handover_to_tray import load_args, run_episode


def select_successful_seeds(
    *,
    task_name: str,
    task_args: dict,
    seed_start: int,
    required: int,
    candidate_count: int,
    offset: int,
) -> tuple[list[int], list[dict]]:
    if candidate_count < required:
        raise ValueError("candidate_count must be at least required")

    selected: list[int] = []
    attempts: list[dict] = []
    for candidate_index in range(candidate_count):
        seed = seed_start + candidate_index
        result = run_episode(task_name, task_args, seed, offset + candidate_index)
        attempts.append(result)
        if result.get("success"):
            selected.append(seed)
            if len(selected) == required:
                break

    if len(selected) != required:
        raise RuntimeError(
            f"only {len(selected)}/{required} scripted-expert-valid seeds were found "
            f"from [{seed_start}, {seed_start + candidate_count})"
        )
    return selected, attempts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="handover_to_tray")
    parser.add_argument("--config", default="handover_to_tray_smoke")
    parser.add_argument("--development-start", type=int, default=30000)
    parser.add_argument("--development-count", type=int, default=20)
    parser.add_argument("--test-start", type=int, default=31000)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument(
        "--output",
        default=os.environ.get(
            "ROBOTWIN_EVAL_SEED_MANIFEST",
            str(
                Path(__file__).resolve().parents[2]
                / "outputs"
                / "manifests"
                / "handover_to_tray_v1_eval_seeds.json"
            ),
        ),
    )
    cli = parser.parse_args()

    if cli.development_count <= 0 or cli.test_count <= 0 or cli.candidate_multiplier < 1:
        raise ValueError("counts must be positive and candidate multiplier must be at least one")

    development_candidates = cli.development_count * cli.candidate_multiplier
    test_candidates = cli.test_count * cli.candidate_multiplier
    development_range = range(cli.development_start, cli.development_start + development_candidates)
    test_range = range(cli.test_start, cli.test_start + test_candidates)
    if set(development_range).intersection(test_range):
        raise ValueError("development and test candidate ranges must not overlap")

    task_args = load_args(cli.task, cli.config)
    development_seeds, development_attempts = select_successful_seeds(
        task_name=cli.task,
        task_args=task_args,
        seed_start=cli.development_start,
        required=cli.development_count,
        candidate_count=development_candidates,
        offset=0,
    )
    test_seeds, test_attempts = select_successful_seeds(
        task_name=cli.task,
        task_args=task_args,
        seed_start=cli.test_start,
        required=cli.test_count,
        candidate_count=test_candidates,
        offset=development_candidates,
    )
    manifest = {
        "task": cli.task,
        "config": cli.config,
        "selection_rule": "first scripted-expert-valid seeds from disjoint fixed candidate ranges",
        "development_seeds": development_seeds,
        "test_seeds": test_seeds,
        "development_attempts": development_attempts,
        "test_attempts": test_attempts,
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "development_count": len(development_seeds),
                "test_count": len(test_seeds),
                "development_first": development_seeds[0],
                "test_first": test_seeds[0],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
