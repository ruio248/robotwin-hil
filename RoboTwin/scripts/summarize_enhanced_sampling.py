"""Compare completed, matched-seed Vanilla/Enhanced episode JSONL logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_suite(directory, mode):
    episodes = {}
    for path in sorted(Path(directory).glob("episode_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows or rows[0].get("event") != "episode_start" or rows[-1].get("event") != "episode_end":
            raise ValueError(f"Incomplete episode log: {path}")
        head, end = rows[0], rows[-1]
        if head["config"]["mode"] != mode or end["error"] is not None:
            raise ValueError(f"Wrong mode or errored episode: {path}")
        seed = head["episode_seed"]
        if seed in episodes:
            raise ValueError(f"Duplicate episode seed: {seed}")
        decisions = [r for r in rows if r["event"] == "decision" and r["active"]]
        actual = [r for r in rows if r["event"] == "executed" and r["active"]]
        episodes[seed] = {"head": head, "end": end, "decisions": decisions, "actual": actual}
    if not episodes:
        raise ValueError(f"No completed episodes in {directory}")
    return episodes


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def compare(vanilla, enhanced):
    if vanilla.keys() != enhanced.keys():
        raise ValueError("Both suites must contain the same episode seeds")
    rows = []
    for seed in sorted(vanilla):
        left, right = vanilla[seed], enhanced[seed]
        for key in ("critic_sha256", "encoder_sha256", "policy_checkpoint", "task_config", "frequency", "instruction", "executor"):
            if left["head"][key] != right["head"][key]:
                raise ValueError(f"Unmatched {key} at seed {seed}")
        lc, rc = dict(left["head"]["config"]), dict(right["head"]["config"])
        lc.pop("mode"); rc.pop("mode")
        if lc != rc:
            raise ValueError(f"Unmatched sampling configuration at seed {seed}")
        row = {"seed": seed}
        for mode, episode in (("vanilla", left), ("enhanced", right)):
            end, decisions, actual = episode["end"], episode["decisions"], episode["actual"]
            row[mode] = {key: end[key] for key in ("success", "window_reached", "active_min", "active_mean",
                                                   "active_low_fraction", "active_low_hit", "active_decisions",
                                                   "active_executed_scores")}
            row[mode]["mean_selected_lookahead_min"] = mean_or_none([d["selected_score"] for d in decisions])
            row[mode]["mean_selection_seconds"] = mean_or_none([d["selection_seconds"] for d in decisions])
            errors = [abs(a["prediction_error"]) for a in actual if a["prediction_error"] is not None]
            row[mode]["max_execution_prediction_error"] = max(errors) if errors else None
        rows.append(row)
    paired = [r for r in rows if all(r[m]["window_reached"] for m in ("vanilla", "enhanced"))]
    result = {"episodes": len(rows), "paired_window_reached": len(paired), "per_seed": rows,
              "paired_enhanced_minus_vanilla_active_min": mean_or_none(
                  [r["enhanced"]["active_min"] - r["vanilla"]["active_min"] for r in paired]),
              "paired_enhanced_minus_vanilla_active_mean": mean_or_none(
                  [r["enhanced"]["active_mean"] - r["vanilla"]["active_mean"] for r in paired]),
              "interpretation": "Coverage is the frozen selection critic score, not a calibrated failure probability. "
                                "Finite-N importance resampling approximates the tilted policy. "
                                "No HIL intervention data is inferred by this evaluator."}
    for mode in ("vanilla", "enhanced"):
        result[mode] = {"success_rate": mean_or_none([int(r[mode]["success"]) for r in rows]),
                        "window_reached": sum(r[mode]["window_reached"] for r in rows),
                        "paired_low_coverage_hit_rate": mean_or_none(
                            [int(r[mode]["active_low_hit"]) for r in paired if r[mode]["active_low_hit"] is not None])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla-dir", required=True)
    parser.add_argument("--enhanced-dir", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(read_suite(args.vanilla_dir, "vanilla"), read_suite(args.enhanced_dir, "enhanced"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=2))


if __name__ == "__main__":
    main()
