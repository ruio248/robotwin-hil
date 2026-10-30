"""Aggregate same-seed noisy-scripted-expert success, without claiming full DART."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def read_arm(directory, expected):
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines() if line]
    by_seed = {int(row["seed"]): row for row in rows}
    if len(rows) != len(by_seed) or set(by_seed) != expected:
        raise ValueError(f"missing or duplicate seeds: {directory}")
    if any(row["status"] != "completed" or row.get("nonfinite_targets") for row in rows):
        raise ValueError(f"error/nonfinite targets: {directory}")
    summary = json.loads((directory / "summary.json").read_text())
    if not summary["complete"] or summary["records"] != len(expected):
        raise ValueError(f"incomplete arm: {directory}")
    return by_seed


def bootstrap(differences):
    rng = random.Random(42)
    n = len(differences)
    draws = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    return [draws[250], draws[9750]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = set(range(120000, 120020))
    all_arms = {}
    for sub in ("formal", "stress"):
        for directory in sorted((args.root / sub).glob("scale_*")):
            scale = float(directory.name.removeprefix("scale_").replace("p", "."))
            all_arms[scale] = read_arm(directory, expected)
    baseline = all_arms[0.0]
    report_rows = []
    for scale, arm in sorted(all_arms.items()):
        success = sum(bool(arm[s]["success"]) for s in expected)
        losses = sorted(s for s in expected if baseline[s]["success"] and not arm[s]["success"])
        gains = sorted(s for s in expected if not baseline[s]["success"] and arm[s]["success"])
        delta = [int(arm[s]["success"]) - int(baseline[s]["success"]) for s in sorted(expected)]
        report_rows.append({
            "covariance_scale": scale,
            "seeds": 20,
            "successes": success,
            "success_rate": success / 20,
            "mean_realized_noise_rms_rad": sum(arm[s]["noise_rms_rad"] for s in expected) / 20,
            "max_realized_abs_noise_rad": max(arm[s]["noise_abs_max_rad"] for s in expected),
            "plan_failures": sum(not bool(arm[s]["plan_success"]) for s in expected),
            "planned_joint_legality_failures": sum(not bool(arm[s]["joints_legal"]) for s in expected),
            "baseline_success_noisy_failure_seeds": losses,
            "baseline_failure_noisy_success_seeds": gains,
            "paired_success_rate_delta": sum(delta) / 20,
            "paired_bootstrap95_delta": bootstrap(delta),
        })
    cov_path = args.root / "covariance_all_train_proxy.npz"
    report = {
        "status": "complete_noisy_scripted_expert_success_scan_not_full_DART",
        "task_config": "handover_to_tray_v2_promptfix",
        "seed_start": 120000,
        "seed_end": 120019,
        "covariance_sha256": hashlib.sha256(cov_path.read_bytes()).hexdigest(),
        "covariance_source": "405 SFT train episodes, 11461 anchors, same-observation first policy candidate vs next-frame qpos proxy",
        "method_limit": "noise injected into scripted expert joint position targets; replanning occurs at stage boundaries, not every control step; no clean action labels, learner retraining, or HIL takeover measured",
        "legality_limit": "planned_joints_legal audits planner output, not every noisy commanded target; all commanded targets were checked finite",
        "rows": report_rows,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "successes_by_scale":
                      {str(row["covariance_scale"]): row["successes"] for row in report_rows}}))


if __name__ == "__main__":
    main()
