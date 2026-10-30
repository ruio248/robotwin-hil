#!/usr/bin/env python3
"""Paired stage-six success-rate calibration for the Gaussian sampling baseline.

The script first evaluates the unmodified policy on a held-out seed set, then
replays the same expert-prefix handoffs for K-candidate uniform averaging plus
each requested output-space Gaussian sigma.  Existing records are resumed, so
the pilot can be extended from (for example) 10 to 20 seeds without rerunning
finished episodes.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROBOTWIN_ROOT = SCRIPT_DIR.parent
EVALUATOR = SCRIPT_DIR / "eval_one_stage_policy.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path,
                        help="Reuse a completed same-seed baseline rather than rerunning it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18304)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", required=True)
    parser.add_argument("--target-success-rate", type=float, default=0.25)
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--policy-microbatch", type=int, default=32)
    parser.add_argument("--step-limit", type=int, default=900)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ckpt-name", default="v2_promptfix_9999")
    return parser.parse_args()


def sigma_tag(sigma: float) -> str:
    return f"{sigma:.9g}".replace("-", "m").replace(".", "p")


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    if trials == 0:
        return None
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - radius, center + radius]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_arm(cli, *, mode: str, output_dir: Path, sigma: float | None, baseline_records: Path | None):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        cli.python, "-u", str(EVALUATOR),
        "--mode", mode,
        "--host", cli.host,
        "--port", str(cli.port),
        "--policy-name", "Pi_05_RobotTwin",
        "--ckpt-name", cli.ckpt_name,
        "--task-config", "handover_to_tray_v2_promptfix",
        "--seed-start", str(cli.seed_start),
        "--episodes", str(cli.episodes),
        "--frequency", "30",
        "--render-freq", "0",
        "--save-freq", "15",
        "--step-limit", str(cli.step_limit),
        "--output-dir", str(output_dir),
    ]
    if (output_dir / "records.jsonl").exists():
        command.append("--resume")
    if mode == "gaussian":
        command.extend([
            "--paired-baseline-records", str(baseline_records),
            "--num-candidates", str(cli.num_candidates),
            "--policy-microbatch", str(cli.policy_microbatch),
            "--gaussian-sigma", repr(float(sigma)),
        ])
    state = {
        "status": "running", "mode": mode, "sigma": sigma,
        "command": command, "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(cli.output_root / "calibration_state.json", state)
    with (output_dir / "run.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=ROBOTWIN_ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if result.returncode:
        state.update(status="failed", returncode=result.returncode,
                     finished_at=datetime.now().isoformat(timespec="seconds"))
        write_json(cli.output_root / "calibration_state.json", state)
        raise RuntimeError(f"{mode} sigma={sigma} exited with {result.returncode}; see {output_dir / 'run.log'}")
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def compact(summary, sigma):
    n = int(summary["ready_stage6"])
    successes = int(summary["autonomous_successes"])
    return {
        "mode": summary["mode"], "sigma": sigma,
        "ready_stage6": n, "successes": successes,
        "success_rate": successes / n if n else None,
        "success_rate_wilson95": wilson_interval(successes, n),
        "prefix_failed": int(summary["prefix_failed"]),
        "paired_prefix_excluded": int(summary["paired_prefix_excluded"]),
        "planned_joints_legal": int(summary.get("planned_joints_legal", 0)),
        "mean_realized_gaussian_noise_rms": summary.get("mean_realized_gaussian_noise_rms"),
    }


def main():
    cli = parse_args()
    if cli.episodes < 1 or cli.seed_start < 0:
        raise ValueError("Require nonnegative seed start and positive episode count")
    if not 0 <= cli.target_success_rate <= 1:
        raise ValueError("target success rate must lie in [0,1]")
    if any(not math.isfinite(sigma) or sigma < 0 for sigma in cli.sigmas):
        raise ValueError("Every sigma must be finite and nonnegative")
    if len(set(cli.sigmas)) != len(cli.sigmas):
        raise ValueError("Sigma grid contains duplicates")
    cli.output_root = cli.output_root.expanduser().resolve()
    cli.output_root.mkdir(parents=True, exist_ok=True)

    baseline_dir = cli.baseline_dir.expanduser().resolve() if cli.baseline_dir else cli.output_root / "baseline"
    if cli.baseline_dir:
        baseline = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
        baseline_rows = [json.loads(line) for line in (baseline_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_seeds = set(range(cli.seed_start, cli.seed_start + cli.episodes))
        if (baseline.get("mode") != "baseline" or baseline.get("seed_start") != cli.seed_start
                or not expected_seeds.issubset({int(row["seed"]) for row in baseline_rows})):
            raise ValueError("Reused baseline is incomplete or does not match the requested seed range")
    else:
        baseline = run_arm(cli, mode="baseline", output_dir=baseline_dir,
                           sigma=None, baseline_records=None)
    rows = [compact(baseline, None)]
    for sigma in cli.sigmas:
        output_dir = cli.output_root / f"gaussian_sigma_{sigma_tag(sigma)}"
        summary = run_arm(
            cli, mode="gaussian", output_dir=output_dir, sigma=sigma,
            baseline_records=baseline_dir / "records.jsonl",
        )
        rows.append(compact(summary, sigma))

    eligible = [row for row in rows[1:] if row["success_rate"] is not None]
    recommended = min(
        eligible,
        key=lambda row: (abs(row["success_rate"] - cli.target_success_rate), row["sigma"]),
    ) if eligible else None
    report = {
        "complete": True,
        "seed_start": cli.seed_start,
        "episodes": cli.episodes,
        "seed_end": cli.seed_start + cli.episodes - 1,
        "target_success_rate": cli.target_success_rate,
        "baseline_records": str(baseline_dir / "records.jsonl"),
        "num_candidates": cli.num_candidates,
        "selection_rule": "minimum absolute success-rate error; smaller sigma breaks ties",
        "recommended_sigma": recommended["sigma"] if recommended else None,
        "rows": rows,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(cli.output_root / "calibration_summary.json", report)
    write_json(cli.output_root / "calibration_state.json", {
        "status": "complete", "summary": str(cli.output_root / "calibration_summary.json"),
        "finished_at": report["finished_at"],
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
