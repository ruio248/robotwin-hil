"""Paired success scan for DART-covariance noise on the scripted supervisor.

This is a *noisy scripted-expert safety scan*, not a full DART learner update:
the RoboTwin supervisor replans at stage boundaries, not every physics step.
No training labels or demonstrations are written by this entry point.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_robotwin(root: Path):
    sys.path.insert(0, str(root))
    from scripts.collect_data import class_decorator
    from scripts.smoke_handover_to_tray import load_args
    return class_decorator, load_args


def install_noise(task, covariance: np.ndarray, scale: float, seed: int):
    """Inject one correlated 12-joint draw per planned simulator control step."""
    original = task.take_dense_action
    rng = np.random.default_rng(seed + 730_101)
    stats = {"steps": 0, "joint_targets": 0, "noise_sq_sum": 0.0,
             "noise_abs_max": 0.0, "nonfinite_targets": 0}

    def noisy_take_dense_action(control_seq, save_freq=-1):
        arms = [control_seq["left_arm"], control_seq["right_arm"]]
        max_len = 0
        for arm in arms:
            if arm is not None:
                max_len = max(max_len, len(arm["position"]))
        for key in ("left_gripper", "right_gripper"):
            grip = control_seq[key]
            if grip is not None:
                max_len = max(max_len, int(grip["num_step"]))
        if max_len == 0:
            return original(control_seq, save_freq=save_freq)
        noise = (rng.multivariate_normal(np.zeros(12), covariance, size=max_len,
                                         check_valid="raise") * math.sqrt(scale)
                 if scale else np.zeros((max_len, 12)))
        modified = dict(control_seq)
        for name, section in (("left_arm", slice(0, 6)), ("right_arm", slice(6, 12))):
            arm = control_seq[name]
            if arm is None:
                continue
            positions = np.asarray(arm["position"], dtype=np.float64)
            if positions.ndim != 2 or positions.shape[1] != 6:
                raise ValueError(f"unexpected {name} positions: {positions.shape}")
            drawn = noise[:len(positions), section]
            targets = positions + drawn
            stats["joint_targets"] += int(targets.size)
            stats["noise_sq_sum"] += float(np.sum(drawn ** 2))
            stats["noise_abs_max"] = max(stats["noise_abs_max"], float(np.max(np.abs(drawn))))
            stats["nonfinite_targets"] += int(np.size(targets) - np.isfinite(targets).sum())
            changed = dict(arm)
            changed["position"] = targets
            modified[name] = changed
        stats["steps"] += max_len
        return original(modified, save_freq=save_freq)

    task.take_dense_action = noisy_take_dense_action
    return stats


def run_one(root, task_name, task_args, covariance, scale, seed, index):
    class_decorator, _ = load_robotwin(root)
    task = class_decorator(task_name)
    row = {"seed": seed, "scale": scale, "status": "started", "success": False}
    try:
        args = deepcopy(task_args)
        args.update(seed=seed, now_ep_num=index, need_plan=True)
        task.setup_demo(**args)
        stats = install_noise(task, covariance, scale, seed)
        task.play_once()
        legal, max_planned = task.planned_joints_legal()
        row.update(status="completed", plan_success=bool(task.plan_success),
                   joints_legal=bool(legal), joint_absmax=float(max_planned),
                   success=bool(task.plan_success and legal and task.check_success()),
                   stage_ids=[int(s["stage_id"]) for s in task.stage_snapshots],
                   noise_steps=stats["steps"],
                   noise_rms_rad=math.sqrt(stats["noise_sq_sum"] / stats["joint_targets"])
                   if stats["joint_targets"] else 0.0,
                   noise_abs_max_rad=stats["noise_abs_max"],
                   nonfinite_targets=stats["nonfinite_targets"],
                   success_metrics=task.success_metrics())
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc(limit=8))
    finally:
        try:
            task.close_env(clear_cache=True)
        except Exception as exc:
            row["close_error"] = f"{type(exc).__name__}: {exc}"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--covariance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="handover_to_tray")
    parser.add_argument("--config", default="handover_to_tray_v2_promptfix")
    parser.add_argument("--seed-start", type=int, default=120000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--scales", type=float, nargs="+", default=[0, .25, .5, 1, 2])
    cli = parser.parse_args()
    if cli.episodes < 1 or len(set(cli.scales)) != len(cli.scales) or any(s < 0 for s in cli.scales):
        raise ValueError("invalid episode count or scales")
    with np.load(cli.covariance, allow_pickle=False) as source:
        covariance = source["covariance"]
    if covariance.shape != (12, 12) or not np.isfinite(covariance).all():
        raise ValueError("expected finite [12,12] covariance")
    if not np.allclose(covariance, covariance.T, atol=1e-10) or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("covariance must be PSD")
    _, load_args = load_robotwin(cli.robotwin_root)
    task_args = load_args(cli.task, cli.config)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"task": cli.task, "task_config": cli.config,
              "seed_start": cli.seed_start, "episodes": cli.episodes,
              "scales": cli.scales, "covariance_trace_rad2": float(np.trace(covariance)),
              "status": "noisy_scripted_expert_safety_scan_not_full_DART",
              "expert_feedback": "replans_at_stage_boundaries_only",
              "training_labels": "not_collected"}
    config_path = cli.output_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("existing output config differs")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    for scale in cli.scales:
        arm_dir = cli.output_dir / f"scale_{str(scale).replace('.', 'p')}"
        arm_dir.mkdir(exist_ok=True)
        records_path = arm_dir / "records.jsonl"
        rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()] if records_path.exists() else []
        expected = list(range(cli.seed_start, cli.seed_start + cli.episodes))
        existing = {int(row["seed"]): row for row in rows}
        if len(existing) != len(rows) or any(seed not in expected for seed in existing):
            raise ValueError(f"duplicate or unexpected seeds in {records_path}")
        for i, seed in enumerate(expected):
            if seed in existing:
                continue
            row = run_one(cli.robotwin_root, cli.task, task_args, covariance, scale, seed, i)
            with records_path.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            rows.append(row)
            print(json.dumps({"scale": scale, "seed": seed, "status": row["status"],
                              "success": row["success"], "completed": len(rows)}), flush=True)
            if row["status"] == "error":
                raise RuntimeError(f"infrastructure/error seed={seed} scale={scale}: {row['error']}")
        summary = {"scale": scale, "records": len(rows), "complete": len(rows) == cli.episodes,
                   "successes": sum(bool(row["success"]) for row in rows),
                   "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
                   "generated_at_utc": datetime.now(timezone.utc).isoformat()}
        (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": True, "output_dir": str(cli.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
