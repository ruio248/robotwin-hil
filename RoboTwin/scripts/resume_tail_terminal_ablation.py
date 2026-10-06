#!/usr/bin/env python3
"""Resume the three recorded-end reward runs from step 10000 to step 30000."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from tail_value.cache import atomic_json, cache_fingerprint, load_manifest, sha256
from tail_value.model import load_checkpoint

SCRIPTS = Path(__file__).resolve().parent
REWARDS = (1, 10, 100)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(args):
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    manifest = load_manifest(args.cache_dir)
    fingerprint = cache_fingerprint(manifest)
    checkpoints = {}
    for reward in REWARDS:
        run_dir = args.runs_root / f"terminal_{reward}"
        starting_checkpoint = run_dir / "step_0010000.pt"
        payload = load_checkpoint(starting_checkpoint)
        config = payload["config"]
        if (payload["step"] != 10000 or config.get("record_end_terminal_reward") != reward
                or config.get("terminal_semantics") != "assumed_record_end_not_verified_success"
                or config.get("alpha") != 1.0 or config.get("seed") != 42
                or payload["cache_fingerprint"] != fingerprint):
            raise ValueError(f"Starting checkpoint protocol mismatch: {starting_checkpoint}")
        if load_checkpoint(run_dir / "last.pt")["step"] != 10000:
            raise ValueError(f"Run is not at step 10000; refusing to overwrite: {run_dir}")
        checkpoints[reward] = sha256(starting_checkpoint)
    status_path = args.runs_root / "resume_to_30000_status.json"
    if status_path.exists() or any((args.runs_root / f"report_{split}_step30000").exists()
                                   for split in ("val", "train")):
        raise FileExistsError("Resume status or step-30000 report already exists")
    state = {"status": "running", "pid": os.getpid(),
             "started_at": datetime.now(timezone.utc).isoformat(),
             "cache_fingerprint": fingerprint,
             "starting_checkpoint_sha256": {str(k): v for k, v in checkpoints.items()},
             "target_steps": 30000, "device": args.device, "jobs": []}

    def save():
        atomic_json(status_path, state)

    save()
    env = dict(os.environ, OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads))
    try:
        for reward in REWARDS:
            run_dir = args.runs_root / f"terminal_{reward}"
            log_path = args.runs_root / f"terminal_{reward}.resume_to_30000.log"
            command = [sys.executable, "-u", str(SCRIPTS / "train_tail.py"),
                       "--cache-dir", str(args.cache_dir), "--output-dir", str(run_dir),
                       "--resume", str(run_dir / "last.pt"), "--record-end-terminal-reward", str(reward),
                       "--steps", "30000", "--seed", "42", "--alpha", "1.0",
                       "--eval-every", "2000", "--log-every", "100",
                       "--device", args.device, "--cpu-threads", str(args.cpu_threads)]
            job = {"reward": reward, "status": "running", "log": str(log_path), "command": command}
            state["jobs"].append(job)
            save()
            with log_path.open("x", encoding="utf-8") as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, check=False)
            job["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"R={reward} resume failed ({result.returncode}); see {log_path}")
            final = load_checkpoint(run_dir / "last.pt")
            if final["step"] != 30000:
                raise ValueError(f"R={reward} finished without a step-30000 checkpoint")
            job["checkpoint_sha256"] = sha256(run_dir / "last.pt")
            job["status"] = "complete"
            save()
        for split in ("val", "train"):
            report = args.runs_root / f"report_{split}_step30000"
            log = args.runs_root / f"report_{split}_step30000.log"
            command = [sys.executable, "-u", str(SCRIPTS / "plot_tail_terminal_ablation.py"),
                       "--cache-dir", str(args.cache_dir), "--runs-root", str(args.runs_root),
                       "--output-dir", str(report), "--split", split,
                       "--expected-steps", "30000", "--device", args.device,
                       "--cpu-threads", str(args.cpu_threads)]
            with log.open("x", encoding="utf-8") as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, check=False)
            if result.returncode:
                raise RuntimeError(f"{split} report failed ({result.returncode}); see {log}")
            state[f"report_{split}"] = str(report)
            save()
        state["status"] = "complete"
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main(parse_args())
