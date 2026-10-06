#!/usr/bin/env python3
"""Sequential, immutable R=1/10/100 recorded-end boundary ablation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from tail_value.cache import atomic_json, cache_fingerprint, load_manifest, sha256

SCRIPTS = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def main(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite ablation: {args.output_dir}")
    if args.steps != 10000 or args.alpha != 1.0:
        raise ValueError("This controlled experiment requires 10000 steps and alpha=1.0")
    manifest = load_manifest(args.cache_dir)
    splits = Counter(e["split"] for e in manifest["episodes"])
    if manifest["config"]["source_format"] != "lerobot" or splits != {"train": 405, "val": 45}:
        raise ValueError(f"Expected the complete 405/45 LeRobot cache, got {splits}")
    source_files = [SCRIPTS / name for name in
                    ("train_tail.py", "eval_tail_value.py", "plot_tail_terminal_ablation.py",
                     "run_tail_terminal_ablation.py")]
    source_files += sorted((SCRIPTS / "tail_value").glob("*.py"))
    state = {"status": "running", "pid": os.getpid(),
             "started_at": datetime.now(timezone.utc).isoformat(),
             "cache_fingerprint": cache_fingerprint(manifest), "seed": args.seed,
             "alpha": args.alpha, "steps": args.steps, "device": args.device,
             "terminal_assumption": "final real adjacent recorded transition is treated as terminal; success unverified",
             "source_sha256": {str(p.relative_to(SCRIPTS)): sha256(p) for p in source_files},
             "jobs": []}
    args.output_dir.mkdir(parents=True)
    status_path = args.output_dir / "status.json"

    def save():
        atomic_json(status_path, state)

    save()
    env = dict(os.environ, OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads))
    try:
        for reward in (1, 10, 100):
            run_dir = args.output_dir / f"terminal_{reward}"
            command = [sys.executable, "-u", str(SCRIPTS / "train_tail.py"),
                       "--cache-dir", str(args.cache_dir), "--output-dir", str(run_dir),
                       "--record-end-terminal-reward", str(reward), "--steps", str(args.steps),
                       "--seed", str(args.seed), "--alpha", str(args.alpha),
                       "--eval-every", "500", "--log-every", "50", "--device", args.device,
                       "--cpu-threads", str(args.cpu_threads)]
            job = {"reward": reward, "status": "running", "command": command,
                   "log": str(args.output_dir / f"terminal_{reward}.log")}
            state["jobs"].append(job)
            save()
            with open(job["log"], "x", encoding="utf-8") as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, check=False)
            job["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"R={reward} training failed ({result.returncode}); see {job['log']}")
            job["checkpoint"] = str(run_dir / "last.pt")
            job["checkpoint_sha256"] = sha256(run_dir / "last.pt")
            job["status"] = "complete"
            save()
        for split in ("val", "train"):
            report = args.output_dir / f"report_{split}"
            command = [sys.executable, "-u", str(SCRIPTS / "plot_tail_terminal_ablation.py"),
                       "--cache-dir", str(args.cache_dir), "--runs-root", str(args.output_dir),
                       "--output-dir", str(report), "--split", split,
                       "--device", args.device, "--cpu-threads", str(args.cpu_threads)]
            log = args.output_dir / f"report_{split}.log"
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
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main(parse_args())
