#!/usr/bin/env python3
"""Run an A800 dump_bin_bigbin milestone on Ubuntu under the same eval protocol."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import eval_checkpoint_lane as lane


EXPECTED_TASK_SHA256 = "9afb1b1cd02f5ddcc69e6502254a4675ed9a750262bcce59c695ce8181971630"
TASK = "dump_bin_bigbin"
GPU = 0
PORT = 18480


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("milestone", type=int, choices=(3000, 4000, 5000))
    args = parser.parse_args()
    root = lane.EVAL
    if subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", "/media/ruio/hdd"], text=True).strip() != "/dev/sda1":
        raise RuntimeError("Ubuntu data disk is not mounted as /dev/sda1")
    actual = hashlib.sha256((lane.ROBOTWIN / "envs/dump_bin_bigbin.py").read_bytes()).hexdigest()
    if actual != EXPECTED_TASK_SHA256:
        raise RuntimeError(f"settling-fix code differs: {actual}")
    if not (lane.ROBOTWIN / "assets/objects/063_tabletrashbin").is_dir():
        raise RuntimeError("RoboTwin assets are unavailable")
    if lane.tcp_open(PORT):
        raise RuntimeError(f"port {PORT} is already in use")
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(GPU)], text=True
    )
    if int(memory.strip()) < 17000:
        raise RuntimeError(f"GPU {GPU} has insufficient free memory: {memory.strip()} MiB")
    lane.validate_checkpoint(TASK, args.milestone)
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    (root / f"protocol_{args.milestone}.json").write_text(json.dumps({
        "source": "A800 state=True full-param SFT milestones",
        "task": TASK,
        "milestone": args.milestone,
        "seeds": list(lane.SEEDS),
        "scene": "demo_clean",
        "action": "absolute joint, complete H=10 chunk",
        "frequency_field": 30,
        "step_limit": 600,
        "prompt": "Pick up the small bin and pour all the balls into the big bin.",
        "settling_fix_sha256": actual,
        "gpu": GPU,
        "port": PORT,
    }, indent=2) + "\n")
    with (control / "ubuntu_eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lane.evaluate_job(0, GPU, PORT, TASK, args.milestone)


if __name__ == "__main__":
    main()
