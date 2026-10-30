#!/usr/bin/env python3
"""Start formal evaluation only after a smoke run exits successfully."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-pid", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    if "--smoke-only" in runner_args:
        raise ValueError("formal runner arguments must not include --smoke-only")

    while True:
        cmdline = Path(f"/proc/{args.smoke_pid}/cmdline")
        try:
            command = cmdline.read_bytes()
        except FileNotFoundError:
            command = b""
        if b"run_public_pi05_candidate.py" not in command:
            break
        time.sleep(15)

    state = json.loads((args.output_root / "control/state.json").read_text())
    record_path = args.output_root / "smoke" / f"seed_{args.seed}" / "eval_records.jsonl"
    records = record_path.read_text().splitlines()
    if state.get("phase") != "smoke_complete" or len(records) != 1:
        raise RuntimeError(f"smoke did not complete cleanly: {state}, records={len(records)}")
    record = json.loads(records[0])
    if record.get("task") != args.task or record.get("seed") != args.seed:
        raise RuntimeError(f"smoke record identity mismatch: {record_path}")
    print(f"smoke verified for {args.task} seed {args.seed}; starting formal", flush=True)
    return subprocess.call([sys.executable, str(args.runner), *runner_args])


if __name__ == "__main__":
    raise SystemExit(main())
