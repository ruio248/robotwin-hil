#!/usr/bin/env python3
"""Launch an isolated handover-stage diagnostic against one public policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--target-stage", type=int, choices=(1, 4, 6), default=6)
    parser.add_argument("--max-stage-actions", type=int, default=300)
    cli = parser.parse_args()
    cfg = json.loads(cli.deployment.read_text())
    if cli.output_dir.exists():
        parser.error("output-dir must be new; refusing to overwrite diagnostic results")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", cfg["port"]))
    free = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
         "-i", str(cfg["gpu"])], text=True).strip())
    if free < cfg["minimum_free_mib"]:
        raise RuntimeError(f"GPU {cfg['gpu']} has only {free} MiB free")
    cli.output_dir.mkdir(parents=True)
    common = os.environ | cfg["common_env"]
    for item in cfg["mkdirs"]:
        Path(item).mkdir(parents=True, exist_ok=True)
    server = None
    completed = []
    try:
        with (cli.output_dir / "server.log").open("w") as log:
            server = subprocess.Popen(cfg["server_command"], cwd=cfg["server_cwd"],
                                      env=common | cfg["server_env"],
                                      stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"policy server exited during startup: {server.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", cfg["port"]), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("policy server startup timed out")
        for seed in cli.seed:
            dest = cli.output_dir / f"seed_{seed}.json"
            cmd = [cfg["sim_python"], "-u",
                   str(Path(__file__).with_name("diagnose_handover_transfer_stage.py")),
                   "--seed", str(seed), "--port", str(cfg["port"]),
                   "--target-stage", str(cli.target_stage),
                   "--output", str(dest), "--renderer-device", cfg["renderer_device"],
                   "--max-stage-actions", str(cli.max_stage_actions)]
            with (cli.output_dir / f"seed_{seed}.log").open("w") as log:
                proc = subprocess.run(cmd, cwd=cfg["sim_cwd"],
                                      env=common | cfg["sim_env"], stdout=log,
                                      stderr=subprocess.STDOUT, check=False)
            row = {"seed": seed, "returncode": proc.returncode,
                   "output": str(dest), "result_exists": dest.exists()}
            completed.append(row)
            print(json.dumps(row), flush=True)
            if proc.returncode != 0 or not dest.exists():
                raise RuntimeError(f"stage diagnostic failed for seed {seed}; see its log")
        return 0
    finally:
        (cli.output_dir / "orchestration.json").write_text(json.dumps({
            "task": "handover_block", "target_stage": cli.target_stage,
            "diagnostic_only": True,
            "accepted_for_training": False, "completed": completed}, indent=2) + "\n")
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                raise RuntimeError("own policy server did not terminate")


if __name__ == "__main__":
    raise SystemExit(main())
