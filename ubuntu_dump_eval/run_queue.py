#!/usr/bin/env python3
"""Wait for direct host-to-host checkpoint copies, then evaluate on Ubuntu."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "control"
EXPECTED = {
    5000: (305678, 43, 12438830619, "2bf13aec1ea7bbb1b7c960885b7321b61bd55641ad3608ace3c5bef82d732660"),
    4000: (307616, 38, 12441407033, "95e934d2da9d88509f0e3f0c78966b229e9c79e6ff11afd7546da5687bf49495"),
}


def status(phase: str, milestone: int, detail: str = "") -> None:
    path = CONTROL / "queue_state.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"phase": phase, "milestone": milestone, "detail": detail,
                                      "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n")
    temporary.replace(path)


def transfer_running(pid: int, milestone: int) -> bool:
    command = Path(f"/proc/{pid}/cmdline")
    try:
        raw = command.read_bytes()
    except FileNotFoundError:
        return False
    return f"/dump_bin_bigbin/{milestone}/" in raw.replace(b"\0", b" ").decode(errors="replace")


def checkpoint_complete(milestone: int) -> bool:
    _, count, total, manifest_sha = EXPECTED[milestone]
    checkpoint = ROOT / "checkpoints" / str(milestone)
    files = [path for path in checkpoint.rglob("*") if path.is_file()]
    if len(files) != count or sum(path.stat().st_size for path in files) != total:
        return False
    manifest = checkpoint / "params/manifest.ocdbt"
    return hashlib.sha256(manifest.read_bytes()).hexdigest() == manifest_sha


def main() -> None:
    if os.geteuid() == 0:
        raise RuntimeError("queue must run as the regular Ubuntu user")
    for milestone, (pid, _, _, _) in EXPECTED.items():
        status("waiting_transfer", milestone)
        while transfer_running(pid, milestone):
            time.sleep(30)
        if not checkpoint_complete(milestone):
            status("failed", milestone, "direct rsync ended but checkpoint bytes or manifest differ")
            raise RuntimeError(f"incomplete checkpoint {milestone}")
        status("evaluating", milestone)
        log = ROOT / f"eval_{milestone}.log"
        with log.open("a") as output:
            completed = subprocess.run([sys.executable, "-u", str(CONTROL / "run_milestone.py"), str(milestone)],
                                       stdout=output, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            status("failed", milestone, f"evaluator exited {completed.returncode}; inspect {log}")
            raise RuntimeError(f"evaluation failed for {milestone}: {completed.returncode}")
        status("milestone_complete", milestone)
    status("complete", 4000)


if __name__ == "__main__":
    main()
