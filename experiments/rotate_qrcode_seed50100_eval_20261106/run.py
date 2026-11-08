"""Guarded QR SFT evaluation on the latest DAgger seed range."""
from __future__ import annotations

import importlib.util
import socket
import subprocess
from pathlib import Path


BASE = Path("/hdd/robotwin-hil-github/experiments/stack_blocks_three_seed48400_eval_20261105/run.py")
spec = importlib.util.spec_from_file_location("seed48400_evaluation", BASE)
assert spec and spec.loader
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

evaluation.ROOT = Path("/media/ruio/hdd/official_nine_task_pi05_sft_5000_20261101/eval_rotate_qrcode_seed50100_20261106_retry2")
evaluation.ENTRY = Path("/hdd/robotwin-hil-github/experiments/rotate_qrcode_seed50100_eval_20261106/seed_entry.py")
evaluation.CHECKPOINT = Path("/media/ruio/hdd/official_nine_task_pi05_sft_5000_20261101/checkpoints/rotate_qrcode/4999")
evaluation.SERVER_CONFIG = Path("/hdd/official_nine_task_pi05_sft_5000_20261101/eval_rotate_qrcode_min_trace_20261103/control/server_setup/server.yaml")
evaluation.PORT = 18511
evaluation.SERVER_PID = 0  # Set to the verified live server PID immediately before launch.
evaluation.TASK = "rotate_qrcode"
evaluation.SEED_START = 50100
evaluation.CHECKPOINT_SHA = "e101733c792130dcc87b6de7dabaa7be2e6ae0654268ef76baa3ee373c499ec4"
evaluation.NORM_SHA = "13c1e756264e3660f33d1b87b595c29c4953aea8ad435edec22ffe7335664ed0"
evaluation.RUN_ID = "rotate_qrcode_seed50100_20261106_retry2"


def preflight() -> None:
    mount = subprocess.check_output(
        ["findmnt", "-no", "SOURCE", "/media/ruio/hdd"], text=True
    ).strip()
    if mount != "/dev/sda1":
        raise RuntimeError(f"wrong data mount: {mount}")
    if evaluation.digest(evaluation.SOURCE / "scripts/eval_policy_xpolicylab_official_trace.py") != evaluation.EVALUATOR_SHA:
        raise RuntimeError("pinned SFT evaluator changed")
    if not (evaluation.SOURCE / "assets/objects/objaverse/list.json").is_file():
        raise RuntimeError("simulation assets missing")
    if evaluation.digest(evaluation.CHECKPOINT / "params/manifest.ocdbt") != evaluation.CHECKPOINT_SHA:
        raise RuntimeError("checkpoint params mismatch")
    norm = evaluation.CHECKPOINT / "assets/rotate_qrcode_official50_v30_20261101/norm_stats.json"
    if evaluation.digest(norm) != evaluation.NORM_SHA:
        raise RuntimeError("checkpoint norm stats mismatch")
    config = evaluation.SERVER_CONFIG.read_text()
    for fragment in ("task_name: rotate_qrcode", str(evaluation.CHECKPOINT), "state", "h50"):
        if fragment.lower() not in config.lower():
            raise RuntimeError(f"server config missing {fragment}")
    cmdline = Path(f"/proc/{evaluation.SERVER_PID}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if ("stack_server_entry.py" not in cmdline or str(evaluation.SERVER_CONFIG) not in cmdline
            or f"--port {evaluation.PORT}" not in cmdline):
        raise RuntimeError(f"policy service identity changed: {cmdline}")
    with socket.create_connection(("127.0.0.1", evaluation.PORT), timeout=2):
        pass
    rows = subprocess.check_output(["ps", "-eo", "comm=,args="], text=True).splitlines()
    python_args = [parts[1] for row in rows if (parts := row.split(None, 1))
                   and len(parts) == 2 and parts[0].startswith("python")]
    if any("-m dagger.runner" in args for args in python_args):
        raise RuntimeError("DAgger runner is active")
    if any("eval_policy_xpolicylab_official_trace.py" in args for args in python_args):
        raise RuntimeError("another SFT evaluator is active")
    if (evaluation.ROOT / "smoke").exists() or (evaluation.ROOT / "formal").exists():
        raise RuntimeError("refusing to overwrite results")
    result_root = evaluation.SOURCE / "eval_result/rotate_qrcode/Pi_05/demo_clean"
    for name in ("smoke", "formal"):
        if (result_root / f"{evaluation.RUN_ID}_{name}").exists():
            raise RuntimeError(f"official result label exists: {name}")
    available = int(subprocess.check_output(
        ["df", "-B1", "--output=avail", str(evaluation.ROOT.parent)], text=True
    ).splitlines()[-1])
    if available < 100 * 1024**3:
        raise RuntimeError("less than 100 GiB free")
    evaluation.write_json(evaluation.ROOT / "control/protocol.json", {
        "task": evaluation.TASK,
        "seed_start_override": evaluation.SEED_START,
        "seed_start_official_default": 100000,
        "only_evaluator_change": "eval_remote_policy start seed argument",
        "evaluator": str(evaluation.SOURCE / "scripts/eval_policy_xpolicylab_official_trace.py"),
        "evaluator_sha256": evaluation.EVALUATOR_SHA,
        "seed_entry_sha256": evaluation.digest(evaluation.ENTRY),
        "checkpoint": str(evaluation.CHECKPOINT),
        "params_sha256": evaluation.CHECKPOINT_SHA,
        "norm_stats_sha256": evaluation.NORM_SHA,
        "server_pid": evaluation.SERVER_PID,
        "server_config": str(evaluation.SERVER_CONFIG),
        "server_config_sha256": evaluation.digest(evaluation.SERVER_CONFIG),
        "port": evaluation.PORT,
        "expert_check": True,
        "instruction_type": "seen",
        "instruction_pool_size": 50,
        "task_config": "demo_clean",
        "frequency_field": 50,
        "action_type": "joint",
        "smoke_count": 1,
        "formal_count": 50,
        "comparison_limit": "DAgger screening additionally checks planned_joints_legal; no interventions here",
    })


evaluation.preflight = preflight


if __name__ == "__main__":
    import os

    evaluation.SERVER_PID = int(os.environ["EVAL_SERVER_PID"])
    evaluation.main()
