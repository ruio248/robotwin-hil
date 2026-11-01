#!/usr/bin/env python3
"""One guarded official-source smoke and 50-scene trace evaluation on Ubuntu.

Only the two passive recorder files are added to the pristine RoboTwin copy.
This runner is experiment-specific and deliberately never resumes partial data.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/media/ruio/hdd/handover_block_official_min_trace_eval_20261102")
REPOSITORY = Path("/media/ruio/hdd/robotwin-hil-github")
ROBOTWIN = ROOT / "official_runtime"
PRIOR_SERVER = Path("/media/ruio/hdd/handover_block_official_sft_5000_20261101/official_eval_20261101")
PRIOR_ROBOTWIN = Path("/media/ruio/hdd/handover_block_official_collect_pr510_20261031/runtime_robotwin")
CHECKPOINT = Path("/media/ruio/hdd/handover_block_official_sft_5000_20261101/checkpoint/4999")
SIM_PYTHON = Path("/hdd/miniconda3/envs/robotwin_hil/bin/python")
SERVER_PYTHON = Path("/hdd/Openpi-PACE-eval-20261022/.venv/bin/python")
PORT = 18496
MODEL_MANIFEST_SHA256 = "e62227471473a1b7bb0b81c2a641453b99223b72440c967c56b7c96421b4b645"
INSTRUCTION_SHA256 = "c67350f7684c5d39cf7458c9775e087670e4f03828cea94536f1a7a7d203edd2"
REPO_ID = "handover_block_pr510_official50_v30_20261101"
RUN_ID = "hb_official_min_trace_20261102"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def state(phase: str, **details: object) -> None:
    save_json(ROOT / "control/state.json", {
        "phase": phase, "time_utc": datetime.now(timezone.utc).isoformat(), **details,
    })


def port_open() -> bool:
    with socket.socket() as connection:
        connection.settimeout(1)
        return connection.connect_ex(("127.0.0.1", PORT)) == 0


def preflight() -> None:
    mount = subprocess.check_output(["findmnt", "-no", "SOURCE", "/media/ruio/hdd"], text=True).strip()
    if mount != "/dev/sda1":
        raise RuntimeError(f"incorrect data mount: {mount}")
    head = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    remote_head = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "origin/main"], text=True).strip()
    if head != remote_head:
        raise RuntimeError(f"Ubuntu source is not at origin/main: {head} != {remote_head}")
    source = REPOSITORY / "RoboTwin"
    for relative in (
        "envs/_base_task.py", "envs/handover_block.py", "env_cfg/task_config/demo_clean.yml",
        "description/task_instruction/handover_block.json", "scripts/eval_policy_xpolicylab.py",
        "XPolicyLab/setup_policy_server.py", "XPolicyLab/policy/Pi_05/model.py",
    ):
        if sha256(source / relative) != sha256(ROBOTWIN / relative):
            raise RuntimeError(f"official source mismatch: {relative}")
    for name in ("eval_policy_xpolicylab_official_trace.py", "official_policy_trace.py"):
        overlay = REPOSITORY / "robotwin_hil/overlay/scripts" / name
        if sha256(overlay) != sha256(ROBOTWIN / "scripts" / name):
            raise RuntimeError(f"trace source mismatch: {name}")
    for relative in ("XPolicyLab/setup_policy_server.py", "XPolicyLab/policy/Pi_05/model.py"):
        if sha256(PRIOR_ROBOTWIN / relative) != sha256(ROBOTWIN / relative):
            raise RuntimeError(f"policy server source mismatch: {relative}")
    if sha256(ROBOTWIN / "description/task_instruction/handover_block.json") != INSTRUCTION_SHA256:
        raise RuntimeError("PR #510 instruction mismatch")
    if sha256(CHECKPOINT / "params/manifest.ocdbt") != MODEL_MANIFEST_SHA256:
        raise RuntimeError("checkpoint params manifest mismatch")
    norm_stats = CHECKPOINT / "assets" / REPO_ID / "norm_stats.json"
    if not norm_stats.is_file():
        raise RuntimeError("checkpoint norm_stats missing")
    if port_open():
        raise RuntimeError(f"port {PORT} already in use")
    gpu_used = int(subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()[0])
    if gpu_used > 2048:
        raise RuntimeError(f"GPU0 already uses {gpu_used} MiB")
    for path in (SIM_PYTHON, SERVER_PYTHON, PRIOR_SERVER / "server_entry.py", PRIOR_SERVER / "server.yaml"):
        if not path.is_file():
            raise RuntimeError(f"missing runtime file: {path}")
    for path in (ROOT / "smoke", ROOT / "formal", ROOT / "summary.json"):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
    for label in (f"{RUN_ID}_smoke", f"{RUN_ID}_formal"):
        result_path = ROBOTWIN / "eval_result/handover_block/Pi_05/demo_clean" / label
        if result_path.exists():
            raise RuntimeError(f"refusing to reuse official result: {result_path}")
    save_json(ROOT / "control/protocol.json", {
        "repository_commit": head,
        "official_source_commit": "4c5c753d66c54c3f6ce855ddf426bcaf3e327250",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_params_manifest_sha256": MODEL_MANIFEST_SHA256,
        "norm_stats_sha256": sha256(norm_stats),
        "base_task_sha256": sha256(ROBOTWIN / "envs/_base_task.py"),
        "task_sha256": sha256(ROBOTWIN / "envs/handover_block.py"),
        "instruction_sha256": INSTRUCTION_SHA256,
        "trace_evaluator_sha256": sha256(ROBOTWIN / "scripts/eval_policy_xpolicylab_official_trace.py"),
        "trace_recorder_sha256": sha256(ROBOTWIN / "scripts/official_policy_trace.py"),
        "server_entry": str(PRIOR_SERVER / "server_entry.py"),
        "server_policy_model_sha256": sha256(ROBOTWIN / "XPolicyLab/policy/Pi_05/model.py"),
        "protocol": "official scene/expert/seen control plus passive query trace",
        "task_config": "demo_clean", "expert_check": True, "instruction_type": "seen",
        "seed": 0, "episodes": 50, "frequency_field": 50, "horizon": 50,
        "state": True, "action_type": "joint", "gpu": 0, "port": PORT,
    })


def evaluate(name: str, count: int, environment: dict[str, str]) -> dict:
    directory = ROOT / name
    directory.mkdir(exist_ok=False)
    label = f"{RUN_ID}_{name}"
    trace = directory / "trace"
    argv = [
        str(SIM_PYTHON), "-u", str(ROBOTWIN / "scripts/eval_policy_xpolicylab_official_trace.py"),
        "--task_name", "handover_block", "--env_cfg_type", "aloha_agilex",
        "--policy_name", "Pi_05", "--host", "127.0.0.1", "--port", str(PORT),
        "--protocol", "ws", "--root_dir", str(ROBOTWIN), "--device_id", "0",
        "--task_config", "demo_clean", "--seed", "0", "--test_num", str(count),
        "--eval_batch", "false", "--expert_check", "true", "--instruction_type", "seen",
        "--frequency", "50", "--additional_info", f"ckpt_setting={label},action_type=joint",
        "--trace_dir", str(trace),
    ]
    save_json(directory / "command.json", {"argv": argv, "cwd": str(ROBOTWIN)})
    with (directory / "eval.log").open("w") as log:
        result = subprocess.run(argv, cwd=ROBOTWIN, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, timeout=40 * 3600, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} evaluator exited {result.returncode}: {directory / 'eval.log'}")
    manifest = json.loads((trace / "manifest.json").read_text())
    counted = [episode for episode in manifest["episodes"] if episode["counted_in_official_result"]]
    seeds = [episode["scene_seed"] for episode in counted]
    if not (manifest["status"] == "complete" and manifest["d_analysis_ready"] and
            len(counted) == count and len(set(seeds)) == count and
            all(episode["queries"] for episode in counted)):
        raise RuntimeError(f"{name} trace cannot be accepted: {trace / 'manifest.json'}")
    official_result = list((ROBOTWIN / "eval_result/handover_block/Pi_05/demo_clean" / label).glob("*/_result.txt"))
    if len(official_result) != 1:
        raise RuntimeError(f"{name} official result file not unique: {official_result}")
    official_rate = float(official_result[0].read_text().strip().splitlines()[-1])
    successes = manifest["successful_episodes"]
    if abs(official_rate - successes / count) > 1e-9:
        raise RuntimeError(f"{name} official result disagrees with trace")
    summary = {
        "complete": True, "count": count, "successes": successes,
        "policy_failures": manifest["policy_failure_episodes"],
        "official_rate": official_rate, "official_result": str(official_result[0]),
        "trace_manifest": str(trace / "manifest.json"),
        "scene_results": [{"scene_seed": episode["scene_seed"], "success": episode["success"]}
                          for episode in counted],
        "expert_rejected": sum(scene["stage"] == "expert_rejected" for scene in manifest["skipped_scenes"]),
    }
    save_json(directory / "summary.json", summary)
    return summary


def main() -> None:
    control = ROOT / "control"
    control.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(control / "runner.pid", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644), "w") as stream:
        stream.write(f"{os.getpid()}\n")
    state("preflight")
    preflight()
    temporary = ROOT / "tmp"
    for path in (temporary, temporary / "xdg", temporary / "warp", ROOT / "cache/jax"):
        path.mkdir(parents=True, exist_ok=True)
    service_env = os.environ.copy()
    service_env.update({
        "PYTHONPATH": f"{PRIOR_SERVER / 'openpi/src'}:{PRIOR_ROBOTWIN}:{PRIOR_ROBOTWIN / 'XPolicyLab'}",
        "CUDA_VISIBLE_DEVICES": "0", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.55",
        "JAX_COMPILATION_CACHE_DIR": str(ROOT / "cache/jax"),
        "HF_HOME": "/media/ruio/hdd/pace_pi05_eval_20261022/.hf-home",
        "OPENPI_DATA_HOME": "/media/ruio/hdd/pace_pi05_eval_20261022/openpi_data",
        "TMPDIR": str(temporary), "XDG_CACHE_HOME": str(temporary / "xdg"),
        "WANDB_MODE": "disabled", "PYTHONDONTWRITEBYTECODE": "1",
    })
    simulation_env = os.environ.copy()
    simulation_env.update({
        "PYTHONPATH": f"{ROBOTWIN}:{ROBOTWIN / 'XPolicyLab'}",
        "CUDA_VISIBLE_DEVICES": "0", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "ROBOTWIN_ROOT": str(ROBOTWIN), "TMPDIR": str(temporary),
        "XDG_CACHE_HOME": str(temporary / "xdg"), "WARP_CACHE_PATH": str(temporary / "warp"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    state("starting_server")
    with (control / "server.log").open("w") as log:
        server = subprocess.Popen(
            [str(SERVER_PYTHON), "-u", str(PRIOR_SERVER / "server_entry.py"),
             "--config_path", str(PRIOR_SERVER / "server.yaml")],
            cwd=ROBOTWIN, env=service_env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (control / "server.pid").write_text(f"{server.pid}\n")
    try:
        for _ in range(600):
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}: {control / 'server.log'}")
            if port_open():
                break
            time.sleep(1)
        else:
            raise TimeoutError("policy server did not bind within 10 minutes")
        state("smoke", server_pid=server.pid)
        smoke = evaluate("smoke", 1, simulation_env)
        state("formal", server_pid=server.pid, smoke=smoke)
        formal = evaluate("formal", 50, simulation_env)
        save_json(ROOT / "summary.json", {"smoke": smoke, "formal": formal})
        state("complete", successes=formal["successes"], episodes=50)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        state("failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
