#!/usr/bin/env python3
"""Evaluate one disjoint lane of the 15 trained Pi0.5 inference checkpoints."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


EVAL = Path(os.environ.get("ROBOTWIN_EVAL_ROOT", Path.cwd())).resolve()
ROBOTWIN = EVAL / "robotwin"
SERVER_PY = Path("/hdd/Openpi-PACE-eval-20261022/.venv/bin/python")
SIM_PY = Path("/hdd/miniconda3/envs/robotwin_hil/bin/python")
TASKS = ("turn_switch", "dump_bin_bigbin", "stack_blocks_three")
MILESTONES = (1000, 2000, 3000, 4000, 5000)
STEP_LIMITS = {"turn_switch": 400, "dump_bin_bigbin": 600, "stack_blocks_three": 1200}
SEEDS = tuple(range(31000, 31050))
JOBS = tuple((task, milestone) for task in TASKS for milestone in MILESTONES)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def status(lane: int, phase: str, task: str = "none", milestone: int = 0, detail: str = "") -> None:
    atomic_json(
        EVAL / "control" / f"lane_{lane}_state.json",
        {"lane": lane, "phase": phase, "task": task, "milestone": milestone,
         "detail": detail, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
    )


def checkpoint_path(task: str, milestone: int) -> Path:
    return EVAL / "checkpoints" / str(milestone)


def validate_checkpoint(task: str, milestone: int) -> Path:
    ckpt = checkpoint_path(task, milestone)
    asset = ckpt / "assets" / "ruio248" / f"robotwin_{task}_demo_clean50_fixedprompt_proprio_20261018" / "norm_stats.json"
    if not (ckpt / "params/manifest.ocdbt").is_file() or not asset.is_file() or asset.stat().st_size == 0:
        raise FileNotFoundError(f"incomplete inference milestone: {ckpt}")
    return ckpt


def tcp_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def server_environment(gpu: int, temporary: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "TMPDIR": str(temporary), "XDG_CACHE_HOME": str(temporary / "xdg"),
        "HF_HOME": "/media/ruio/hdd/pace_pi05_eval_20261022/.hf-home",
        "OPENPI_DATA_HOME": "/media/ruio/hdd/pace_pi05_eval_20261022/openpi_data",
        "JAX_COMPILATION_CACHE_DIR": str(EVAL / "cache/jax"),
        "CUDA_VISIBLE_DEVICES": str(gpu), "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.60",
        "PYTHONPATH": f"{EVAL / 'openpi/src'}:{ROBOTWIN}:{ROBOTWIN / 'XPolicyLab'}", "WANDB_MODE": "disabled",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def sim_environment(gpu: int, temporary: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "TMPDIR": str(temporary), "XDG_CACHE_HOME": str(temporary / "xdg"),
        "WARP_CACHE_PATH": str(temporary / "warp"), "CUDA_VISIBLE_DEVICES": str(gpu),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONPATH": str(ROBOTWIN),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def validated_seed_record(output: Path, seed: int) -> dict | None:
    records = output / "eval_records.jsonl"
    summaries = list(output.glob("summary_*.json"))
    if not records.exists() and not summaries:
        return None
    if not records.exists() or len(summaries) != 1:
        raise RuntimeError(f"partial seed output requires inspection: {output}")
    lines = [line for line in records.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one record in {records}")
    record = json.loads(lines[0])
    summary = json.loads(summaries[0].read_text())
    if int(record["seed"]) != seed or summary.get("episodes") != 1 or summary.get("successes") != int(bool(record["success"])):
        raise RuntimeError(f"invalid seed result: {output}")
    return record


def run_rollout(task: str, milestone: int, seed: int, output: Path, gpu: int, port: int, temporary: Path) -> dict:
    prior = validated_seed_record(output, seed)
    if prior is not None:
        return prior
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(SIM_PY), "-u", str(EVAL / "control/policy_eval_three_tasks.py"),
        "--host", "127.0.0.1", "--port", str(port),
        "--policy-name", "Pi_05_RobotTwin", "--ckpt-name", f"{task}_proprio_5000_{milestone}",
        "--task-name", task, "--task-config", "demo_clean",
        "--seed-start", str(seed), "--episodes", "1", "--frequency", "30",
        "--render-freq", "0", "--save-freq", "15", "--step-limit", str(STEP_LIMITS[task]),
        "--save-videos", "none", "--output-dir", str(output),
    ]
    with (output / "eval.log").open("a") as log:
        completed = subprocess.run(command, cwd=ROBOTWIN, env=sim_environment(gpu, temporary),
                                   stdout=log, stderr=subprocess.STDOUT, timeout=7200, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"infrastructure/evaluation error rc={completed.returncode}, inspect {output / 'eval.log'}")
    record = validated_seed_record(output, seed)
    if record is None:
        raise RuntimeError(f"missing result despite zero exit: {output}")
    return record


def stop_own_server(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=30)


def evaluate_job(lane: int, gpu: int, port: int, task: str, milestone: int) -> None:
    ckpt = validate_checkpoint(task, milestone)
    output = EVAL / "results" / task / str(milestone)
    formal = output / "formal"
    job_summary = output / "aggregate_summary.json"
    if job_summary.is_file():
        saved = json.loads(job_summary.read_text())
        if (saved.get("complete") is not True or saved.get("seeds") != list(SEEDS)
                or len(saved.get("records", [])) != len(SEEDS)):
            raise RuntimeError(f"invalid existing aggregate: {job_summary}")
        status(lane, "already_complete", task, milestone)
        return
    config = EVAL / "configs" / f"{task}_{milestone}.yml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config_text = (
        "policy_name: Pi_05_RobotTwin\n"
        f"task_name: {task}\n"
        "action_type: joint\nenv_cfg_type: aloha_agilex\n"
        f"train_config_name: pi05_robotwin_{task}_demo_clean50_proprio_5000_state_true_w2_20261019\n"
        f"repo_id: ruio248/robotwin_{task}_demo_clean50_fixedprompt_proprio_20261018\n"
        f"model_path: {ckpt}\nhost: 127.0.0.1\nport: {port}\nprotocol: ws\n"
    )
    if config.exists():
        if config.read_text() != config_text:
            raise RuntimeError(f"existing config differs from locked protocol: {config}")
    else:
        config.write_text(config_text)
    if tcp_open(port):
        raise RuntimeError(f"port {port} already occupied; refusing to touch its service")
    temporary = EVAL / "tmp" / f"lane_{lane}"
    for name in ("xdg", "warp"):
        (temporary / name).mkdir(parents=True, exist_ok=True)
    status(lane, "starting_server", task, milestone)
    (output / "server.log").parent.mkdir(parents=True, exist_ok=True)
    with (output / "server.log").open("a") as server_log:
        command = [str(SERVER_PY), "-u", str(ROBOTWIN / "XPolicyLab/setup_policy_server.py"),
                   "--config_path", str(config), "--host", "127.0.0.1", "--port", str(port)]
        server = subprocess.Popen(command, cwd=ROBOTWIN, env=server_environment(gpu, temporary),
                                  stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(output / "server_process.json", {"pid": server.pid, "command": command, "gpu": gpu, "port": port})
        try:
            deadline = time.monotonic() + 900
            while not tcp_open(port):
                if server.poll() is not None:
                    raise RuntimeError(f"server exited rc={server.returncode}; inspect {output / 'server.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"server startup timed out: {output / 'server.log'}")
                time.sleep(2)
            status(lane, "smoke", task, milestone, "seed_31000_not_counted")
            run_rollout(task, milestone, 31000, output / "smoke" / "seed_31000", gpu, port, temporary)
            records = []
            for seed in SEEDS:
                status(lane, "formal", task, milestone, f"seed={seed}")
                records.append(run_rollout(task, milestone, seed, formal / f"seed_{seed}", gpu, port, temporary))
            if [int(row["seed"]) for row in records] != list(SEEDS):
                raise RuntimeError(f"seed mismatch for {task}/{milestone}")
            aggregate = {
                "complete": True, "task": task, "milestone_updates": milestone,
                "checkpoint": str(ckpt), "checkpoint_manifest_sha256": hashlib.sha256((ckpt / "params/manifest.ocdbt").read_bytes()).hexdigest(),
                "config": str(config), "seeds": list(SEEDS), "episodes": len(SEEDS),
                "successes": sum(bool(row["success"]) for row in records),
                "success_rate": sum(bool(row["success"]) for row in records) / len(SEEDS),
                "records": records,
            }
            atomic_json(job_summary, aggregate)
            status(lane, "job_complete", task, milestone,
                   f"successes={aggregate['successes']}/{len(SEEDS)}")
        finally:
            stop_own_server(server)


def aggregate_all() -> None:
    results = []
    for task, milestone in JOBS:
        path = EVAL / "results" / task / str(milestone) / "aggregate_summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text())
        if value.get("complete") is not True or value.get("seeds") != list(SEEDS):
            raise RuntimeError(f"incomplete aggregate: {path}")
        results.append({k: value[k] for k in ("task", "milestone_updates", "episodes", "successes", "success_rate", "checkpoint")})
    atomic_json(EVAL / "aggregate_all.json", {"complete": True, "total_jobs": len(JOBS),
                                               "total_rollouts": len(JOBS) * len(SEEDS), "results": results})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=int, choices=(0, 1))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate_all()
        return
    if args.lane is None or args.gpu is None or args.port is None:
        parser.error("--lane, --gpu and --port required unless --aggregate")
    EVAL.joinpath("control").mkdir(parents=True, exist_ok=True)
    with (EVAL / "control" / f"lane_{args.lane}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status(args.lane, "running")
        try:
            for index, (task, milestone) in enumerate(JOBS):
                if index % 2 == args.lane:
                    evaluate_job(args.lane, args.gpu, args.port, task, milestone)
            status(args.lane, "complete")
        except BaseException as error:
            status(args.lane, "failed", task, milestone, repr(error))
            raise


if __name__ == "__main__":
    main()
