#!/usr/bin/env python3
"""Rerun selected A800 Pi0.5 failure seeds with video, without changing evaluations."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/data-training/robotwin-hil-lyt")
ROBOTWIN = ROOT / "RoboTwin"
SERVER_PY = ROOT / "piperx/openpi/.venv/bin/python"
SIM_PY = Path("/data-training/qplanning_repro/.venv-robotwin-copy/bin/python")
SIM_DEPS = ROOT / "outputs/state_true_three_checkpoint_eval_31000_31099_20261015/sim_py_deps"
OUTPUT = Path(os.environ.get(
    "REVIEW_OUTPUT_DIR", str(ROOT / "outputs/seven_task_failure_review_20261027")
))
REVIEW_OUTCOME = os.environ.get("REVIEW_OUTCOME", "failure")
PORT = int(os.environ.get("REVIEW_PORT", "18486"))
GPU = int(os.environ.get("REVIEW_GPU", "4"))

JOBS = [
    {
        "task": "move_can_pot",
        "original": ROOT / "outputs/robotwin_dagger_screen_20261024/results/move_can_pot_h50/aggregate_summary.json",
        "server_config": ROOT / "outputs/robotwin_dagger_screen_20261024/results/move_can_pot_h50/control/server.yml",
        "prompt": "there is a can and a pot on the table, use one arm to pick up the can and move it beside the pot",
        "horizon": 50, "frequency": 50, "step_limit": 400,
    },
    {
        "task": "rotate_qrcode",
        "original": ROOT / "outputs/robotwin_dagger_screen_20261024/results/rotate_qrcode_h50/aggregate_summary.json",
        "server_config": ROOT / "outputs/robotwin_dagger_screen_20261024/results/rotate_qrcode_h50/control/server.yml",
        "prompt": "Use arm to catch the qrcode board on the table, pick it up and rotate to let the qrcode face towards you",
        "horizon": 50, "frequency": 50, "step_limit": 400,
    },
    {
        "task": "stack_blocks_three",
        "original": ROOT / "outputs/three_task_pi05_15_checkpoint_eval_31000_31049_20261021/results/stack_blocks_three/5000/aggregate_summary.json",
        "server_config": ROOT / "outputs/three_task_pi05_15_checkpoint_eval_31000_31049_20261021/configs/stack_blocks_three_5000.yml",
        "prompt": "Stack the blocks with green on red and blue on green.",
        "horizon": 10, "frequency": 30, "step_limit": 1200,
    },
    {
        "task": "put_object_cabinet",
        "original": ROOT / "outputs/robotwin_dagger_screen_20261024/results/put_object_cabinet_h50/aggregate_summary.json",
        "server_config": ROOT / "outputs/robotwin_dagger_screen_20261024/results/put_object_cabinet_h50/control/server.yml",
        "prompt": "use one arm to hold the object and the other arm to open the cabinet drawer, then place the object inside",
        "horizon": 50, "frequency": 50, "step_limit": 700,
    },
]


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def port_open() -> bool:
    with socket.socket() as connection:
        connection.settimeout(1)
        return connection.connect_ex(("127.0.0.1", PORT)) == 0


def wait_ready(process: subprocess.Popen, timeout=900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited rc={process.returncode}")
        if port_open():
            return
        time.sleep(2)
    raise TimeoutError("server readiness timed out")


def gpu_free() -> int:
    return int(subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
        "-i", str(GPU),
    ], text=True).strip())


def server_env(task_dir: Path, task: str) -> dict[str, str]:
    env = os.environ.copy()
    temp = task_dir / "tmp"
    for name in ("xdg", "warp"):
        (temp / name).mkdir(parents=True, exist_ok=True)
    if task == "stack_blocks_three":
        policy_source = ROOT / "outputs/three_task_pi05_proprio_sft_5000_state_true_w2_20261019/openpi/src"
        policy_path = f"{ROBOTWIN}:{policy_source}"
    else:
        public_root = ROOT / "outputs/public_pi05_dump_bin_bigbin_matched_eval_20261024/openpi_pace"
        policy_path = f"{ROBOTWIN}:{public_root / 'src'}:{public_root / 'pace/src'}"
    env.update({
        "TMPDIR": str(temp), "XDG_CACHE_HOME": str(temp / "xdg"),
        "HF_HOME": str(OUTPUT / "hf"), "OPENPI_DATA_HOME": str(OUTPUT / "openpi"),
        "JAX_COMPILATION_CACHE_DIR": str(OUTPUT / "jax"),
        "CUDA_VISIBLE_DEVICES": str(GPU), "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
        "PYTHONPATH": policy_path,
        "WANDB_MODE": "disabled", "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def sim_env(task_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    temp = task_dir / "tmp"
    env.update({
        "TMPDIR": str(temp), "XDG_CACHE_HOME": str(temp / "xdg"),
        "WARP_CACHE_PATH": str(temp / "warp"), "CUDA_VISIBLE_DEVICES": str(GPU),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONPATH": f"{SIM_DEPS}:{ROBOTWIN}", "PYTHONDONTWRITEBYTECODE": "1",
        "ROBOTWIN_ROOT": str(ROBOTWIN),
    })
    return env


def run_one(job: dict) -> None:
    task = job["task"]
    task_dir = OUTPUT / task
    task_dir.mkdir(parents=True, exist_ok=True)
    original = json.loads(job["original"].read_text())
    want_success = REVIEW_OUTCOME == "success"
    candidate_seeds = [int(row["seed"]) for row in original["records"]
                       if row["success"] is want_success]
    if len(candidate_seeds) < 3:
        raise RuntimeError(f"{task}: fewer than three original {REVIEW_OUTCOME} cases")
    if port_open():
        raise RuntimeError(f"port {PORT} occupied before {task}")
    if gpu_free() < 40000:
        raise RuntimeError(f"GPU {GPU} insufficient free memory before {task}")
    config_lines = job["server_config"].read_text().splitlines()
    if sum(line.startswith("port:") for line in config_lines) != 1:
        raise RuntimeError(f"unexpected server configuration: {job['server_config']}")
    config_copy = task_dir / "server_config.yml"
    config_copy.write_text("\n".join(
        f"port: {PORT}" if line.startswith("port:") else line for line in config_lines
    ) + "\n")
    command = [str(SERVER_PY), "-u", str(ROBOTWIN / "XPolicyLab/setup_policy_server.py"),
               "--config_path", str(config_copy), "--host", "127.0.0.1",
               "--port", str(PORT)]
    write_json(task_dir / "protocol.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in job.items()},
        "note": f"Fresh stochastic reruns on original {REVIEW_OUTCOME} raw seeds, not original action traces.",
        "gpu": GPU, "port": PORT, "server_command": command,
    })
    rows_path = task_dir / "replay_records.jsonl"
    prior = [json.loads(line) for line in rows_path.read_text().splitlines() if line] if rows_path.exists() else []
    seen = {int(row["seed"]) for row in prior}
    confirmed = sum(row["rerun_success"] is want_success for row in prior)
    with (task_dir / "server.log").open("a") as server_log:
        server = subprocess.Popen(command, cwd=ROBOTWIN, env=server_env(task_dir, task),
                                  stdout=server_log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            wait_ready(server)
            for seed in candidate_seeds:
                if confirmed >= 3:
                    break
                if seed in seen:
                    continue
                seed_dir = task_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                if task == "stack_blocks_three":
                    evaluator = OUTPUT / "control/policy_eval_three_tasks_case_video.py"
                    call = [str(SIM_PY), "-u", str(evaluator), "--host", "127.0.0.1",
                            "--port", str(PORT), "--task-name", task,
                            "--task-config", "demo_clean", "--seed-start", str(seed),
                            "--episodes", "1", "--frequency", "30", "--render-freq", "0",
                            "--save-freq", "5", "--step-limit", str(job["step_limit"]),
                            "--save-videos", "all" if want_success else "failure",
                            "--output-dir", str(seed_dir)]
                else:
                    evaluator = OUTPUT / "control/record_public_pi05_case_video.py"
                    call = [str(SIM_PY), "-u", str(evaluator), "--task-name", task,
                            "--prompt", job["prompt"], "--seed", str(seed),
                            "--host", "127.0.0.1", "--port", str(PORT),
                            "--frequency", "50", "--execute-horizon", "50",
                            "--step-limit", str(job["step_limit"]),
                            "--record-all-video" if want_success else "--record-failure-video",
                            "--output-dir", str(seed_dir)]
                with (seed_dir / "run.log").open("w") as log:
                    result = subprocess.run(call, cwd=ROBOTWIN, env=sim_env(task_dir),
                                            stdout=log, stderr=subprocess.STDOUT,
                                            timeout=1800, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"{task} seed {seed}: evaluator rc={result.returncode}; see {seed_dir}/run.log")
                records = [json.loads(line) for line in (seed_dir / "eval_records.jsonl").read_text().splitlines() if line]
                if len(records) != 1 or int(records[0]["seed"]) != seed:
                    raise RuntimeError(f"{task} seed {seed}: invalid records")
                record = records[0]
                if record["success"] is want_success:
                    video = seed_dir / "video/episode_0000000.mp4"
                    if not video.is_file() or video.stat().st_size < 1000:
                        raise RuntimeError(f"{task} seed {seed}: target rollout has no valid video")
                row = {
                    "task": task, "seed": seed, "original_success": want_success,
                    "rerun_success": bool(record["success"]),
                    "steps": int(record["policy_steps"]),
                    "video": str((seed_dir / "video/episode_0000000.mp4").relative_to(task_dir))
                    if record["success"] is want_success else None,
                    "protocol": "raw_seed_pi05_h10_own" if task == "stack_blocks_three" else "raw_seed_public_pi05_h50",
                }
                with rows_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                seen.add(seed)
                confirmed += record["success"] is want_success
                print(f"{task} seed={seed} rerun_success={record['success']} "
                      f"confirmed_{REVIEW_OUTCOME}={confirmed}/3", flush=True)
            write_json(task_dir / "summary.json", {
                "task": task, f"confirmed_{REVIEW_OUTCOME}_videos": confirmed,
                "target": 3, "attempted": len(seen), "complete": confirmed >= 3,
            })
            if confirmed < 3:
                raise RuntimeError(f"{task}: only {confirmed}/3 {REVIEW_OUTCOME} on rerun")
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def main() -> None:
    if REVIEW_OUTCOME not in {"failure", "success"}:
        raise ValueError(f"unknown REVIEW_OUTCOME={REVIEW_OUTCOME}")
    if not SIM_PY.is_file() or not SERVER_PY.is_file():
        raise RuntimeError("missing validated A800 runtime")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    only_task = os.environ.get("REVIEW_ONLY_TASK")
    if only_task and only_task not in {job["task"] for job in JOBS}:
        raise ValueError(f"unknown REVIEW_ONLY_TASK={only_task}")
    selected = [job for job in JOBS if not only_task or job["task"] == only_task]
    for job in selected:
        print(f"starting {job['task']}", flush=True)
        write_json(OUTPUT / "state.json", {"task": job["task"], "status": "running"})
        run_one(job)
        write_json(OUTPUT / "state.json", {"task": job["task"], "status": "complete"})
    write_json(OUTPUT / "state.json", {"status": "complete", "tasks": [job["task"] for job in selected]})


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_json(OUTPUT / "state.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
