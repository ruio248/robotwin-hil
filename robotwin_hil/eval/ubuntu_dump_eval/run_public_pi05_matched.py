#!/usr/bin/env python3
"""Compare the released dump_bin_bigbin pi0.5 checkpoint on matched seeds."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


BASE = Path("/data-training/robotwin-hil-lyt")
ROOT = BASE / "outputs/public_pi05_dump_bin_bigbin_matched_eval_20261024"
ROBOTWIN = BASE / "RoboTwin"
CKPT = ROOT / "checkpoints/dump_bin_bigbin"
PACE = ROOT / "openpi_pace"
SERVER_PY = BASE / "piperx/openpi/.venv/bin/python"
SIM_PY = Path("/data-training/qplanning_repro/.venv-robotwin-copy/bin/python")
SIM_DEPS = BASE / "outputs/state_true_three_checkpoint_eval_31000_31099_20261015/sim_py_deps"
SEEDS = range(31000, 31050)
GPU = 4
PORT = 18456
DOWNLOADER_PID = 1647967
TASK_SHA256 = "9afb1b1cd02f5ddcc69e6502254a4675ed9a750262bcce59c695ce8181971630"


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def status(phase, detail=""):
    save_json(ROOT / "control/state.json", {
        "phase": phase, "detail": detail, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")
    })


def port_open():
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", PORT)) == 0


def wait_for_download():
    log = ROOT / "control/download.log"
    deadline = time.monotonic() + 7200
    while True:
        if log.is_file() and "all selected task files verified" in log.read_text():
            return
        try:
            command = (Path("/proc") / str(DOWNLOADER_PID) / "cmdline").read_bytes()
        except FileNotFoundError:
            command = b""
        if b"download_from_manifest.py" not in command or b"dump_bin_bigbin" not in command:
            raise RuntimeError("checkpoint downloader exited before manifest verification completed")
        if time.monotonic() > deadline:
            raise TimeoutError("checkpoint download still incomplete after two hours")
        status("waiting_checkpoint_download")
        time.sleep(15)


def preflight():
    if hashlib.sha256((ROBOTWIN / "envs/dump_bin_bigbin.py").read_bytes()).hexdigest() != TASK_SHA256:
        raise RuntimeError("settling-fix simulator code differs from the self-trained evaluation")
    if not (CKPT / "assets/robotwin/norm_stats.json").is_file():
        raise RuntimeError("missing publisher normalization statistics")
    manifest = json.loads((ROOT / "control/manifest.json").read_text())
    model = next(item for item in manifest["models"] if item["task"] == "dump_bin_bigbin")
    for item in model["files"]:
        path = ROOT / "checkpoints" / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"publisher checkpoint file incomplete: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise RuntimeError(f"publisher checkpoint hash mismatch: {path}")
    if port_open():
        raise RuntimeError(f"port {PORT} is already occupied")
    free_mib = int(subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(GPU)
    ], text=True).strip())
    if free_mib < 40000:
        raise RuntimeError(f"GPU {GPU} only has {free_mib} MiB available")
    return model


def run_seed(horizon, seed, stage, sim_env):
    output = ROOT / "results" / f"h{horizon}" / stage / f"seed_{seed}"
    records = output / "eval_records.jsonl"
    if records.is_file():
        lines = records.read_text().splitlines()
        if len(lines) != 1:
            raise RuntimeError(f"partial/duplicate seed record: {records}")
        record = json.loads(lines[0])
        if record["seed"] != seed or record["execute_horizon"] != horizon:
            raise RuntimeError(f"seed or horizon mismatch: {records}")
        return record
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(SIM_PY), "-u", str(ROOT / "control/policy_eval_public_matched.py"),
        "--host", "127.0.0.1", "--port", str(PORT),
        "--policy-name", "Pi_05_RobotTwin", "--ckpt-name", "public_pi05_dump_bin_bigbin",
        "--task-name", "dump_bin_bigbin", "--task-config", "demo_clean",
        "--seed-start", str(seed), "--episodes", "1", "--frequency", "30",
        "--render-freq", "0", "--save-freq", "15", "--step-limit", "600",
        "--save-videos", "none", "--execute-horizon", str(horizon),
        "--output-dir", str(output),
    ]
    with (output / "eval.log").open("a") as log:
        result = subprocess.run(command, cwd=ROBOTWIN, env=sim_env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=7200, check=False)
    if result.returncode:
        raise RuntimeError(f"simulator infrastructure error h={horizon} seed={seed} rc={result.returncode}; {output / 'eval.log'}")
    if not records.is_file():
        raise RuntimeError(f"missing seed record: {records}")
    lines = records.read_text().splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"unexpected seed record count: {records}")
    return json.loads(lines[0])


def main():
    ROOT.joinpath("control").mkdir(parents=True, exist_ok=True)
    with (ROOT / "control/runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        server = None
        try:
            wait_for_download()
            model = preflight()
            save_json(ROOT / "control/protocol.json", {
                "checkpoint": str(CKPT), "publisher": "niejunnan25/pi05-robotwin",
                "manifest_bytes": model["bytes"], "simulator_sha256": TASK_SHA256,
                "seeds": list(SEEDS), "horizons": [50, 10],
                "scene": "demo_clean", "expert_prefilter": False,
                "prompt": "Pick up the small bin and pour all the balls into the big bin.",
                "frequency_field": 30, "step_limit": 600,
                "action": "absolute joint; full 50-step prediction, execute first H before replanning",
                "gpu": GPU, "port": PORT,
            })
            cfg = ROOT / "control/public_server.yml"
            cfg.write_text(
                "policy_name: Pi_05_RobotTwin\n"
                "task_name: dump_bin_bigbin\n"
                "action_type: joint\nenv_cfg_type: aloha_agilex\n"
                "train_config_name: pi05_robotwin_dump_bin_bigbin\n"
                "repo_id: robotwin\n"
                f"model_path: {CKPT}\nhost: 127.0.0.1\nport: {PORT}\nprotocol: ws\n"
            )
            tmp = ROOT / "tmp"
            for name in ("xdg", "warp"):
                (tmp / name).mkdir(parents=True, exist_ok=True)
            base_env = os.environ.copy()
            base_env.update({
                "TMPDIR": str(tmp), "XDG_CACHE_HOME": str(tmp / "xdg"),
                "CUDA_VISIBLE_DEVICES": str(GPU), "PYTHONDONTWRITEBYTECODE": "1",
            })
            server_env = base_env | {
                "HF_HOME": str(ROOT / "cache/hf"), "OPENPI_DATA_HOME": str(ROOT / "cache/openpi"),
                "JAX_COMPILATION_CACHE_DIR": str(ROOT / "cache/jax"),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.38", "WANDB_MODE": "disabled",
                "PYTHONPATH": f"{ROBOTWIN}:{PACE / 'src'}:{PACE / 'pace/src'}",
            }
            sim_env = base_env | {
                "WARP_CACHE_PATH": str(tmp / "warp"), "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "PYTHONPATH": f"{SIM_DEPS}:{ROBOTWIN}",
            }
            status("starting_server")
            with (ROOT / "control/server.log").open("a") as server_log:
                command = [str(SERVER_PY), "-u", str(ROBOTWIN / "XPolicyLab/setup_policy_server.py"),
                           "--config_path", str(cfg), "--host", "127.0.0.1", "--port", str(PORT)]
                server = subprocess.Popen(command, cwd=ROBOTWIN, env=server_env,
                                          stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
                save_json(ROOT / "control/server_process.json", {"pid": server.pid, "command": command})
                deadline = time.monotonic() + 1200
                while not port_open():
                    if server.poll() is not None:
                        raise RuntimeError(f"policy server exited {server.returncode}; see control/server.log")
                    if time.monotonic() > deadline:
                        raise TimeoutError("policy server start timed out; see control/server.log")
                    time.sleep(2)
                for horizon in (50, 10):
                    status("smoke", f"H={horizon}, seed=31000")
                    run_seed(horizon, 31000, "smoke", sim_env)
                    records = []
                    for seed in SEEDS:
                        status("formal", f"H={horizon}, seed={seed}")
                        records.append(run_seed(horizon, seed, "formal", sim_env))
                    if [r["seed"] for r in records] != list(SEEDS):
                        raise RuntimeError("seed mismatch")
                    successes = sum(bool(r["success"]) for r in records)
                    save_json(ROOT / "results" / f"h{horizon}" / "aggregate_summary.json", {
                        "complete": True, "horizon": horizon, "seeds": list(SEEDS),
                        "episodes": len(SEEDS), "successes": successes,
                        "success_rate": successes / len(SEEDS), "records": records,
                    })
                status("complete")
        except Exception as exc:
            status("failed", str(exc))
            raise
        finally:
            if server is not None and server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=30)


if __name__ == "__main__":
    main()
