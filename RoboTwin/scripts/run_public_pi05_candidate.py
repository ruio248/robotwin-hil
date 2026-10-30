#!/usr/bin/env python3
"""Verify and evaluate one released task-specific pi0.5 checkpoint.

The smoke seed is separate from the formal 50-seed cohort.  Each formal seed
has one durable JSON record.  Simulator exceptions stop the run rather than
being counted as policy failures.
"""

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
import time


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--pace-root", type=Path, required=True)
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--sim-deps", type=Path, required=True)
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--min-free-gpu-mib", type=int, default=40000,
                        help="Preflight GPU free-memory floor; 40000 keeps the A800 default")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=31000)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--replace-unstable-setup-seeds", action="store_true",
                        help="Only replace seeds that fail before policy with setup_demo UnStableError")
    parser.add_argument("--max-unstable-replacements", type=int, default=10,
                        help="Maximum extra consecutive seeds inspected when replacement is enabled")
    parser.add_argument("--frequency", type=int, default=50)
    parser.add_argument("--step-limit", type=int, required=True)
    parser.add_argument("--horizon", type=int, choices=(10, 50), default=50)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--trace-handover-state", action="store_true",
                        help="Record privileged handover diagnostics in separate runs")
    parser.add_argument("--trace-task-state", action="store_true",
                        help="Diagnostic-only basket/can-pot state at every chunk boundary")
    parser.add_argument("--recover-at-step", type=int, default=None,
                        help="Diagnostic-only expert rescue at this chunk boundary")
    parser.add_argument("--handover-handback-stage-actions", type=int, default=None,
                        help="Smoke-only staged expert actions before handing control back to policy")
    parser.add_argument("--task-state-handback-stage-actions", type=int, default=None,
                        help="Smoke-only move_can_pot expert actions before handing control back")
    parser.add_argument("--trace-burger-final", action="store_true",
                        help="Smoke-only physical placement diagnosis for place_burger_fries")
    parser.add_argument("--burger-settle-steps", type=int, default=250)
    return parser.parse_args()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    partial.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class UnstableSetupSeed(RuntimeError):
    """The simulator rejected a scene before any policy action was executed."""


def is_unstable_setup_log(log_text: str) -> bool:
    """Recognize only the task initialization stability gate, not policy failure."""
    return (
        "task_env.setup_demo(" in log_text
        and "envs.utils.create_actor.UnStableError:" in log_text
        and "Objects is unstable in seed(" in log_text
        and "task_env.take_action(" not in log_text
    )


def preflight(cli: argparse.Namespace) -> dict:
    manifest = json.loads(cli.manifest.read_text())
    models = {item["task"]: item for item in manifest["models"]}
    if cli.task not in models:
        raise RuntimeError(f"task absent from publisher manifest: {cli.task}")
    model = models[cli.task]
    for item in model["files"]:
        path = cli.model_root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"missing or incomplete checkpoint file: {path}")
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"checkpoint SHA256 mismatch: {path}")
    norm_stats = cli.model_root / model["norm_stats"]
    if not norm_stats.is_file() or sha256(norm_stats) != model["norm_stats_sha256"]:
        raise RuntimeError("missing or incorrect task-specific norm_stats")
    if port_open(cli.port):
        raise RuntimeError(f"port already occupied: {cli.port}")
    available = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(cli.gpu)],
        text=True,
    ).strip())
    if cli.min_free_gpu_mib <= 0:
        raise ValueError("--min-free-gpu-mib must be positive")
    if available < cli.min_free_gpu_mib:
        raise RuntimeError(
            f"GPU {cli.gpu} has only {available} MiB free; "
            f"requires {cli.min_free_gpu_mib} MiB"
        )
    return model


def run_seed(cli: argparse.Namespace, stage: str, seed: int, sim_env: dict) -> dict:
    folder = cli.output_root / stage / f"seed_{seed}"
    record_path = folder / "eval_records.jsonl"
    if record_path.exists():
        lines = record_path.read_text().splitlines()
        if len(lines) != 1:
            raise RuntimeError(f"partial or duplicate seed record: {record_path}")
        record = json.loads(lines[0])
        if record["seed"] != seed or record["task"] != cli.task:
            raise RuntimeError(f"mismatched existing seed record: {record_path}")
        return record
    folder.mkdir(parents=True, exist_ok=True)
    command = [
        str(cli.sim_python), "-u", str(cli.eval_script),
        "--task-name", cli.task, "--prompt", cli.prompt,
        "--seed", str(seed), "--port", str(cli.port),
        "--frequency", str(cli.frequency), "--execute-horizon", str(cli.horizon),
        "--step-limit", str(cli.step_limit), "--output-dir", str(folder),
    ]
    if cli.trace_handover_state:
        command.append("--trace-handover-state")
    if cli.trace_task_state:
        command.append("--trace-task-state")
    if cli.recover_at_step is not None:
        command.extend(("--recover-at-step", str(cli.recover_at_step)))
    if cli.handover_handback_stage_actions is not None:
        command.extend(("--handover-handback-stage-actions",
                        str(cli.handover_handback_stage_actions)))
    if cli.task_state_handback_stage_actions is not None:
        command.extend(("--task-state-handback-stage-actions",
                        str(cli.task_state_handback_stage_actions)))
    if cli.trace_burger_final:
        command.extend(("--trace-burger-final", "--burger-settle-steps",
                        str(cli.burger_settle_steps)))
    log_path = folder / "eval.log"
    previous_log_bytes = log_path.stat().st_size if log_path.exists() else 0
    with log_path.open("a") as log:
        result = subprocess.run(
            command, cwd=cli.robotwin_root, env=sim_env,
            stdout=log, stderr=subprocess.STDOUT, timeout=7200, check=False,
        )
    if result.returncode != 0:
        with log_path.open("rb") as log:
            log.seek(previous_log_bytes)
            new_log_text = log.read().decode(errors="replace")
        if is_unstable_setup_log(new_log_text):
            raise UnstableSetupSeed(
                f"pre-policy unstable scene task={cli.task} seed={seed}; {log_path}"
            )
        raise RuntimeError(f"infrastructure error task={cli.task} seed={seed} rc={result.returncode}; {log_path}")
    if not record_path.is_file() or len(record_path.read_text().splitlines()) != 1:
        raise RuntimeError(f"missing or duplicate seed record: {record_path}")
    record = json.loads(record_path.read_text())
    if record["seed"] != seed or record["task"] != cli.task:
        raise RuntimeError(f"seed record identity mismatch: {record_path}")
    return record


def main() -> None:
    cli = args_parser()
    if cli.seed_count < 1 or cli.max_unstable_replacements < 0:
        raise ValueError("seed-count must be positive and max-unstable-replacements nonnegative")
    if cli.trace_task_state and (cli.task not in ("place_can_basket", "move_can_pot")
                                 or not cli.smoke_only):
        raise ValueError("basket/can-pot state trace is diagnostic-only with --smoke-only")
    if cli.trace_task_state and cli.trace_handover_state:
        raise ValueError("select only one state trace")
    if cli.recover_at_step is not None and (
        not cli.smoke_only or not (cli.trace_handover_state or cli.trace_task_state)
    ):
        raise ValueError("expert rescue requires a task state trace and --smoke-only")
    if cli.handover_handback_stage_actions is not None and (
        not cli.trace_handover_state or cli.recover_at_step is None
        or cli.handover_handback_stage_actions <= 0
    ):
        raise ValueError("handback requires smoke-only recovery and a positive stage-action count")
    if cli.task_state_handback_stage_actions is not None and (
        not cli.trace_task_state or cli.task != "move_can_pot"
        or cli.recover_at_step is None or not cli.smoke_only
        or cli.task_state_handback_stage_actions <= 0
    ):
        raise ValueError("task-state handback is smoke-only for recovered move_can_pot states")
    if cli.trace_burger_final and (cli.task != "place_burger_fries" or not cli.smoke_only):
        raise ValueError("burger physical diagnosis is smoke-only for place_burger_fries")
    if cli.burger_settle_steps < 0:
        raise ValueError("--burger-settle-steps must be nonnegative")
    cli.output_root.mkdir(parents=True, exist_ok=True)
    control = cli.output_root / "control"
    control.mkdir(parents=True, exist_ok=True)

    def status(phase: str, detail: str = "") -> None:
        save_json(control / "state.json", {
            "phase": phase, "detail": detail, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        })

    with (control / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        server = None
        try:
            status("verifying_checkpoint")
            model = preflight(cli)
            checkpoint = cli.model_root / model["path"]
            save_json(control / "protocol.json", {
                "task": cli.task, "checkpoint": str(checkpoint),
                "config": model["config"], "norm_stats": str(cli.model_root / model["norm_stats"]),
                "prompt": cli.prompt, "seeds": list(range(cli.seed_start, cli.seed_start + cli.seed_count)),
                "seeds_field_meaning": "initial planned cohort; actual completed seeds are in aggregate_summary.json",
                "seed_candidates": list(range(
                    cli.seed_start, cli.seed_start + cli.seed_count +
                    (cli.max_unstable_replacements if cli.replace_unstable_setup_seeds else 0)
                )),
                "seed_rule": ("first N valid scene initializations from consecutive seeds"
                              if cli.replace_unstable_setup_seeds else "fixed consecutive seeds"),
                "replace_unstable_setup_seeds": cli.replace_unstable_setup_seeds,
                "max_unstable_replacements": cli.max_unstable_replacements,
                "smoke_seed": cli.seed_start, "smoke_excluded_from_formal": True,
                "scene": "demo_clean", "expert_prefilter": False,
                "action": "absolute joint; predict 50, execute H",
                "generated_horizon": 50, "execute_horizon": cli.horizon,
                "frequency_field": cli.frequency, "step_limit": cli.step_limit,
                "gpu": cli.gpu, "port": cli.port,
                "min_free_gpu_mib": cli.min_free_gpu_mib,
                "simulator_sha256": sha256(cli.robotwin_root / "envs" / f"{cli.task}.py"),
                "trace_handover_state": cli.trace_handover_state,
                "trace_task_state": cli.trace_task_state,
                "recover_at_step": cli.recover_at_step,
                "handover_handback_stage_actions": cli.handover_handback_stage_actions,
                "trace_burger_final": cli.trace_burger_final,
                "burger_settle_steps": cli.burger_settle_steps if cli.trace_burger_final else None,
                "manifest": str(cli.manifest),
            })
            cfg = control / "server.yml"
            cfg.write_text(
                "policy_name: Pi_05_RobotTwin\n"
                f"task_name: {cli.task}\n"
                "action_type: joint\nenv_cfg_type: aloha_agilex\n"
                f"train_config_name: {model['config']}\n"
                "repo_id: robotwin\n"
                f"model_path: {checkpoint}\n"
                f"host: 127.0.0.1\nport: {cli.port}\nprotocol: ws\n"
            )
            tmp = cli.output_root / "tmp"
            (tmp / "xdg").mkdir(parents=True, exist_ok=True)
            (tmp / "warp").mkdir(parents=True, exist_ok=True)
            base_env = os.environ.copy()
            base_env.update({
                "TMPDIR": str(tmp), "XDG_CACHE_HOME": str(tmp / "xdg"),
                "CUDA_VISIBLE_DEVICES": str(cli.gpu), "PYTHONDONTWRITEBYTECODE": "1",
                "ROBOTWIN_ROOT": str(cli.robotwin_root),
            })
            server_env = base_env | {
                "HF_HOME": str(cli.output_root / "cache/hf"),
                "OPENPI_DATA_HOME": str(cli.output_root / "cache/openpi"),
                "JAX_COMPILATION_CACHE_DIR": str(cli.output_root / "cache/jax"),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.38",
                "WANDB_MODE": "disabled",
                "PYTHONPATH": f"{cli.robotwin_root}:{cli.pace_root / 'src'}:{cli.pace_root / 'pace/src'}",
            }
            sim_env = base_env | {
                "WARP_CACHE_PATH": str(tmp / "warp"),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "PYTHONPATH": f"{cli.sim_deps}:{cli.robotwin_root}",
            }
            status("starting_server")
            command = [
                str(cli.server_python), "-u", str(cli.robotwin_root / "XPolicyLab/setup_policy_server.py"),
                "--config_path", str(cfg), "--host", "127.0.0.1", "--port", str(cli.port),
            ]
            with (control / "server.log").open("a") as log:
                server = subprocess.Popen(
                    command, cwd=cli.robotwin_root, env=server_env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            save_json(control / "server_process.json", {"pid": server.pid, "command": command})
            deadline = time.monotonic() + 1200
            while not port_open(cli.port):
                if server.poll() is not None:
                    raise RuntimeError(f"policy server exited {server.returncode}; inspect {control / 'server.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"policy server startup timed out; inspect {control / 'server.log'}")
                time.sleep(2)
            status("smoke", f"seed={cli.seed_start}")
            run_seed(cli, "smoke", cli.seed_start, sim_env)
            if cli.smoke_only:
                status("smoke_complete")
                return
            records = []
            excluded_infra = []
            upper_exclusive = cli.seed_start + cli.seed_count + (
                cli.max_unstable_replacements if cli.replace_unstable_setup_seeds else 0
            )
            for seed in range(cli.seed_start, upper_exclusive):
                if len(records) >= cli.seed_count:
                    break
                status("formal", f"seed={seed}")
                try:
                    records.append(run_seed(cli, "formal", seed, sim_env))
                except UnstableSetupSeed as exc:
                    if not cli.replace_unstable_setup_seeds:
                        raise
                    excluded_infra.append({"seed": seed, "reason": str(exc)})
                    save_json(control / "excluded_infra_seeds.json", excluded_infra)
                    continue
            seeds = [record["seed"] for record in records]
            if len(seeds) != cli.seed_count or len(set(seeds)) != cli.seed_count:
                raise RuntimeError("formal seed count or uniqueness mismatch")
            successes = sum(bool(record["success"]) for record in records)
            save_json(cli.output_root / "aggregate_summary.json", {
                "complete": True, "task": cli.task, "seeds": seeds,
                "episodes": len(records), "successes": successes,
                "success_rate": successes / len(records), "records": records,
                "excluded_infra_seeds": excluded_infra,
                "seed_rule": ("first N valid scene initializations from consecutive seeds"
                              if cli.replace_unstable_setup_seeds else "fixed consecutive seeds"),
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
