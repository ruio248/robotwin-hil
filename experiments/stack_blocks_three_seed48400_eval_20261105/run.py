"""Guarded, sequential SFT evaluation on the DAgger seed range.

Reuses the already-running, checkpoint-verified stack policy service. It does
not start, stop or restart the service. Smoke is excluded from formal results.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/media/ruio/hdd/official_nine_task_pi05_sft_5000_20261101/eval_stack_blocks_three_seed48400_20261105")
SOURCE = Path("/media/ruio/hdd/handover_block_official_min_trace_eval_20261102/official_runtime")
ENTRY = Path("/hdd/robotwin-hil-github/experiments/stack_blocks_three_seed48400_eval_20261105/seed_entry.py")
CHECKPOINT = Path("/media/ruio/hdd/official_nine_task_pi05_sft_5000_20261101/checkpoints/stack_blocks_three/4999")
SERVER_CONFIG = Path("/hdd/official_nine_task_pi05_sft_5000_20261101/eval_stack_blocks_three_20261101/server.yaml")
PYTHON = Path("/hdd/miniconda3/envs/robotwin_hil/bin/python")
PORT = 18510
SERVER_PID = 196268
TASK = "stack_blocks_three"
SEED_START = 48400
CHECKPOINT_SHA = "d44166ee36d40ce0bb36adb3ad57ab9dea007671fee2fce18ff15e917a5d3df8"
NORM_SHA = "3da00d7d1f3a10482373ddf821508a05173d7b5be26d45c0de87f496bdd1546f"
EVALUATOR_SHA = "cbae8d114278c86171e46ed3bf36563614b8ce8398c3d250a01b3d4b93f12800"
RUN_ID = "stack_blocks_three_seed48400_20261105"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def state(phase: str, **details) -> None:
    write_json(ROOT / "control/state.json", {
        "phase": phase, "time_utc": datetime.now(timezone.utc).isoformat(), **details,
    })


def preflight() -> None:
    mount = subprocess.check_output(["findmnt", "-no", "SOURCE", "/media/ruio/hdd"], text=True).strip()
    if mount != "/dev/sda1":
        raise RuntimeError(f"wrong data mount: {mount}")
    if not ENTRY.is_file() or not PYTHON.is_file():
        raise RuntimeError("launcher or simulation Python is missing")
    if digest(SOURCE / "scripts/eval_policy_xpolicylab_official_trace.py") != EVALUATOR_SHA:
        raise RuntimeError("pinned SFT evaluator changed")
    if not (SOURCE / "assets/objects/objaverse/list.json").is_file():
        raise RuntimeError("simulation object assets missing")
    if digest(CHECKPOINT / "params/manifest.ocdbt") != CHECKPOINT_SHA:
        raise RuntimeError("checkpoint params mismatch")
    norm = CHECKPOINT / "assets/stack_blocks_three_official50_v30_20261101/norm_stats.json"
    if digest(norm) != NORM_SHA:
        raise RuntimeError("checkpoint norm stats mismatch")
    config = SERVER_CONFIG.read_text()
    for fragment in (f"task_name: {TASK}", str(CHECKPOINT), "state", "h50"):
        if fragment.lower() not in config.lower():
            raise RuntimeError(f"server config missing {fragment}")
    cmdline = Path(f"/proc/{SERVER_PID}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "stack_server_entry.py" not in cmdline or str(SERVER_CONFIG) not in cmdline or f"--port {PORT}" not in cmdline:
        raise RuntimeError(f"policy service identity changed: {SERVER_PID} {cmdline}")
    with socket.create_connection(("127.0.0.1", PORT), timeout=2):
        pass
    process_lines = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
    if any("python" in line and "-m dagger.runner" in line for line in process_lines):
        raise RuntimeError("a DAgger runner is active; do not share its policy service during evaluation")
    if any("python" in line and "eval_policy_xpolicylab_official_trace.py" in line
           for line in process_lines):
        raise RuntimeError("another SFT evaluator is active")
    if (ROOT / "smoke").exists() or (ROOT / "formal").exists():
        raise RuntimeError("refusing to overwrite existing smoke/formal results")
    result_root = SOURCE / "eval_result" / TASK / "Pi_05" / "demo_clean"
    for name in ("smoke", "formal"):
        if (result_root / f"{RUN_ID}_{name}").exists():
            raise RuntimeError(f"official result label already exists: {name}")
    available = int(subprocess.check_output(["df", "-B1", "--output=avail", str(ROOT.parent)], text=True).splitlines()[-1])
    if available < 100 * 1024**3:
        raise RuntimeError("less than 100 GiB free on data disk")
    write_json(ROOT / "control/protocol.json", {
        "task": TASK, "seed_start_override": SEED_START, "seed_start_official_default": 100000,
        "only_evaluator_change": "eval_remote_policy start seed argument",
        "evaluator": str(SOURCE / "scripts/eval_policy_xpolicylab_official_trace.py"),
        "evaluator_sha256": EVALUATOR_SHA, "seed_entry_sha256": digest(ENTRY),
        "checkpoint": str(CHECKPOINT), "params_sha256": CHECKPOINT_SHA, "norm_stats_sha256": NORM_SHA,
        "server_pid": SERVER_PID, "server_config": str(SERVER_CONFIG),
        "server_config_sha256": digest(SERVER_CONFIG), "port": PORT,
        "expert_check": True, "instruction_type": "seen", "instruction_pool_size": 50,
        "task_config": "demo_clean", "frequency_field": 50, "action_type": "joint",
        "smoke_count": 1, "formal_count": 50,
        "comparison_limit": "DAgger test seeds match; aborted/intervened DAgger episodes are not autonomous outcomes",
    })


def evaluate(name: str, count: int, env: dict[str, str]) -> dict:
    folder = ROOT / name
    folder.mkdir(exist_ok=False)
    label = f"{RUN_ID}_{name}"
    argv = [str(PYTHON), "-u", str(ENTRY), "--task_name", TASK,
            "--env_cfg_type", "aloha_agilex", "--policy_name", "Pi_05",
            "--host", "127.0.0.1", "--port", str(PORT), "--protocol", "ws",
            "--root_dir", str(SOURCE), "--device_id", "0", "--task_config", "demo_clean",
            "--seed", "0", "--test_num", str(count), "--eval_batch", "false",
            "--expert_check", "true", "--instruction_type", "seen", "--frequency", "50",
            "--additional_info", f"ckpt_setting={label},action_type=joint",
            "--trace_dir", str(folder / "trace")]
    write_json(folder / "command.json", {"argv": argv, "cwd": str(SOURCE)})
    with (folder / "eval.log").open("x") as log:
        process = subprocess.Popen(argv, cwd=SOURCE, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        write_json(ROOT / f"control/{name}_process.json", {"pid": process.pid, "argv": argv})
        code = process.wait()
    if code:
        raise RuntimeError(f"{name} evaluator exited {code}; see {folder / 'eval.log'}")
    manifest_path = folder / "trace/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    episodes = [row for row in manifest["episodes"] if row["counted_in_official_result"]]
    seeds = [row["scene_seed"] for row in episodes]
    if (manifest.get("status") != "complete" or manifest.get("d_analysis_ready") is not True
            or manifest.get("infra_error_episodes") != 0 or manifest.get("infra_error_skipped_scenes") != 0
            or len(episodes) != count or len(set(seeds)) != count
            or any(row["status"] not in ("success", "policy_failure") or not row["queries"]
                   or row["rollout_failed"] for row in episodes)):
        raise RuntimeError(f"{name} trace did not pass clean acceptance: {manifest_path}")
    matches = list((SOURCE / "eval_result" / TASK / "Pi_05" / "demo_clean" / label).glob("*/_result.txt"))
    if len(matches) != 1:
        raise RuntimeError(f"{name} official result not unique: {matches}")
    successes = sum(row["success"] is True for row in episodes)
    rate = float(matches[0].read_text().strip().splitlines()[-1])
    if abs(rate - successes / count) > 1e-9:
        raise RuntimeError("official result disagrees with trace successes")
    result = {"complete": True, "count": count, "successes": successes,
              "policy_failures": count - successes, "success_rate": successes / count,
              "scene_seed_min": min(seeds), "scene_seed_max": max(seeds),
              "expert_rejected": sum(row["stage"] == "expert_rejected" for row in manifest["skipped_scenes"]),
              "trace_manifest": str(manifest_path), "official_result": str(matches[0]),
              "scene_results": [{"scene_seed": row["scene_seed"], "success": row["success"],
                                 "instruction": row.get("instruction")}
                                for row in episodes]}
    write_json(folder / "summary.json", result)
    return result


def main() -> None:
    (ROOT / "control").mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(ROOT / "control/runner.pid", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644), "w") as stream:
        stream.write(f"{os.getpid()}\n")
    try:
        state("preflight")
        preflight()
        tmp = ROOT / "tmp"
        for directory in (tmp, tmp / "xdg", tmp / "warp"):
            directory.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({"PYTHONPATH": f"{SOURCE / 'scripts'}:{SOURCE}:{SOURCE / 'XPolicyLab'}",
                    "CUDA_VISIBLE_DEVICES": "0", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "ROBOTWIN_ROOT": str(SOURCE), "EVAL_SCENE_SEED_START": str(SEED_START),
                    "TMPDIR": str(tmp), "XDG_CACHE_HOME": str(tmp / "xdg"),
                    "WARP_CACHE_PATH": str(tmp / "warp"), "PYTHONDONTWRITEBYTECODE": "1"})
        state("smoke", server_pid=SERVER_PID)
        smoke = evaluate("smoke", 1, env)
        state("formal", server_pid=SERVER_PID, smoke_successes=smoke["successes"])
        formal = evaluate("formal", 50, env)
        write_json(ROOT / "summary.json", {"smoke": smoke, "formal": formal})
        state("complete", successes=formal["successes"], episodes=50)
    except Exception as exc:
        state("failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
