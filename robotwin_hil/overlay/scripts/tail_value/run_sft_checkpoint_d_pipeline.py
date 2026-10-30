#!/usr/bin/env python3
"""Collect and train a same-observation H50 expert-support D for an SFT model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def gpu_used(gpu: int) -> int:
    result = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    return int(float(result.strip()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--reference-code", type=Path, required=True)
    parser.add_argument("--tail-value-code", type=Path, required=True)
    parser.add_argument("--r3m-asset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=7)
    args = parser.parse_args()
    experiment = args.experiment_root.resolve()
    ref = args.reference_code.resolve()
    tail_value_code = args.tail_value_code.resolve()
    output = args.output_root.resolve()
    collector = args.collector.resolve()
    sft_state_path = experiment / "control/sft_state.json"
    checkpoint = experiment / "checkpoints/4999"
    r3m_asset = args.r3m_asset.resolve()
    encoder = ref / "encode_pairs_h50.py"
    trainer = ref / "train_native_h50.py"
    for path in (sft_state_path, checkpoint / "params", checkpoint / "assets",
                 r3m_asset, encoder, trainer, collector):
        if not path.exists():
            raise FileNotFoundError(path)
    sft = json.loads(sft_state_path.read_text())
    if sft.get("status") != "sft_complete" or sft.get("formal_updates") != 5000:
        raise RuntimeError("source SFT is not complete at 5000 updates")
    if output.exists():
        forbidden = [output / name for name in ("state.json", "data", "models", "logs")]
        if any(path.exists() for path in forbidden):
            raise FileExistsError(f"refusing existing run artifacts under: {output}")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    state_path = output / "state.json"
    state = {
        "status": "starting", "task": "place_empty_cup",
        "stage": "preflight", "gpu": args.gpu,
        "sft_checkpoint": str(checkpoint), "sft_updates": 5000,
        "expert_episodes": 50, "same_observation_candidate_count": 4,
        "action_horizon": 50, "split": "frozen source cache split; 25/25",
        "D_training": {"architecture": "ChunkCoverageCritic width256, 3 Linear-LayerNorm-SiLU blocks",
                       "updates": [1000, 10000], "train_episodes": 40,
                       "heldout_episodes": 10, "seed": 42, "batch": 256,
                       "learning_rate": 0.0001},
        "completed_stages": [], "child_pid": None,
    }
    write_json(state_path, state)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.55"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(output / "tmp")
    env["XDG_CACHE_HOME"] = str(output / "cache")
    env["MPLCONFIGDIR"] = str(output / "cache/matplotlib")
    env["CUDA_CACHE_PATH"] = str(output / "cache/cuda")
    env["PYTHONPATH"] = os.pathsep.join([
        str(experiment / "code"),
        "/data-training/robotwin-hil-lyt/piperx/openpi/src",
        str(ref.parent / "openpi/packages/openpi-client/src"),
        str(tail_value_code), str(ref),
        env.get("PYTHONPATH", ""),
    ])
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    py = Path("/data-training/robotwin-hil-lyt/piperx/openpi/.venv/bin/python")

    def stage(name: str, command: list[str]) -> None:
        used = gpu_used(args.gpu)
        if used > 4096:
            raise RuntimeError(f"GPU{args.gpu} has {used} MiB in use; refusing to share")
        if shutil.disk_usage(output).free < 8 * 1024**3:
            raise RuntimeError("less than 8 GiB free for D artifacts")
        state.update(status="running", stage=name, command=command,
                     child_pid=None, updated=time.time())
        write_json(state_path, state)
        with (logs / f"{name}.log").open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, cwd=ref.parent)
            state["child_pid"] = process.pid
            write_json(state_path, state)
            rc = process.wait()
        if rc:
            raise RuntimeError(f"stage {name} exited {rc}; see {logs / f'{name}.log'}")
        state["completed_stages"].append(name)
        state.update(child_pid=None, updated=time.time())
        write_json(state_path, state)

    try:
        stage("smoke_policy_io", [
            str(py), "-u", str(collector), "--experiment-root", str(experiment),
            "--checkpoint", str(checkpoint), "--output-root", str(output / "smoke_pairs"),
            "--gpu", str(args.gpu), "--smoke",
        ])
        smoke_state = json.loads((output / "smoke_pairs/state.json").read_text())
        smoke_summary = json.loads((output / "smoke_pairs/smoke_summary.json").read_text())
        if (smoke_state.get("status") != "smoke_complete"
                or not smoke_summary.get("complete")
                or smoke_state.get("accepted_for_training") is not False
                or smoke_state.get("completed_episodes") != 1
                or smoke_state.get("total_pairs") != 1):
            raise RuntimeError("SFT D policy smoke did not produce one diagnostic-only H50 pair")
        stage("collect_same_observation_pairs", [
            str(py), "-u", str(collector), "--experiment-root", str(experiment),
            "--checkpoint", str(checkpoint), "--output-root", str(output / "data"),
            "--gpu", str(args.gpu),
        ])
        for split in (0, 1):
            stage(f"encode_split_{split}", [
                str(py), "-u", str(encoder), "--task-root",
                str(output / "data" / f"seed{split}_n25"), "--asset", str(r3m_asset),
                "--device", "cuda:0",
            ])
        stage("train_D_10000", [
            str(py), "-u", str(trainer), "--task", "place_empty_cup",
            "--data", str(output / "data"), "--output", str(output / "models"),
            "--asset", str(r3m_asset),
        ])
        protocol_path = output / "models/protocol.json"
        protocol = json.loads(protocol_path.read_text())
        if protocol.get("status") != "complete":
            raise RuntimeError("D training protocol is not complete")
        state.update(status="complete", stage="complete", completed=time.time(),
                     D_protocol=str(protocol_path),
                     D_checkpoints=[str(output / "models/disc_s42_step_1000.pt"),
                                    str(output / "models/disc_s42_step_10000.pt")])
        write_json(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}",
                     traceback=traceback.format_exc(), updated=time.time())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
