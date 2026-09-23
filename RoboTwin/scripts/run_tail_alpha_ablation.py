#!/usr/bin/env python3
"""Run one fresh policy-bootstrap critic per idle GPU, then evaluate validation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tail_value.cache import atomic_json, cache_fingerprint, load_manifest, sha256
from tail_value.model import POLICY_BOOTSTRAP

SCRIPTS = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--gpus", nargs="+", required=True, help="Physical indices from nvidia-smi, one per alpha")
    parser.add_argument("--alphas", nargs="+", type=float, default=[.1, .5, .01, 1.])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-plots", type=int, default=10)
    parser.add_argument("--code-revision", required=True, help="Commit of the deployed source archive")
    return parser.parse_args(argv)


def idle_gpu_mapping(indices):
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    inventory = {}
    for line in output.splitlines():
        index, uuid, used, util = (part.strip() for part in line.split(","))
        inventory[index] = {"index": index, "uuid": uuid, "memory_used_mib": int(used), "utilization": int(util)}
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"
    ], text=True)
    occupied = set(apps.splitlines())
    selected = []
    for index in indices:
        if index not in inventory:
            raise ValueError(f"GPU index not found: {index}")
        gpu = inventory[index]
        if gpu["uuid"] in occupied or gpu["memory_used_mib"] > 256 or gpu["utilization"] > 5:
            raise RuntimeError(f"Requested GPU is occupied; no existing process will be stopped: {gpu}")
        selected.append(gpu)
    return selected


def run(args):
    if len(args.gpus) != len(args.alphas) or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Require one distinct GPU per alpha")
    if len(set(args.alphas)) != len(args.alphas) or any(not math.isfinite(a) or a < 0 for a in args.alphas):
        raise ValueError("Alphas must be unique finite nonnegative values")
    if any(getattr(args, key) < 1 for key in ("steps", "eval_every", "log_every", "batch_size", "cpu_threads")) or args.max_plots < 0:
        raise ValueError("Invalid step, batch, thread or plot count")
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new ablation output directory: {args.output_dir}")
    # Fail before training if report dependencies are missing.
    import matplotlib  # noqa: F401
    import pyarrow  # noqa: F401
    manifest = load_manifest(args.cache_dir)
    splits = Counter(entry["split"] for entry in manifest["episodes"])
    if manifest["config"]["source_format"] != "lerobot" or not splits["train"] or not splits["val"]:
        raise ValueError("Require the completed LeRobot train/validation candidate cache")
    gpus = idle_gpu_mapping(args.gpus)
    args.output_dir.mkdir(parents=True)
    logs = args.output_dir / "logs"
    logs.mkdir()
    source_files = [SCRIPTS / name for name in ("train_tail.py", "eval_tail_value.py", "run_tail_alpha_ablation.py")]
    source_files += sorted((SCRIPTS / "tail_value").glob("*.py"))
    state = {"status": "running", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
             "bootstrap": POLICY_BOOTSTRAP, "code_revision": args.code_revision,
             "source_sha256": {str(p.relative_to(SCRIPTS)): sha256(p) for p in source_files},
             "cache_dir": str(args.cache_dir), "cache_fingerprint": cache_fingerprint(manifest),
             "split_episodes": dict(splits), "steps": args.steps, "seed": args.seed,
             "python": sys.executable, "jobs": []}
    status_path = args.output_dir / "status.json"
    active = {}

    def save():
        atomic_json(status_path, state)

    def launch(job, stage):
        log_path = logs / f"{job['name']}.{stage}.log"
        if stage == "train":
            command = [sys.executable, "-u", str(SCRIPTS / "train_tail.py"),
                       "--cache-dir", str(args.cache_dir), "--output-dir", job["run_dir"],
                       "--alpha", str(job["alpha"]), "--steps", str(args.steps), "--seed", str(args.seed),
                       "--batch-size", str(args.batch_size), "--eval-every", str(args.eval_every),
                       "--log-every", str(args.log_every), "--cpu-threads", str(args.cpu_threads), "--device", "cuda:0"]
        else:
            command = [sys.executable, "-u", str(SCRIPTS / "eval_tail_value.py"),
                       "--cache-dir", str(args.cache_dir), "--checkpoint", str(Path(job["run_dir"]) / "last.pt"),
                       "--output-dir", job["report_dir"], "--suite", "val", "--device", "cuda:0",
                       "--batch-size", str(args.batch_size), "--cpu-threads", str(args.cpu_threads),
                       "--max-plots", str(args.max_plots)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=job["gpu"]["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads))
        with log_path.open("x") as stream:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        active[job["name"]] = process
        job.update({"stage": stage, "status": "running", "pid": process.pid})
        job[f"{stage}_pid"] = process.pid
        job[f"{stage}_command"] = command
        job[f"{stage}_log"] = str(log_path)
        save()
        print(f"{job['name']} GPU {job['gpu']['index']} {stage} pid={process.pid}", flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index, (alpha, gpu) in enumerate(zip(args.alphas, gpus, strict=True)):
            name = f"alpha_{str(alpha).replace('.', 'p')}"
            job = {"name": name, "alpha": alpha, "gpu": gpu,
                   "run_dir": str(args.output_dir / "runs" / name),
                   "report_dir": str(args.output_dir / "reports" / name)}
            state["jobs"].append(job)
            launch(job, "train")
        while active:
            for job in state["jobs"]:
                process = active.get(job["name"])
                if process is None or process.poll() is None:
                    continue
                stage, code = job["stage"], process.returncode
                del active[job["name"]]
                job[f"{stage}_returncode"] = code
                if code:
                    job["status"] = "failed"
                elif stage == "train":
                    launch(job, "eval")
                else:
                    job["status"] = "completed"
                save()
            if active:
                time.sleep(5)
        comparison, starts, configs = [], [], []
        for job in state["jobs"]:
            item = {key: job[key] for key in ("name", "alpha", "status", "run_dir", "report_dir")}
            if job["status"] == "completed":
                report = json.loads((Path(job["report_dir"]) / "metrics.json").read_text())
                item["val"] = next(iter(report["checkpoints"].values()))["suites"]["val"]
                records = [json.loads(line) for line in (Path(job["run_dir"]) / "metrics.jsonl").read_text().splitlines()]
                starts.append(records[0]["model_sha256"])
                item["final_validation"] = [r for r in records if r["event"] == "val"][-1]
                config = json.loads((Path(job["run_dir"]) / "config.json").read_text())
                config.pop("alpha")
                configs.append(config)
            comparison.append(item)
        complete = all(job["status"] == "completed" for job in state["jobs"])
        matched_init = complete and len(set(starts)) == 1
        matched_config = complete and all(config == configs[0] for config in configs)
        atomic_json(args.output_dir / "comparison.json", {
            "bootstrap": POLICY_BOOTSTRAP, "code_revision": args.code_revision,
            "cache_fingerprint": state["cache_fingerprint"], "seed": args.seed,
            "initializations_identical": matched_init, "configs_identical_except_alpha": matched_config,
            "runs": comparison,
            "interpretation": "One seed; internal critic validation, not robot success or independent heldout performance."})
        state["status"] = "completed" if complete and matched_init and matched_config else "failed"
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"Ablation {state['status']}: {args.output_dir / 'comparison.json'}", flush=True)
        return 0 if state["status"] == "completed" else 1
    except BaseException as exc:
        # Only terminate subprocesses created by this orchestrator.
        for process in active.values():
            if process.poll() is None:
                process.terminate()
        for process in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for job in state["jobs"]:
            if job["name"] in active:
                job.update({"status": "interrupted", "returncode": active[job["name"]].returncode})
        state.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error": str(exc)})
        save()
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
