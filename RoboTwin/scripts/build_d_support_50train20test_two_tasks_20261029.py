#!/usr/bin/env python3
"""Build matched expert/policy D data: 50 train and 20 independent test episodes.

The existing 30 successful turn_switch episodes are preserved as the first part
of the train cohort. No previous output is overwritten; new collections and
post-processing live in a fresh experiment root.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/media/ruio/hdd/pace_pi05_eval_20261022")
OPENPI = Path("/media/ruio/hdd/Openpi-PACE-eval-20261022")
ROBOTWIN = Path("/media/ruio/hdd/robotwin-hil-enhanced-sampling/RoboTwin")
PIPELINE = ROOT / "control/goal_d_pair_pipeline_20261026"
COLLECTOR = PIPELINE / "collect_matched_pairs_v2.py"
ANNOTATOR = PIPELINE / "coverage_benchmark/annotate_pair_stages.py"
AUGMENTER = PIPELINE / "coverage_benchmark/augment_matched_policy_candidates.py"
ENCODER = PIPELINE / "coverage_benchmark/encode_matched_pairs_r3m.py"
CLIENT_PY = Path("/hdd/miniconda3/envs/robotwin_hil/bin/python")
SERVER_PY = OPENPI / ".venv/bin/python"
R3M = ROOT / "control/d_pair_seed1_regeneration_20261026/assets/r3m18/pytorch_model.bin"
GPU = 0
PORTS = {"turn_switch_train50": 18494, "turn_switch_test20": 18495,
         "dump_bin_bigbin_train50": 18496, "dump_bin_bigbin_test20": 18497}
TASKS = ("turn_switch", "dump_bin_bigbin")
CONTROL = ROOT / "control/d_pair_support_50train20test_two_tasks_20261029"
DATA_ROOT = ROOT / "outputs/d_action_pairs_support_turn_switch_dump_bin_bigbin_20261029_train50_test20"
STAGING = ROOT / "outputs/d_action_pairs_support_turn_switch_dump_bin_bigbin_20261029_train50_test20_staging"
PREVIOUS_TURN30 = ROOT / "outputs/d_action_pairs_support_turn_switch_dump_bin_bigbin_20261029_seed1_n30/turn_switch_seed1_n30"
STATE = CONTROL / "state.json"
LOG = CONTROL / "run.log"
EXPECTED_DUMP_SHA = "3bd57fe9172d59731db1ccf1bc9605dfe3ef08a0f7224edc720ea4812c214611"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def state(status: str, task: str = "", phase: str = "", detail: str = "") -> None:
    atomic_json(STATE, {
        "status": status, "task": task or None, "phase": phase or None,
        "train_successful_episodes_per_task": 50,
        "independent_test_successful_episodes_per_task": 20,
        "train_scene_seed_split": 1, "test_scene_seed_split": 2,
        "horizon": 10, "candidates_per_observation": 4,
        "policy_checkpoint_source": "public task-specific pi0.5 checkpoint",
        "action_pair_semantics": "same RGB/state/prompt; expert future H10 vs K=4 policy H10",
        "detail": detail or None,
    })


def run(label: str, command: list[str], *, env: dict[str, str] | None = None,
        cwd: Path = ROOT) -> None:
    log(f"start step={label} command={' '.join(command)}")
    with (CONTROL / f"{label}.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = (CONTROL / f"{label}.log").read_text(errors="replace")[-4000:]
        raise RuntimeError(f"{label} failed rc={result.returncode}:\n{tail}")
    log(f"complete step={label}")


def wait_slot(port: int | None = None) -> None:
    while True:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(GPU)],
            capture_output=True, text=True, check=True)
        used = int(query.stdout.strip().splitlines()[0])
        port_free = True
        if port is not None:
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    port_free = False
        if used <= 4096 and port_free:
            return
        log(f"wait gpu={GPU} used_mib={used} port={port} port_free={port_free}")
        time.sleep(30)


def task_env(*, client: bool = True) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"OPENPI_DATA_HOME": str(ROOT / "openpi_data"),
                "HF_HOME": str(ROOT / ".hf-home"),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.55",
                "CUDA_VISIBLE_DEVICES": str(GPU)})
    pieces = [str(OPENPI / "src"), str(OPENPI / "packages/openpi-client/src"),
              str(ROOT / "pace/src"), str(ROBOTWIN / "scripts"), str(PIPELINE)]
    if env.get("PYTHONPATH"):
        pieces.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pieces)
    if client:
        env["PATH"] = f"{CLIENT_PY.parent}:{env.get('PATH', '')}"
    return env


def collector_command(task: str, split: int, offset: int, candidates: int,
                      successes: int, output: Path, port: int) -> list[str]:
    return [str(CLIENT_PY), "-u", str(COLLECTOR), "--task", task,
            "--checkpoint-root", str(ROOT / "checkpoints"),
            "--checkpoint-id", f"public_pi05_{task}", "--output", str(output),
            "--pace-root", str(ROOT / "pace"), "--robotwin-root", str(ROBOTWIN),
            "--openpi-root", str(OPENPI), "--server-python", str(SERVER_PY),
            "--scene-seed", str(split), "--scene-offset", str(offset),
            "--episodes", str(candidates), "--target-episodes", str(successes),
            "--gpu", str(GPU), "--port", str(port), "--horizon", "10"]


def validate_collected(root: Path, expected: int, split: int) -> list[Path]:
    run_state = json.loads((root / "state.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    metas = sorted(root.glob("episode_*_scene_*/metadata.json"))
    seeds = [int(json.loads(path.read_text())["scene_seed"]) for path in metas]
    if not (run_state.get("status") == "complete"
            and run_state.get("completed_episodes") == expected
            and run_state.get("scene_seed_split") == split
            and summary.get("complete") is True
            and summary.get("expert_successes") == expected
            and len(metas) == expected and len(set(seeds)) == expected):
        raise ValueError(f"collection validation failed at {root}: state={run_state.get('status')} "
                         f"episodes={len(metas)} expected={expected} unique_seeds={len(set(seeds))}")
    for path in metas:
        meta = json.loads(path.read_text())
        if meta.get("expert_success") is not True or meta.get("complete_pairs", 0) < 1:
            raise ValueError(f"invalid successful pair episode: {path}")
        meta["scene_seed_split"] = split
        meta["episode_dir"] = str(path.parent.resolve())
        atomic_json(path, meta)
    return metas


def copy_episodes(source: Path, destination: Path, *, start_index: int = 0,
                  split: int = 1) -> int:
    metas = sorted(source.glob("episode_*_scene_*/metadata.json"))
    for offset, meta_path in enumerate(metas):
        meta = json.loads(meta_path.read_text())
        scene_seed = int(meta["scene_seed"])
        target = destination / f"episode_{start_index + offset:03d}_scene_{scene_seed}"
        if target.exists():
            raise FileExistsError(target)
        shutil.copytree(meta_path.parent, target)
        copied_meta = json.loads((target / "metadata.json").read_text())
        copied_meta["episode_dir"] = str(target.resolve())
        copied_meta["scene_seed_split"] = split
        atomic_json(target / "metadata.json", copied_meta)
    return len(metas)


def write_combined_collection_metadata(root: Path, task: str, split: int,
                                       expected: int, candidate_scenes: int) -> None:
    metas = sorted(root.glob("episode_*_scene_*/metadata.json"))
    records = [json.loads(path.read_text()) for path in metas]
    seeds = [int(meta["scene_seed"]) for meta in records]
    if len(records) != expected or len(set(seeds)) != expected:
        raise ValueError(f"combined cohort count/uniqueness mismatch {root}")
    for meta in records:
        if meta.get("expert_success") is not True or meta.get("complete_pairs", 0) < 1:
            raise ValueError(f"combined cohort contains invalid episode under {root}")
    summary = {"complete": True, "task": task, "episodes": expected,
               "expert_successes": expected, "pairs": sum(m["complete_pairs"] for m in records),
               "scene_seeds": seeds, "scene_seed_split": split,
               "candidate_scenes": candidate_scenes,
               "checkpoint_id": f"public_pi05_{task}",
               "pair_semantics": "same RGB/state/prompt expert H10 vs policy H10"}
    atomic_json(root / "summary.json", summary)
    atomic_json(root / "state.json", {
        "status": "complete", "task": task, "completed_episodes": expected,
        "expected_episodes": expected, "expert_successes": expected,
        "scene_seed_split": split, "scene_seeds": seeds,
        "strictly_same_observation": True, "pair_count": summary["pairs"],
        "candidate_scenes": candidate_scenes,
        "pair_semantics": summary["pair_semantics"],
    })


def collect_one(task: str, name: str, split: int, offset: int, candidates: int,
                successes: int, port: int) -> Path:
    output = STAGING / name
    if output.exists():
        raise FileExistsError(f"Refusing to reuse collection output: {output}")
    wait_slot(port)
    state("running", task, f"collect_{name}")
    run(name, collector_command(task, split, offset, candidates, successes, output, port),
        env=task_env(), cwd=ROOT)
    validate_collected(output, successes, split)
    return output


def annotate_and_encode(task: str, split_name: str, root: Path, port: int) -> None:
    env = task_env(client=True)
    state("running", task, f"annotate_{split_name}")
    run(f"{task}_{split_name}_stages", [str(CLIENT_PY), "-u", str(ANNOTATOR),
        "--task", task, "--task-root", str(root), "--pace-root", str(ROOT / "pace"),
        "--robotwin-root", str(ROBOTWIN), "--collector", str(COLLECTOR)], env=env)
    wait_slot(port)
    state("running", task, f"augment_k4_{split_name}")
    run(f"{task}_{split_name}_k4", [str(CLIENT_PY), "-u", str(AUGMENTER),
        "--task", task, "--input-root", str(root), "--checkpoint-root", str(ROOT / "checkpoints"),
        "--openpi-root", str(OPENPI), "--server-python", str(SERVER_PY),
        "--gpu", str(GPU), "--port", str(port), "--candidates", "4"], env=env, cwd=ROOT)
    wait_slot()
    state("running", task, f"encode_r3m_{split_name}")
    run(f"{task}_{split_name}_r3m", [str(SERVER_PY), "-u", str(ENCODER),
        "--task-root", str(root), "--asset", str(R3M), "--device", "cuda:0", "--batch-size", "64"],
        env=env, cwd=OPENPI)


def manifest_dataset() -> dict:
    tasks = []
    for task in TASKS:
        splits = {}
        for split_name, expected, seed_split in (("train50", 50, 1), ("test20", 20, 2)):
            task_root = DATA_ROOT / f"{task}_{split_name}"
            state_data = json.loads((task_root / "state.json").read_text())
            summary = json.loads((task_root / "summary.json").read_text())
            feature = json.loads((task_root / "r3m_feature_manifest.json").read_text())
            metas = sorted(task_root.glob("episode_*_scene_*/metadata.json"))
            seeds = [int(json.loads(p.read_text())["scene_seed"]) for p in metas]
            if (len(metas) != expected or len(set(seeds)) != expected
                    or any(int(json.loads(p.read_text())["scene_seed_split"]) != seed_split for p in metas)
                    or state_data.get("status") != "complete"
                    or summary.get("expert_successes") != expected
                    or feature.get("complete") is not True
                    or feature.get("episodes") != expected):
                raise ValueError(f"final dataset validation failed: {task}/{split_name}")
            files = {}
            for meta_path in metas:
                episode = meta_path.parent
                meta = json.loads(meta_path.read_text())
                for filename in ("metadata.json", "pairs.npz", "stage_labels.npz",
                                 "policy_candidates_k4.npz", "features_r3m18.npz"):
                    path = episode / filename
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    files[str(path.relative_to(DATA_ROOT))] = sha256(path)
                with __import__("numpy").load(episode / "pairs.npz", allow_pickle=False) as raw, \
                     __import__("numpy").load(episode / "policy_candidates_k4.npz", allow_pickle=False) as policy, \
                     __import__("numpy").load(episode / "features_r3m18.npz", allow_pickle=False) as features:
                    n = len(raw["state"])
                    if raw["expert_chunks"].shape != (n, 10, 14) or policy["policy_chunks"].shape != (n, 4, 10, 14):
                        raise ValueError(f"invalid pair/K4 shape in {episode}")
                    if features["features"].shape != (n, 1536):
                        raise ValueError(f"invalid R3M feature shape in {episode}")
            splits[split_name] = {"episodes": expected, "scene_seed_split": seed_split,
                                  "scene_seeds": sorted(seeds),
                                  "successful_expert_episodes": expected,
                                  "anchors": int(summary["pairs"]), "files_sha256": files}
        if set(splits["train50"]["scene_seeds"]) & set(splits["test20"]["scene_seeds"]):
            raise ValueError(f"train/test scene overlap for {task}")
        tasks.append({"task": task, **splits})
    return {"complete": True, "protocol": {
        "policy": "public task-specific pi0.5 checkpoint", "config": "demo_clean",
        "action_space": "absolute_joint", "expert_and_policy_horizon": 10,
        "policy_candidates_per_observation": 4, "sample_hz": 50, "physics_hz": 250,
        "train_scene_seed_split": 1, "test_scene_seed_split": 2,
        "train_successful_episodes_per_task": 50, "independent_test_episodes_per_task": 20,
        "test_episodes_are_never_used_for_training_or_calibration": True,
        "source_screening_scene_cache_sha256": sha256(ROOT / "pace/configs/expert_scenes.json")},
        "tasks": tasks}


def main() -> None:
    CONTROL.mkdir(parents=True, exist_ok=True)
    if STATE.exists() or DATA_ROOT.exists() or STAGING.exists():
        raise FileExistsError("New 50-train/20-test output already exists; refusing to overwrite")
    if subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", "--target", str(ROOT)], text=True).strip() != "/dev/sda1":
        raise RuntimeError("Expected /dev/sda1 mounted at experiment root")
    if not all(path.exists() for path in (CLIENT_PY, SERVER_PY, COLLECTOR, ANNOTATOR, AUGMENTER, ENCODER, R3M)):
        raise FileNotFoundError("Missing D-pair pipeline dependency")
    if sha256(ROBOTWIN / "envs/dump_bin_bigbin.py") != EXPECTED_DUMP_SHA:
        raise RuntimeError("dump_bin_bigbin source differs from the published scene-cache source")
    for task in TASKS:
        for path in (ROOT / f"checkpoints/{task}/params",
                     ROOT / f"checkpoints/{task}/assets/robotwin/norm_stats.json"):
            if not path.exists():
                raise FileNotFoundError(path)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    state("running", "", "preflight")
    DATA_ROOT.mkdir(parents=True)
    STAGING.mkdir(parents=True)

    # Reuse only the already-complete turn_switch seed-1 training cohort.
    old_state = json.loads((PREVIOUS_TURN30 / "state.json").read_text())
    old_summary = json.loads((PREVIOUS_TURN30 / "summary.json").read_text())
    old_metas = sorted(PREVIOUS_TURN30.glob("episode_*_scene_*/metadata.json"))
    old_seeds = [int(json.loads(p.read_text())["scene_seed"]) for p in old_metas]
    if not (old_state.get("status") == "complete" and old_state.get("completed_episodes") == 30
            and old_summary.get("complete") is True and old_summary.get("expert_successes") == 30
            and len(old_metas) == 30 and len(set(old_seeds)) == 30):
        raise ValueError("Existing turn_switch 30-episode cohort failed validation")
    turn_train = DATA_ROOT / "turn_switch_train50"
    turn_train.mkdir()
    copied = copy_episodes(PREVIOUS_TURN30, turn_train, split=1)
    if copied != 30:
        raise ValueError(f"Expected 30 existing turn_switch episodes; copied {copied}")

    # The next 20 train scenes are disjoint from the original first 40 cache entries.
    state("running", "turn_switch", "collect_train_extra20")
    extra = collect_one("turn_switch", "turn_switch_train_extra20", 1, 40, 30, 20, 18494)
    copied_extra = copy_episodes(extra, turn_train, start_index=30, split=1)
    if copied_extra != 20:
        raise ValueError(f"Expected 20 additional turn_switch train episodes; got {copied_extra}")
    write_combined_collection_metadata(turn_train, "turn_switch", 1, 50, 70)

    state("running", "turn_switch", "collect_independent_test20")
    turn_test = collect_one("turn_switch", "turn_switch_test20", 2, 0, 30, 20, 18495)

    state("running", "dump_bin_bigbin", "collect_train50")
    dump_train = collect_one("dump_bin_bigbin", "dump_bin_bigbin_train50", 1, 0, 65, 50, 18496)
    state("running", "dump_bin_bigbin", "collect_independent_test20")
    dump_test = collect_one("dump_bin_bigbin", "dump_bin_bigbin_test20", 2, 0, 30, 20, 18497)

    roots = (("turn_switch", "train50", turn_train, 18494),
             ("turn_switch", "test20", turn_test, 18495),
             ("dump_bin_bigbin", "train50", dump_train, 18496),
             ("dump_bin_bigbin", "test20", dump_test, 18497))
    for task, split_name, root, port in roots:
        expected, split = (50, 1) if split_name == "train50" else (20, 2)
        metas = validate_collected(root, expected, split)
        if len(metas) != expected:
            raise ValueError(f"wrong cohort size {task}/{split_name}")
        annotate_and_encode(task, split_name, root, port)

    dataset_manifest = manifest_dataset()
    atomic_json(DATA_ROOT / "dataset_manifest.json", dataset_manifest)
    state("complete", "", "done", str(DATA_ROOT / "dataset_manifest.json"))
    log(f"COMPLETE dataset={DATA_ROOT} manifest={DATA_ROOT / 'dataset_manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        CONTROL.mkdir(parents=True, exist_ok=True)
        state("failed", detail=f"{type(error).__name__}: {error}")
        log(f"FAILED {type(error).__name__}: {error}")
        traceback.print_exc()
        raise
