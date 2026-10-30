#!/usr/bin/env python3
"""Train matched-action D on 50 episodes and evaluate on 20 held-out episodes."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


BASE = Path("/home/ruihao/d_pair_support_turn_switch_dump_bin_bigbin_20261029_50train20test")
DATA = BASE / "data"
RESULTS = BASE / "results"
CONTROL = BASE / "control"
CODE = Path("/home/ruihao/d_pair_screening_seed1_20261026/code")
PY = Path("/home/ruihao/train_critic/.venv/bin/python")
TASKS = ("turn_switch", "dump_bin_bigbin")
STEPS = 10000
BATCH_SIZE = 256
SAVE_EVERY = 1000
SEED = 42

sys.path.insert(0, str(CODE))
from coverage_benchmark import train_screened_task_discriminators as base  # noqa: E402


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_state(status: str, task: str = "", phase: str = "", detail: str = "") -> None:
    atomic_json(CONTROL / "state.json", {
        "status": status, "task": task or None, "phase": phase or None,
        "completed_tasks": sum((RESULTS / t / "metrics.json").is_file() for t in TASKS),
        "tasks": list(TASKS), "train_successful_episodes": 50,
        "independent_test_successful_episodes": 20,
        "training_steps": STEPS, "batch_size": BATCH_SIZE,
        "model": "ChunkCoverageCritic; frozen R3M RGB + state + absolute H10 chunk",
        "detail": detail or None,
    })


def verify_manifest() -> dict:
    manifest_path = DATA / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError("Ubuntu dataset manifest is incomplete")
    rows = {row["task"]: row for row in manifest.get("tasks", [])}
    if set(rows) != set(TASKS):
        raise ValueError(f"Dataset manifest tasks differ: {set(rows)}")
    for task in TASKS:
        row = rows[task]
        for split, expected in (("train50", 50), ("test20", 20)):
            data = row[split]
            if data.get("episodes") != expected or len(data.get("scene_seeds", [])) != expected:
                raise ValueError(f"Wrong manifest count for {task}/{split}")
            if len(set(data["scene_seeds"])) != expected:
                raise ValueError(f"Duplicate manifest seed for {task}/{split}")
            for relative, expected_sha in data.get("files_sha256", {}).items():
                path = DATA / relative
                if not path.is_file() or digest(path) != expected_sha:
                    raise ValueError(f"Dataset checksum mismatch: {path}")
        if set(row["train50"]["scene_seeds"]) & set(row["test20"]["scene_seeds"]):
            raise ValueError(f"Train/test leakage for {task}")
    return manifest


def load_cohort(task: str, split: str, expected: int) -> list[dict]:
    task_root = DATA / f"{task}_{split}"
    state = json.loads((task_root / "state.json").read_text())
    summary = json.loads((task_root / "summary.json").read_text())
    feature_manifest = json.loads((task_root / "r3m_feature_manifest.json").read_text())
    if state.get("status") != "complete" or state.get("completed_episodes") != expected:
        raise ValueError(f"Incomplete {task}/{split} collector state")
    if summary.get("complete") is not True or summary.get("expert_successes") != expected:
        raise ValueError(f"Incomplete {task}/{split} episode summary")
    if feature_manifest.get("complete") is not True or feature_manifest.get("episodes") != expected:
        raise ValueError(f"Incomplete R3M feature manifest for {task}/{split}")
    metas = sorted(task_root.glob("episode_*_scene_*/metadata.json"))
    if len(metas) != expected:
        raise ValueError(f"Expected {expected} episode files, got {len(metas)} for {task}/{split}")
    episodes = [base.read_episode(task_root, path) for path in metas]
    seeds = [episode["scene_seed"] for episode in episodes]
    if len(set(seeds)) != expected:
        raise ValueError(f"Duplicate scene seeds for {task}/{split}")
    return episodes


def score_many(model, episodes, device):
    expert_rows, policy_rows = [], []
    episode_metrics = []
    for episode in episodes:
        expert, policy = base.score_episode(model, episode, device)
        expert_rows.append(expert)
        policy_rows.append(policy)
    return np.concatenate(expert_rows), np.concatenate(policy_rows)


def summarize(rows: list[dict], seed: int) -> dict:
    names = ("expert_rank", "d_b_mean", "d_b_median", "normalized_range_median",
             "normalized_std_median", "ess_fraction_median", "max_weight_median",
             "candidate_pair_joint_rms_mrad_median", "candidate_pair_gripper_rms_median",
             "candidate_exact_duplicate_pair_fraction", "weighted_minus_uniform_joint_rms_mrad",
             "weighted_minus_uniform_gripper_rms", "action_gap_spearman_median")
    return {name: base.bootstrap_episode([row[name] for row in rows], seed + i)
            for i, name in enumerate(names)}


def train_task(task: str, device: torch.device) -> dict:
    train = load_cohort(task, "train50", 50)
    test = load_cohort(task, "test20", 20)
    train_seeds = {e["scene_seed"] for e in train}
    test_seeds = {e["scene_seed"] for e in test}
    if train_seeds & test_seeds:
        raise ValueError(f"Actual episode seeds overlap in {task}")

    out = RESULTS / task
    out.mkdir(parents=True, exist_ok=False)
    indices = list(range(len(train)))
    base.seed_all(SEED)
    model, norm = base.build_model(train, indices, device)
    training = base.train_fold(model, norm, train, indices, out / "final",
                               steps=STEPS, batch_size=BATCH_SIZE, seed=SEED,
                               save_every=SAVE_EVERY)

    # Calibrate only with training-cohort scores; the independent 20 episodes
    # are never used to choose beta, normalization, checkpoints, or thresholds.
    train_expert, train_policy = score_many(model, train, device)
    calibration = base.calibrate(train_expert, train_policy, target=.5, beta_max=100.)
    test_rows = []
    for episode in test:
        expert_scores, policy_scores = base.score_episode(model, episode, device)
        test_rows.append(base.per_episode_metrics(expert_scores, policy_scores,
                                                  calibration, episode))

    train_metrics_path = out / "final/train_metrics.jsonl"
    lines = [line for line in train_metrics_path.read_text().splitlines() if line.strip()]
    final_train_metrics = json.loads(lines[-1]) if lines else None
    metrics = {
        "complete": True,
        "task": task,
        "scope": "same-observation expert-vs-policy action-chunk support discrimination; not success/failure prediction or rescue probability",
        "data": {"train_successful_expert_episodes": 50,
                 "independent_test_successful_expert_episodes": 20,
                 "train_scene_seed_split": 1, "test_scene_seed_split": 2,
                 "train_scene_seeds": sorted(train_seeds), "test_scene_seeds": sorted(test_seeds),
                 "train_anchors": int(sum(len(e["state"]) for e in train)),
                 "test_anchors": int(sum(len(e["state"]) for e in test)),
                 "train_test_scene_overlap": 0, "candidate_count": 4,
                 "pair_semantics": "same RGB/state/prompt; expert next H10 vs four stochastic policy H10 chunks",
                 "test_used_for_training_or_calibration": False},
        "protocol": {"model": "ChunkCoverageCritic: frozen R3M 1536D + normalized 14D state + normalized absolute H10x14 chunk; 3x256 MLP",
                     "objective": "balanced source BCE; expert=positive, policy=negative",
                     "training_steps": STEPS, "batch_size": BATCH_SIZE, "seed": SEED,
                     "optimizer": "AdamW lr=1e-4 wd=1e-4; grad norm clip=1",
                     "calibration_source": "training cohort only (in-sample score scale); no independent-test tuning",
                     "calibration": calibration,
                     "weight_sign": "softmax(-beta * D / scale); lower-D candidates receive greater weight"},
        "final_train_metrics": final_train_metrics,
        "independent_test_summary": summarize(test_rows, SEED + 1000),
        "per_episode_test": test_rows,
        "test_stage_means": {},
        "source_hashes": {"dataset_manifest": digest(DATA / "dataset_manifest.json"),
                          "trainer": digest(Path(__file__)),
                          "base_trainer": digest(CODE / "coverage_benchmark/train_screened_task_discriminators.py"),
                          "final_checkpoint": digest(out / "final/last.pt")},
        "interpretation": [
            "Expert rank is computed at identical RGB/state/prompt observations and compares expert H10 against four policy H10 chunks.",
            "D_B, ESS/K, and weighted action displacement are reported on the independent 20-episode test cohort with episode-level bootstrap intervals.",
            "These diagnostics measure action-source discrimination and sampler behavior, not whether the policy episode succeeds or an expert takeover rescues it.",
        ],
    }
    stage_values = {}
    for row in test_rows:
        for stage_id, value in row["stage_summaries"].items():
            stage_values.setdefault(stage_id, {"stage_name": value["stage_name"], "episodes": []})
            stage_values[stage_id]["episodes"].append({k: value[k] for k in
                ("expert_rank", "d_b_mean", "d_b_median", "ess_fraction_median")})
    for stage_id, entry in stage_values.items():
        entry["episode_count"] = len(entry["episodes"])
        entry["means"] = {name: float(np.mean([row[name] for row in entry["episodes"]]))
                          for name in ("expert_rank", "d_b_mean", "d_b_median", "ess_fraction_median")}
    metrics["test_stage_means"] = stage_values
    atomic_json(out / "metrics.json", metrics)
    atomic_json(out / "manifest.json", {"complete": True, "task": task,
        "train_episodes": 50, "independent_test_episodes": 20,
        "final_checkpoint": "final/last.pt", "checkpoint_sha256": metrics["source_hashes"]["final_checkpoint"],
        "metrics_sha256": digest(out / "metrics.json")})
    return {"task": task, "expert_rank": metrics["independent_test_summary"]["expert_rank"],
            "d_b_median": metrics["independent_test_summary"]["d_b_median"],
            "ess_fraction_median": metrics["independent_test_summary"]["ess_fraction_median"],
            "checkpoint": str(out / "final/last.pt")}


def main() -> None:
    CONTROL.mkdir(parents=True, exist_ok=True)
    if (CONTROL / "state.json").exists() or RESULTS.exists():
        raise FileExistsError("JG training output already exists; refusing overwrite/relaunch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on JG")
    manifest = verify_manifest()
    device = torch.device("cuda:0")
    summaries = []
    for task in TASKS:
        write_state("running", task, "validate_and_train_full_50")
        summaries.append(train_task(task, device))
        write_state("running", task, "task_complete")
    aggregate = {"complete": True, "protocol": manifest["protocol"], "tasks": summaries}
    atomic_json(RESULTS / "aggregate_d_support_50train20test.json", aggregate)
    write_state("complete", phase="done")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        CONTROL.mkdir(parents=True, exist_ok=True)
        write_state("failed", detail=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
