#!/usr/bin/env python3
"""Episode-held-out training and diagnostics for same-observation D(o, a).

Labels are source labels (expert chunk vs task policy chunk), not success labels.
Every score comparison uses the exact same stored RGB, joint state, prompt, and
task stage. Episodes, never adjacent anchors, define the five CV folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from tail_value.cache import sha256
from tail_value.chunk import ChunkCoverageCritic
from coverage_benchmark.common import JOINTS, calibrate, weights


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_episode(root: Path, meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    folder = meta_path.parent
    with np.load(folder / "pairs.npz", allow_pickle=False) as source:
        raw = {k: source[k] for k in source.files}
    stage_file = folder / "stage_labels.npz"
    stage_manifest_path = folder / "stage_replay.json"
    stage_manifest = json.loads(stage_manifest_path.read_text()) if stage_manifest_path.is_file() else {}
    if stage_file.is_file():
        with np.load(stage_file, allow_pickle=False) as source:
            stage = source["stage_id"].astype(np.int32)
            stage_name = source["stage_name"].astype(str)
            if not np.array_equal(source["sampled_state_index"], raw["sampled_state_index"]):
                raise ValueError(f"{folder}: replayed stages do not align with sampled state indices")
            if not np.array_equal(source["sim_step"], raw["sim_step"]):
                raise ValueError(f"{folder}: replayed stages do not align with simulator steps")
        stage_hash = sha256(stage_file)
    elif "stage_name" in raw:
        stage = raw["stage_id"].astype(np.int32)
        stage_name = raw["stage_name"].astype(str)
        stage_hash = sha256(folder / "pairs.npz")
    else:
        raise ValueError(f"{folder}: no verified expert-subplan stage labels")
    with np.load(folder / "policy_candidates_k4.npz", allow_pickle=False) as source:
        policy = source["policy_chunks"]
    with np.load(folder / "features_r3m18.npz", allow_pickle=False) as source:
        features = source["features"]
    n = len(raw["state"])
    expected = {"state": (n, 14), "expert_chunks": (n, 10, 14),
                "stage_id": (n,), "sim_step": (n,)}
    for key, shape in expected.items():
        if key not in raw or raw[key].shape != shape:
            raise ValueError(f"{folder}: invalid {key} shape {raw.get(key, np.empty(0)).shape}; expected {shape}")
    image_shapes = [raw.get(k, np.empty(0)).shape for k in ("images_head", "images_left", "images_right")]
    if any(len(shape) != 4 or shape[0] != n or shape[-1] != 3 for shape in image_shapes):
        raise ValueError(f"{folder}: invalid camera shapes {image_shapes}")
    if len({shape[1:] for shape in image_shapes}) != 1:
        raise ValueError(f"{folder}: camera image dimensions differ {image_shapes}")
    if policy.shape != (n, 4, 10, 14) or features.shape != (n, 1536):
        raise ValueError(f"{folder}: K4/features shape mismatch: {policy.shape}, {features.shape}")
    for key, value in (("state", raw["state"]), ("expert", raw["expert_chunks"]),
                       ("policy", policy), ("features", features)):
        if not np.isfinite(value).all():
            raise ValueError(f"{folder}: nonfinite {key}")
    if not all(raw[k].dtype == np.uint8 for k in ("images_head", "images_left", "images_right")):
        raise ValueError(f"{folder}: image dtype is not uint8")
    if meta.get("complete_pairs") != n or meta.get("expert_success") is not True:
        raise ValueError(f"{folder}: incomplete or unsuccessful expert episode")
    if stage.shape != (n,) or stage_name.shape != (n,) or np.any(stage < 0):
        raise ValueError(f"{folder}: stage annotation arrays are missing or invalid")
    stage_names = {}
    for stage_id, name in zip(stage.tolist(), stage_name.tolist(), strict=True):
        key = str(int(stage_id))
        if stage_id == 0:
            if str(name) != "unverified_alignment":
                raise ValueError(f"{folder}: stage 0 must mean unverified alignment")
            continue
        previous = stage_names.setdefault(key, str(name))
        if previous != str(name) or not name or name in {"unlabeled", "pre_subplan"} or ":" not in name:
            raise ValueError(f"{folder}: inconsistent stage label for id {key}")
    return {"episode": folder.name, "scene_seed": int(meta["scene_seed"]),
            "state": raw["state"].astype(np.float32),
            "features": features.astype(np.float32),
            "expert": raw["expert_chunks"].astype(np.float32),
            "policy": policy.astype(np.float32),
            "stage": stage,
            "stage_names": stage_names,
            "stage_annotation_mode": stage_manifest.get("mode", "online_instrumented_expert_subplan_callsite"),
            "stage_verified_anchors": int(np.sum(stage > 0)),
            "stage_unverified_anchors": int(np.sum(stage == 0)),
            "stage_label_sha256": stage_hash,
            "sim_step": raw["sim_step"].astype(np.int64),
            "metadata_sha256": sha256(meta_path),
            "pairs_sha256": sha256(folder / "pairs.npz"),
            "candidate_sha256": sha256(folder / "policy_candidates_k4.npz"),
            "feature_sha256": sha256(folder / "features_r3m18.npz")}


def normalization(episodes: list[dict], indices: list[int]) -> dict[str, torch.Tensor]:
    states = np.concatenate([episodes[i]["state"] for i in indices])
    actions = np.concatenate([episodes[i]["expert"].reshape(-1, 14) for i in indices])
    return {
        "state_mean": torch.as_tensor(states.mean(0), dtype=torch.float32),
        "state_scale": torch.as_tensor(np.maximum(states.std(0), .05), dtype=torch.float32),
        "action_mean": torch.as_tensor(actions.mean(0), dtype=torch.float32),
        "action_scale": torch.as_tensor(np.maximum(actions.std(0), .05), dtype=torch.float32),
    }


def build_model(episodes, train_indices, device):
    norm = normalization(episodes, train_indices)
    return ChunkCoverageCritic(norm, horizon=10, width=256).to(device), norm


def train_fold(model, norm, episodes, train_indices, output: Path, *, steps: int,
               batch_size: int, seed: int, save_every: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "last.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    start = 0
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol") != "source_bce_v1" or payload.get("target_steps") != steps:
            raise ValueError(f"Refusing incompatible fold checkpoint: {checkpoint}")
        if payload.get("train_scenes") != [episodes[i]["scene_seed"] for i in train_indices]:
            raise ValueError(f"Fold checkpoint train split changed: {checkpoint}")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        rng.bit_generator.state = payload["numpy_rng_state"]
        start = int(payload["step"])
        if start >= steps:
            return {"checkpoint": str(checkpoint), "steps": start, "resumed": False}

    device = next(model.parameters()).device
    train_groups = [episodes[i] for i in train_indices]
    group_sizes = np.asarray([len(ep["state"]) for ep in train_groups], dtype=np.int64)
    if np.any(group_sizes == 0):
        raise ValueError("Empty train episode")
    metrics_path = output / "train_metrics.jsonl"
    for step in range(start + 1, steps + 1):
        group_ids = rng.integers(len(train_groups), size=batch_size)
        rows = np.asarray([rng.integers(group_sizes[g]) for g in group_ids], dtype=np.int64)
        selected = [train_groups[g] for g in group_ids]
        features = torch.as_tensor(np.stack([ep["features"][row] for ep, row in zip(selected, rows)]), device=device)
        states = torch.as_tensor(np.stack([ep["state"][row] for ep, row in zip(selected, rows)]), device=device)
        expert = torch.as_tensor(np.stack([ep["expert"][row] for ep, row in zip(selected, rows)]), device=device)
        all_policy = np.stack([ep["policy"][row] for ep, row in zip(selected, rows)])
        negative_ids = rng.integers(4, size=batch_size)
        negative = torch.as_tensor(all_policy[np.arange(batch_size), negative_ids], device=device)
        optimizer.zero_grad(set_to_none=True)
        positive_logit = model(features, states, expert)
        negative_logit = model(features, states, negative)
        loss = .5 * (F.softplus(-positive_logit).mean() + F.softplus(negative_logit).mean())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite D loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % save_every == 0 or step == steps:
            record = {"step": step, "loss": float(loss.detach()),
                      "expert_logit": float(positive_logit.detach().mean()),
                      "policy_logit": float(negative_logit.detach().mean()),
                      "grad_norm": float(grad_norm), "seed": seed}
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            payload = {"protocol": "source_bce_v1", "target_steps": steps, "step": step,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "numpy_rng_state": rng.bit_generator.state,
                       "train_scenes": [episodes[i]["scene_seed"] for i in train_indices],
                       "normalization": norm, "seed": seed}
            tmp = checkpoint.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            os.replace(tmp, checkpoint)
            print(json.dumps({"event": "D_train", **record}), flush=True)
    return {"checkpoint": str(checkpoint), "steps": steps, "resumed": start > 0}


@torch.inference_mode()
def score_episode(model, episode: dict, device: torch.device, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    expert_scores, policy_scores = [], []
    n = len(episode["state"])
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        features = torch.as_tensor(episode["features"][start:end], dtype=torch.float32, device=device)
        states = torch.as_tensor(episode["state"][start:end], dtype=torch.float32, device=device)
        expert = torch.as_tensor(episode["expert"][start:end], dtype=torch.float32, device=device)
        policy = torch.as_tensor(episode["policy"][start:end], dtype=torch.float32, device=device)
        expert_scores.append(model(features, states, expert).cpu().numpy())
        policy_scores.append(model.score_many(features, states, policy).cpu().numpy())
    return np.concatenate(expert_scores), np.concatenate(policy_scores)


def per_episode_metrics(expert: np.ndarray, policy: np.ndarray, calibration: dict,
                        episode: dict) -> dict:
    ranks = ((expert[:, None] > policy).astype(np.float64)
             + .5 * (expert[:, None] == policy)).mean(1)
    pairs_i, pairs_j = np.triu_indices(policy.shape[1], 1)
    raw_pair_gaps = np.abs(policy[:, pairs_i] - policy[:, pairs_j])
    scale = max(float(calibration["scale"]), 1e-12)
    d_b = raw_pair_gaps.mean(1) / scale
    score_range = np.ptp(policy, axis=1) / scale
    score_std = np.std(policy, axis=1) / scale
    w = np.stack([weights(row, calibration) for row in policy])
    ess = 1.0 / np.sum(w * w, axis=1) / policy.shape[1]
    chunks = episode["policy"]
    # Quantify whether repeated stochastic policy queries actually produce
    # distinct action chunks.  K=4 can otherwise look like four candidates
    # while effectively containing duplicates.
    candidate_i, candidate_j = np.triu_indices(chunks.shape[1], 1)
    candidate_delta = chunks[:, candidate_i] - chunks[:, candidate_j]
    candidate_joint_rms_mrad = np.sqrt(np.mean(
        candidate_delta[:, :, :, JOINTS] ** 2, axis=(2, 3))) * 1000
    candidate_gripper_rms = np.sqrt(np.mean(
        candidate_delta[:, :, :, [6, 13]] ** 2, axis=(2, 3)))
    candidate_duplicate = np.max(np.abs(candidate_delta), axis=(2, 3)) <= 1e-6
    weighted = np.einsum("nk,nkhd->nhd", w, chunks)
    uniform = chunks.mean(1)
    difference = weighted - uniform
    joint_shift_mrad = np.sqrt(np.mean(difference[:, :, JOINTS] ** 2, axis=(1, 2))) * 1000
    gripper_shift = np.sqrt(np.mean(difference[:, :, [6, 13]] ** 2, axis=(1, 2)))
    rho_values = []
    for candidates, scores in zip(chunks, policy, strict=True):
        # Use all six action-score pairs; robust to candidate zero being duplicated.
        ai, aj = np.triu_indices(4, 1)
        candidate_pairs = candidates[ai] - candidates[aj]
        action_distances = np.linalg.norm(candidate_pairs[:, :, JOINTS].reshape(6, -1), axis=1)
        score_gaps = np.abs(scores[ai] - scores[aj])
        if np.ptp(action_distances) > 1e-12 and np.ptp(score_gaps) > 1e-12:
            corr = float(spearmanr(action_distances, score_gaps).statistic)
            if np.isfinite(corr):
                rho_values.append(corr)
    return {
        "scene_seed": episode["scene_seed"], "anchors": len(ranks),
        "expert_rank": float(ranks.mean()),
        "d_b_mean": float(d_b.mean()), "d_b_median": float(np.median(d_b)),
        "d_b_q25_q75": np.quantile(d_b, [.25, .75]).tolist(),
        "normalized_range_median": float(np.median(score_range)),
        "normalized_std_median": float(np.median(score_std)),
        "ess_fraction_median": float(np.median(ess)),
        "max_weight_median": float(np.median(w.max(1))),
        "candidate_pair_joint_rms_mrad_median": float(np.median(candidate_joint_rms_mrad)),
        "candidate_pair_gripper_rms_median": float(np.median(candidate_gripper_rms)),
        "candidate_exact_duplicate_pair_fraction": float(np.mean(candidate_duplicate)),
        "weighted_minus_uniform_joint_rms_mrad": float(np.mean(joint_shift_mrad)),
        "weighted_minus_uniform_gripper_rms": float(np.mean(gripper_shift)),
        "action_gap_spearman_median": float(np.median(rho_values)) if rho_values else None,
        "action_gap_spearman_defined_anchors": len(rho_values),
        "stage_summaries": _stage_summary(episode["stage"], ranks, d_b, ess,
                                           episode["stage_names"]),
    }


def _stage_summary(stage: np.ndarray, ranks: np.ndarray, d_b: np.ndarray,
                   ess: np.ndarray, stage_names: dict[str, str]) -> dict:
    out = {}
    for stage_id in sorted(np.unique(stage).tolist()):
        if int(stage_id) <= 0:
            continue
        mask = stage == stage_id
        key = str(int(stage_id))
        out[key] = {"stage_name": stage_names[key], "anchors": int(mask.sum()),
            "expert_rank": float(ranks[mask].mean()), "d_b_mean": float(d_b[mask].mean()),
            "d_b_median": float(np.median(d_b[mask])), "ess_fraction_median": float(np.median(ess[mask]))}
    return out


def bootstrap_episode(values: list[float | None], seed: int, repeats: int = 10000) -> dict:
    x = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(x):
        return {"mean": None, "ci95": None, "episodes": 0}
    rng = np.random.default_rng(seed)
    sample = x[rng.integers(len(x), size=(repeats, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": np.quantile(sample, [.025, .975]).tolist(),
            "episodes": len(x)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--task-data-dir", type=Path,
        help="Explicit task pair-data directory (defaults to <data-root>/<task>_seed0_n30)",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.folds != 5 or args.steps < 1 or args.batch_size < 1:
        parser.error("Protocol requires five episode folds and positive training settings")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    task_root = args.task_data_dir or (args.data_root / f"{args.task}_seed0_n30")
    task_root = task_root.resolve()
    state = json.loads((task_root / "state.json").read_text())
    if state.get("task") != args.task:
        raise ValueError(f"Pair-data task mismatch: expected {args.task}, got {state.get('task')}")
    if state.get("status") != "complete" or state.get("completed_episodes") != 30:
        raise ValueError(f"Pair collection incomplete for {args.task}")
    summary = json.loads((task_root / "summary.json").read_text())
    if summary.get("complete") is not True or summary.get("episodes") != 30:
        raise ValueError(f"Pair summary incomplete for {args.task}")
    feature_manifest = json.loads((task_root / "r3m_feature_manifest.json").read_text())
    if feature_manifest.get("complete") is not True or feature_manifest.get("episodes") != 30:
        raise ValueError(f"R3M encoding incomplete for {args.task}")
    metas = sorted(task_root.glob("episode_*_scene_*/metadata.json"))
    if len(metas) != 30:
        raise ValueError(f"Expected 30 episode metadata files, got {len(metas)}")
    episodes = [read_episode(task_root, path) for path in metas]
    scene_ids = [e["scene_seed"] for e in episodes]
    if len(set(scene_ids)) != 30:
        raise ValueError("Training cohort contains duplicate scene seeds")

    out = args.output_root / args.task
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(episodes)).tolist()
    groups = [perm[i::args.folds] for i in range(args.folds)]
    if any(len(group) != 6 for group in groups):
        raise ValueError("Expected 30 scenes split into five groups of six")
    assignment = {str(episodes[i]["scene_seed"]): {"test_fold": None, "calibration_folds": []}
                  for i in range(len(episodes))}
    oof_by_seed: dict[int, dict] = {}
    fold_records = []
    for fold in range(args.folds):
        test_indices = groups[fold]
        cal_indices = groups[(fold + 1) % args.folds]
        train_indices = [i for i in range(len(episodes)) if i not in set(test_indices + cal_indices)]
        if len(set(train_indices) & set(cal_indices + test_indices)) or len(train_indices) != 18:
            raise ValueError("Fold episode leakage or wrong training cohort")
        for i in test_indices:
            assignment[str(scene_ids[i])]["test_fold"] = fold
        for i in cal_indices:
            assignment[str(scene_ids[i])]["calibration_folds"].append(fold)
        seed = args.seed + fold
        seed_all(seed)
        model, norm = build_model(episodes, train_indices, device)
        fold_dir = out / f"fold_{fold}"
        train_record = train_fold(model, norm, episodes, train_indices, fold_dir,
                                  steps=args.steps, batch_size=args.batch_size, seed=seed,
                                  save_every=args.save_every)

        cal_expert, cal_policy = [], []
        for i in cal_indices:
            e, p = score_episode(model, episodes[i], device)
            cal_expert.append(e); cal_policy.append(p)
        cal_expert = np.concatenate(cal_expert)
        cal_policy = np.concatenate(cal_policy)
        calibration = calibrate(cal_expert, cal_policy, target=.5, beta_max=100.)
        atomic_json(fold_dir / "calibration.json", calibration)
        test_rows = []
        for i in test_indices:
            e, p = score_episode(model, episodes[i], device)
            row = per_episode_metrics(e, p, calibration, episodes[i])
            row["fold"] = fold
            oof_by_seed[row["scene_seed"]] = row
            test_rows.append(row)
        fold_records.append({"fold": fold,
            "train_scene_seeds": [scene_ids[i] for i in train_indices],
            "calibration_scene_seeds": [scene_ids[i] for i in cal_indices],
            "test_scene_seeds": [scene_ids[i] for i in test_indices],
            "training": train_record, "calibration": calibration,
            "test_episode_metrics": test_rows,
            "checkpoint_sha256": sha256(fold_dir / "last.pt")})
        atomic_json(out / "folds_progress.json", {"completed_folds": fold + 1,
                                                   "folds": fold_records})
        print(json.dumps({"event": "fold_complete", "task": args.task, "fold": fold,
                          "test_expert_rank": float(np.mean([x["expert_rank"] for x in test_rows])),
                          "test_d_b_median": float(np.median([x["d_b_median"] for x in test_rows])),
                          "test_ess_k_median": float(np.median([x["ess_fraction_median"] for x in test_rows]))}), flush=True)

    if set(oof_by_seed) != set(scene_ids) or any(len(assignment[str(s)]["calibration_folds"]) != 1 for s in scene_ids):
        raise ValueError("OOF coverage or role assignment is incomplete")
    rows = [oof_by_seed[s] for s in sorted(oof_by_seed)]
    metric_names = ("expert_rank", "d_b_mean", "d_b_median", "normalized_range_median",
                    "normalized_std_median", "ess_fraction_median", "max_weight_median",
                    "candidate_pair_joint_rms_mrad_median",
                    "candidate_pair_gripper_rms_median",
                    "candidate_exact_duplicate_pair_fraction",
                    "weighted_minus_uniform_joint_rms_mrad", "weighted_minus_uniform_gripper_rms",
                    "action_gap_spearman_median")
    aggregate = {name: bootstrap_episode([row[name] for row in rows], args.seed + j)
                 for j, name in enumerate(metric_names)}
    result = {
        "complete": True,
        "task": args.task,
        "scope": "D(o,a) source discrimination on expert-visited observations; not success prediction, failure detection, or takeover rescue probability",
        "data": {"pair_data_dir": str(task_root),
                 "scene_seed_split": state.get("scene_seed_split"),
                 "scene_offset": state.get("scene_offset"),
                 "episodes": 30, "unique_scene_seeds": sorted(scene_ids),
                 "anchors": int(sum(len(e["state"]) for e in episodes)),
                 "candidate_count": 4, "expert_success_episodes": 30,
                 "stage_annotation": "exact online labels where available; otherwise monotone qpos-DTW replay with per-anchor confidence filtering (unverified anchors excluded only from stage summaries)",
                 "stage_annotation_modes": sorted({e["stage_annotation_mode"] for e in episodes}),
                 "stage_verified_anchor_fraction": float(
                     sum(e["stage_verified_anchors"] for e in episodes)
                     / max(1, sum(len(e["state"]) for e in episodes))
                 ),
                 "stage_unverified_anchors": int(sum(e["stage_unverified_anchors"] for e in episodes)),
                 "expert_pair_positive": "future ten 50Hz joint states from scripted expert",
                 "policy_pair_negative": "four stochastic policy H10 chunks queried at the exact stored RGB/state/prompt",
                 "stage_labels_present": True, "d_act_on_policy_visited_states": False,
                 "pair_files_sha256": {e["episode"]: {"pairs": e["pairs_sha256"],
                     "stage_labels": e["stage_label_sha256"],
                     "candidates": e["candidate_sha256"], "features": e["feature_sha256"]} for e in episodes}},
        "protocol": {"model": "existing ChunkCoverageCritic: 1536 R3M RGB + normalized 14D state + normalized absolute H10x14 chunk; 3x256 MLP",
                     "objective": "balanced source BCE: expert=positive, policy=negative; negatives sampled from same-o candidates",
                     "folds": 5, "episode_split": "18 train / 6 calibration / 6 test; rotate disjoint 6-episode groups; every scene is test exactly once",
                     "steps_per_fold": args.steps, "batch_size": args.batch_size,
                     "optimizer": "AdamW lr=1e-4 wd=1e-4; grad norm clip=1",
                     "calibration": "fold-local calibration episodes; global balanced-class score scale; beta capped at 100 to target median ESS/K=0.5",
                     "weight_sign": "softmax(-beta * B / scale), same sign as deployed coverage sampler; therefore low-D candidates receive greater weight",
                     "stage_alignment": "expert planner subplans are identified by the frozen task source-file call-site line; approximate replay labels are assigned only when the DTW path has one positive stage and mapped qpos max-absolute error <= 0.05 rad; stage 0 is omitted from stage-only summaries but remains in global D metrics",
            "bootstrap": "10,000 episode-level resamples; all 30 scenes receive one out-of-fold test prediction; intervals are conditional on this single five-fold training/calibration realization and do not capture model-initialization or new-dataset variance"},
        "aggregate_out_of_fold": aggregate,
        "per_episode_out_of_fold": rows,
        "folds": fold_records,
        "interpretation": [
            "Expert rank above 0.5 means D assigned larger logits to the scripted expert chunk than to policy candidates at the same observation.",
            "D_B is the mean pairwise absolute score gap among the four same-observation policy candidates, normalized by the disjoint calibration score scale; higher resolution is useful only if stable and not merely saturation.",
            "ESS/K and weighted-minus-uniform joint displacement quantify the sampler's score-induced concentration and action change, not task success.",
            "Candidate-pair joint/gripper RMS and exact-duplicate fraction verify whether K=4 stochastic queries add meaningful action diversity; low diversity weakens the interpretation of D_B and sampler response.",
            "High D quality is a task-screening signal for action distinguishability/support, not evidence that D can detect failure or that an expert takeover will rescue the episode.",
            "All examples are from successful expert-visited states; this dataset does not include policy-visited failure states or takeover outcome labels."
        ],
        "source_hashes": {"training_script": sha256(Path(__file__)),
                          "feature_manifest": sha256(task_root / "r3m_feature_manifest.json"),
                          "collector_summary": sha256(task_root / "summary.json")},
    }
    atomic_json(out / "metrics.json", result)
    atomic_json(out / "manifest.json", {"complete": True, "task": args.task,
        "metrics_sha256": sha256(out / "metrics.json"),
        "fold_checkpoints": {str(i): sha256(out / f"fold_{i}/last.pt") for i in range(args.folds)},
        "episodes": len(rows), "anchors": result["data"]["anchors"]})
    print(json.dumps({"event": "task_complete", "task": args.task,
                      "metrics": str(out / "metrics.json"), "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
