#!/usr/bin/env python3
"""Train a same-observation support D on 50 source demos; score 20 held-out demos.

Labels are expert-vs-own-policy actions, never episode success/failure labels.
The 50 source episodes get five episode-disjoint OOF folds (30 fit / 10
calibration / 10 OOF test); a final model is fit on all 50, calibrated from
the OOF training-cohort scores, and evaluated once on the independent 20.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_trainer(path: Path):
    code_root = path.resolve().parent.parent
    sys.path.insert(0, str(code_root))
    spec = importlib.util.spec_from_file_location("d_support_base_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base D trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_cohort(root: Path, task: str, expected: int, split_id: int) -> list[dict]:
    state = json.loads((root / "state.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (state.get("task") != task or state.get("status") != "complete"
            or state.get("completed_episodes") != expected
            or summary.get("complete") is not True
            or summary.get("expert_successes") != expected):
        raise ValueError(f"incomplete expert cohort: {root}")
    metas = sorted(root.glob("episode_*_scene_*/metadata.json"))
    if len(metas) != expected:
        raise ValueError(f"expected {expected} metadata files under {root}, got {len(metas)}")
    episodes = []
    for meta_path in metas:
        meta = json.loads(meta_path.read_text())
        folder = meta_path.parent
        if (meta.get("expert_success") is not True or meta.get("complete_pairs", 0) < 1
                or meta.get("source_frequency_hz") != 15 or meta.get("chunk_horizon") != 10
                or meta.get("scene_seed_split") != split_id):
            raise ValueError(f"invalid cohort metadata: {meta_path}")
        with np.load(folder / "pairs.npz", allow_pickle=False) as z:
            raw = {k: z[k] for k in z.files}
        with np.load(folder / "policy_candidates_k4.npz", allow_pickle=False) as z:
            policy = z["policy_chunks"]
        with np.load(folder / "features_r3m18.npz", allow_pickle=False) as z:
            features = z["features"]
        n = len(raw["state"])
        expected_shapes = {
            "state": (n, 14), "expert_chunks": (n, 10, 14),
            "sampled_state_index": (n,), "sim_step": (n,),
        }
        for key, shape in expected_shapes.items():
            if key not in raw or raw[key].shape != shape:
                raise ValueError(f"{folder}: {key} shape {raw.get(key, np.empty(0)).shape}, expected {shape}")
        if (policy.shape != (n, 4, 10, 14) or features.shape != (n, 1536)
                or not np.array_equal(raw["sampled_state_index"], raw["sim_step"])):
            raise ValueError(f"{folder}: invalid K4/features/anchor alignment")
        for name in ("images_head", "images_left", "images_right"):
            if raw[name].ndim != 4 or raw[name].shape[-1] != 3 or raw[name].dtype != np.uint8:
                raise ValueError(f"{folder}: invalid RGB array {name}: {raw[name].shape}/{raw[name].dtype}")
        for name, value in (("state", raw["state"]), ("expert", raw["expert_chunks"]),
                            ("policy", policy), ("features", features)):
            if not np.isfinite(value).all():
                raise ValueError(f"{folder}: non-finite {name}")
        # These source demos have no verified subtask annotations. Use an
        # explicit all-zero unknown label, never infer a stage from outcome.
        n = len(raw["state"])
        episodes.append({
            "episode": folder.name,
            "scene_seed": int(meta["scene_seed"]),
            "state": raw["state"].astype(np.float32),
            "features": features.astype(np.float32),
            "expert": raw["expert_chunks"].astype(np.float32),
            "policy": policy.astype(np.float32),
            "stage": np.zeros(n, dtype=np.int32),
            "stage_names": {},
            "stage_annotation_mode": "unverified_alignment_no_source_stage_labels",
            "stage_verified_anchors": 0,
            "stage_unverified_anchors": n,
            "stage_label_sha256": None,
            "sim_step": raw["sim_step"].astype(np.int64),
            "metadata_sha256": sha256(meta_path),
            "pairs_sha256": sha256(folder / "pairs.npz"),
            "candidate_sha256": sha256(folder / "policy_candidates_k4.npz"),
            "feature_sha256": sha256(folder / "features_r3m18.npz"),
        })
    seeds = [ep["scene_seed"] for ep in episodes]
    if len(set(seeds)) != expected:
        raise ValueError(f"duplicate scene seeds in {root}")
    return episodes


def bootstrap_metrics(rows: list[dict], trainer, seed: int) -> dict:
    names = ("expert_rank", "d_b_mean", "d_b_median", "normalized_range_median",
             "normalized_std_median", "ess_fraction_median", "max_weight_median",
             "candidate_pair_joint_rms_mrad_median", "candidate_pair_gripper_rms_median",
             "candidate_exact_duplicate_pair_fraction", "weighted_minus_uniform_joint_rms_mrad",
             "weighted_minus_uniform_gripper_rms", "action_gap_spearman_median")
    return {name: trainer.bootstrap_episode([row.get(name) for row in rows], seed + i)
            for i, name in enumerate(names)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--test-root", type=Path, required=True)
    p.add_argument("--dataset-manifest", type=Path, required=True)
    p.add_argument("--trainer-script", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    manifest = json.loads(args.dataset_manifest.read_text())
    if manifest.get("complete") is not True or manifest.get("protocol", {}).get("candidate_count") != 4:
        raise ValueError("dataset manifest is incomplete or not K=4")
    task_row = next((r for r in manifest.get("tasks", []) if r.get("task") == args.task), None)
    if task_row is None or task_row.get("train50", {}).get("episodes") != 50 or task_row.get("test20", {}).get("episodes") != 20:
        raise ValueError(f"manifest lacks validated 50/20 split for {args.task}")

    train = read_cohort(args.train_root, args.task, 50, 1)
    test = read_cohort(args.test_root, args.task, 20, 2)
    train_seeds = {ep["scene_seed"] for ep in train}
    test_seeds = {ep["scene_seed"] for ep in test}
    if train_seeds & test_seeds:
        raise ValueError("train/test scene seed leakage")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    trainer = load_trainer(args.trainer_script)
    args.output.mkdir(parents=True, exist_ok=True)

    # Five episode-disjoint 30/10/10 folds on the 50 original source demos.
    permutation = np.random.default_rng(args.seed).permutation(50).tolist()
    groups = [permutation[i * 10:(i + 1) * 10] for i in range(5)]
    oof_rows: dict[int, dict] = {}
    oof_expert_scores, oof_policy_scores = [], []
    fold_records = []
    for fold in range(5):
        test_ids = groups[fold]
        calibration_ids = groups[(fold + 1) % 5]
        train_ids = [i for i in range(50) if i not in set(test_ids + calibration_ids)]
        if (len(train_ids), len(calibration_ids), len(test_ids)) != (30, 10, 10):
            raise AssertionError("unexpected grouped fold sizes")
        fold_seed = args.seed + fold
        trainer.seed_all(fold_seed)
        model, norm = trainer.build_model(train, train_ids, device)
        fold_dir = args.output / f"fold_{fold}"
        training = trainer.train_fold(model, norm, train, train_ids, fold_dir,
                                     steps=args.steps, batch_size=args.batch_size,
                                     seed=fold_seed, save_every=args.save_every)
        cal_e, cal_p = [], []
        for i in calibration_ids:
            e, q = trainer.score_episode(model, train[i], device)
            cal_e.append(e); cal_p.append(q)
        calibration = trainer.calibrate(np.concatenate(cal_e), np.concatenate(cal_p),
                                        target=.5, beta_max=100.)
        fold_rows = []
        for i in test_ids:
            e, q = trainer.score_episode(model, train[i], device)
            row = trainer.per_episode_metrics(e, q, calibration, train[i])
            row["fold"] = fold
            oof_rows[row["scene_seed"]] = row
            oof_expert_scores.append(e)
            oof_policy_scores.append(q)
            fold_rows.append(row)
        record = {
            "fold": fold,
            "train_scene_seeds": [train[i]["scene_seed"] for i in train_ids],
            "calibration_scene_seeds": [train[i]["scene_seed"] for i in calibration_ids],
            "test_scene_seeds": [train[i]["scene_seed"] for i in test_ids],
            "training": training,
            "calibration": calibration,
            "test_episode_metrics": fold_rows,
            "checkpoint_sha256": sha256(fold_dir / "last.pt"),
        }
        fold_records.append(record)
        trainer.atomic_json(args.output / "folds_progress.json", {
            "completed_folds": fold + 1, "folds": fold_records})
        print(json.dumps({"event": "fold_complete", "task": args.task, "fold": fold,
                          "expert_rank": float(np.mean([r["expert_rank"] for r in fold_rows])),
                          "d_b_median": float(np.median([r["d_b_median"] for r in fold_rows]))}), flush=True)

    if set(oof_rows) != train_seeds:
        raise ValueError("OOF results do not cover exactly the 50 training episodes")
    oof_e = np.concatenate(oof_expert_scores)
    oof_p = np.concatenate(oof_policy_scores)
    final_calibration = trainer.calibrate(oof_e, oof_p, target=.5, beta_max=100.)

    # Fit the deployable D on all 50 source episodes; temperature/score scale
    # remain fixed from episode-held-out training-cohort predictions above.
    final_seed = args.seed + 1000
    trainer.seed_all(final_seed)
    final_model, final_norm = trainer.build_model(train, list(range(50)), device)
    final_dir = args.output / "final_full50"
    final_training = trainer.train_fold(final_model, final_norm, train, list(range(50)),
                                        final_dir, steps=args.steps,
                                        batch_size=args.batch_size, seed=final_seed,
                                        save_every=args.save_every)
    final_rows = []
    for episode in test:
        e, q = trainer.score_episode(final_model, episode, device)
        final_rows.append(trainer.per_episode_metrics(e, q, final_calibration, episode))

    oof_list = [oof_rows[s] for s in sorted(oof_rows)]
    result = {
        "complete": True,
        "task": args.task,
        "scope": "same-observation expert-vs-own-policy action-chunk support discrimination; not success/failure trajectory classification, failure detection, or takeover rescue probability",
        "data": {
            "train_episodes": 50, "test_episodes": 20,
            "train_scene_seeds": sorted(train_seeds), "test_scene_seeds": sorted(test_seeds),
            "train_anchors": int(sum(len(ep["state"]) for ep in train)),
            "test_anchors": int(sum(len(ep["state"]) for ep in test)),
            "source_frequency_hz": 15, "horizon": 10, "candidate_count": 4,
            "action_space": "absolute joint qpos, H10x14",
            "stage_annotation": "unverified_alignment_no_source_stage_labels",
            "test_usage": "20 independent successful expert episodes; excluded from fit and calibration",
            "train_files_sha256": {ep["episode"]: {"pairs": ep["pairs_sha256"],
                "candidates": ep["candidate_sha256"], "features": ep["feature_sha256"]} for ep in train},
            "test_files_sha256": {ep["episode"]: {"pairs": ep["pairs_sha256"],
                "candidates": ep["candidate_sha256"], "features": ep["feature_sha256"]} for ep in test},
        },
        "protocol": {
            "architecture": "existing ChunkCoverageCritic: frozen R3M RGB(1536) + normalized state(14) + normalized absolute H10x14; 3x256 MLP",
            "objective": "balanced source BCE; expert positive, one random own-policy K4 candidate negative per sampled anchor",
            "train_cohort_oof": "five episode-disjoint folds; 30 fit / 10 calibration / 10 test; every source demo tested once",
            "final_fit": "all 50 original source demos; no use of independent test20 for fitting or calibration",
            "steps_per_fold": args.steps, "final_steps": args.steps,
            "batch_size": args.batch_size, "optimizer": "AdamW lr=1e-4, wd=1e-4; grad norm clip=1",
            "calibration": "final ESS/K temperature from pooled OOF predictions on train50; target 0.5, beta cap 100",
            "stage_labels": "not verified; all stage IDs remain unknown and no stage metrics are claimed",
            "bootstrap": "10,000 episode-level resamples for OOF train50 and independent test20 summaries",
        },
        "train_oof": {
            "episodes": 50,
            "aggregate": bootstrap_metrics(oof_list, trainer, args.seed + 2000),
            "per_episode": oof_list,
            "folds": fold_records,
        },
        "heldout_test20": {
            "episodes": 20,
            "calibration_from_train_oof": final_calibration,
            "aggregate": bootstrap_metrics(final_rows, trainer, args.seed + 3000),
            "per_episode": final_rows,
        },
        "interpretation": [
            "expert_rank is the same-observation pairwise ranking AUC: each expert chunk is compared with its four own-policy candidates at that identical RGB/state/prompt",
            "D_B, ESS/K, and weighted action shift measure score resolution and sampler concentration, not task success or rescue rate",
            "the independent test20 contains successful scripted-expert episodes only; it does not support success-vs-failure trajectory AUC or takeover-event detection claims",
        ],
    }
    trainer.atomic_json(args.output / "metrics.json", result)
    trainer.atomic_json(args.output / "manifest.json", {
        "complete": True, "task": args.task, "episodes_train": 50,
        "episodes_test": 20, "anchors_train": result["data"]["train_anchors"],
        "anchors_test": result["data"]["test_anchors"],
        "metrics_sha256": sha256(args.output / "metrics.json"),
        "final_checkpoint_sha256": sha256(final_dir / "last.pt"),
        "fold_checkpoints_sha256": {str(i): sha256(args.output / f"fold_{i}/last.pt") for i in range(5)},
    })
    print(json.dumps({"event": "task_complete", "task": args.task,
                      "metrics": str(args.output / "metrics.json"),
                      "train_oof_expert_rank": result["train_oof"]["aggregate"]["expert_rank"],
                      "heldout_test20_expert_rank": result["heldout_test20"]["aggregate"]["expert_rank"]}), flush=True)


if __name__ == "__main__":
    main()
