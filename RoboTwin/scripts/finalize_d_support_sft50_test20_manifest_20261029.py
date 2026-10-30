#!/usr/bin/env python3
"""Validate the two exact 50/20 cohorts and write a checksum-complete manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


TASKS = ("turn_switch", "dump_bin_bigbin")
SPLITS = (("train50", 50, 1), ("test20", 20, 2))
FILES = ("metadata.json", "pairs.npz", "policy_candidates_k4.npz", "features_r3m18.npz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS),
                        help="Tasks to validate; defaults to both registered tasks")
    args = parser.parse_args()
    root = args.data_root.resolve()
    rows = []
    for task in args.tasks:
        task_row = {"task": task}
        seed_sets = {}
        for split, expected, split_id in SPLITS:
            task_root = root / f"{task}_{split}"
            state = json.loads((task_root / "state.json").read_text())
            summary = json.loads((task_root / "summary.json").read_text())
            feature_manifest_path = task_root / "r3m_feature_manifest.json"
            feature_manifest = json.loads(feature_manifest_path.read_text())
            metas = sorted(task_root.glob("episode_*_scene_*/metadata.json"))
            if (state.get("status") != "complete" or state.get("completed_episodes") != expected
                    or summary.get("complete") is not True
                    or summary.get("expert_successes") != expected
                    or feature_manifest.get("complete") is not True
                    or feature_manifest.get("episodes") != expected
                    or len(metas) != expected):
                raise ValueError(f"incomplete cohort: {task}/{split}")
            records = [json.loads(path.read_text()) for path in metas]
            seeds = [int(record["scene_seed"]) for record in records]
            if len(set(seeds)) != expected:
                raise ValueError(f"duplicate scene seeds: {task}/{split}")
            if any(int(record.get("scene_seed_split", -1)) != split_id for record in records):
                raise ValueError(f"wrong scene split label: {task}/{split}")
            files = {}
            anchors = 0
            for meta_path, record in zip(metas, records, strict=True):
                episode = meta_path.parent
                if record.get("expert_success") is not True or record.get("complete_pairs", 0) < 1:
                    raise ValueError(f"invalid expert episode: {episode}")
                if record.get("source_frequency_hz") != 15:
                    raise ValueError(f"wrong source frequency at {episode}")
                if record.get("chunk_horizon") != 10:
                    raise ValueError(f"wrong horizon at {episode}")
                with np.load(episode / "pairs.npz", allow_pickle=False) as pairs, \
                     np.load(episode / "policy_candidates_k4.npz", allow_pickle=False) as policy, \
                     np.load(episode / "features_r3m18.npz", allow_pickle=False) as features:
                    n = len(pairs["state"])
                    if pairs["state"].shape != (n, 14) or pairs["expert_chunks"].shape != (n, 10, 14):
                        raise ValueError(f"invalid state/expert shape at {episode}")
                    if policy["policy_chunks"].shape != (n, 4, 10, 14):
                        raise ValueError(f"invalid own-policy K4 shape at {episode}")
                    if features["features"].shape != (n, 1536):
                        raise ValueError(f"invalid R3M feature shape at {episode}")
                    if not np.isfinite(pairs["state"]).all() or not np.isfinite(pairs["expert_chunks"]).all():
                        raise ValueError(f"nonfinite expert pair values at {episode}")
                    if not np.isfinite(policy["policy_chunks"]).all() or not np.isfinite(features["features"]).all():
                        raise ValueError(f"nonfinite policy/feature values at {episode}")
                    if not np.array_equal(pairs["sampled_state_index"], pairs["sim_step"]):
                        raise ValueError(f"anchor alignment mismatch at {episode}")
                anchors += n
                for filename in FILES:
                    path = episode / filename
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    files[str(path.relative_to(root))] = sha256(path)
            seed_sets[split] = set(seeds)
            task_row[split] = {
                "episodes": expected, "scene_seed_split": split_id,
                "scene_seeds": sorted(seeds), "successful_expert_episodes": expected,
                "anchors": anchors, "files_sha256": files,
            }
        if seed_sets["train50"] & seed_sets["test20"]:
            raise ValueError(f"train/test scene seed overlap: {task}")
        rows.append(task_row)

    manifest = {
        "complete": True,
        "protocol": {
            "task_data": "exact native SFT HDF5 training demos for train50; separately generated demo_clean scripted-expert HDF5 for test20",
            "source_frequency_hz": 15,
            "anchor_stride_frames": 5,
            "action_horizon": 10,
            "candidate_count": 4,
            "pair_semantics": "same RGB/state/fixed prompt; expert next H10 absolute joint targets vs own 5k SFT checkpoint K4 H10 outputs",
            "model_source": "task-specific user-trained 5000-step state=True Pi0.5 checkpoint; no public checkpoint negatives",
            "stage_labels": "unverified; no invented stage segmentation",
            "test_usage": "20 independent successful expert episodes per task; excluded from D fitting and calibration",
        },
        "tasks": rows,
    }
    destination = args.output.resolve() if args.output else root / "dataset_manifest.json"
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, destination)
    print(json.dumps({"complete": True, "manifest": str(destination),
                      "tasks": {row["task"]: {
                          split: {"episodes": row[split]["episodes"],
                                  "anchors": row[split]["anchors"]}
                          for split, _, _ in SPLITS} for row in rows}}, indent=2))


if __name__ == "__main__":
    main()
