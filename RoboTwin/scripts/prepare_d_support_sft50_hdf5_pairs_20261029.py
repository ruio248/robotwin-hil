#!/usr/bin/env python3
"""Convert original native-15-Hz SFT HDF5 demos into same-observation D pairs.

Expert labels are the next ten native HDF5 action targets.  The policy
candidates are added in a later step by querying the task's own 5k SFT
checkpoint on the exact stored observation.  No public checkpoint or rollout
data is read here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


CAMERAS = {
    "images_head": "cam_head",
    "images_left": "cam_left_wrist",
    "images_right": "cam_right_wrist",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def decode_rgb(value) -> np.ndarray:
    raw = bytes(value).rstrip(b"\x00")
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def qpos(group) -> np.ndarray:
    return np.concatenate([
        np.asarray(group["left_arm_joint_states"], dtype=np.float32),
        np.asarray(group["left_ee_joint_states"], dtype=np.float32),
        np.asarray(group["right_arm_joint_states"], dtype=np.float32),
        np.asarray(group["right_ee_joint_states"], dtype=np.float32),
    ], axis=1)


def read_seed_list(path: Path, expected: int) -> list[int]:
    seeds = [int(value) for value in path.read_text().split()]
    if len(seeds) != expected or len(set(seeds)) != expected:
        raise ValueError(f"seed list must contain {expected} unique values: {path}")
    return seeds


def convert_episode(source: Path, output: Path, scene_seed: int, prompt: str,
                    horizon: int, anchor_stride: int, split_id: int) -> dict:
    with h5py.File(source, "r") as h5:
        frequency = int(h5["additional_info/frequency"][()])
        if frequency != 15:
            raise ValueError(f"{source}: expected native 15 Hz, got {frequency}")
        state = qpos(h5["state"])
        action = qpos(h5["action"])
        if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
            raise ValueError(f"{source}: state/action shape mismatch {state.shape}/{action.shape}")
        frames = len(state)
        indices = [index for index in range(0, frames - horizon, anchor_stride)
                   if index + horizon < frames]
        if not indices:
            raise ValueError(f"{source}: no complete H{horizon} anchors")

        images = {}
        for key, camera in CAMERAS.items():
            dataset = h5[f"vision/{camera}/colors"]
            decoded = [decode_rgb(dataset[index]) for index in indices]
            if len({image.shape for image in decoded}) != 1:
                raise ValueError(f"{source}: inconsistent image shape for {camera}")
            images[key] = np.stack(decoded).astype(np.uint8)

        states = state[indices].astype(np.float32)
        expert = np.stack([action[index + 1:index + horizon + 1] for index in indices]).astype(np.float32)
        if expert.shape != (len(indices), horizon, 14):
            raise ValueError(f"{source}: malformed expert H{horizon} chunks {expert.shape}")
        if not all(np.isfinite(value).all() for value in (states, expert)):
            raise ValueError(f"{source}: nonfinite state/action values")

    n = len(indices)
    atomic_npz(
        output / "pairs.npz",
        state=states,
        expert_chunks=expert,
        sampled_state_index=np.asarray(indices, dtype=np.int64),
        sim_step=np.asarray(indices, dtype=np.int64),
        stage_id=np.zeros(n, dtype=np.int32),
        stage_name=np.full(n, "unverified_alignment", dtype="<U32"),
        **images,
    )
    metadata = {
        "complete": True,
        "task": output.parent.name.removesuffix("_train50"),
        "scene_seed": int(scene_seed),
        "scene_seed_split": int(split_id),
        "episode_index": int(re.search(r"episode_(\d+)", source.name).group(1)),
        "expert_success": True,
        "expert_success_evidence": (
            "member of the exact clean-50 SFT training dataset"
            if split_id == 1 else
            "successfully collected by the scripted expert in the independent test cohort"
        ),
        "source_dataset": (
            "three_task_sft_20261018"
            if split_id == 1 else
            "independently_generated_demo_clean_native15_test20"
        ),
        "source_hdf5": str(source),
        "source_hdf5_sha256": sha256(source),
        "source_frequency_hz": frequency,
        "anchor_stride_frames": int(anchor_stride),
        "anchor_rate_hz": frequency / anchor_stride,
        "chunk_horizon": int(horizon),
        "action_semantics": "absolute joint target qpos, next H actions at native 15 Hz",
        "instruction": prompt,
        "complete_pairs": n,
        "stage_annotation": "unavailable in original SFT HDF5; all anchors explicitly unverified",
        "pair_semantics": "same native RGB/state/prompt; expert next H10 vs own-checkpoint K4 H10 to be added",
        "diagnostic_only": split_id == 2,
        "accepted_for_training": split_id != 2,
    }
    atomic_json(output / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("turn_switch", "dump_bin_bigbin"))
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Original task/aloha_agilex directory containing data/ and seed.txt")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--anchor-stride", type=int, default=5)
    parser.add_argument("--scene-seed-split", type=int, default=1)
    args = parser.parse_args()
    expected_episodes = 50 if args.scene_seed_split == 1 else 20
    if (args.episodes != expected_episodes or args.horizon != 10
            or args.anchor_stride < 1 or args.scene_seed_split not in (1, 2)):
        parser.error("protocol requires train split=50 or independent test split=20, both H10")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    source_root = args.source_root.resolve()
    data = source_root / "data"
    files = sorted(data.glob("episode_*.hdf5"))
    seeds = read_seed_list(source_root / "seed.txt", args.episodes)
    if len(files) != args.episodes:
        raise ValueError(f"{args.task}: expected 50 HDF5 demos, found {len(files)}")
    output_root = args.output_root
    output_root.mkdir(parents=True)
    atomic_json(output_root / "state.json", {
        "status": "running", "task": args.task, "expected_episodes": args.episodes,
        "source": "exact original SFT training demos" if args.scene_seed_split == 1
                  else "independent scripted-expert test demos",
        "frequency_hz": 15,
        "horizon": args.horizon, "anchor_stride_frames": args.anchor_stride,
    })
    records = []
    try:
        for index, source in enumerate(files):
            match = re.fullmatch(r"episode_(\d+)\.hdf5", source.name)
            episode_index = int(match.group(1))
            if episode_index != index:
                raise ValueError(f"episode numbering is not contiguous: {source.name}")
            out = output_root / f"episode_{index:03d}_scene_{seeds[index]}"
            out.mkdir()
            record = convert_episode(source, out, seeds[index], args.prompt,
                                     args.horizon, args.anchor_stride,
                                     args.scene_seed_split)
            records.append(record)
            print(f"PACKED {args.task} {index + 1}/50 seed={seeds[index]} "
                  f"anchors={record['complete_pairs']}", flush=True)
        anchors = int(sum(row["complete_pairs"] for row in records))
        summary = {
            "complete": True, "task": args.task, "episodes": args.episodes,
            "expert_successes": args.episodes, "pairs": anchors,
            "scene_seeds": seeds, "scene_seed_split": args.scene_seed_split,
            "native_frequency_hz": 15, "anchor_stride_frames": args.anchor_stride,
            "chunk_horizon": args.horizon,
            "source_hdf5_sha256": {row["source_hdf5"]: row["source_hdf5_sha256"] for row in records},
            "diagnostic_only": args.scene_seed_split == 2,
            "accepted_for_training": args.scene_seed_split != 2,
        }
        atomic_json(output_root / "summary.json", summary)
        atomic_json(output_root / "state.json", {
            "status": "complete", "task": args.task,
            "expected_episodes": args.episodes,
            "completed_episodes": args.episodes,
            "expert_successes": args.episodes,
            "scene_seed_split": args.scene_seed_split, "scene_seeds": seeds,
            "anchor_count": anchors, "frequency_hz": 15, "horizon": args.horizon,
        })
    except BaseException as error:
        atomic_json(output_root / "state.json", {
            "status": "failed", "task": args.task, "completed_episodes": len(records),
            "detail": f"{type(error).__name__}: {error}",
        })
        raise


if __name__ == "__main__":
    main()
