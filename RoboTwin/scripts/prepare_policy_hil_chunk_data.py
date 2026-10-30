#!/usr/bin/env python3
"""Prepare auditable policy-chunk and HIL sources from one-stage collections.

Policy observations are saved every 15 *executed* 30 Hz actions.  A policy
training label is therefore reconstructed from the high-rate execution log:
steps 15*i+1, 15*i+3, ..., 15*i+19 produce a 10-target, nominal-15-Hz
chunk.  We require that all targets precede the next saved policy frame in
the same uninterrupted segment.  Sparse saved frames are never treated as
ten adjacent actions, and policy actions are never relabelled as expert.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))
from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video  # noqa: E402


CAMERAS = {"cam_high": "head_camera", "cam_left_wrist": "left_camera",
           "cam_right_wrist": "right_camera"}
JOINT_FIELDS = ("left_arm_joint_states", "left_ee_joint_states",
                "right_arm_joint_states", "right_ee_joint_states")
PROMPT = "Pass the red bar from the left arm to the right arm and place it in the blue tray."


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sources(roots: list[Path]) -> tuple[list[dict], list[dict]]:
    hil, policy = [], []
    for root in roots:
        sessions = sorted(root.glob("session_*.json"))
        if len(sessions) != 1 or read_json(sessions[0]).get("aborted") is not False:
            raise ValueError(f"need one completed, unaborted session in {root}")
        episodes = sorted((root / "raw").glob("episode_*/episode.json"))
        for ep in episodes:
            meta = read_json(ep)
            mask = [str(x).lower() for x in meta.get("control_mask", [])]
            indices = [i for i, x in enumerate(mask) if x == "hil"]
            frames = sorted((ep.parent / "frames").glob("*.pkl"), key=lambda p: int(p.stem))
            if (meta.get("supervisor_label") != "success" or len(indices) < 2
                    or [int(p.stem) for p in frames] != list(range(len(mask)))):
                raise ValueError(f"invalid saved HIL episode: {ep}")
            hil.append({"root": root, "metadata": ep, "meta": meta, "hil_indices": indices})
        for p in sorted((root / "policy_rollouts").glob("*/rollout.json")):
            policy.append({"root": root, "metadata": p, "meta": read_json(p)})
    return hil, policy


def inspect_sources(roots: list[Path], expected_hil: int) -> tuple[list[dict], list[dict]]:
    if len(roots) != 2 or any(not root.is_dir() for root in roots):
        raise ValueError("pass exactly two existing source roots: main and topup")
    hil, policy = sources(roots)
    if len(hil) != expected_hil:
        raise ValueError(f"expected {expected_hil} HIL episodes, found {len(hil)}")
    seeds = [int(item["meta"]["seed"]) for item in hil]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate HIL seeds across main and topup")
    policy_keys = [(str(item["root"]), int(item["meta"]["rollout_index"])) for item in policy]
    if len(set(policy_keys)) != len(policy_keys):
        raise ValueError("duplicate archived policy rollout")
    return hil, policy


def executed_actions(path: Path) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actions: dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") != "executed":
            continue
        step = int(event["rollout_step"])
        action = np.asarray(event["action"], dtype=np.float32)
        if step in actions or action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError(f"duplicate/invalid executed step {step} in {path}")
        actions[step] = action
    return actions


def policy_rows(policy: list[dict]):
    for item in policy:
        root, meta, metadata = item["root"], item["meta"], item["metadata"]
        if meta.get("source") != "policy_rollout_not_expert_or_hil" or int(meta["save_freq"]) != 15:
            raise ValueError(f"unexpected policy archive schema: {metadata}")
        rollout = int(meta["rollout_index"])
        seed = int(meta["seed"])
        log = root / "sampling" / f"episode_{rollout:04d}_seed_{seed}.jsonl"
        actions = executed_actions(log)
        mask = [str(x).lower() for x in meta["control_mask"]]
        for segment in meta["segments"]:
            indices = [int(i) for i in segment["source_frame_indices"]]
            if not indices or any(mask[i] != "policy" for i in indices):
                raise ValueError(f"segment crosses intervention: {metadata}")
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError(f"noncontiguous segment indices: {metadata}")
            frames = metadata.parent / f"segment_{int(segment['segment_index']):03d}" / "frames"
            if len(list(frames.glob("*.pkl"))) != len(indices):
                raise ValueError(f"policy frame count mismatch: {frames}")
            # The next saved policy frame bounds the action interval.  If the
            # final saved frame is immediately followed by HIL, no invented
            # unseen policy future is admitted.
            for local_index, frame_index in enumerate(indices):
                steps = [frame_index * 15 + 1 + 2 * j for j in range(10)]
                if max(steps) > (indices[-1] + 1) * 15 or any(s not in actions for s in steps):
                    continue
                frame = frames / f"{local_index}.pkl"
                if not frame.is_file():
                    raise FileNotFoundError(frame)
                yield {
                    "root": root, "rollout": rollout, "seed": seed,
                    "segment": int(segment["segment_index"]), "frame_index": frame_index,
                    "frame": frame, "sampling_log": log, "rollout_metadata": metadata,
                    "steps": steps, "actions": np.stack([actions[s] for s in steps]),
                    "autonomous_success": bool(meta["autonomous_success"]),
                    "sampling_mode": meta["sampling_mode"],
                }


def export_policy(policy: list[dict], output: Path, expected_mode: str) -> dict:
    target = output / "policy_chunks.h5"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    rows = list(policy_rows(policy))
    if not rows:
        raise ValueError("no valid contiguous policy chunk labels")
    if any(row["sampling_mode"] != expected_mode for row in rows):
        raise ValueError("mixed policy sampling modes")
    temp = target.with_suffix(".h5.tmp")
    records = []
    try:
        with h5py.File(temp, "w") as h5:
            h5.attrs["schema_version"] = 1
            h5.attrs["source"] = "executed_policy_actions_not_expert"
            h5.attrs["observation_interval_actions"] = 15
            h5.attrs["online_action_frequency_hz"] = 30
            h5.attrs["label_action_stride"] = 2
            h5.attrs["label_nominal_frequency_hz"] = 15
            h5.attrs["horizon"] = 10
            h5.attrs["sampling_mode"] = expected_mode
            states = h5.create_dataset("observation.state", (len(rows), 14), dtype="f4")
            labels = h5.create_dataset("action.chunk", (len(rows), 10, 14), dtype="f4")
            image_sets = {}
            for index, row in enumerate(rows):
                with row["frame"].open("rb") as stream:
                    obs = pickle.load(stream)  # trusted collector output only
                state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
                if state.shape != (14,) or not np.isfinite(state).all():
                    raise ValueError(f"invalid 14D state: {row['frame']}")
                states[index] = state
                labels[index] = row["actions"]
                for name, source in CAMERAS.items():
                    image = np.asarray(obs["observation"][source]["rgb"])
                    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                        raise ValueError(f"invalid {source} RGB: {row['frame']}")
                    if name not in image_sets:
                        image_sets[name] = h5.create_dataset(
                            f"observation.images.{name}",
                            (len(rows), *image.shape), dtype="u1", compression="lzf")
                    if image.shape != image_sets[name].shape[1:]:
                        raise ValueError(f"camera dimensions changed: {row['frame']}")
                    image_sets[name][index] = image
                records.append({
                    "sample_index": index, "source_root": str(row["root"]),
                    "rollout_index": row["rollout"], "seed": row["seed"],
                    "segment_index": row["segment"], "source_frame_index": row["frame_index"],
                    "executed_action_steps": row["steps"],
                    "autonomous_success": row["autonomous_success"],
                    "source_frame": str(row["frame"]), "source_frame_sha256": digest(row["frame"]),
                    "sampling_log": str(row["sampling_log"]),
                    "sampling_log_sha256": digest(row["sampling_log"]),
                    "rollout_metadata_sha256": digest(row["rollout_metadata"]),
                })
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()
    result = {"complete": True, "samples": len(rows), "rollouts": len(set((str(r["root"]), r["rollout"]) for r in rows)),
              "hdf5": str(target), "sha256": digest(target), "records": records}
    write_json(output / "policy_chunk_manifest.json", result)
    return result


def verify_hil_file(path: Path, seed: int, expected_transitions: int) -> None:
    with h5py.File(path) as h5:
        if int(h5["additional_info"]["frequency"][()]) != 15:
            raise ValueError(f"HIL frequency mismatch: {path}")
        if sorted(set(int(x) for x in h5["additional_info"]["seed"][:])) != [seed]:
            raise ValueError(f"HIL seed mismatch: {path}")
        if any(field not in h5["state"] or field not in h5["action"] for field in JOINT_FIELDS):
            raise ValueError(f"missing 14D joint state/action: {path}")
        if len(h5["state"][JOINT_FIELDS[0]]) != expected_transitions:
            raise ValueError(f"HIL transition mismatch: {path}")


def export_hil(hil: list[dict], output: Path) -> dict:
    native = output / "native" / "hil"
    data = native / "data"
    data.mkdir(parents=True, exist_ok=True)
    records = []
    for index, item in enumerate(hil):
        seed = int(item["meta"]["seed"])
        target = data / f"episode_{index:07d}.hdf5"
        if target.exists():
            verify_hil_file(target, seed, len(item["hil_indices"]) - 1)
        else:
            stage = native / ".staging" / f"episode_{index:07d}"
            if stage.exists():
                raise FileExistsError(f"unresolved staging episode: {stage}")
            stage.mkdir(parents=True)
            try:
                for new_idx, source_idx in enumerate(item["hil_indices"]):
                    source = item["metadata"].parent / "frames" / f"{source_idx}.pkl"
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    try:
                        os.link(source, stage / f"{new_idx}.pkl")
                    except OSError:
                        shutil.copy2(source, stage / f"{new_idx}.pkl")
                process_folder_to_hdf5_video(
                    str(stage), str(target), str(native / "video" / f"episode_{index:07d}.mp4"),
                    instructions=[PROMPT],
                    frequency=15,
                    episode_metadata={"source": "hil", "seed": seed,
                                      "source_metadata_sha256": digest(item["metadata"])},
                    save_video=False,
                )
            finally:
                shutil.rmtree(stage)
            verify_hil_file(target, seed, len(item["hil_indices"]) - 1)
        records.append({"episode_index": index, "seed": seed, "source_root": str(item["root"]),
                        "source_metadata": str(item["metadata"]),
                        "source_metadata_sha256": digest(item["metadata"]),
                        "hil_frames": len(item["hil_indices"]),
                        "hil_transitions": len(item["hil_indices"]) - 1,
                        "hdf5": str(target), "hdf5_sha256": digest(target)})
    result = {"complete": True, "episodes": len(hil),
              "transitions": sum(row["hil_transitions"] for row in records), "records": records}
    write_json(native / "export_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--topup-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("gaussian", "enhanced"), required=True)
    parser.add_argument("--expected-hil", type=int, default=30)
    args = parser.parse_args()
    roots = [args.main_root.resolve(), args.topup_root.resolve()]
    hil, policy = inspect_sources(roots, args.expected_hil)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy_result = (read_json(output / "policy_chunk_manifest.json")
                     if (output / "policy_chunk_manifest.json").exists()
                     else export_policy(policy, output, args.mode))
    if not Path(policy_result["hdf5"]).is_file() or digest(Path(policy_result["hdf5"])) != policy_result["sha256"]:
        raise ValueError("policy data hash mismatch")
    hil_result = export_hil(hil, output)
    summary = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "complete": True, "group": args.mode,
               "source_main_root": str(roots[0]), "source_topup_root": str(roots[1]),
               "policy": {k: policy_result[k] for k in ("samples", "rollouts", "hdf5", "sha256")},
               "hil": {k: hil_result[k] for k in ("episodes", "transitions")},
               "training_contract": "SFT + (policy rollout + HIL), original SFT norm_stats, state=False, H=10; policy source uses explicit observation/action.chunk pairs and requires a chunk-aware loader, not LeRobot temporal indexing",
               "hil_timing_caveat": "HIL native export retains the earlier C-group 15-fps metadata convention, but underlying HIL frames were saved at save_freq=15 control actions. Do not interpret ten successive saved HIL frames as ten adjacent 15-Hz actions without an independent timing audit.",
               "paired_rate_note": "topup episodes are training sources only, not paired takeover-rate observations"}
    write_json(output / "training_data_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
