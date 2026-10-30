#!/usr/bin/env python3
"""Pair a frozen PI0.5 checkpoint's H50 candidates with the exact saved demos.

The expert observations, prompts, and absolute-joint targets are read from the
frozen ``expert50`` PNG/NPZ files used by the SFT run. No simulator rollout or
expert-script replay is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"invalid RGB image {path}: {value.shape}")
    return value


def collect(args: argparse.Namespace) -> None:
    experiment_root = args.experiment_root.resolve()
    checkpoint = args.checkpoint.resolve()
    source_root = (experiment_root / "expert50").resolve()
    output_root = args.output_root.resolve()
    source_manifest = source_root / "manifest.json"
    sft_state = experiment_root / "control/sft_state.json"
    validation = experiment_root / "control/dataset_validation.json"
    for required in (source_manifest, sft_state, validation,
                     checkpoint / "params", checkpoint / "assets"):
        if not required.exists():
            raise FileNotFoundError(required)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite D pair output: {output_root}")

    manifest = json.loads(source_manifest.read_text())
    train_state = json.loads(sft_state.read_text())
    data_validation = json.loads(validation.read_text())
    if manifest.get("status") != "complete" or not manifest.get("complete"):
        raise ValueError("frozen expert50 manifest is not complete")
    if train_state.get("status") != "sft_complete" or train_state.get("formal_updates") != 5000:
        raise ValueError("the source SFT run is not complete at 5000 updates")
    if not data_validation.get("complete") or data_validation.get("episodes") != 50:
        raise ValueError("frozen SFT dataset validation did not confirm 50 episodes")
    if len(manifest.get("episodes", [])) != 50:
        raise ValueError("expected exactly 50 frozen expert demonstrations")

    # Import only after the launcher has established the intended OpenPI source
    # tree, CUDA visibility, and allocator limit.
    from experiment_config import config as sft_config
    from openpi.training import config as openpi_config
    from openpi.policies.policy_config import create_trained_policy

    # The SFT training launcher accepted checkpoint_dir_override, but serving
    # does not need that training-only field. Filter only constructor fields
    # unsupported by the installed OpenPI version while preserving the model,
    # data transforms, asset id, and policy metadata verbatim.
    original_train_config = openpi_config.TrainConfig
    accepted_fields = {field.name for field in __import__("dataclasses").fields(original_train_config)}

    def compatible_train_config(**kwargs):
        return original_train_config(**{key: value for key, value in kwargs.items() if key in accepted_fields})

    openpi_config.TrainConfig = compatible_train_config
    try:
        cfg = sft_config(smoke=False)
    finally:
        openpi_config.TrainConfig = original_train_config
    if cfg.model.action_horizon != 50 or cfg.model.discrete_state_input is not True:
        raise ValueError("SFT policy config must be state=True with H=50")
    policy = create_trained_policy(cfg, checkpoint)

    output_root.mkdir(parents=True)
    split_dirs = {0: output_root / "seed0_n25", 1: output_root / "seed1_n25"}
    for split_dir in split_dirs.values():
        split_dir.mkdir()
    state = {
        "status": "running", "task": "place_empty_cup", "stage": "collect_pairs",
        "sft_checkpoint": str(checkpoint), "sft_updates": 5000,
        "expert_source": str(source_root),
        "expert_manifest_sha256": sha256(source_manifest),
        "source_dataset_manifest_sha256": data_validation["source_manifest_sha256"],
        "state_input": True, "horizon": 50, "candidate_count": 4,
        "anchor_source": "frozen native-save PNG/state/action files; complete_h50_anchor_indices",
        "pair_semantics": "same frozen RGB/state/prompt: next 50 expert qpos vs K4 SFT-policy H50 chunks",
        "episodes_expected": 50, "completed_episodes": 0, "total_pairs": 0,
        "gpu": args.gpu,
        "diagnostic_only": bool(args.smoke),
        "accepted_for_training": not bool(args.smoke),
    }
    write_json(output_root / "state.json", state)
    summaries = {0: [], 1: []}
    try:
        entries = sorted(manifest["episodes"], key=lambda row: row["episode_index"])
        if args.smoke:
            entries = entries[:1]
        for entry in entries:
            index = int(entry["episode_index"])
            split = int(entry["cache_split"])
            source_episode = source_root / entry["folder"].split("/expert50/", 1)[-1]
            meta_path = source_episode / "metadata.json"
            trajectory_path = source_episode / "trajectory.npz"
            if not meta_path.is_file() or not trajectory_path.is_file():
                raise FileNotFoundError(f"missing frozen source episode: {source_episode}")
            meta = json.loads(meta_path.read_text())
            if (not meta.get("complete") or not meta.get("expert_success")
                    or not meta.get("accepted_for_sft") or not meta.get("accepted_for_d_expert")):
                raise ValueError(f"source episode is not an accepted successful expert demo: {meta_path}")
            if split not in split_dirs or split != int(meta["cache_split"]):
                raise ValueError(f"bad source split for scene {entry['scene_seed']}")
            prompt = meta.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"missing frozen prompt in {meta_path}")

            with np.load(trajectory_path, allow_pickle=False) as trajectory:
                qpos = trajectory["native_qpos"].astype(np.float32)
                native_step = trajectory["native_physics_step"].astype(np.int64)
                stage_id = trajectory["native_stage_id"].astype(np.int64)
            if qpos.ndim != 2 or qpos.shape[1] != 14 or not np.isfinite(qpos).all():
                raise ValueError(f"invalid qpos trajectory: {trajectory_path}")
            anchors = [int(x) for x in meta["complete_h50_anchor_indices"]]
            if args.smoke:
                anchors = anchors[:1]
            pairs = []
            for anchor_index, t in enumerate(anchors):
                if t < 0 or t + 50 >= len(qpos):
                    raise ValueError(f"invalid complete H50 anchor t={t}: {trajectory_path}")
                expert_chunk = qpos[t + 1:t + 51].copy()
                images_hwc = [
                    read_rgb(source_episode / camera / f"{t:06d}.png")
                    for camera in ("head_camera", "left_camera", "right_camera")
                ]
                obs = {
                    "state": qpos[t].copy(),
                    "images": {
                        "cam_high": images_hwc[0].transpose(2, 0, 1),
                        "cam_left_wrist": images_hwc[1].transpose(2, 0, 1),
                        "cam_right_wrist": images_hwc[2].transpose(2, 0, 1),
                    },
                    "prompt": prompt,
                }
                candidates = []
                for _ in range(4):
                    response = policy.infer(obs)
                    actions = np.asarray(response["actions"], dtype=np.float32)
                    if actions.shape != (50, 14) or not np.isfinite(actions).all():
                        raise ValueError(f"policy returned invalid H50 chunk {actions.shape} at scene {entry['scene_seed']} t={t}")
                    candidates.append(actions.copy())
                pairs.append({
                    "state": qpos[t].copy(), "images_hwc": images_hwc,
                    "expert_chunk": expert_chunk, "policy_candidates": np.stack(candidates),
                    "sim_step": int(native_step[t]), "sampled_state_index": t,
                    "stage_id": int(stage_id[t]),
                })

            if not pairs:
                raise ValueError(f"no H50 anchors in {source_episode}")
            scene_seed = int(entry["scene_seed"])
            name = f"episode_{index:03d}_scene_{scene_seed}"
            episode_out = split_dirs[split] / name
            episode_out.mkdir()
            candidates = np.stack([p["policy_candidates"] for p in pairs]).astype(np.float32)
            np.savez_compressed(
                episode_out / "pairs.npz",
                state=np.stack([p["state"] for p in pairs]).astype(np.float32),
                images_head=np.stack([p["images_hwc"][0] for p in pairs]).astype(np.uint8),
                images_left=np.stack([p["images_hwc"][1] for p in pairs]).astype(np.uint8),
                images_right=np.stack([p["images_hwc"][2] for p in pairs]).astype(np.uint8),
                expert_chunks=np.stack([p["expert_chunk"] for p in pairs]).astype(np.float32),
                policy_chunks=candidates[:, 0], policy_candidates=candidates,
                sim_step=np.asarray([p["sim_step"] for p in pairs], dtype=np.int64),
                sampled_state_index=np.asarray([p["sampled_state_index"] for p in pairs], dtype=np.int64),
                stage_id=np.asarray([p["stage_id"] for p in pairs], dtype=np.int64),
                stage_name=np.asarray([meta.get("stage_labels", {}).get(str(p["stage_id"]), "") for p in pairs], dtype=np.str_),
            )
            episode_meta = {
                "task": "place_empty_cup", "scene_seed": scene_seed,
                "expert_success": True, "expert_plan_success": True,
                "source_episode": str(source_episode),
                "source_episode_manifest_sha256": entry["manifest_sha256"],
                "frozen_observation_action_source": True,
                "prompt": prompt, "native_save_events": True,
                "chunk_horizon": 50, "candidate_count": 4, "action_dim": 14,
                "action_semantics": "absolute_joint qpos",
                "same_observation_check": "expert target and K4 policy candidates use identical saved RGB/state/prompt at each anchor",
                "accepted_for_training": True,
            }
            write_json(episode_out / "metadata.json", episode_meta)
            summary = {"scene_seed": scene_seed, "pairs": len(pairs), "expert_success": True,
                       "episode_dir": str(episode_out), "cache_split": split}
            summaries[split].append(summary)
            state["completed_episodes"] += 1
            state["total_pairs"] += len(pairs)
            state["last_completed_scene_seed"] = scene_seed
            write_json(output_root / "state.json", state)
            print(f"PAIRED {state['completed_episodes']}/50 scene={scene_seed} split={split} H50_pairs={len(pairs)}", flush=True)

        if args.smoke:
            state.update(status="smoke_complete", stage="smoke_complete",
                         completed_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())
            write_json(output_root / "smoke_summary.json", {
                "complete": True, "diagnostic_only": True, "accepted_for_training": False,
                "task": "place_empty_cup", "episodes": state["completed_episodes"],
                "pairs": state["total_pairs"], "sft_checkpoint": str(checkpoint),
            })
            write_json(output_root / "state.json", state)
            return

        for split, split_dir in split_dirs.items():
            summaries[split].sort(key=lambda row: row["scene_seed"])
            if len(summaries[split]) != 25 or len({x["scene_seed"] for x in summaries[split]}) != 25:
                raise ValueError(f"split {split} did not produce 25 unique episodes")
            split_meta = {
                "complete": True, "task": "place_empty_cup", "split": split,
                "episodes": 25, "expert_successes": 25,
                "scene_seeds": [x["scene_seed"] for x in summaries[split]],
                "pairs": sum(x["pairs"] for x in summaries[split]),
                "checkpoint": str(checkpoint), "candidate_count": 4,
                "action_horizon": 50, "pair_semantics": state["pair_semantics"],
                "episode_summaries": summaries[split],
            }
            write_json(split_dir / "summary.json", split_meta)
            write_json(split_dir / "state.json", split_meta)
        state.update(status="complete", stage="pairs_complete", completed_episodes=50,
                     completed_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())
        write_json(output_root / "state.json", state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_json(output_root / "state.json", state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--smoke", action="store_true",
                        help="one saved observation only; diagnostic, excluded from D training")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
