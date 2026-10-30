#!/usr/bin/env python3
"""Attach confidence-filtered expert-subplan labels to paired observations.

New collector outputs already contain online call-site labels. Earlier batches
stored stage_id=0, so this tool replays the cached expert scene and aligns its
qpos path monotonically to the stored path. Since PhysX replays are not always
bitwise deterministic, approximate replay labels are kept only for unambiguous
DTW matches with <=0.05 rad max-absolute qpos error; unknown anchors remain 0.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCHEME = "expert_play_once_move_callsite_source_line_v1"
DTW_SCHEME = "expert_play_once_move_callsite_source_line_dtw_v1"
DTW_QPOS_MAX_ABS_ERROR = 0.05


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_npz(path: Path, **arrays) -> None:
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_collector(path: Path):
    spec = importlib.util.spec_from_file_location("matched_pair_collector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import collector at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def embedded_stage_labels(meta: dict, pair_path: Path, n: int):
    if meta.get("stage_label_scheme") != SCHEME:
        return None
    with np.load(pair_path, allow_pickle=False) as source:
        if "stage_name" not in source.files:
            return None
        stage_id = source["stage_id"].astype(np.int32)
        stage_name = source["stage_name"].astype(str)
        sampled_state_index = source["sampled_state_index"].astype(np.int64)
        sim_step = source["sim_step"].astype(np.int64)
    # Older online collectors could save valid call-site names while storing
    # stage_id=0 from RoboTwin.task_metadata.  Treat those arrays as
    # incomplete and fall through to deterministic, same-seed expert replay;
    # never promote the names alone to verified stage labels.
    if stage_id.shape != (n,) or stage_name.shape != (n,):
        return None
    if np.any(stage_id <= 0):
        return None
    labels = {}
    for stage, name in zip(stage_id.tolist(), stage_name.tolist(), strict=True):
        previous = labels.setdefault(str(int(stage)), name)
        if previous != name or not name or name in {"unlabeled", "pre_subplan"} or ":" not in name:
            raise ValueError(f"inconsistent embedded stage name for {stage} in {pair_path}")
    return stage_id, stage_name, sampled_state_index, sim_step, labels


def reconstruct_saved_qpos(pair_path: Path):
    """Rebuild the contiguous stored 50 Hz expert qpos path from H=10 pairs."""
    with np.load(pair_path, allow_pickle=False) as source:
        anchors = source["state"].astype(np.float32)
        indices = source["sampled_state_index"].astype(np.int64)
        chunks = source["expert_chunks"].astype(np.float32)
    length = int(indices[-1] + chunks.shape[1] + 1)
    states = np.full((length, anchors.shape[1]), np.nan, dtype=np.float32)
    for anchor, index, chunk in zip(anchors, indices, chunks, strict=True):
        index = int(index)
        values = [(index, anchor), *enumerate(chunk, start=index + 1)]
        for state_index, value in values:
            previous = states[state_index]
            if np.isfinite(previous).all() and not np.array_equal(previous, value):
                raise ValueError(f"overlapping stored expert chunks disagree at state {state_index}")
            states[state_index] = value
    valid = np.isfinite(states).all(axis=1)
    if not valid.all():
        first_gap = int(np.flatnonzero(~valid)[0])
        if valid[first_gap:].any():
            raise ValueError(f"interior gap in reconstructed expert qpos at state {first_gap}")
        states = states[:first_gap]
    return states


def monotone_qpos_dtw(saved: np.ndarray, replay: np.ndarray, band: int = 24):
    """Return a monotone saved-index -> replay-index path and path ambiguity."""
    n, m = len(saved), len(replay)
    band = max(int(band), abs(n - m) + 8)
    scale = np.maximum(np.std(np.concatenate([saved, replay]), axis=0), 0.02)
    a, b = saved / scale, replay / scale
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    previous = np.zeros((n + 1, m + 1), dtype=np.uint8)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            local = float(np.sum((a[i - 1] - b[j - 1]) ** 2))
            options = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            move = int(np.argmin(options))
            cost[i, j] = local + options[move]
            previous[i, j] = move + 1
    if not np.isfinite(cost[n, m]):
        raise ValueError(f"no monotone qpos alignment for saved/replay lengths {n}/{m}")
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        move = int(previous[i, j])
        if move == 1:
            i, j = i - 1, j - 1
        elif move == 2:
            i -= 1
        elif move == 3:
            j -= 1
        else:
            raise ValueError(f"invalid DTW backtrace at {(i, j)}")
    path.reverse()
    matches: list[list[int]] = [[] for _ in range(n)]
    for saved_i, replay_i in path:
        matches[saved_i].append(replay_i)
    mapped = np.asarray([
        int(np.median(row)) if row else -1 for row in matches
    ], dtype=np.int64)
    return mapped, matches, float(cost[n, m]), band


def annotate_episode(task: str, episode_dir: Path, scenes_by_seed: dict,
                     env_loader, collector, args) -> dict:
    pair_path = episode_dir / "pairs.npz"
    meta_path = episode_dir / "metadata.json"
    replay_path = episode_dir / "stage_replay.json"
    stage_file = episode_dir / "stage_labels.npz"
    if replay_path.is_file():
        previous = json.loads(replay_path.read_text())
        current_stage_hash = sha256(stage_file) if stage_file.is_file() else None
        if (previous.get("complete") is True
                and previous.get("pair_file_sha256") == sha256(pair_path)
                and previous.get("stage_file_sha256") == current_stage_hash):
            return previous
    metadata = json.loads(meta_path.read_text())
    with np.load(pair_path, allow_pickle=False) as source:
        states = source["state"].astype(np.float32)
        expert_chunks = source["expert_chunks"].astype(np.float32)
        sampled_state_index = source["sampled_state_index"].astype(np.int64)
        sim_step = source["sim_step"].astype(np.int64)
    n = len(states)
    if metadata.get("expert_success") is not True or metadata.get("complete_pairs") != n:
        raise ValueError(f"refusing incomplete or unsuccessful episode {episode_dir}")

    embedded = embedded_stage_labels(metadata, pair_path, n)
    if embedded is not None:
        stage_id, stage_name, idx, steps, labels = embedded
        if not np.array_equal(idx, sampled_state_index) or not np.array_equal(steps, sim_step):
            raise ValueError(f"embedded stage labels misaligned in {episode_dir}")
        state_error = 0.0
        chunk_error = 0.0
        replay_mode = "online_instrumented_expert_subplan_callsite"
        alignment_errors = np.zeros(n, dtype=np.float32)
        replay_success = True
        stage_scheme = SCHEME
    else:
        seed = int(metadata["scene_seed"])
        if seed not in scenes_by_seed:
            raise ValueError(f"scene seed {seed} not found in frozen expert scene cache")
        match = re.match(r"episode_(\d+)_scene_", episode_dir.name)
        if not match:
            raise ValueError(f"unexpected episode directory name: {episode_dir.name}")
        episode_index = int(match.group(1))
        scene = {**scenes_by_seed[seed], "task": task}
        env, env_args = env_loader(task)
        try:
            env.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **env_args)
            replay_args = SimpleNamespace(
                physics_stride=5, anchor_stride=1, horizon=10, capture_images=False,
            )
            states50, anchors, counters, _, _, replay_success, source_labels = collector._collect_expert_trace(
                env, scene, env_args, replay_args, episode_index
            )
        finally:
            env.close_env()
        states50 = np.asarray(states50, dtype=np.float32)
        saved_states = reconstruct_saved_qpos(pair_path)
        mapped, path_matches, _dtw_cost, dtw_band = monotone_qpos_dtw(saved_states, states50)
        replay_ids = np.asarray([int(row["stage_id"]) for row in anchors], dtype=np.int32)
        replay_names = np.asarray([str(row["stage_name"]) for row in anchors], dtype=str)
        if len(replay_ids) != len(states50) or len(replay_names) != len(states50):
            raise ValueError(f"replay stage trace length mismatch in {episode_dir}")
        stage_id = np.zeros(n, dtype=np.int32)
        stage_name = np.full(n, "unverified_alignment", dtype="<U128")
        alignment_errors = np.full(n, np.inf, dtype=np.float32)
        for row, state_index in enumerate(sampled_state_index):
            index = int(state_index)
            if index >= len(mapped) or mapped[index] < 0:
                continue
            replay_index = int(mapped[index])
            alignment_errors[row] = float(np.max(np.abs(saved_states[index] - states50[replay_index])))
            matched_stages = {int(replay_ids[j]) for j in path_matches[index]}
            matched_names = {str(replay_names[j]) for j in path_matches[index]}
            if (replay_success and len(matched_stages) == 1 and len(matched_names) == 1
                    and next(iter(matched_stages)) > 0
                    and alignment_errors[row] <= DTW_QPOS_MAX_ABS_ERROR):
                stage_id[row] = next(iter(matched_stages))
                stage_name[row] = next(iter(matched_names))
        labels = {str(key): str(value) for key, value in source_labels.items()}
        labels = {key: value for key, value in labels.items()
                  if value and value not in {"unlabeled", "pre_subplan"} and ":" in value}
        positive_ids = set(int(value) for value in np.unique(stage_id) if int(value) > 0)
        labels = {key: value for key, value in labels.items() if int(key) in positive_ids}
        if np.any(stage_id == 0):
            labels["0"] = "unverified_alignment"
        state_error = float(np.max(alignment_errors[np.isfinite(alignment_errors)])) if np.isfinite(alignment_errors).any() else None
        chunk_error = None  # Replay is time-warped; future action chunks are not claimed to match.
        replay_mode = "monotone_qpos_dtw_expert_replay_approx"
        stage_scheme = DTW_SCHEME
        atomic_npz(episode_dir / "stage_labels.npz", stage_id=stage_id,
                   stage_name=stage_name, sampled_state_index=sampled_state_index,
                   sim_step=sim_step, alignment_qpos_abs_error=alignment_errors,
                   verified_mask=(stage_id > 0))

    manifest = {
        "complete": True,
        "task": task,
        "episode": episode_dir.name,
        "scene_seed": int(metadata["scene_seed"]),
        "scheme": stage_scheme,
        "mode": replay_mode,
        "pair_file_sha256": sha256(pair_path),
        "stage_file_sha256": sha256(episode_dir / "stage_labels.npz") if (episode_dir / "stage_labels.npz").exists() else None,
        "matched_anchors": n,
        "stage_count": len(set(int(v) for v in stage_id if int(v) > 0)),
        "stage_labels": labels,
        "max_qpos_abs_error": state_error,
        "max_expert_chunk_abs_error": chunk_error,
        "stage_ids": sorted(set(int(v) for v in stage_id if int(v) > 0)),
        "verified_stage_anchors": int(np.sum(stage_id > 0)),
        "unverified_stage_anchors": int(np.sum(stage_id == 0)),
        "verified_stage_anchor_fraction": float(np.mean(stage_id > 0)),
        "alignment_qpos_abs_error_p50_p95_max": (
            [float(np.percentile(alignment_errors[np.isfinite(alignment_errors)], p))
             for p in (50, 95, 100)] if np.isfinite(alignment_errors).any() else None
        ),
        "dtw_scaled_squared_cost": (None if embedded is not None else _dtw_cost),
        "dtw_band": (None if embedded is not None else dtw_band),
        "replay_success": replay_success,
        "replay_label_scheme": (SCHEME if embedded is not None else DTW_SCHEME),
        "alignment_method": ("exact_online" if embedded is not None else "monotone_qpos_dtw"),
        "alignment_error_threshold_rad": (None if embedded is not None else DTW_QPOS_MAX_ABS_ERROR),
        "time_warped_expert_chunk_equality_claimed": False,
    }
    atomic_json(episode_dir / "stage_replay.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--pace-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    args = parser.parse_args()

    task_root = args.task_root.resolve()
    collection_state = json.loads((task_root / "state.json").read_text())
    summary = json.loads((task_root / "summary.json").read_text())
    if collection_state.get("status") != "complete" or summary.get("complete") is not True:
        raise ValueError(f"pair collection is incomplete for {args.task}")
    episodes = sorted(task_root.glob("episode_*_scene_*"))
    if len(episodes) != 30:
        raise ValueError(f"expected 30 episode directories, found {len(episodes)}")
    seeds = [int(json.loads((episode / "metadata.json").read_text())["scene_seed"])
             for episode in episodes]
    if len(set(seeds)) != 30:
        raise ValueError(f"duplicate scene seeds in {args.task}")

    sys.path.insert(0, str(args.pace_root.resolve() / "src"))
    sys.path.insert(0, str(args.robotwin_root.resolve()))
    sys.path.insert(0, str(args.robotwin_root.resolve() / "description/utils"))
    from pace.robotwin import load_cached_scenes, load_environment

    cache_seed = int(collection_state.get("scene_seed_split", 0))
    scene_count = int(collection_state.get("scene_offset", 0)) + int(
        collection_state.get("candidate_scenes", 0))
    scenes, cache_meta = load_cached_scenes(args.task, cache_seed, scene_count)
    scenes_by_seed = {int(scene["scene_seed"]): scene for scene in scenes}
    if not set(seeds).issubset(scenes_by_seed):
        raise ValueError(f"frozen scene cache does not contain all collected seeds for {args.task}")
    collector = load_collector(args.collector.resolve())
    replay_args = SimpleNamespace(physics_stride=5, anchor_stride=1, horizon=10,
                                  capture_images=False)
    results = []
    for episode in episodes:
        results.append(annotate_episode(args.task, episode, scenes_by_seed,
                                        load_environment, collector, replay_args))
        print(json.dumps({"task": args.task, "episode": episode.name,
                          "mode": results[-1]["mode"],
                          "matched_anchors": results[-1]["matched_anchors"],
                          "stage_count": results[-1]["stage_count"]}), flush=True)
    manifest = {
        "complete": True,
        "task": args.task,
        "episodes": len(results),
        "unique_scene_seeds": len(set(seeds)),
        "scheme": SCHEME,
        "scene_cache": cache_meta,
        "episodes_verified": results,
    }
    atomic_json(task_root / "stage_annotation_manifest.json", manifest)
    print(json.dumps({"complete": True, "task": args.task,
                      "episodes": len(results), "manifest": str(task_root / "stage_annotation_manifest.json")}))


if __name__ == "__main__":
    main()
