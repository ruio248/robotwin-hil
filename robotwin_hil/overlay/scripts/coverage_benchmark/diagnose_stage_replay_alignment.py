#!/usr/bin/env python3
"""Read-only validation of qpos-DTW stage-label recovery for paired episodes.

This diagnostic never modifies pair data. It replays selected expert scenes,
reconstructs the stored 50 Hz qpos path from overlapping H=10 chunks, aligns
the two paths monotonically, and reports whether expert-subplan labels remain
unambiguous at the stored pair anchors.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct_saved_qpos(pair_path: Path):
    with np.load(pair_path, allow_pickle=False) as source:
        anchors = source["state"].astype(np.float32)
        anchor_indices = source["sampled_state_index"].astype(np.int64)
        chunks = source["expert_chunks"].astype(np.float32)
        sim_steps = source["sim_step"].astype(np.int64)
    horizon = int(chunks.shape[1])
    length = int(anchor_indices[-1] + horizon + 1)
    states = np.full((length, anchors.shape[1]), np.nan, dtype=np.float32)
    for anchor, index, chunk in zip(anchors, anchor_indices, chunks, strict=True):
        index = int(index)
        for state_index, value in [(index, anchor), *enumerate(chunk, start=index + 1)]:
            previous = states[state_index]
            if np.isfinite(previous).all() and not np.array_equal(previous, value):
                raise ValueError(f"overlapping stored qpos chunks disagree at {state_index}")
            states[state_index] = value
    valid = np.isfinite(states).all(axis=1)
    if not valid.all():
        # The final observed states can be absent because no complete H=10
        # target remained. Interior gaps would indicate corrupt pair data.
        first_gap = int(np.flatnonzero(~valid)[0])
        if valid[first_gap:].any():
            raise ValueError(f"interior qpos reconstruction gap at {first_gap}")
        states = states[:first_gap]
    return states, anchor_indices, sim_steps


def dtw_map(saved: np.ndarray, replay: np.ndarray, band: int):
    n, m = len(saved), len(replay)
    band = max(int(band), abs(n - m) + 8)
    scale = np.maximum(np.std(np.concatenate([saved, replay]), axis=0), 0.02)
    a, b = saved / scale, replay / scale
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    previous = np.zeros((n + 1, m + 1), dtype=np.uint8)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lower, upper = max(1, i - band), min(m, i + band)
        for j in range(lower, upper + 1):
            local = float(np.sum((a[i - 1] - b[j - 1]) ** 2))
            options = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            move = int(np.argmin(options))
            cost[i, j] = local + options[move]
            previous[i, j] = move + 1
    if not np.isfinite(cost[n, m]):
        raise ValueError(f"DTW path not found (n={n}, m={m}, band={band})")
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
            raise ValueError(f"broken DTW backtrace at {(i, j)}")
    path.reverse()
    mapping: list[list[int]] = [[] for _ in range(n)]
    for saved_i, replay_i in path:
        mapping[saved_i].append(replay_i)
    mapped = np.asarray([
        int(np.median(row)) if row else -1 for row in mapping
    ], dtype=np.int64)
    return mapped, mapping, float(cost[n, m]), band


def run_episode(task: str, episode_dir: Path, root: Path, pace_root: Path,
                collector_path: Path, band: int):
    collector = load_module(collector_path, "diagnostic_collector")
    sys.path[:0] = [str(pace_root / "src"), str(root), str(root / "description/utils")]
    from pace.robotwin import load_cached_scenes, load_environment

    metadata = json.loads((episode_dir / "metadata.json").read_text())
    seed = int(metadata["scene_seed"])
    episode_index = int(episode_dir.name.split("_scene_")[0].removeprefix("episode_"))
    saved, anchor_indices, sim_steps = reconstruct_saved_qpos(episode_dir / "pairs.npz")
    collection_state = json.loads((episode_dir.parent / "state.json").read_text())
    scene_count = int(collection_state.get("scene_offset", 0)) + int(
        collection_state["candidate_scenes"]
    )
    scenes, _ = load_cached_scenes(task, 0, scene_count)
    scene = next((row for row in scenes if int(row["scene_seed"]) == seed), None)
    if scene is None:
        raise ValueError(f"scene seed {seed} not in frozen cache")
    env, env_args = load_environment(task)
    started = time.monotonic()
    try:
        env.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **env_args)
        replay_args = SimpleNamespace(
            physics_stride=5, anchor_stride=1, horizon=10, capture_images=False,
        )
        replay, anchors, counters, _prompt, _info, success, labels = collector._collect_expert_trace(
            env, {**scene, "task": task}, env_args, replay_args, episode_index,
        )
    finally:
        env.close_env()
    replay = np.asarray(replay, dtype=np.float32)
    mapped, path_by_saved, total_cost, actual_band = dtw_map(saved, replay, band)
    stage_by_replay = np.asarray([int(row["stage_id"]) for row in anchors], dtype=np.int64)
    name_by_id = {int(row["stage_id"]): str(row["stage_name"]) for row in anchors}
    pair_labels, ambiguous = [], []
    for index in anchor_indices:
        path_indices = path_by_saved[int(index)]
        stage_set = {int(stage_by_replay[j]) for j in path_indices}
        if len(stage_set) == 1 and next(iter(stage_set)) > 0:
            pair_labels.append(next(iter(stage_set)))
            ambiguous.append(False)
        else:
            pair_labels.append(0)
            ambiguous.append(True)
    mapped_safe = np.clip(mapped, 0, len(replay) - 1)
    errors = np.max(np.abs(saved - replay[mapped_safe]), axis=1)
    transition_indices = [
        int(i) for i in range(1, len(pair_labels))
        if pair_labels[i] != pair_labels[i - 1] and pair_labels[i] > 0 and pair_labels[i - 1] > 0
    ]
    replay_transitions = [
        int(i) for i in range(1, len(stage_by_replay))
        if stage_by_replay[i] != stage_by_replay[i - 1]
    ]
    transition_alignment = []
    for row in transition_indices:
        previous_stage, next_stage = int(pair_labels[row - 1]), int(pair_labels[row])
        candidates = [
            i for i in replay_transitions
            if int(stage_by_replay[i - 1]) == previous_stage
            and int(stage_by_replay[i]) == next_stage
        ]
        sample_index = int(anchor_indices[row])
        mapped_index = int(mapped[sample_index])
        nearest = min(candidates, key=lambda i: abs(i - mapped_index)) if candidates else None
        nearby = errors[max(0, sample_index - 5):min(len(errors), sample_index + 6)]
        transition_alignment.append({
            "pair_row": int(row),
            "saved_sample_index": sample_index,
            "mapped_replay_index": mapped_index,
            "replay_stage_transition_index": nearest,
            "transition_offset_samples": (mapped_index - nearest) if nearest is not None else None,
            "stage_from": previous_stage,
            "stage_to": next_stage,
            "qpos_abs_error_at_anchor": float(errors[sample_index]),
            "qpos_abs_error_near_boundary_max": float(np.max(nearby)),
        })
    return {
        "task": task,
        "episode": episode_dir.name,
        "scene_seed": seed,
        "replay_success": bool(success),
        "saved_samples_reconstructed": int(len(saved)),
        "replay_samples": int(len(replay)),
        "saved_sim_steps_first_last": [int(sim_steps[0]), int(sim_steps[-1])],
        "replay_physics_steps": int(counters["physics_steps"]),
        "stage_labels": name_by_id,
        "stage_transition_indices_replay": [
            int(i) for i in range(1, len(stage_by_replay))
            if stage_by_replay[i] != stage_by_replay[i - 1]
        ],
        "stage_transition_indices_saved_pair_rows": transition_indices,
        "stage_transition_alignment": transition_alignment,
        "ambiguous_pair_anchors": int(sum(ambiguous)),
        "pair_anchor_count": int(len(anchor_indices)),
        "unambiguous_pair_anchor_fraction": float(1.0 - np.mean(ambiguous)),
        "qpos_abs_error_percentiles": {
            str(p): float(np.percentile(errors, p)) for p in (50, 90, 95, 99, 100)
        },
        "dtw_total_scaled_squared_cost": total_cost,
        "dtw_band": actual_band,
        "elapsed_seconds": float(time.monotonic() - started),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--pace-root", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--episode-indices", nargs="+", type=int, default=[0, 15, 29])
    parser.add_argument("--band", type=int, default=24)
    args = parser.parse_args()
    reports = []
    for task in args.tasks:
        task_root = args.pair_root / f"{task}_seed0_n30"
        episodes = sorted(task_root.glob("episode_*_scene_*"))
        for index in args.episode_indices:
            if index >= len(episodes):
                raise ValueError(f"{task}: requested episode {index}, only {len(episodes)} available")
            report = run_episode(task, episodes[index], args.root, args.pace_root,
                                 args.collector, args.band)
            reports.append(report)
            print(json.dumps(report, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"complete": True, "reports": reports}, indent=2) + "\n")
    print(json.dumps({"complete": True, "episodes": len(reports), "output": str(args.output)}))


if __name__ == "__main__":
    main()
