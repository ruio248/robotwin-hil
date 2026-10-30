#!/usr/bin/env python3
"""Recompute OpenPI mixed norm stats without crossing LeRobot episodes.

The RoboTwin training transform stores 14-D absolute joint targets on disk,
converts the 12 arm joints to ``action - current state`` before Normalize, and
keeps both grippers absolute. This utility mirrors that transform and the
10-step action windows used by the loader. Source datasets are mixed 50:50 by
using the same number of anchor frames from each source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ACTION_DIM = 14
ARM_ACTION_MASK = np.asarray([True] * 6 + [False] + [True] * 6 + [False])


def _read_episode_tables(root: Path) -> list[dict[str, Any]]:
    """Load and validate frame tables, grouping rows by episode_index."""
    import pyarrow.parquet as pq

    files = sorted(path for path in (root / "data").rglob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no LeRobot frame parquet files under {root / 'data'}")

    pieces: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for path in files:
        table = pq.read_table(
            path,
            columns=["observation.state", "action", "episode_index", "frame_index"],
        )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        episode_ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_ids = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        if states.shape != (len(table), ACTION_DIM) or actions.shape != (len(table), ACTION_DIM):
            raise ValueError(f"{path}: expected state/action shape [N,14], got {states.shape}/{actions.shape}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"{path}: state/action contains non-finite values")
        for episode_id in np.unique(episode_ids):
            mask = episode_ids == episode_id
            pieces.setdefault(int(episode_id), []).append(
                (frame_ids[mask], states[mask], actions[mask])
            )

    episodes: list[dict[str, Any]] = []
    for episode_id in sorted(pieces):
        groups = pieces[episode_id]
        frame_ids = np.concatenate([group[0] for group in groups])
        states = np.concatenate([group[1] for group in groups])
        actions = np.concatenate([group[2] for group in groups])
        order = np.argsort(frame_ids, kind="stable")
        frame_ids, states, actions = frame_ids[order], states[order], actions[order]
        if len(np.unique(frame_ids)) != len(frame_ids):
            raise ValueError(f"episode_index={episode_id}: duplicate frame_index values")
        if len(frame_ids) > 1 and not np.all(np.diff(frame_ids) == 1):
            raise ValueError(f"episode_index={episode_id}: frame_index is not contiguous")
        episodes.append(
            {"episode_index": episode_id, "frame_index": frame_ids, "state": states, "action": actions}
        )
    return episodes


def _sample_equal_source_anchors(
    episodes_by_source: dict[str, list[dict[str, Any]]], budget: int
) -> dict[str, list[tuple[int, int]]]:
    """Select an equal number of uniformly spread anchors from both sources."""
    counts = {name: sum(len(ep["state"]) for ep in episodes) for name, episodes in episodes_by_source.items()}
    if len(counts) != 2 or min(counts.values()) <= 0:
        raise ValueError(f"expected two non-empty sources, got frame counts {counts}")
    take = min(budget, *counts.values())
    selected: dict[str, list[tuple[int, int]]] = {}
    for source, episodes in episodes_by_source.items():
        positions = np.linspace(0, counts[source] - 1, num=take, dtype=np.int64)
        lengths = np.asarray([len(ep["state"]) for ep in episodes], dtype=np.int64)
        boundaries = np.cumsum(lengths)
        episode_slots = np.searchsorted(boundaries, positions, side="right")
        starts = np.concatenate(([0], boundaries[:-1]))
        selected[source] = [
            (int(ep_slot), int(position - starts[ep_slot]))
            for ep_slot, position in zip(episode_slots, positions, strict=True)
        ]
    return selected


def action_window_delta(episode: dict[str, Any], anchor: int, horizon: int) -> np.ndarray:
    """Return training-space [H,14] action window, clamped only within episode."""
    states = episode["state"]
    actions = episode["action"]
    if not 0 <= anchor < len(states):
        raise IndexError(f"anchor {anchor} outside episode of length {len(states)}")
    indices = np.minimum(anchor + np.arange(horizon, dtype=np.int64), len(actions) - 1)
    window = actions[indices].copy()
    window[:, ARM_ACTION_MASK] -= states[anchor, ARM_ACTION_MASK]
    return window


def build_mixed_samples(
    episodes_by_source: dict[str, list[dict[str, Any]]], *, horizon: int, budget: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    anchors = _sample_equal_source_anchors(episodes_by_source, budget)
    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    counts: dict[str, Any] = {}
    for source, episodes in episodes_by_source.items():
        source_anchors = anchors[source]
        states = np.stack([episodes[ep_index]["state"][frame_index] for ep_index, frame_index in source_anchors])
        windows = np.stack(
            [action_window_delta(episodes[ep_index], frame_index, horizon) for ep_index, frame_index in source_anchors]
        )
        state_parts.append(states)
        action_parts.append(windows.reshape(-1, ACTION_DIM))
        counts[source] = {
            "episodes": len(episodes),
            "frames": sum(len(ep["state"]) for ep in episodes),
            "sampled_anchor_frames": len(source_anchors),
            "sampled_action_rows": int(windows.shape[0] * windows.shape[1]),
        }
    return np.concatenate(state_parts), np.concatenate(action_parts), counts


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _stats_to_json(stats: Any) -> dict[str, Any]:
    return {
        field: _jsonable(getattr(stats, field))
        for field in ("mean", "std", "q01", "q99")
    }


def compute_and_save(args: argparse.Namespace) -> dict[str, Any]:
    from openpi.shared import normalize

    source_roots = {"baseline": args.baseline_root.resolve(), "dagger": args.dagger_root.resolve()}
    episodes_by_source = {name: _read_episode_tables(root) for name, root in source_roots.items()}
    states, actions, counts = build_mixed_samples(
        episodes_by_source, horizon=args.horizon, budget=args.frames_per_source
    )
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    state_stats.update(states)
    action_stats.update(actions)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(output_dir, {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()})

    provenance = {
        "schema_version": 1,
        "action_semantics_on_disk": "14d_absolute_joint_target",
        "action_transform_for_stats": "12 arm joints: action - current observation.state; grippers absolute",
        "action_horizon": args.horizon,
        "episode_boundary_policy": "group by episode_index; clamp tail windows to the final action within that episode; never cross episodes",
        "source_distribution": "exactly 50:50 by sampled anchor frames; same anchor count per source",
        "frames_per_source_budget": args.frames_per_source,
        "sampled": counts,
        "sources": {name: str(root) for name, root in source_roots.items()},
        "state_summary": _stats_to_json(state_stats.get_statistics()),
        "action_summary": _stats_to_json(action_stats.get_statistics()),
    }
    (output_dir / "norm_stats_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.manifest_out is not None:
        manifest = {
            "schema_version": 3,
            "action_semantics": "14d_absolute_joint_target",
            "training_action_transform": provenance["action_transform_for_stats"],
            "episode_boundary_policy": provenance["episode_boundary_policy"],
            "baseline": {"repo_id": args.baseline_repo_id, "root": str(source_roots["baseline"])},
            "dagger": {"repo_id": args.dagger_repo_id, "root": str(source_roots["dagger"])},
            "mix": {
                "baseline_fraction": 0.5,
                "dagger_fraction": 0.5,
                "baseline_loss_weight": 1.0,
                "dagger_loss_weight": 1.0,
            },
            "norm_stats_asset_id": args.norm_stats_asset_id,
            "norm_stats_dir": str(output_dir),
            "action_horizon": args.horizon,
            "sampled": counts,
        }
        args.manifest_out.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.resolve().write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--dagger-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--baseline-repo-id", default="ruio248/robotwin_handover_to_tray_v2_promptfix")
    parser.add_argument("--dagger-repo-id", default="iwr_hg_dagger_corrected_compat")
    parser.add_argument("--norm-stats-asset-id", default="iwr_hg_dagger_episode_safe_v1")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--frames-per-source", type=int, default=100_000)
    args = parser.parse_args()
    if args.horizon <= 0 or args.frames_per_source <= 0:
        parser.error("--horizon and --frames-per-source must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(compute_and_save(parse_args()), indent=2, ensure_ascii=False), flush=True)
