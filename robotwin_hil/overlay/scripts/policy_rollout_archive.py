"""Archive policy-controlled frames separately from expert HIL labels.

Each uninterrupted policy segment is stored on its own.  Joining policy
frames across a human-control gap would invent nonexistent transitions.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def policy_runs(control_mask: list[str]) -> list[list[int]]:
    runs: list[list[int]] = []
    for index, source in enumerate(control_mask):
        if str(source).lower() != "policy":
            continue
        if not runs or index != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(index)
    return runs


def archive_policy_rollout(
    cache: Path | None,
    output_dir: Path,
    *,
    rollout_index: int,
    seed: int,
    control_mask: list[str],
    save_freq: int,
    sampling_mode: str,
    autonomous_success: bool,
) -> Path | None:
    """Save complete policy runs as raw frames; never label them as expert.

    Returns None when there are fewer than two contiguous policy frames.
    This archive is opt-in and is not part of the HIL training dataset.
    """
    runs = [run for run in policy_runs(control_mask) if len(run) >= 2]
    if not runs:
        return None
    if cache is None or not cache.is_dir():
        raise FileNotFoundError(f"missing policy frame cache: {cache}")
    available = sorted(int(path.stem) for path in cache.glob("*.pkl"))
    if available != list(range(len(control_mask))):
        raise ValueError("policy frame cache and control mask do not align")
    target = output_dir / "policy_rollouts" / f"rollout_{rollout_index:07d}_seed_{seed}"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite policy archive: {target}")
    target.mkdir(parents=True)
    segment_metadata: list[dict[str, Any]] = []
    for segment_index, indices in enumerate(runs):
        frames = target / f"segment_{segment_index:03d}" / "frames"
        frames.mkdir(parents=True)
        for new_index, source_index in enumerate(indices):
            source = cache / f"{source_index}.pkl"
            destination = frames / f"{new_index}.pkl"
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        segment_metadata.append({
            "segment_index": segment_index,
            "source_frame_indices": indices,
            "frames": len(indices),
            "transitions": len(indices) - 1,
        })
    metadata = {
        "schema_version": 1,
        "source": "policy_rollout_not_expert_or_hil",
        "rollout_index": int(rollout_index),
        "seed": int(seed),
        "save_freq": int(save_freq),
        "sampling_mode": sampling_mode,
        "autonomous_success": bool(autonomous_success),
        "control_mask": list(control_mask),
        "segments": segment_metadata,
    }
    (target / "rollout.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target
