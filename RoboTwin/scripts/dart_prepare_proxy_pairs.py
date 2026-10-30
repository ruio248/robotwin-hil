"""Prepare an explicitly *proxy* first-iteration DART residual cohort.

The cached SFT 'expert' action is the next recorded qpos, NOT the clean
intended supervisor command. This cohort may calibrate scales but must not
be represented as a genuine DART supervisor-action dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError("cache manifest is not complete")
    if manifest["config"]["action_semantics"] != "14D absolute joint targets; expert action[t] equals next recorded state":
        raise ValueError("unexpected expert action semantics")
    if manifest["config"]["source_fps"] != 15 or manifest["config"]["num_candidates"] != 4:
        raise ValueError("unexpected cache frequency or candidate count")
    expert_parts, learner_parts, offsets = [], [], [0]
    episode_ids = []
    for entry in sorted(manifest["episodes"], key=lambda item: item["file"]):
        if entry["split"] != "train":
            continue
        path = args.cache / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"cache hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            keep = np.asarray(data["valid_transition"], dtype=bool)
            expert = np.asarray(data["expert_chunks"], dtype=np.float32)
            learner = np.asarray(data["policy_chunks"], dtype=np.float32)
        if expert.ndim != 3 or expert.shape[1:] != (10, 14):
            raise ValueError(f"unexpected expert chunk shape in {path}")
        if learner.shape != (len(expert), 4, 10, 14) or keep.shape != (len(expert),):
            raise ValueError(f"unexpected policy/valid shape in {path}")
        if not keep.any():
            continue
        expert_parts.append(expert[keep, 0, :])
        learner_parts.append(learner[keep, 0, 0, :])  # first saved draw, same observation
        offsets.append(offsets[-1] + int(keep.sum()))
        episode_ids.append(entry["id"])
    expert_all = np.concatenate(expert_parts)
    learner_all = np.concatenate(learner_parts)
    if len(episode_ids) != 405 or len(expert_all) != 11461:
        raise ValueError(f"unexpected train cohort: {len(episode_ids)} episodes, {len(expert_all)} anchors")
    if not np.isfinite(expert_all).all() or not np.isfinite(learner_all).all():
        raise ValueError("nonfinite actions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, expert_intended_actions=expert_all,
                        learner_actions=learner_all,
                        episode_offsets=np.asarray(offsets, dtype=np.int64))
    metadata = {
        "status": "proxy_only_not_true_dart_intended_actions",
        "source_manifest_sha256": sha256(manifest_path),
        "source_fps": 15,
        "policy_candidate_index": 0,
        "action_timestep": 0,
        "episodes": len(episode_ids),
        "anchors": len(expert_all),
        "output_sha256": sha256(args.output),
        "limitation": "expert target is next recorded qpos, not clean scripted supervisor command",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
