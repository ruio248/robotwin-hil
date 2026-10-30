"""Verify finished seed-1 pair postprocessing and write its transfer manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path


HASHED_NAMES = {
    "pairs.npz", "metadata.json", "stage_labels.npz", "stage_replay.json",
    "policy_candidates_k4.npz", "policy_candidates_k4.json",
    "features_r3m18.npz", "r3m_feature_manifest.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("tasks", nargs="+")
    args = parser.parse_args()
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("duplicate task names")

    rows = []
    for task in args.tasks:
        task_root = args.root / f"{task}_seed1_n30"
        state = json.loads((task_root / "state.json").read_text())
        summary = json.loads((task_root / "summary.json").read_text())
        features = json.loads((task_root / "r3m_feature_manifest.json").read_text())
        assert state.get("status") == "complete" and state.get("scene_seed_split") == 1, task
        assert summary.get("complete") is True and summary.get("episodes") == 30, task
        assert len(set(summary.get("scene_seeds", []))) == 30, task
        assert features.get("complete") is True and features.get("episodes") == 30, task
        hashes = {}
        for path in sorted(task_root.rglob("*")):
            if path.is_file() and path.name in HASHED_NAMES:
                hashes[str(path.relative_to(task_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(hashes) >= 30 * 3, (task, len(hashes))
        rows.append({"task": task, "episode_count": 30, "scene_seed_split": 1, "files_sha256": hashes})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps({"complete": True, "tasks": rows}, indent=2) + "\n")
    os.replace(tmp, args.output)


if __name__ == "__main__":
    main()
