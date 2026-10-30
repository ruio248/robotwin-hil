"""Check the converted RoboTwin LeRobot training split before transfer."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def resolve_root(repo_id: str, dataset_root: str | None) -> Path:
    if dataset_root:
        return Path(dataset_root).expanduser().resolve()
    home = os.environ.get("HF_LEROBOT_HOME")
    if not home:
        raise ValueError("Pass --dataset-root or set HF_LEROBOT_HOME")
    return (Path(home) / repo_id).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="ruio248/robotwin_handover_to_tray_v1")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--expected-episodes", type=int, default=450)
    parser.add_argument("--expected-task", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = resolve_root(args.repo_id, args.dataset_root)
    report = {"root": str(root), "ok": False, "errors": []}
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        report["errors"].append(f"missing {info_path}")
    else:
        info = json.loads(info_path.read_text())
        report["episodes"] = int(info.get("total_episodes", -1))
        report["frames"] = int(info.get("total_frames", -1))
        features = info.get("features", {})
        for key in ("observation.state", "action"):
            shape = features.get(key, {}).get("shape")
            if shape != [14]:
                report["errors"].append(f"{key} shape must be [14], got {shape}")
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            if key not in features:
                report["errors"].append(f"missing feature {key}")
        if report["episodes"] != args.expected_episodes:
            report["errors"].append(
                f"expected {args.expected_episodes} train episodes, got {report['episodes']}"
            )

    if args.expected_task is not None:
        tasks_path = root / "meta" / "tasks.jsonl"
        episodes_path = root / "meta" / "episodes.jsonl"
        if not tasks_path.is_file() or not episodes_path.is_file():
            report["errors"].append("missing task or episode metadata")
        else:
            task_rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line]
            expected_rows = [{"task_index": 0, "task": args.expected_task}]
            if task_rows != expected_rows:
                report["errors"].append("tasks.jsonl does not contain the expected single task")
            episode_rows = [
                json.loads(line) for line in episodes_path.read_text().splitlines() if line
            ]
            bad_rows = sum(row.get("tasks") != [args.expected_task] for row in episode_rows)
            report["episode_task_rows_checked"] = len(episode_rows)
            if bad_rows:
                report["errors"].append(
                    f"{bad_rows} episode metadata rows do not contain the expected task"
                )

    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        report["errors"].append("no parquet data files")
    else:
        rng = random.Random(0)
        samples = parquet_files if len(parquet_files) <= 5 else rng.sample(parquet_files, 5)
        checked_rows = 0
        for path in samples:
            table = pq.read_table(path, columns=["observation.state", "action"])
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if state.ndim != 2 or state.shape[1] != 14 or action.shape != state.shape:
                report["errors"].append(f"{path}: invalid state/action shapes {state.shape}/{action.shape}")
            elif not np.isfinite(state).all() or not np.isfinite(action).all():
                report["errors"].append(f"{path}: non-finite state/action")
            checked_rows += len(state)
        report["sampled_parquet_files"] = [str(path) for path in samples]
        report["sampled_rows"] = checked_rows

    video_files = sorted((root / "videos").rglob("*.mp4"))
    if not video_files:
        report["errors"].append("no LeRobot video files")
    report["video_files"] = len(video_files)
    report["ok"] = not report["errors"]

    report_path = Path(args.report) if args.report else root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
