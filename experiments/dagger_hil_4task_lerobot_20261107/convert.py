#!/usr/bin/env python3
"""Convert saved DAgger HIL spans to an auditable LeRobot v3 dataset.

The diagnostic variant includes every saved HIL span and MUST NOT be used for
training. The eligible variant follows the recorded training acceptance gate.
No policy-controlled frames or cross-controller transitions are exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("diagnostic_all", "eligible"), required=True)
    parser.add_argument("--one-segment-per-source", action="store_true",
                        help="Keep one HIL segment per saved source episode: accepted first, then longest")
    parser.add_argument("--expected-raw-episodes", type=int, default=15)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "robotwin_hil" / "overlay" / "scripts"))
    from dagger.recording import accepted_hil_spans, training_eligible
    from prepare_hg_dagger_dataset import (
        CAMERA_MAP, _add_frame, _image, _make_dataset, _read_episode,
        _save_episode, _segment_runs, _vector,
    )

    raw = args.raw_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    paths = sorted(raw.glob("episode_*/episode.json"))
    if len(paths) != args.expected_raw_episodes:
        raise ValueError(f"Expected {args.expected_raw_episodes} raw episodes, found {len(paths)}")
    episodes = [_read_episode(raw, path) for path in paths]
    if len({ep.episode_index for ep in episodes}) != len(episodes):
        raise ValueError("Duplicate raw episode indices")
    seeds = [ep.metadata.get("seed") for ep in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate raw seeds")

    # v3 dataset creation is exclusive; this path is dedicated to one mode.
    output.mkdir(parents=True)
    dataset = _make_dataset(output, "dagger", (240, 320), True)
    rows = []
    total_frames = 0
    for ep in episodes:
        all_spans = [(start, end) for start, end, source in _segment_runs(ep.mask) if source == "hil"]
        eligible = training_eligible(raw / f"episode_{ep.episode_index:07d}", ep.metadata)
        allowed = set(accepted_hil_spans(ep.metadata, all_spans)) if eligible else set()
        candidates = [(start, end) for start, end in all_spans if end - start - 1 >= 2]
        chosen = (max(candidates, key=lambda span: (
            span in allowed, span[1] - span[0] - 1, span[0],
        )) if args.one_segment_per_source and candidates else None)
        frames = None
        for start, end in all_spans:
            transitions = end - start - 1
            selected = (transitions >= 2
                        and (chosen is None or (start, end) == chosen)
                        and (args.mode == "diagnostic_all" or (eligible and (start, end) in allowed)))
            row = {
                "source_episode_index": ep.episode_index,
                "seed": ep.metadata.get("seed"),
                "start_frame": start,
                "end_frame_exclusive": end,
                "transitions": transitions,
                "record_accepted_for_training": eligible,
                "record_accepted_segment": (start, end) in allowed,
                "exported": selected,
            }
            if selected:
                if frames is None:
                    frames = [pickle.loads(path.read_bytes()) for path in ep.frames]
                for index in range(start, end - 1):
                    current, following = frames[index], frames[index + 1]
                    record = {
                        "observation.state": _vector(current, f"{raw}:{ep.episode_index}:{index}:state"),
                        "action": _vector(following, f"{raw}:{ep.episode_index}:{index}:action"),
                    }
                    for output_camera, input_camera in CAMERA_MAP.items():
                        record[f"observation.images.{output_camera}"] = _image(current, input_camera, (240, 320))
                    if not np.isfinite(record["observation.state"]).all() or not np.isfinite(record["action"]).all():
                        raise ValueError(f"Non-finite vector in {ep.episode_index}:{index}")
                    _add_frame(dataset, record, ep.instruction)
                _save_episode(dataset, ep.instruction)
                total_frames += transitions
            rows.append(row)
    dataset.finalize()
    exported = [row for row in rows if row["exported"]]
    source_count = len({row["source_episode_index"] for row in exported})
    if not exported:
        raise ValueError("No qualifying HIL segments")
    info = json.loads((output / "dagger" / "meta" / "info.json").read_text())
    if int(info["total_episodes"]) != len(exported) or int(info["total_frames"]) != total_frames:
        raise ValueError(f"LeRobot metadata count mismatch: {info['total_episodes']=} {info['total_frames']=}")
    videos = sorted((output / "dagger" / "videos").rglob("*.mp4"))
    # LeRobot v3 packs many episodes into each per-camera MP4 file.
    expected_video_files = len(CAMERA_MAP) * math.ceil(len(exported) / int(info["chunks_size"]))
    if len(videos) != expected_video_files:
        raise ValueError(f"Expected {expected_video_files} packed camera videos, found {len(videos)}")
    manifest = {
        "mode": args.mode,
        "training_usable": args.mode == "eligible",
        "warning": "diagnostic_all contains rejected expert-motion records; NEVER train on it" if args.mode == "diagnostic_all" else None,
        "source_raw": str(raw),
        "raw_episode_count": len(episodes),
        "source_episodes_exported": source_count,
        "lerobot_episodes": len(exported),
        "transitions": total_frames,
        "fps": 15,
        "camera_keys": list(CAMERA_MAP),
        "video_files": len(videos),
        "min_segment_transitions": 2,
        "one_segment_per_source": args.one_segment_per_source,
        "segment_selection_rule": "record-accepted first, then most transitions, then later start" if args.one_segment_per_source else "all qualifying spans",
        "segmentation": "one continuous HIL span per LeRobot episode; no policy frames or boundary crossing",
        "rows": rows,
    }
    manifest_path = output / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(json.dumps({"output": str(output), "source_episodes": source_count, "lerobot_episodes": len(exported), "transitions": total_frames, "videos": len(videos), "manifest_sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
