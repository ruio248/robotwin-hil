#!/usr/bin/env python3
"""Offline export of raw HG-DAgger episodes to HDF5 / MP4.

Collection writes only ``raw/episode_*/frames/*.pkl`` plus ``episode.json``.
This script turns those raw episodes into the native RoboTwin HDF5 (and
optionally an MP4), either as the full episode or as a HIL-only view.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))

from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "hil"], default="full")
    parser.add_argument("--save-video", default="false")
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_frames(frames_dir: Path, control_mask: list[Any], mode: str, staging_dir: Path) -> Path:
    """Return the frame directory to merge (a renumbered copy for HIL-only)."""
    if mode == "full":
        return frames_dir

    staging_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        index
        for index, source in enumerate(control_mask)
        if str(source).lower() == "hil"
    ]
    for new_index, old_index in enumerate(selected):
        source = frames_dir / f"{old_index}.pkl"
        if not source.is_file():
            continue
        target = staging_dir / f"{new_index}.pkl"
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return staging_dir


def main() -> int:
    cli = parse_args()
    cli.raw_root = cli.raw_root.expanduser().resolve()
    cli.output_dir = cli.output_dir.expanduser().resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    save_video = parse_bool(cli.save_video)

    episode_dirs = sorted(path for path in cli.raw_root.iterdir() if path.is_dir())
    if not episode_dirs:
        raise SystemExit(f"no raw episodes under {cli.raw_root}")

    manifest: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        meta_path = episode_dir / "episode.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frames_dir = episode_dir / "frames"
        if not frames_dir.is_dir():
            continue

        episode_index = int(meta.get("episode_index", len(manifest)))
        instruction = str(meta.get("instruction", "Pass the red bar from the left arm to the right arm and place it in the blue tray."))
        frequency = int(meta.get("save_freq", 15) or 15)
        control_mask = list(meta.get("control_mask", []))

        staging = cli.output_dir / ".staging" / f"episode_{episode_index:07d}_{cli.mode}"
        if staging.exists():
            shutil.rmtree(staging)
        merge_frames = stage_frames(frames_dir, control_mask, cli.mode, staging)

        suffix = "" if cli.mode == "full" else "_hil"
        episode_name = f"episode_{episode_index:07d}{suffix}"
        hdf5_path = cli.output_dir / "data" / f"{episode_name}.hdf5"
        video_path = cli.output_dir / "video" / f"{episode_name}.mp4"
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.parent.mkdir(parents=True, exist_ok=True)

        process_folder_to_hdf5_video(
            str(merge_frames),
            str(hdf5_path),
            str(video_path),
            instructions=[instruction],
            frequency=frequency,
            episode_metadata=meta.get("episode_metadata"),
            save_video=save_video,
        )
        write_json(
            cli.output_dir / "instruction" / f"{episode_name}.json",
            {
                "episode_index": episode_index,
                "seen": [instruction],
                "unseen": [instruction],
                "supervisor_label": meta.get("supervisor_label"),
                "segments": meta.get("segments"),
                "control_mask": control_mask,
                "mode": cli.mode,
            },
        )
        manifest.append(
            {
                "episode_index": episode_index,
                "mode": cli.mode,
                "seed": meta.get("seed"),
                "supervisor_label": meta.get("supervisor_label"),
                "hdf5": str(hdf5_path),
                "video": str(video_path) if save_video else None,
            }
        )
        print(
            f"[EXPORT] episode={episode_index} mode={cli.mode} "
            f"hdf5={hdf5_path.name} video={save_video}",
            flush=True,
        )

    write_json(
        cli.output_dir / f"manifest_{cli.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        {"mode": cli.mode, "episodes": manifest},
    )
    print(f"exported={len(manifest)} mode={cli.mode} output={cli.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
