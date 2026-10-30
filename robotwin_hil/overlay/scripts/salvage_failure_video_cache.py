#!/usr/bin/env python3
"""Finish exporting a completed RoboTwin rollout whose metadata write failed."""

from __future__ import annotations

import argparse
from pathlib import Path

from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--save-frequency", type=int, required=True)
    cli = parser.parse_args()
    cache = cli.episode_dir / ".cache/episode0"
    frames = list(cache.glob("*.pkl"))
    if len(frames) < 2 or not (cache / f"{len(frames) - 1}.pkl").is_file():
        raise RuntimeError(f"cache missing complete ordered frames: {cache}")
    data = cli.episode_dir / "data/episode_0000000.hdf5"
    video = cli.episode_dir / "video/episode_0000000.mp4"
    if data.exists() or video.exists():
        raise RuntimeError("refusing to overwrite an existing export")
    data.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    process_folder_to_hdf5_video(
        str(cache), str(data), str(video), instructions=[cli.prompt],
        frequency=cli.save_frequency, episode_metadata={"export_recovered_from_cache": True},
        save_video=True,
    )
    print(f"exported {len(frames)} cached samples to {video}", flush=True)


if __name__ == "__main__":
    main()
