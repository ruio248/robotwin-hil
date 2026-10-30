#!/usr/bin/env python3
"""Convert one validated IWR native source with the existing RoboTwin converter.

The standard converter normally discovers inputs under RoboTwin/data/.  This
wrapper points discovery at an explicit native expert or HIL directory and
refuses to overwrite an existing LeRobot dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-data-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-episodes", type=int, default=30)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--resolution", default="240x320")
    args = parser.parse_args()

    native_data = args.native_data_dir.expanduser().resolve()
    files = sorted(native_data.glob("episode_*.hdf5"))
    if len(files) != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} native episodes; found {len(files)}")
    if args.expected_episodes < 1 or args.fps < 1:
        raise ValueError("episode count and fps must be positive")

    from XPolicyLab.scripts import transform_lerobot_v21_format as converter

    destination = converter.HF_LEROBOT_HOME / args.repo_id
    if destination.exists():
        raise FileExistsError(f"refusing converter overwrite: {destination}")
    target = ("iwr_native", "handover_to_tray", "aloha_agilex")
    converter._discover_conversion_targets = lambda patterns: [target]
    converter._resolve_input_dir = lambda *parts: native_data
    sys.argv = [
        str(Path(__file__)), ".".join(target),
        "--repo_id", args.repo_id,
        "--max_episode", str(args.expected_episodes),
        "--fps", str(args.fps),
        "--resolution", args.resolution,
    ]
    converter.main()

    episode_index = destination / "meta" / "episodes.jsonl"
    info_path = destination / "meta" / "info.json"
    if not episode_index.is_file() or not info_path.is_file():
        raise RuntimeError(f"conversion did not produce metadata at {destination}")
    episodes = sum(bool(line.strip()) for line in episode_index.open(encoding="utf-8"))
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if episodes != args.expected_episodes or int(info["fps"]) != args.fps:
        raise ValueError(f"converted episode/fps mismatch: {episodes=} {info.get('fps')=}")
    print(f"[LEROBOT] validated {episodes} episodes, fps={info['fps']}: {destination}")


if __name__ == "__main__":
    main()
