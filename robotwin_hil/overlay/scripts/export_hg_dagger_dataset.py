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
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))

from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video  # noqa: E402
from dagger.recording import MIN_HIL_SEGMENT_TRANSITIONS, training_eligible  # noqa: E402


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


def is_diagnostic_episode(episode_dir: Path, meta: dict[str, Any]) -> bool:
    """Keep headless smoke and frozen test seeds out of training exports."""
    return not training_eligible(episode_dir, meta)


def validated_frame_count(frames_dir: Path, control_mask: list[Any], *, require_mask: bool = True) -> int:
    """Check contiguous frames; require aligned ownership labels for HIL export."""
    files = sorted(int(path.stem) for path in frames_dir.glob("*.pkl") if path.stem.isdigit())
    if not files or files != list(range(len(files))):
        raise ValueError(f"Non-contiguous raw frames in {frames_dir}")
    if require_mask and (len(control_mask) != len(files) or any(
        str(source).lower() not in {"hil", "policy"} for source in control_mask
    )):
        raise ValueError(f"Invalid control_mask/frame alignment in {frames_dir}")
    return len(files)


def hil_runs(control_mask: list[Any]):
    """Yield HIL runs with enough within-run action transitions for training."""
    start = None
    for index, source in enumerate([*control_mask, "policy"]):
        if source == "hil" and start is None:
            start = index
        elif source != "hil" and start is not None:
            if index - start - 1 >= MIN_HIL_SEGMENT_TRANSITIONS:
                yield start, index
            start = None


def stage_frames(frames_dir: Path, start: int, end: int, staging_dir: Path) -> Path:
    """Renumber one contiguous run while retaining original raw files."""
    staging_dir.mkdir(parents=True, exist_ok=False)
    for new_index, old_index in enumerate(range(start, end)):
        source = frames_dir / f"{old_index}.pkl"
        if not source.is_file():
            raise FileNotFoundError(source)
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
        if is_diagnostic_episode(episode_dir, meta):
            print(f"[EXPORT] {episode_dir.name} skipped (diagnostic-only)", flush=True)
            continue
        frames_dir = episode_dir / "frames"
        if not frames_dir.is_dir():
            continue

        episode_index = int(meta.get("episode_index", len(manifest)))
        instruction = str(meta.get("instruction", "Pass the red bar from the left arm to the right arm and place it in the blue tray."))
        frequency = int(meta.get("save_freq", 15) or 15)
        control_mask = list(meta.get("control_mask", []))

        frame_count = validated_frame_count(frames_dir, control_mask, require_mask=cli.mode == "hil")
        runs = list(hil_runs([str(source).lower() for source in control_mask])) if cli.mode == "hil" else [(0, frame_count)]
        if not runs:
            print(f"[EXPORT] episode={episode_index} mode=hil skipped "
                  f"(no segment with {MIN_HIL_SEGMENT_TRANSITIONS} transitions)", flush=True)
            continue
        for segment_index, (start, end) in enumerate(runs):
            suffix = "" if cli.mode == "full" else ("_hil" if len(runs) == 1 else f"_hil_s{segment_index:02d}")
            episode_name = f"episode_{episode_index:07d}{suffix}"
            hdf5_path = cli.output_dir / "data" / f"{episode_name}.hdf5"
            video_path = cli.output_dir / "video" / f"{episode_name}.mp4"
            instruction_path = cli.output_dir / "instruction" / f"{episode_name}.json"
            if hdf5_path.exists() or instruction_path.exists() or (save_video and video_path.exists()):
                raise FileExistsError(f"Refusing to overwrite an exported segment: {episode_name}")
            hdf5_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            instruction_path.parent.mkdir(parents=True, exist_ok=True)
            segment_metadata = dict(meta.get("episode_metadata") or {})
            segment_metadata.update(source_episode_index=episode_index, source_seed=meta.get("seed"),
                                    source_frame_start=start, source_frame_end_exclusive=end,
                                    source_control_mask=control_mask,
                                    transition_count=end - start - 1)
            with tempfile.TemporaryDirectory(prefix=f"{episode_name}_", dir=cli.output_dir) as temporary:
                temporary_dir = Path(temporary)
                staged_hdf5 = temporary_dir / hdf5_path.name
                staged_video = temporary_dir / video_path.name
                staged_instruction = temporary_dir / instruction_path.name
                merge_frames = (frames_dir if cli.mode == "full" else
                                stage_frames(frames_dir, start, end, temporary_dir / "frames"))
                converted = process_folder_to_hdf5_video(
                    str(merge_frames), str(staged_hdf5), str(staged_video),
                    instructions=[instruction], frequency=frequency,
                    episode_metadata=segment_metadata, save_video=save_video,
                )
                if converted != end - start - 1 or not staged_hdf5.is_file() or (save_video and not staged_video.is_file()):
                    raise ValueError(f"Incomplete conversion for {episode_name}: transitions={converted}")
                write_json(staged_instruction, {
                    "episode_index": episode_index, "seed": meta.get("seed"),
                    "seen": [instruction], "unseen": [instruction],
                    "supervisor_label": meta.get("supervisor_label"),
                    "segments": meta.get("segments"), "control_mask": control_mask,
                    "source_frame_start": start, "source_frame_end_exclusive": end,
                    "transition_count": end - start - 1, "mode": cli.mode,
                })
                published: list[Path] = []
                outputs = [(staged_hdf5, hdf5_path)]
                if save_video:
                    outputs.append((staged_video, video_path))
                outputs.append((staged_instruction, instruction_path))
                try:
                    for source, target in outputs:
                        os.link(source, target)  # fails rather than replacing an existing result
                        published.append(target)
                except Exception:
                    for path in reversed(published):
                        path.unlink()
                    raise
            manifest.append({
                "episode_index": episode_index, "segment_index": segment_index,
                "source_frame_start": start, "source_frame_end_exclusive": end,
                "transition_count": end - start - 1, "mode": cli.mode,
                "seed": meta.get("seed"), "instruction": instruction,
                "supervisor_label": meta.get("supervisor_label"),
                "hdf5": str(hdf5_path), "video": str(video_path) if save_video else None,
            })
            print(f"[EXPORT] episode={episode_index} segment={segment_index} mode={cli.mode} "
                  f"hdf5={hdf5_path.name} video={save_video}", flush=True)

    write_json(
        cli.output_dir / f"manifest_{cli.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        {"mode": cli.mode, "episodes": manifest},
    )
    print(f"exported={len(manifest)} mode={cli.mode} output={cli.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
