#!/usr/bin/env python3
"""Extract Ubuntu HG-DAgger raw episodes into balanced LeRobot sources.

The collector stores one pickle per saved observation and a ``control_mask`` in
``episode.json``.  This script keeps only adjacent frames from the same
controller segment, so an action never jumps across a policy/HIL boundary and
the end of a file is never fabricated into a self-transition.

By default the policy-controlled segments are written as a local ``baseline``
source and the human-controlled segments as ``dagger``.  For a real SFT
baseline, pass ``--baseline-root`` pointing at an existing LeRobot dataset;
then only the DAgger source is created and the manifest records that external
baseline.  The default policy-segment baseline is useful for an Ubuntu smoke
experiment, but it is not a replacement for the original SFT demonstrations.

The output also contains mixed-source normalization statistics.  State and
chunked-action statistics use equal source probability, independent of the
number of frames in each source, matching ``train_iwr_balanced_5050.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Any, Iterable

import numpy as np


ACTION_DIM = 14
DEFAULT_CONFIG = "pi05_robotwin_handover_to_tray_v2_promptfix"
DEFAULT_ASSET_ID = "iwr_hg_dagger_balanced_5050"
CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


@dataclass(frozen=True)
class Episode:
    root: Path
    episode_index: int
    instruction: str
    mask: tuple[str, ...]
    frames: tuple[Path, ...]
    metadata: dict[str, Any]


def _parse_image_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"image size must be HxW, got {value!r}")
    height, width = (int(part) for part in parts)
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    return height, width


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_episode(root: Path, metadata_path: Path) -> Episode:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mask = tuple(str(value).strip().lower() for value in metadata.get("control_mask", []))
    if not mask:
        raise ValueError(f"{metadata_path}: missing control_mask")
    if any(value not in {"policy", "hil"} for value in mask):
        raise ValueError(f"{metadata_path}: control_mask contains a value other than policy/hil")

    frames_dir = metadata_path.parent / "frames"
    frame_paths = sorted(frames_dir.glob("*.pkl"), key=lambda path: int(path.stem))
    expected = list(range(len(mask)))
    actual = [int(path.stem) for path in frame_paths]
    if actual != expected:
        raise ValueError(f"{metadata_path}: expected frame files {expected[:3]}...{expected[-3:]}, got {actual[:3]}...")
    instruction = str(metadata.get("instruction", "")).strip()
    if not instruction:
        raise ValueError(f"{metadata_path}: missing instruction")
    return Episode(
        root=root,
        episode_index=int(metadata.get("episode_index", int(metadata_path.parent.name.split("_")[-1]))),
        instruction=instruction,
        mask=mask,
        frames=tuple(frame_paths),
        metadata=metadata,
    )


def discover_episodes(raw_roots: Iterable[Path], limit: int | None) -> list[Episode]:
    episodes: list[Episode] = []
    for raw_root in raw_roots:
        raw_root = raw_root.expanduser().resolve()
        paths = sorted(raw_root.glob("episode_*/episode.json"), key=lambda path: int(path.parent.name.split("_")[-1]))
        if not paths:
            raise FileNotFoundError(f"No raw episodes under {raw_root}")
        for metadata_path in paths:
            episodes.append(_read_episode(raw_root, metadata_path))
            if limit is not None and len(episodes) >= limit:
                return episodes
    return episodes


def _vector(frame: dict[str, Any], key: str) -> np.ndarray:
    joint_action = frame.get("joint_action", {})
    value = joint_action.get("vector")
    if value is None:
        pieces = [
            joint_action.get("left_arm"),
            joint_action.get("left_gripper"),
            joint_action.get("right_arm"),
            joint_action.get("right_gripper"),
        ]
        if any(piece is None for piece in pieces):
            raise ValueError(f"{key}: frame has no 14-D joint_action vector")
        value = np.concatenate([np.asarray(piece, dtype=np.float32).reshape(-1) for piece in pieces])
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"{key}: expected a 14-D absolute joint vector, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{key}: non-finite joint vector")
    return vector


def _image(frame: dict[str, Any], camera_name: str, image_size: tuple[int, int]) -> np.ndarray:
    import cv2

    camera = frame.get("observation", {}).get(camera_name, {})
    value = camera.get("rgb")
    if value is None:
        raise ValueError(f"missing RGB image for {camera_name}")
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{camera_name}: expected HxWx3 RGB image, got {image.shape}")
    image = image.astype(np.uint8, copy=False)
    height, width = image_size
    if image.shape[:2] != image_size:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _add_frame(dataset: Any, frame: dict[str, Any], instruction: str) -> None:
    try:
        dataset.add_frame(frame, task=instruction)
    except TypeError:
        dataset.add_frame({**frame, "task": instruction})


def _save_episode(dataset: Any, instruction: str) -> None:
    try:
        dataset.save_episode(task=instruction)
    except TypeError:
        dataset.save_episode()


def _make_dataset(output_root: Path, repo_id: str, image_size: tuple[int, int], use_videos: bool) -> Any:
    """Create a LeRobot dataset after setting its home to the requested output."""
    os.environ["HF_LEROBOT_HOME"] = str(output_root)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    height, width = image_size
    motors = [
        *(f"left_arm_joint_{index}" for index in range(6)),
        "left_gripper",
        *(f"right_arm_joint_{index}" for index in range(6)),
        "right_gripper",
    ]
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (ACTION_DIM,), "names": [motors]},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": [motors]},
    }
    for camera in CAMERA_MAP:
        features[f"observation.images.{camera}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    kwargs = {
        "repo_id": repo_id,
        "fps": 15,
        "robot_type": "aloha_agilex",
        "features": features,
        "use_videos": use_videos,
        "tolerance_s": 0.0001,
        "image_writer_processes": 0,
        "image_writer_threads": 4,
        "video_backend": "pyav" if use_videos else None,
    }
    try:
        return LeRobotDataset.create(**kwargs, root=output_root / repo_id)
    except TypeError as exc:
        if "unexpected keyword argument 'root'" not in str(exc):
            raise
        return LeRobotDataset.create(**kwargs)


def _segment_runs(mask: tuple[str, ...]) -> Iterable[tuple[int, int, str]]:
    start = 0
    for index in range(1, len(mask) + 1):
        if index == len(mask) or mask[index] != mask[start]:
            yield start, index, mask[start]
            start = index


def extract_source(
    episodes: list[Episode],
    datasets: dict[str, Any],
    *,
    image_size: tuple[int, int],
    min_segment_frames: int,
) -> dict[str, int]:
    counts = {"baseline": 0, "dagger": 0, "discarded_short_segments": 0, "skipped_policy_segments": 0, "episodes": 0}
    for episode in episodes:
        frames = [pickle.loads(path.read_bytes()) for path in episode.frames]
        for start, end, source in _segment_runs(episode.mask):
            target = "baseline" if source == "policy" else "dagger"
            transition_count = end - start - 1
            if target not in datasets:
                counts["skipped_policy_segments"] += 1
                continue
            if transition_count < min_segment_frames:
                counts["discarded_short_segments"] += 1
                continue
            dataset = datasets[target]
            for index in range(start, end - 1):
                current = frames[index]
                following = frames[index + 1]
                row = {
                    "observation.state": _vector(current, f"{episode.root}:{episode.episode_index}:{index}:state"),
                    "action": _vector(following, f"{episode.root}:{episode.episode_index}:{index}:action"),
                }
                for output_camera, input_camera in CAMERA_MAP.items():
                    row[f"observation.images.{output_camera}"] = _image(current, input_camera, image_size)
                _add_frame(dataset, row, episode.instruction)
                counts[target] += 1
            _save_episode(dataset, episode.instruction)
            counts["episodes"] += 1
    return counts


def _read_matrix(table: Any, key: str, path: Path) -> np.ndarray:
    values = np.asarray(table[key].to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError(f"{path}: {key} shape {values.shape}, expected [N,14]")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite values in {key}")
    return values


def _source_files(root: Path) -> list[Path]:
    # LeRobot also writes ``meta/*.parquet`` (tasks and episode metadata).
    # Those files are not frame tables and must never enter norm-stat scans.
    # Restrict the search to the dataset's data tree and require a regular
    # file so a partially written/metadata path cannot be mistaken for data.
    data_root = root / "data"
    if not data_root.is_dir():
        return []
    files = sorted(path for path in data_root.rglob("*.parquet") if path.is_file())
    return files


def _equal_sample_indices(lengths: list[int], budget: int) -> list[np.ndarray]:
    total = sum(lengths)
    if total == 0 or budget <= 0:
        return [np.empty(0, dtype=np.int64) for _ in lengths]
    budget = min(total, budget)
    if budget == total:
        return [np.arange(length, dtype=np.int64) for length in lengths]
    positions = np.linspace(0, total - 1, budget, dtype=np.int64)
    boundaries = np.cumsum(np.asarray(lengths, dtype=np.int64))
    file_ids = np.searchsorted(boundaries, positions, side="right")
    starts = np.concatenate(([0], boundaries[:-1]))
    return [(positions[file_ids == i] - starts[i]).astype(np.int64) for i in range(len(lengths))]


def compute_mixed_norm_stats(baseline_root: Path, dagger_root: Path, output_dir: Path, horizon: int, budget: int) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from openpi.shared import normalize

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    sampled: dict[str, int] = {}
    for source, root in (("baseline", baseline_root), ("dagger", dagger_root)):
        files = _source_files(root)
        if not files:
            raise FileNotFoundError(f"no LeRobot parquet files under {root}")
        lengths = [int(pq.ParquetFile(path).metadata.num_rows) for path in files]
        indices_by_file = _equal_sample_indices(lengths, budget)
        sampled[source] = int(sum(len(indices) for indices in indices_by_file))
        for path, indices in zip(files, indices_by_file, strict=True):
            if len(indices) == 0:
                continue
            table = pq.read_table(path, columns=["observation.state", "action"])
            state = _read_matrix(table, "observation.state", path)
            action = _read_matrix(table, "action", path)
            state_stats.update(state[indices])
            offsets = np.arange(horizon, dtype=np.int64)
            action_indices = np.minimum(indices[:, None] + offsets[None, :], len(action) - 1)
            action_stats.update(action[action_indices].reshape(-1, ACTION_DIM))
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(output_dir, {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()})
    return {"frames_per_source": sampled, "action_horizon": horizon, "distribution": "equal source probability"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", action="append", type=Path, required=True, help="Raw HIL root; repeat for multiple collection rounds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, help="Existing SFT LeRobot root; omit to extract policy segments as baseline")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--norm-stats-asset-id", default=DEFAULT_ASSET_ID)
    parser.add_argument("--assets-base-dir", type=Path)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--stats-frames-per-source", type=int, default=100_000)
    parser.add_argument("--min-segment-frames", type=int, default=2)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--image-size", default="240x320")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action_horizon <= 0 or args.stats_frames_per_source < 2 or args.min_segment_frames < 2:
        raise ValueError("action horizon/stats budget must be positive and segments need at least two frames")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"non-empty output directory: {output}; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    image_size = _parse_image_size(args.image_size)
    episodes = discover_episodes(args.raw_root, args.episode_limit)

    datasets: dict[str, Any] = {}
    baseline_root: Path
    baseline_mode: str
    if args.baseline_root is None:
        datasets["baseline"] = _make_dataset(output, "baseline", image_size, not args.no_videos)
        baseline_root = output / "baseline"
        baseline_mode = "policy_segments_from_same_hil_rollouts"
    else:
        baseline_root = args.baseline_root.expanduser().resolve()
        if not (baseline_root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"baseline LeRobot root is missing meta/info.json: {baseline_root}")
        baseline_mode = "external_lerobot"
    datasets["dagger"] = _make_dataset(output, "dagger", image_size, not args.no_videos)
    counts = extract_source(episodes, datasets, image_size=image_size, min_segment_frames=args.min_segment_frames)

    # LeRobot 0.4 keeps parquet writers open until ``finalize``.  Close both
    # the frame and metadata writers before PyArrow scans the files for norm
    # statistics; otherwise the parquet footer is not present yet.
    for dataset in datasets.values():
        finalize = getattr(dataset, "finalize", None)
        if finalize is not None:
            finalize()

    if "baseline" not in datasets:
        # Keep the generated DAgger dataset under output while using the
        # caller-supplied SFT root for the equal-probability statistics.
        dagger_root = output / "dagger"
    else:
        dagger_root = output / "dagger"
    assets_base = (args.assets_base_dir or (output / "assets")).expanduser().resolve()
    asset_dir = assets_base / args.config_name / args.norm_stats_asset_id
    stats = compute_mixed_norm_stats(baseline_root, dagger_root, asset_dir, args.action_horizon, args.stats_frames_per_source)
    manifest = {
        "schema_version": 1,
        "action_semantics": "14d_absolute_joint_target",
        "transition_semantics": "state_t_to_next_saved_frame_joint_target_within_same_control_segment",
        "baseline": {"root": str(baseline_root), "mode": baseline_mode},
        "dagger": {"root": str(dagger_root), "mode": "hil_segments"},
        "mix": {"baseline_fraction": 0.5, "dagger_fraction": 0.5, "baseline_loss_weight": 1.0, "dagger_loss_weight": 1.0},
        "source_roots": [str(path.expanduser().resolve()) for path in args.raw_root],
        "episodes": len(episodes),
        "counts": counts,
        "norm_stats": {"asset_dir": str(asset_dir), **stats},
        "limitations": [
            "The default baseline is policy-controlled rollout data, not original SFT demonstrations.",
            "The final frame of every control segment is excluded because it has no same-segment successor.",
            "The next saved joint target is an approximate transition at the collection save frequency.",
        ],
    }
    (output / "iwr_balanced_mix.json").write_text(json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
