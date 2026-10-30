"""Validate native RoboTwin handover_to_tray HDF5 demonstrations.

This is a pre-transfer gate. A report with any failed episode is a hard stop:
do not convert or upload the dataset until the issue is resolved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))

from data.decode_image_bit import decode_image_bit


JOINT_FIELDS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array[:, None] if array.ndim == 1 else array


def _episode_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def validate_episode(path: Path, scene_info: dict) -> dict:
    result = {"episode": _episode_index(path), "path": str(path), "ok": False, "errors": []}
    try:
        with h5py.File(path, "r") as handle:
            states = []
            actions = []
            for field in JOINT_FIELDS:
                state_path = f"state/{field}"
                action_path = f"action/{field}"
                if state_path not in handle or action_path not in handle:
                    result["errors"].append(f"missing {state_path} or {action_path}")
                    continue
                states.append(_as_2d(handle[state_path][...]))
                actions.append(_as_2d(handle[action_path][...]))

            if len(states) != len(JOINT_FIELDS):
                return result
            state = np.concatenate(states, axis=1)
            action = np.concatenate(actions, axis=1)
            if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
                result["errors"].append(f"state/action shapes must both be (T,14), got {state.shape}/{action.shape}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                result["errors"].append("non-finite state or action")
            if np.max(np.abs(state[:, [*range(6), *range(7, 13)]])) > np.pi + 1e-3:
                result["errors"].append("arm joint state outside +/- pi")
            if len(state) < 2:
                result["errors"].append("fewer than two aligned frames")
            elif not np.allclose(action[:-1], state[1:], rtol=1e-4, atol=1e-4):
                result["errors"].append("action[t] is not state[t+1]")

            for camera in CAMERAS:
                colors_path = f"vision/{camera}/colors"
                if colors_path not in handle:
                    result["errors"].append(f"missing {colors_path}")
                    continue
                colors = handle[colors_path]
                if len(colors) != len(state):
                    result["errors"].append(f"{camera} length {len(colors)} != state length {len(state)}")
                    continue
                for frame_idx in sorted({0, len(colors) // 2, len(colors) - 1}):
                    image = decode_image_bit(colors[frame_idx])
                    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                        result["errors"].append(f"{camera} frame {frame_idx} has invalid decoded shape/dtype")
                    elif int(image.max()) <= 5 or float(image.std()) <= 1.0:
                        result["errors"].append(f"{camera} frame {frame_idx} is black or near-constant")

            info = handle.get("additional_info")
            if info is None:
                result["errors"].append("missing additional_info")
            else:
                for key in ("stage_id", "seed", "episode_metadata_json"):
                    if key not in info:
                        result["errors"].append(f"missing additional_info/{key}")
                if "stage_id" in info:
                    stages = np.asarray(info["stage_id"][...], dtype=np.int32)
                    if len(stages) != len(state):
                        result["errors"].append("stage_id length mismatch")
                    elif not np.all((1 <= stages) & (stages <= 6)):
                        result["errors"].append("stage_id outside [1, 6]")
                    elif np.any(np.diff(stages) < 0):
                        result["errors"].append("stage_id is not monotonic")
                    elif set(range(1, 7)) - set(stages.tolist()):
                        result["errors"].append("stage_id does not cover all six stages")
                if "seed" in info:
                    seeds = np.asarray(info["seed"][...], dtype=np.int64)
                    if len(seeds) != len(state) or len(np.unique(seeds)) != 1:
                        result["errors"].append("seed metadata is not constant per episode")
                if "episode_metadata_json" in info:
                    raw = info["episode_metadata_json"][()]
                    raw = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    metadata = json.loads(raw)
                    result["metadata"] = metadata
                    if not metadata.get("success", False) or not metadata.get("plan_success", False):
                        result["errors"].append("episode metadata is not a successful planned trajectory")
                    if metadata.get("stage_count") != 6:
                        result["errors"].append("episode metadata stage_count is not six")

        scene_key = f"episode_{result['episode']}"
        scene_episode = scene_info.get(scene_key)
        if scene_episode is None:
            result["errors"].append(f"scene_info missing {scene_key}")
        else:
            snapshots = scene_episode.get("stage_snapshots", [])
            if [snapshot.get("stage_id") for snapshot in snapshots] != list(range(1, 7)):
                result["errors"].append("scene_info stage snapshots are not exactly [1..6]")
            task_metadata = scene_episode.get("task_metadata", {})
            if not task_metadata.get("success", False):
                result["errors"].append("scene_info marks the episode unsuccessful")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")

    result["ok"] = not result["errors"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help=".../<task_config>/handover_to_tray/aloha_agilex")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    files = sorted((root / "data").glob("episode_*.hdf5"), key=_episode_index)
    if not files:
        raise FileNotFoundError(f"No HDF5 episodes under {root / 'data'}")
    scene_info_path = root / "scene_info.json"
    if not scene_info_path.is_file():
        raise FileNotFoundError(scene_info_path)
    scene_info = json.loads(scene_info_path.read_text(encoding="utf-8"))
    results = [validate_episode(path, scene_info) for path in files]
    report = {
        "dataset_root": str(root),
        "episodes": len(results),
        "valid_episodes": sum(item["ok"] for item in results),
        "invalid_episodes": sum(not item["ok"] for item in results),
        "ok": all(item["ok"] for item in results),
        "results": results,
    }
    report_path = Path(args.report) if args.report else root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"report={report_path}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
