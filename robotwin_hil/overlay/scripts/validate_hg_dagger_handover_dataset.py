#!/usr/bin/env python3
"""Validate native RoboTwin HG-DAgger recovery-only episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


PROMPT = "Pass the red bar from the left arm to the right arm and place it in the blue tray."
JOINT_FIELDS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_text(value.item())
    return str(value)


def episode_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def as_2d(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array[:, None] if array.ndim == 1 else array


def validate_episode(path: Path, scene_info: dict) -> dict:
    index = episode_index(path)
    result = {"episode": index, "path": str(path), "ok": False, "errors": []}
    errors: list[str] = result["errors"]
    try:
        with h5py.File(path, "r") as handle:
            instructions = json.loads(decode_text(handle["instructions"][()]))
            if instructions != [PROMPT]:
                errors.append(f"prompt mismatch: {instructions!r}")

            states = []
            actions = []
            for field in JOINT_FIELDS:
                state_key = f"state/{field}"
                action_key = f"action/{field}"
                if state_key not in handle or action_key not in handle:
                    errors.append(f"missing {state_key} or {action_key}")
                    continue
                states.append(as_2d(handle[state_key][...]))
                actions.append(as_2d(handle[action_key][...]))
            if len(states) == len(JOINT_FIELDS):
                state = np.concatenate(states, axis=1)
                action = np.concatenate(actions, axis=1)
                if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
                    errors.append(f"invalid state/action shapes {state.shape}/{action.shape}")
                if len(state) < 2:
                    errors.append("fewer than two state/action pairs")
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    errors.append("non-finite state/action value")
                if len(state) >= 2 and not np.allclose(
                    action[:-1], state[1:], rtol=1e-4, atol=1e-4
                ):
                    errors.append("action[t] is not state[t+1]")
                arm_columns = [*range(6), *range(7, 13)]
                if np.max(np.abs(state[:, arm_columns])) > np.pi + 1e-3:
                    errors.append("arm joint state outside +/- pi")

                for camera in CAMERAS:
                    colors_key = f"vision/{camera}/colors"
                    if colors_key not in handle:
                        errors.append(f"missing {colors_key}")
                    elif len(handle[colors_key]) != len(state):
                        errors.append(f"{camera} frame count mismatch")

            additional = handle.get("additional_info")
            if additional is None:
                errors.append("missing additional_info")
            else:
                for key in ("stage_id", "seed", "episode_metadata_json"):
                    if key not in additional:
                        errors.append(f"missing additional_info/{key}")
                if "stage_id" in additional:
                    stages = np.asarray(additional["stage_id"][...], dtype=np.int32)
                    if np.any((stages < 1) | (stages > 6)):
                        errors.append("stage_id outside [1, 6]")
                    if np.any(np.diff(stages) < 0):
                        errors.append("recovery stage_id decreases")
                if "seed" in additional:
                    seeds = np.asarray(additional["seed"][...], dtype=np.int64)
                    if len(np.unique(seeds)) != 1:
                        errors.append("episode seed is not constant")
                if "episode_metadata_json" in additional:
                    metadata = json.loads(decode_text(additional["episode_metadata_json"][()]))
                    result["metadata"] = metadata
                    recovery = metadata.get("recovery", {})
                    hg = metadata.get("hg_dagger", {})
                    if not metadata.get("success") or not metadata.get("plan_success"):
                        errors.append("episode metadata is not successful")
                    if not recovery.get("success") or not recovery.get("plan_success"):
                        errors.append("expert recovery metadata is not successful")
                    if not hg.get("accepted_for_training"):
                        errors.append("HG-DAgger record is not accepted_for_training")
                    if hg.get("instruction") != PROMPT:
                        errors.append("HG-DAgger metadata prompt mismatch")

        scene_episode = scene_info.get(f"episode_{index}")
        if not isinstance(scene_episode, dict):
            errors.append(f"scene_info missing episode_{index}")
        else:
            if not scene_episode.get("recovery", {}).get("success"):
                errors.append("scene_info recovery is not successful")
            if not scene_episode.get("hg_dagger", {}).get("accepted_for_training"):
                errors.append("scene_info HG-DAgger record is not accepted")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    result["ok"] = not errors
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    files = sorted((root / "data").glob("episode_*.hdf5"), key=episode_index)
    if not files:
        raise FileNotFoundError(f"No HDF5 episodes under {root / 'data'}")
    indexes = [episode_index(path) for path in files]
    if indexes != list(range(indexes[-1] + 1)):
        raise ValueError(f"Episode indexes are not contiguous from zero: {indexes}")
    scene_info_path = root / "scene_info.json"
    if not scene_info_path.is_file():
        raise FileNotFoundError(scene_info_path)
    scene_info = json.loads(scene_info_path.read_text(encoding="utf-8"))
    results = [validate_episode(path, scene_info) for path in files]
    report = {
        "dataset_root": str(root),
        "episodes": len(results),
        "expected_episodes": args.expected_episodes,
        "valid_episodes": sum(item["ok"] for item in results),
        "invalid_episodes": sum(not item["ok"] for item in results),
        "ok": all(item["ok"] for item in results)
        and (args.expected_episodes is None or len(results) == args.expected_episodes),
        "results": results,
    }
    if args.expected_episodes is not None and len(results) != args.expected_episodes:
        report["count_error"] = (
            f"expected {args.expected_episodes} episodes, found {len(results)}"
        )
    report_path = args.report or root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"report={report_path}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
