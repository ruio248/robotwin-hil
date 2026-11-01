"""Opt-in, query-aligned archive for the RoboTwin XPolicyLab eval loop.

This module does not choose scenes, call the policy, or advance the simulator.
An incomplete manifest is never a valid source of outcome labels for D.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class TraceRecordingError(RuntimeError):
    """Recording failed; the evaluation must stop instead of counting a policy failure."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


class OfficialPolicyTrace:
    """One archive per evaluation invocation; never overwrite a previous run."""

    def __init__(self, root: str | os.PathLike[str], *, run_info: Mapping[str, Any]):
        self.root = Path(root).expanduser().resolve()
        try:
            self.root.mkdir(parents=True, exist_ok=False)
            self.episodes_dir = self.root / "episodes"
            self.episodes_dir.mkdir()
            self.manifest: dict[str, Any] = {
                "format_version": 1,
                "status": "running",
                "d_analysis_ready": False,
                "run_info": dict(run_info),
                "episodes": [],
                "skipped_scenes": [],
            }
            self.active: dict[str, Any] | None = None
            _write_json(self.root / "manifest.json", self.manifest)
        except Exception as exc:
            raise TraceRecordingError(f"could not initialize trace at {self.root}: {exc}") from exc

    def skip_scene(self, *, scene_seed: int, stage: str, reason: str) -> None:
        try:
            self.manifest["skipped_scenes"].append({
                "scene_seed": int(scene_seed), "stage": stage, "reason": reason,
            })
            _write_json(self.root / "manifest.json", self.manifest)
        except Exception as exc:
            raise TraceRecordingError(f"could not record skipped scene {scene_seed}: {exc}") from exc

    def begin_episode(self, *, scene_seed: int, instruction: str, official_index: int) -> None:
        if self.active is not None:
            raise TraceRecordingError("previous trace episode was not finalized")
        directory = self.episodes_dir / f"episode_{len(self.manifest['episodes']):04d}_seed_{scene_seed}"
        try:
            directory.mkdir(exist_ok=False)
            self.active = {
                "scene_seed": int(scene_seed),
                "instruction": str(instruction),
                "official_episode_index": int(official_index),
                "status": "running",
                "queries": [],
                "directory": str(directory.relative_to(self.root)),
            }
            _write_json(directory / "episode.json", self.active)
        except Exception as exc:
            raise TraceRecordingError(f"could not begin trace episode {scene_seed}: {exc}") from exc

    def record_query(
        self,
        *,
        xpl_obs: Mapping[str, Any],
        absolute_joint_chunk: np.ndarray,
        policy_steps_before_query: int,
    ) -> None:
        if self.active is None:
            raise TraceRecordingError("no active trace episode")
        try:
            chunk = np.asarray(absolute_joint_chunk, dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != 14 or not len(chunk) or not np.isfinite(chunk).all():
                raise ValueError(f"expected finite [H,14] absolute-joint chunk, got {chunk.shape}")
            arrays: dict[str, np.ndarray] = {"absolute_joint_chunk": chunk}
            vision = xpl_obs["vision"]
            state = xpl_obs["state"]
            for camera, payload in vision.items():
                if "color" not in payload:
                    continue
                color = np.asarray(payload["color"])
                if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
                    raise ValueError(f"{camera} color must be uint8 HWC RGB, got {color.shape}/{color.dtype}")
                arrays[f"rgb_{camera}"] = color
            for required in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                if f"rgb_{required}" not in arrays:
                    raise ValueError(f"missing RGB input: {required}")
            for key, value in state.items():
                array = np.asarray(value, dtype=np.float32)
                if not np.isfinite(array).all():
                    raise ValueError(f"nonfinite state: {key}")
                arrays[f"state_{key}"] = array
            for required in (
                "left_arm_joint_state", "left_ee_joint_state",
                "right_arm_joint_state", "right_ee_joint_state",
            ):
                if f"state_{required}" not in arrays:
                    raise ValueError(f"missing state: {required}")

            query_index = len(self.active["queries"])
            filename = f"query_{query_index:04d}.npz"
            directory = self.root / self.active["directory"]
            temporary = directory / (filename + ".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, directory / filename)
            self.active["queries"].append({
                "query_index": query_index,
                "policy_steps_before_query": int(policy_steps_before_query),
                "returned_chunk_length": int(len(chunk)),
                "executed_actions": 0,
                "file": filename,
            })
            _write_json(directory / "episode.json", self.active)
        except Exception as exc:
            raise TraceRecordingError(f"could not record policy query: {exc}") from exc

    def action_executed(self) -> None:
        if self.active is None or not self.active["queries"]:
            raise TraceRecordingError("executed action has no recorded query")
        self.active["queries"][-1]["executed_actions"] += 1

    def finish_episode(
        self, *, success: bool, rollout_failed: bool, exception: str | None,
        rollout_steps: int, counted_in_official_result: bool,
    ) -> None:
        if self.active is None:
            raise TraceRecordingError("no active trace episode to finish")
        try:
            episode = self.active
            episode.update({
                "status": "infra_error" if rollout_failed else ("success" if success else "policy_failure"),
                "success": bool(success) if not rollout_failed else None,
                "rollout_failed": bool(rollout_failed),
                "exception": exception,
                "rollout_steps": int(rollout_steps),
                "counted_in_official_result": bool(counted_in_official_result),
                "d_eligible": not rollout_failed and bool(counted_in_official_result),
            })
            if sum(query["executed_actions"] for query in episode["queries"]) != rollout_steps:
                raise ValueError("recorded action count does not match rollout_steps")
            _write_json(self.root / episode["directory"] / "episode.json", episode)
            self.manifest["episodes"].append(episode)
            _write_json(self.root / "manifest.json", self.manifest)
            self.active = None
        except Exception as exc:
            raise TraceRecordingError(f"could not finalize trace episode: {exc}") from exc

    def finish_run(self, *, expected_count: int, official_success_count: int | None, error: str | None = None) -> None:
        try:
            episodes = self.manifest["episodes"]
            counted = [ep for ep in episodes if ep["counted_in_official_result"]]
            infra = [ep for ep in episodes if ep["rollout_failed"]]
            infra_scenes = [scene for scene in self.manifest["skipped_scenes"]
                            if scene["stage"] != "expert_rejected"]
            successful = [ep for ep in counted if ep["success"] is True]
            policy_failures = [ep for ep in counted if ep["status"] == "policy_failure"]
            queried = [ep for ep in counted if ep["queries"]]
            unique_seeds = len({ep["scene_seed"] for ep in counted}) == len(counted)
            matching_success = (official_success_count is not None and
                                len(successful) == official_success_count)
            complete = (error is None and self.active is None and len(counted) == expected_count
                        and len(queried) == len(counted) and unique_seeds and matching_success)
            self.manifest.update({
                "status": ("aborted" if not complete else
                           "infra_error" if infra or infra_scenes else "complete"),
                "d_analysis_ready": complete and not infra and not infra_scenes,
                "expected_official_episodes": int(expected_count),
                "counted_official_episodes": len(counted),
                "successful_episodes": len(successful),
                "policy_failure_episodes": len(policy_failures),
                "query_covered_official_episodes": len(queried),
                "infra_error_episodes": len(infra),
                "infra_error_skipped_scenes": len(infra_scenes),
                "official_success_count": official_success_count,
                "unique_counted_scene_seeds": unique_seeds,
                "success_count_matches_trace": matching_success,
                "error": error,
            })
            _write_json(self.root / "manifest.json", self.manifest)
        except Exception as exc:
            raise TraceRecordingError(f"could not finalize trace run: {exc}") from exc
