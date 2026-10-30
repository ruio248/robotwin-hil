#!/usr/bin/env python3
"""Opt-in lossless query trace for the published fixed-H50 RoboTwin evaluator.

Copy this file beside ``pace/scripts/evaluate.py`` and invoke it with the same
arguments.  It delegates scene selection, actions, and outcome accounting to
the unchanged evaluator while saving policy-visited observations for a later
held-out D(o, a) outcome check.  Traces are diagnostics, never training HIL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import evaluate as original


_run_episode = original.run_episode
_evaluate = original.evaluate
_trace_root: Path | None = None
_episode = 0


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _save_query(root: Path, index: int, observation: dict, actions: np.ndarray) -> None:
    state = np.asarray(observation["state"], dtype=np.float32)
    action = np.asarray(actions[:10], dtype=np.float32)
    if state.shape != (14,) or action.shape != (10, 14):
        raise ValueError(f"invalid D trace state/action shape: {state.shape}/{action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("nonfinite D trace state/action")
    stem = f"query_{index:04d}"
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        chw = np.asarray(observation["images"][camera])
        if chw.ndim != 3 or chw.shape[0] != 3 or chw.dtype != np.uint8:
            raise ValueError(f"invalid camera input {camera}: {chw.shape}/{chw.dtype}")
        target = root / f"{stem}_{camera}.png"
        temporary = root / f"{stem}_{camera}.tmp.png"
        Image.fromarray(chw.transpose(1, 2, 0), mode="RGB").save(temporary, format="PNG")
        os.replace(temporary, target)
    target = root / f"{stem}.npz"
    temporary = root / f"{stem}.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, state=state, policy_h10=action)
    os.replace(temporary, target)


def traced_run_episode(env, policy, choose_horizon, on_query=None):
    global _episode
    assert _trace_root is not None
    episode_index = _episode
    root = _trace_root / f"episode_{episode_index:03d}"
    root.mkdir(parents=True, exist_ok=False)
    queries: list[dict] = []

    class RecordingPolicy:
        def infer(self, observation):
            response = policy.infer(observation)
            actions = np.asarray(response["actions"])
            index = len(queries)
            _save_query(root, index, observation, actions)
            queries.append({"query": index, "prompt": str(observation["prompt"]),
                            "planned_chunk_length": len(actions)})
            return response

    def record(step, actions, horizon):
        if not queries:
            raise ValueError("query callback without recorded observation")
        queries[-1].update(step=int(step), executed_horizon=int(horizon))
        if on_query is not None:
            on_query(step, actions, horizon)

    result = _run_episode(env, RecordingPolicy(), choose_horizon, record)
    if len(queries) != result["queries"] or any("step" not in row for row in queries):
        raise ValueError("query trace does not match evaluator count")
    _atomic_json(root / "trace.json", {
        "complete": True, "diagnostic_only": True, "accepted_for_training": False,
        "episode": episode_index, "success": result["success"],
        "steps": result["steps"], "queries": queries,
        "scoring_action": "first H10 of the published H50 policy prediction at each exact observation",
    })
    _episode += 1
    return result


def traced_evaluate(args):
    global _trace_root, _episode
    if args.method != "fixed" or args.horizon != 50 or not args.ultrafast:
        raise ValueError("outcome trace requires fixed H50 and published cached scenes")
    _trace_root = args.output / "trace"
    _episode = 0
    _evaluate(args)
    metadata = json.loads((args.output / "metadata.json").read_text())
    episodes = [json.loads(line) for line in (args.output / "episodes.jsonl").read_text().splitlines()]
    if metadata["status"] != "complete" or len(episodes) != args.episodes:
        raise ValueError("evaluation incomplete after trace collection")
    for row in episodes:
        path = _trace_root / f"episode_{row['episode']:03d}" / "trace.json"
        trace = json.loads(path.read_text())
        if trace["success"] != row["success"] or len(trace["queries"]) != row["queries"]:
            raise ValueError(f"trace/outcome mismatch: {path}")
        trace["scene_seed"] = row["scene_seed"]
        trace["instruction"] = row["instruction"]
        _atomic_json(path, trace)


if __name__ == "__main__":
    original.run_episode = traced_run_episode
    original.evaluate = traced_evaluate
    original.main()
