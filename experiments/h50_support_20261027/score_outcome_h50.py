#!/usr/bin/env python3
"""Score full H50 policy chunks on held-out rollout outcomes, one vote per episode."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18

from encode_pairs_h50 import image_tensor, r3m_state
from tail_value.chunk import ChunkCoverageCritic


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def auc(success: np.ndarray, failure: np.ndarray) -> float:
    if len(success) < 1 or len(failure) < 1:
        raise ValueError("AUC requires both outcomes")
    gap = success[:, None] - failure[None, :]
    return float(((gap > 0).sum() + .5 * (gap == 0).sum()) / gap.size)


def bootstrap(success: np.ndarray, failure: np.ndarray, seed: int) -> dict:
    if len(success) < 1 or len(failure) < 1:
        return {"auc": None, "ci95": None, "success_count": len(success), "failure_count": len(failure)}
    rng = np.random.default_rng(seed)
    draws = np.empty(10000)
    for i in range(len(draws)):
        draws[i] = auc(rng.choice(success, len(success)), rng.choice(failure, len(failure)))
    return {"auc": auc(success, failure), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "success_count": len(success), "failure_count": len(failure),
            "success_mean": float(success.mean()), "failure_mean": float(failure.mean()),
            "success_minus_failure": float(success.mean() - failure.mean())}


def rgb_feature(model: torch.nn.Module, paths: list[Path], device: torch.device) -> np.ndarray:
    blocks = []
    for start in range(0, len(paths), 32):
        images = []
        for path in paths[start:start + 32]:
            with Image.open(path) as image:
                images.append(image_tensor(np.asarray(image.convert("RGB"), dtype=np.uint8)))
        with torch.inference_mode():
            blocks.append(model(torch.stack(images).to(device)).float().cpu().numpy())
    return np.concatenate(blocks)


def score_episode(folder: Path, r3m: torch.nn.Module, critic: torch.nn.Module,
                  device: torch.device) -> tuple[np.ndarray, dict]:
    trace = json.loads((folder / "trace.json").read_text())
    if trace.get("complete") is not True or trace.get("scoring_action") != "complete H50 of the published policy prediction at each exact observation":
        raise ValueError(f"incomplete or wrong horizon trace: {folder}")
    q = len(trace["queries"])
    if q < 1:
        raise ValueError(f"empty trace: {folder}")
    states, actions = [], []
    for i in range(q):
        if trace["queries"][i]["query"] != i or trace["queries"][i]["planned_chunk_length"] != 50:
            raise ValueError(f"query index or output horizon mismatch: {folder}/{i}")
        with np.load(folder / f"query_{i:04d}.npz", allow_pickle=False) as source:
            state = source["state"].astype(np.float32)
            action = source["policy_h50"].astype(np.float32)
        if state.shape != (14,) or action.shape != (50, 14) or not np.isfinite(action).all():
            raise ValueError(f"bad H50 D input: {folder}/{i}")
        states.append(state)
        actions.append(action)
    feature_blocks = []
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        paths = [folder / f"query_{i:04d}_{camera}.png" for i in range(q)]
        feature_blocks.append(rgb_feature(r3m, paths, device))
    features = np.concatenate(feature_blocks, axis=1)
    with torch.inference_mode():
        scores = critic(torch.from_numpy(features).to(device),
                        torch.from_numpy(np.stack(states)).to(device),
                        torch.from_numpy(np.stack(actions)).to(device)).float().cpu().numpy()
    return scores, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--r3m-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "h50_same_observation_source_bce_v1" or payload.get("task") != args.task or payload.get("horizon") != 50:
        raise ValueError("D checkpoint protocol/task/horizon mismatch")
    critic = ChunkCoverageCritic(payload["normalization"], horizon=50, width=256)
    critic.load_state_dict(payload["model"], strict=True)
    critic = critic.to(device).eval()
    encoder = resnet18(weights=None)
    encoder.fc = torch.nn.Identity()
    encoder.load_state_dict(r3m_state(torch.load(args.r3m_asset, map_location="cpu", weights_only=True), encoder.state_dict()), strict=True)
    encoder = encoder.to(device).eval().requires_grad_(False)
    eval_root = args.eval_root / "evaluation"
    meta = json.loads((eval_root / "metadata.json").read_text())
    records = [json.loads(line) for line in (eval_root / "episodes.jsonl").read_text().splitlines()]
    if meta.get("status") != "complete" or len(records) != 50:
        raise ValueError("evaluation has not completed 50 episodes")
    seeds = [int(row["scene_seed"]) for row in records]
    if len(set(seeds)) != 50 or set(seeds).intersection(payload["scene_seeds"]):
        raise ValueError("test scenes are duplicated or overlap D training")
    rows = []
    for record in records:
        folder = eval_root / "trace" / f"episode_{record['episode']:03d}"
        scores, trace = score_episode(folder, encoder, critic, device)
        if int(trace["scene_seed"]) != int(record["scene_seed"]) or bool(trace["success"]) != bool(record["success"]):
            raise ValueError(f"outcome/trace mismatch: {folder}")
        rows.append({"scene_seed": int(record["scene_seed"]), "success": bool(record["success"]),
                     "queries": len(scores), "score_first1": float(scores[0]),
                     "score_full": float(scores.mean()), "score_series": scores.tolist()})
    success = [row for row in rows if row["success"]]
    failure = [row for row in rows if not row["success"]]
    result = {"complete": True, "task": args.task, "scope": "50 held-out fixed-H50 public-policy rollouts",
              "score": "D logit on each complete H50 action suggestion; episode-equally weighted query mean",
              "outcome": {name: bootstrap(np.asarray([r[f"score_{name}"] for r in success]),
                                          np.asarray([r[f"score_{name}"] for r in failure]), seed)
                          for name, seed in (("first1", 42), ("full", 43))},
              "query_count_control": bootstrap(np.asarray([r["queries"] for r in success]),
                                               np.asarray([r["queries"] for r in failure]), 44),
              "train_scene_count": 50, "test_scene_count": 50, "test_scene_seeds": seeds,
              "checkpoint_sha256": sha256(args.checkpoint), "r3m_sha256": sha256(args.r3m_asset),
              "per_episode": rows,
              "limits": ["Full trajectory score can reflect episode length and task stage.",
                         "D predicts expert versus policy action source, not calibrated success probability."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"task": args.task, "outcome": result["outcome"]}), flush=True)


if __name__ == "__main__":
    main()
