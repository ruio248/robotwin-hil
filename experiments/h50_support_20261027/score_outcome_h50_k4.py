#!/usr/bin/env python3
"""Handover-style K4-then-episode mean on held-out fixed-H50 queries."""
from __future__ import annotations

import argparse
import bisect
import json
import os
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18

from encode_pairs_h50 import r3m_state
from score_outcome_h50 import bootstrap, rgb_feature, sha256
from tail_value.chunk import ChunkCoverageCritic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--r3m-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (payload.get("protocol"), payload.get("task"), payload.get("horizon")) != (
            "h50_same_observation_source_bce_v1", args.task, 50):
        raise ValueError("D model protocol mismatch")
    critic = ChunkCoverageCritic(payload["normalization"], horizon=50, width=256)
    critic.load_state_dict(payload["model"], strict=True)
    critic = critic.to(device).eval()
    encoder = resnet18(weights=None)
    encoder.fc = torch.nn.Identity()
    encoder.load_state_dict(r3m_state(torch.load(args.r3m_asset, map_location="cpu", weights_only=True),
                                      encoder.state_dict()), strict=True)
    encoder = encoder.to(device).eval().requires_grad_(False)
    evaluation = args.eval_root / "evaluation"
    meta = json.loads((evaluation / "metadata.json").read_text())
    records = [json.loads(line) for line in (evaluation / "episodes.jsonl").read_text().splitlines()]
    manifest = json.loads((args.candidates / "manifest.json").read_text())
    if meta.get("status") != "complete" or meta.get("task") != args.task or meta.get("method") != "fixed_h50":
        raise ValueError("incomplete or wrong evaluation")
    if (len(records), len({row["scene_seed"] for row in records})) != (50, 50):
        raise ValueError("expected 50 unique episodes")
    if (not manifest.get("complete") or manifest.get("episodes") != 50 or
            manifest.get("candidate_count") != 4 or manifest.get("action_horizon") != 50 or
            manifest.get("trace_metadata_sha256") != sha256(evaluation / "metadata.json")):
        raise ValueError("candidate manifest/protocol mismatch")
    if set(row["scene_seed"] for row in records).intersection(payload["scene_seeds"]):
        raise ValueError("evaluation scenes overlap D training")
    if len(manifest["files"]) != 50:
        raise ValueError("incomplete K4 file list")
    rows = []
    for record in records:
        scene = int(record["scene_seed"])
        folder = evaluation / "trace" / f"episode_{record['episode']:03d}"
        trace = json.loads((folder / "trace.json").read_text())
        if not trace.get("complete") or int(trace["scene_seed"]) != scene or bool(trace["success"]) != bool(record["success"]):
            raise ValueError(f"trace/outcome mismatch: {folder}")
        name = f"episode_{record['episode']:03d}_scene_{scene}.npz"
        candidate_path = args.candidates / name
        item = manifest["files"][name]
        if item["sha256"] != sha256(candidate_path) or int(item["scene_seed"]) != scene:
            raise ValueError(f"candidate identity/hash mismatch: {name}")
        with np.load(candidate_path, allow_pickle=False) as source:
            actions = source["candidates"].astype(np.float32)
        count = int(record["queries"])
        if actions.shape != (count, 4, 50, 14) or len(trace["queries"]) != count:
            raise ValueError(f"candidate/query shape mismatch: {name}")
        states = []
        for index in range(count):
            with np.load(folder / f"query_{index:04d}.npz", allow_pickle=False) as source:
                state = source["state"].astype(np.float32)
                original = source["policy_h50"].astype(np.float32)
            if state.shape != (14,) or not np.array_equal(actions[index, 0], original):
                raise ValueError(f"candidate zero is not executed policy proposal: {folder}/{index}")
            states.append(state)
        if not np.isfinite(actions).all():
            raise ValueError(f"nonfinite candidates: {name}")
        feature_blocks = []
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            paths = [folder / f"query_{index:04d}_{camera}.png" for index in range(count)]
            feature_blocks.append(rgb_feature(encoder, paths, device))
        features = np.concatenate(feature_blocks, axis=1).astype(np.float32)
        with torch.inference_mode():
            logits = critic(
                torch.from_numpy(np.repeat(features, 4, axis=0)).to(device),
                torch.from_numpy(np.repeat(np.stack(states), 4, axis=0)).to(device),
                torch.from_numpy(actions.reshape(count * 4, 50, 14)).to(device),
            ).float().cpu().numpy().reshape(count, 4)
        per_query = logits.mean(axis=1)
        rows.append({"episode": int(record["episode"]), "scene_seed": scene,
                     "success": bool(record["success"]), "queries": count,
                     "score_first1": float(per_query[:1].mean()),
                     "score_first3": float(per_query[:3].mean()),
                     "score_first5": float(per_query[:5].mean()) if count >= 5 else None,
                     "score_full": float(per_query.mean()),
                     "per_query_mean": per_query.tolist(),
                     "candidate_score_range_median": float(np.median(logits.max(axis=1) - logits.min(axis=1)))})
        print(f"SCORED_K4 {len(rows)}/50 scene={scene}", flush=True)
    outcome = {}
    for i, key in enumerate(("first1", "first3", "first5", "full")):
        eligible = [row for row in rows if row[f"score_{key}"] is not None]
        positive = np.asarray([row[f"score_{key}"] for row in eligible if row["success"]])
        negative = np.asarray([row[f"score_{key}"] for row in eligible if not row["success"]])
        outcome[key] = {**bootstrap(positive, negative, 42 + i),
                        "eligible_episodes": len(eligible)}
    ordered = sorted(float(row["score_full"]) for row in rows)
    for row in rows:
        value = float(row["score_full"])
        row["score_full_percentile"] = 100 * (bisect.bisect_left(ordered, value) +
                                                bisect.bisect_right(ordered, value)) / (2 * len(ordered))
    success = [row for row in rows if row["success"]]
    failure = [row for row in rows if not row["success"]]
    result = {"complete": True, "task": args.task,
              "method": "per-observation mean of K4 raw D logits; per-episode mean of observed query scores; episode-equal outcome AUC",
              "candidate_count": 4, "action_horizon": 50,
              "observation_schedule": "fixed-H50 executed-policy query boundaries; not historical every-saved-frame H10 schedule",
              "outcome_auc": outcome,
              "query_count_control": bootstrap(np.asarray([r["queries"] for r in success]),
                                               np.asarray([r["queries"] for r in failure]), 77),
              "success_minus_failure_percentile_pp": float(np.mean([r["score_full_percentile"] for r in success]) -
                                                           np.mean([r["score_full_percentile"] for r in failure])),
              "candidate_score_range_median": float(np.median([r["candidate_score_range_median"] for r in rows])),
              "test_scene_count": 50, "train_scene_count": 50,
              "d_checkpoint_sha256": sha256(args.checkpoint),
              "candidate_manifest_sha256": sha256(args.candidates / "manifest.json"),
              "r3m_sha256": sha256(args.r3m_asset), "per_episode": rows,
              "limits": ["K4 and two-stage averaging match the historical statistic, but H50 query schedule differs from historical H10 saved frames.",
                         "Full-trajectory scores mix different task stages and lengths.",
                         "D predicts expert/action source, not calibrated success probability or HIL takeover need."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"outcome_auc": outcome, "query_count_control": result["query_count_control"]}), flush=True)


if __name__ == "__main__":
    main()
