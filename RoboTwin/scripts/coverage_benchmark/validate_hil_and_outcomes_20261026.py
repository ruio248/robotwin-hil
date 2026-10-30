#!/usr/bin/env python3
"""Retrospective event and outcome checks for the handover-to-tray R3M D.

This uses the previously frozen HIL and autonomous rollout caches.  It scores
policy suggestions, not the human's executed action.  The ten new RoboTwin
task-specific D models need their own HIL cohorts for the same check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coverage_encoder_discriminator_bigmetrics import auc, load_model, score
from tail_value.cache import atomic_json, sha256


def ci_auc(pos: np.ndarray, neg: np.ndarray, seed: int = 42) -> dict:
    value = auc(pos, neg)
    rng = np.random.default_rng(seed)
    draws = np.empty(10000, dtype=np.float64)
    for i in range(len(draws)):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        draws[i] = auc(p, n)
    return {"auc": value, "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "positive_episodes": len(pos), "negative_episodes": len(neg)}


def ci_mean(values: np.ndarray, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "episodes": len(values)}


def score_archive(model, path: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        frames = data["frame_index"].astype(np.int64)
        scores = score(model, data["features"], data["state"], data["candidates"], device).mean(axis=1)
    if len(frames) != len(scores) or len(frames) < 15 or not np.isfinite(scores).all():
        raise ValueError(f"bad score cache: {path}")
    return frames, scores


def window_drop(values: np.ndarray, t: int) -> float:
    # Both windows contain exactly five saved observations before t.
    if t < 15 or t > len(values):
        raise ValueError(f"bad event index {t}/{len(values)}")
    return float(values[t - 5:t].mean() - values[t - 15:t - 10].mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--hil-cache", type=Path, required=True)
    p.add_argument("--hil-source", type=Path, required=True)
    p.add_argument("--historical-cache", type=Path, required=True)
    p.add_argument("--existing-metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    metrics = json.loads(args.existing_metrics.read_text())
    if sha256(args.model) != metrics["source_sha256"]["model"]:
        raise ValueError("checkpoint differs from frozen R3M D diagnostic")
    if sha256(args.hil_cache / "manifest.json") != metrics["source_sha256"]["hil_cache_manifest"]:
        raise ValueError("HIL feature cache differs from prior validated cohort")
    if sha256(args.historical_cache / "manifest.json") != metrics["source_sha256"]["historical_cache_manifest"]:
        raise ValueError("historical feature cache differs from prior validated cohort")
    if sha256(args.hil_source) != metrics["source_sha256"]["hil_source_manifest"]:
        raise ValueError("HIL labels differ from prior validated cohort")
    source = json.loads(args.hil_source.read_text())
    hil_manifest = json.loads((args.hil_cache / "manifest.json").read_text())
    historical_manifest = json.loads((args.historical_cache / "manifest.json").read_text())
    if not hil_manifest.get("complete") or not historical_manifest.get("complete"):
        raise ValueError("cache incomplete")
    takeovers = {int(x["rollout_index"]): int(x["takeover"])
                 for x in source["identity"]["episodes"]}
    if len(takeovers) != 50 or len(historical_manifest["episodes"]) != 100:
        raise ValueError("unexpected cohort size")
    device = torch.device(args.device)
    model = load_model(args.model, device)
    hil = []
    for entry in hil_manifest["episodes"]:
        path = args.hil_cache / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"changed HIL cache: {path}")
        rid = int(entry["rollout_index"])
        frame, values = score_archive(model, path, device)
        t = int(np.searchsorted(frame, takeovers[rid]))
        if t < 15 or t >= len(frame) or int(frame[t]) != takeovers[rid]:
            raise ValueError(f"takeover frame missing: {rid}")
        hil.append({"rollout_index": rid, "takeover_index": t,
                    "frames": len(frame), "score_drop": window_drop(values, t)})
    pseudo_time = int(np.median([x["takeover_index"] for x in hil]))
    hist = []
    for entry in historical_manifest["episodes"]:
        path = args.historical_cache / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"changed historical cache: {path}")
        _, values = score_archive(model, path, device)
        t = min(pseudo_time, len(values))
        hist.append({"episode": entry["name"], "success": bool(entry["success"]),
                     "frames": len(values), "first10": float(values[:10].mean()),
                     "first15": float(values[:15].mean()), "full": float(values.mean()),
                     "pseudo_event_index": t, "pseudo_drop": window_drop(values, t)})
    if (sum(x["success"] for x in hist), sum(not x["success"] for x in hist)) != (31, 69):
        raise ValueError("historical outcome cohort changed")
    event = np.asarray([x["score_drop"] for x in hil])
    nonevent = np.asarray([x["pseudo_drop"] for x in hist])
    outcome = {}
    for key in ("first10", "first15", "full"):
        pos = np.asarray([x[key] for x in hist if x["success"]])
        neg = np.asarray([x[key] for x in hist if not x["success"]])
        outcome[key] = {**ci_auc(pos, neg),
                        "success_mean": float(pos.mean()), "failure_mean": float(neg.mean()),
                        "success_minus_failure": float(pos.mean() - neg.mean())}
    output = {
        "scope": "Retrospective handover_to_tray R3M source D; saved HIL episodes and historical autonomous rollouts",
        "model_sha256": sha256(args.model),
        "hil_source_sha256": sha256(args.hil_source),
        "hil_cache_manifest_sha256": sha256(args.hil_cache / "manifest.json"),
        "historical_cache_manifest_sha256": sha256(args.historical_cache / "manifest.json"),
        "event": {
            "definition": "first saved human-control frame, with two five-frame windows entirely before takeover",
            "metric": "late[-5,-1] minus early[-15,-11] mean raw D score; lower means more event-like",
            "non_event": f"historical autonomous trajectory pseudo-boundary at saved frame index min({pseudo_time}, length)",
            "hil_drop": ci_mean(event), "autonomous_pseudo_drop": ci_mean(nonevent),
            "event_vs_non_event_auc": ci_auc(-event, -nonevent),
            "per_hil_episode": hil, "per_autonomous_episode": hist,
        },
        "outcome": {"metric": "mean raw D score over the indicated saved prefix or full saved trajectory",
                    "by_window": outcome},
        "limits": [
            "The 50 HIL trajectories were selected because a takeover was saved; this does not estimate deployment precision or recall.",
            "Historical pseudo-boundaries are phase approximations; event and non-event cohorts differ in collection protocol.",
            "The source D scores policy suggestions and was trained on expert-versus-policy action origin, not takeover or success labels.",
            "Outcome AUC on this retrospective cohort does not establish incremental value over observation-only or task-stage controls.",
            "These handover_to_tray results cannot be transferred numerically to the ten new task-specific D models.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, output)
    print(json.dumps({"event_auc": output["event"]["event_vs_non_event_auc"],
                      "outcome_auc": {k: v["auc"] for k, v in outcome.items()}}, indent=2))


if __name__ == "__main__":
    main()
