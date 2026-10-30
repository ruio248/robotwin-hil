#!/usr/bin/env python3
"""Held-out success/failure AUC for a task-specific D on policy-visited states.

This is a separate validation from same-observation expert-vs-policy ranking.
It consumes only complete seed2 diagnostic rollouts, a finished seed1 five-fold
D ensemble, and the exact frozen R3M asset used for D training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from coverage_benchmark.encode_matched_pairs_r3m import image_tensor, r3m_state, sha256
from tail_value.chunk import ChunkCoverageCritic


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if not len(positive) or not len(negative):
        raise ValueError("AUC requires both outcomes")
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0).sum() + .5 * (delta == 0).sum()) / delta.size)


def boot_auc(positive: np.ndarray, negative: np.ndarray, seed: int) -> dict:
    if not len(positive) or not len(negative):
        return {"auc": None, "ci95": None, "successful_episodes": len(positive),
                "failed_episodes": len(negative)}
    rng = np.random.default_rng(seed)
    values = np.empty(10000, dtype=np.float64)
    for i in range(len(values)):
        p = rng.choice(positive, len(positive), replace=True)
        n = rng.choice(negative, len(negative), replace=True)
        values[i] = auc(p, n)
    return {"auc": auc(positive, negative),
            "ci95": np.quantile(values, [.025, .975]).tolist(),
            "successful_episodes": len(positive), "failed_episodes": len(negative)}


def r3m(asset: Path, device: torch.device):
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = torch.nn.Identity()
    payload = torch.load(asset, map_location="cpu", weights_only=True)
    model.load_state_dict(r3m_state(payload, model.state_dict()), strict=True)
    return model.to(device).eval().requires_grad_(False)


def d_folds(root: Path, device: torch.device) -> list[tuple[torch.nn.Module, dict]]:
    metrics = json.loads((root / "metrics.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if not metrics.get("complete") or not manifest.get("complete"):
        raise ValueError("D ensemble incomplete")
    if sha256(root / "metrics.json") != manifest["metrics_sha256"]:
        raise ValueError("D metric manifest hash mismatch")
    folds = []
    for i in range(5):
        checkpoint = root / f"fold_{i}" / "last.pt"
        if sha256(checkpoint) != manifest["fold_checkpoints"][str(i)]:
            raise ValueError(f"D fold {i} checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol") != "source_bce_v1" or payload.get("step") != 10000:
            raise ValueError(f"D fold {i} protocol mismatch")
        model = ChunkCoverageCritic(payload["normalization"], horizon=10, width=256)
        model.load_state_dict(payload["model"], strict=True)
        calibration = json.loads((root / f"fold_{i}" / "calibration.json").read_text())
        if calibration["scale"] <= 0:
            raise ValueError(f"D fold {i} degenerate calibration")
        folds.append((model.to(device).eval(), calibration))
    return folds


def read_episode(root: Path, record: dict) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    folder = root / "evaluation" / "trace" / f"episode_{record['episode']:03d}"
    trace = json.loads((folder / "trace.json").read_text())
    if not trace.get("complete") or not trace.get("diagnostic_only") or trace.get("accepted_for_training"):
        raise ValueError(f"bad diagnostic trace {folder}")
    if trace["scene_seed"] != record["scene_seed"] or trace["success"] != record["success"]:
        raise ValueError(f"trace/outcome mismatch {folder}")
    states, actions, views = [], [], [[], [], []]
    for index, query in enumerate(trace["queries"]):
        if query["query"] != index or query["planned_chunk_length"] != 50:
            raise ValueError(f"bad query index/horizon {folder}")
        with np.load(folder / f"query_{index:04d}.npz", allow_pickle=False) as data:
            state = data["state"]
            action = data["policy_h10"]
        if state.shape != (14,) or action.shape != (10, 14):
            raise ValueError(f"bad D input shape {folder}/{index}")
        states.append(state)
        actions.append(action)
        for camera_index, name in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist")):
            with Image.open(folder / f"query_{index:04d}_{name}.png") as image:
                views[camera_index].append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    if len(states) != record["queries"] or not states:
        raise ValueError(f"query count mismatch {folder}")
    return np.stack(states), np.stack(actions), views


@torch.inference_mode()
def features(model, views: list[list[np.ndarray]], device: torch.device) -> np.ndarray:
    blocks = []
    for images in views:
        chunks = []
        for start in range(0, len(images), 32):
            batch = torch.stack([image_tensor(im) for im in images[start:start + 32]]).to(device)
            chunks.append(model(batch).float().cpu().numpy())
        blocks.append(np.concatenate(chunks))
    return np.concatenate(blocks, axis=1).astype(np.float32)


@torch.inference_mode()
def score(folds, feature: np.ndarray, state: np.ndarray,
          action: np.ndarray, device: torch.device) -> np.ndarray:
    f = torch.as_tensor(feature, device=device)
    s = torch.as_tensor(state, device=device)
    a = torch.as_tensor(action, device=device)
    values = []
    for model, calibration in folds:
        logit = model(f, s, a).float().cpu().numpy()
        values.append((logit - calibration["mean"]) / calibration["scale"])
    result = np.mean(values, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite D trace scores")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--trace-run", type=Path, required=True)
    p.add_argument("--d-root", type=Path, required=True)
    p.add_argument("--r3m-asset", type=Path, required=True)
    p.add_argument("--feature-manifest", type=Path, required=True,
                   help="Seed1 pair-data R3M manifest copied with the finished D; do not infer a cross-host path")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    launch = json.loads((args.trace_run / "launch.json").read_text())
    meta = json.loads((args.trace_run / "evaluation" / "metadata.json").read_text())
    records = [json.loads(x) for x in (args.trace_run / "evaluation" / "episodes.jsonl").read_text().splitlines() if x]
    if launch["status"] != "complete" or meta["status"] != "complete" or len(records) != 50:
        raise ValueError("trace evaluation incomplete")
    if meta["task"] != args.task or meta["method"] != "fixed_h50" or meta["seed"] != 2:
        raise ValueError("wrong trace task/protocol/split")
    if len({r["scene_seed"] for r in records}) != 50 or any(r.get("infra_error") for r in records):
        raise ValueError("duplicate scenes or infrastructure failure")
    d_metric = json.loads((args.d_root / "metrics.json").read_text())
    if d_metric["task"] != args.task or d_metric["data"]["scene_seed_split"] != 1:
        raise ValueError("D must be trained on the independent seed1 cohort")
    feature_manifest = json.loads(args.feature_manifest.read_text())
    if sha256(args.feature_manifest) != d_metric["source_hashes"]["feature_manifest"]:
        raise ValueError("R3M feature manifest differs from D training")
    if feature_manifest["encoder_sha256"] != sha256(args.r3m_asset):
        raise ValueError("outcome scoring R3M differs from D training")
    device = torch.device(args.device)
    encoder, folds = r3m(args.r3m_asset, device), d_folds(args.d_root, device)
    rows = []
    for record in records:
        states, actions, views = read_episode(args.trace_run, record)
        scored = score(folds, features(encoder, views, device), states, actions, device)
        row = {"episode": record["episode"], "scene_seed": record["scene_seed"],
               "success": record["success"], "steps": record["steps"],
               "queries": len(scored), "full": float(scored.mean())}
        # H=50 often ends successful RoboTwin rollouts in fewer than five queries.
        # Preserve the requested 5/10/15 prefixes, but include 1/2/3 so that
        # early comparison retains both outcome classes on these short tasks.
        for prefix in (1, 2, 3, 5, 10, 15):
            row[f"first{prefix}"] = (float(scored[:prefix].mean())
                                     if len(scored) >= prefix else None)
        rows.append(row)
        print(f"SCORED {len(rows)}/50 episode={record['episode']}", flush=True)
    outcome = {}
    for i, key in enumerate(("first1", "first2", "first3", "first5", "first10", "first15", "full")):
        eligible = [r for r in rows if r[key] is not None]
        positive = np.asarray([r[key] for r in eligible if r["success"]], dtype=np.float64)
        negative = np.asarray([r[key] for r in eligible if not r["success"]], dtype=np.float64)
        outcome[key] = {**boot_auc(positive, negative, 42 + i),
                        "eligible_episodes": len(eligible),
                        "excluded_short_episodes": 50 - len(eligible)}
    length_pos = np.asarray([r["queries"] for r in rows if r["success"]], dtype=np.float64)
    length_neg = np.asarray([r["queries"] for r in rows if not r["success"]], dtype=np.float64)
    output = {"complete": True, "task": args.task,
              "scope": "held-out seed2 published policy rollouts; D trained only on seed1 expert-visited same-observation source pairs",
              "d_metrics_sha256": sha256(args.d_root / "metrics.json"),
              "r3m_sha256": sha256(args.r3m_asset),
              "r3m_feature_manifest_sha256": sha256(args.feature_manifest),
              "trace_metadata_sha256": sha256(args.trace_run / "evaluation" / "metadata.json"),
              "method": "mean of five fold-calibrated D logits on actual policy H10 suggestion at every fixed-H50 query",
              "higher_score": "more expert-source-like, not a calibrated success probability",
              "outcome_auc": outcome, "query_count_auc": boot_auc(length_pos, length_neg, 77),
              "per_episode": rows,
              "limits": ["Only episodes with both outcome classes can yield AUC.",
                         "Fixed prefixes exclude shorter episodes; report eligibility and class balance.",
                         "Full-trajectory scores may be confounded by length and task stage.",
                         "This does not test human takeover-event detection or rescue probability."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"task": args.task, "outcome_auc": outcome,
                      "query_count_auc": output["query_count_auc"]}, indent=2))


if __name__ == "__main__":
    main()
