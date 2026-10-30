#!/usr/bin/env python3
"""Score fresh turn_switch policy traces with the frozen 5-fold D ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from coverage_benchmark.encode_matched_pairs_r3m import image_tensor, r3m_state, sha256
from tail_value.chunk import ChunkCoverageCritic


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC needs both successful and failed episodes")
    delta = pos[:, None] - neg[None, :]
    return float(((delta > 0).sum() + .5 * (delta == 0).sum()) / delta.size)


def bootstrap_auc(pos: np.ndarray, neg: np.ndarray, seed: int) -> dict:
    if not len(pos) or not len(neg):
        return {"auc_success_higher": None, "ci95": None,
                "successful_episodes": len(pos), "failed_episodes": len(neg)}
    rng = np.random.default_rng(seed)
    draws = np.empty(10000, dtype=np.float64)
    for i in range(len(draws)):
        draws[i] = auc(rng.choice(pos, len(pos), replace=True),
                       rng.choice(neg, len(neg), replace=True))
    value = auc(pos, neg)
    return {"auc_success_higher": value,
            "auc_failure_higher": 1.0 - value,
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "successful_episodes": len(pos), "failed_episodes": len(neg)}


def load_d(d_root: Path, device: torch.device):
    metrics_path, manifest_path = d_root / "metrics.json", d_root / "manifest.json"
    metrics, manifest = json.loads(metrics_path.read_text()), json.loads(manifest_path.read_text())
    if not metrics.get("complete") or not manifest.get("complete") or metrics.get("task") != "turn_switch":
        raise ValueError("D training results are incomplete or for another task")
    if sha256(metrics_path) != manifest["metrics_sha256"]:
        raise ValueError("D metrics hash mismatch")
    folds = []
    for i, record in enumerate(metrics["train_oof"]["folds"]):
        checkpoint = d_root / f"fold_{i}" / "last.pt"
        expected = manifest["fold_checkpoints_sha256"][str(i)]
        if sha256(checkpoint) != expected or sha256(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError(f"fold {i} checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol") != "source_bce_v1" or payload.get("step") != 10000:
            raise ValueError(f"fold {i} checkpoint protocol/step mismatch")
        calibration = record["calibration"]
        if calibration.get("degenerate") or calibration.get("scale", 0) <= 0:
            raise ValueError(f"fold {i} calibration is invalid")
        model = ChunkCoverageCritic(payload["normalization"], horizon=10, width=256)
        model.load_state_dict(payload["model"], strict=True)
        folds.append((model.to(device).eval().requires_grad_(False), calibration))
    return metrics, manifest, folds


def load_r3m(path: Path, device: torch.device):
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = torch.nn.Identity()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(r3m_state(state, model.state_dict()), strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def score_episode(model_r3m, folds, folder: Path, record: dict, device: torch.device) -> np.ndarray:
    trace = json.loads((folder / "trace.json").read_text())
    if (not trace.get("complete") or not trace.get("diagnostic_only")
            or trace.get("accepted_for_training") is not False
            or trace.get("scene_seed") != record["scene_seed"]
            or trace.get("success") != record["success"]):
        raise ValueError(f"incomplete/label-mismatched diagnostic trace: {folder}")
    states, actions = [], []
    views: list[list[np.ndarray]] = [[], [], []]
    names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    for i, query in enumerate(trace["queries"]):
        if query.get("query") != i or query.get("planned_chunk_length") != 10:
            raise ValueError(f"invalid query metadata {folder}/{i}")
        with np.load(folder / f"query_{i:04d}.npz", allow_pickle=False) as data:
            state, action = data["state"], data["policy_h10"]
        if state.shape != (14,) or action.shape != (10, 14):
            raise ValueError(f"bad state/action shape {folder}/{i}: {state.shape}/{action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"nonfinite D input {folder}/{i}")
        states.append(state.astype(np.float32))
        actions.append(action.astype(np.float32))
        for camera_index, name in enumerate(names):
            with Image.open(folder / f"query_{i:04d}_{name}.png") as image:
                views[camera_index].append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    if len(states) != record["queries"] or not states:
        raise ValueError(f"trace query count mismatch {folder}")
    features = []
    for camera_images in views:
        batches = []
        for start in range(0, len(camera_images), 32):
            batch = torch.stack([image_tensor(im) for im in camera_images[start:start + 32]]).to(device)
            batches.append(model_r3m(batch).float().cpu().numpy())
        features.append(np.concatenate(batches, axis=0))
    feature = torch.as_tensor(np.concatenate(features, axis=1), dtype=torch.float32, device=device)
    state_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    action_t = torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device)
    fold_scores = []
    for model, calibration in folds:
        logits = model(feature, state_t, action_t).float().cpu().numpy()
        fold_scores.append((logits - calibration["mean"]) / calibration["scale"])
    values = np.mean(fold_scores, axis=0)
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite D scores in {folder}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--d-root", type=Path, required=True)
    parser.add_argument("--r3m-asset", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    rows = [json.loads(line) for line in
            (args.trace_root / "evaluation" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    meta = json.loads((args.trace_root / "evaluation" / "metadata.json").read_text())
    if meta.get("status") != "complete" or meta.get("task") != "turn_switch":
        raise ValueError("rollout metadata incomplete or task mismatch")
    if len(rows) != 50 or len({r["scene_seed"] for r in rows}) != 50:
        raise ValueError("expected exactly 50 unique fixed scene seeds")
    if any(r.get("infra_error") for r in rows):
        raise ValueError("infrastructure errors must be resolved/excluded before AUC")
    if not any(r["success"] for r in rows) or all(r["success"] for r in rows):
        raise ValueError("both policy-success and policy-failure trajectories are required")

    metrics, manifest, folds = load_d(args.d_root, device)
    feature_manifest = json.loads(args.feature_manifest.read_text())
    if feature_manifest["encoder_sha256"] != sha256(args.r3m_asset):
        raise ValueError("R3M asset differs from frozen D training encoder")
    r3m = load_r3m(args.r3m_asset, device)
    results = []
    for row in rows:
        folder = args.trace_root / "evaluation" / "trace" / f"episode_{int(row['episode']):03d}"
        values = score_episode(r3m, folds, folder, row, device)
        scored = {"episode": int(row["episode"]), "scene_seed": int(row["scene_seed"]),
                  "success": bool(row["success"]), "steps": int(row["steps"]),
                  "queries": len(values), "query_scores": values.tolist(),
                  "full": float(values.mean())}
        for prefix in (1, 2, 3, 5, 10, 15):
            scored[f"first{prefix}"] = float(values[:prefix].mean()) if len(values) >= prefix else None
        results.append(scored)
        print(f"SCORED {len(results)}/50 seed={row['scene_seed']} success={row['success']} "
              f"queries={len(values)}", flush=True)

    outcome = {}
    for i, key in enumerate(("first1", "first2", "first3", "first5", "first10", "first15", "full")):
        eligible = [r for r in results if r[key] is not None]
        pos = np.asarray([r[key] for r in eligible if r["success"]], dtype=np.float64)
        neg = np.asarray([r[key] for r in eligible if not r["success"]], dtype=np.float64)
        outcome[key] = {**bootstrap_auc(pos, neg, 42 + i),
                        "eligible_episodes": len(eligible),
                        "excluded_short_episodes": 50 - len(eligible)}
    pos_len = np.asarray([r["queries"] for r in results if r["success"]], dtype=np.float64)
    neg_len = np.asarray([r["queries"] for r in results if not r["success"]], dtype=np.float64)
    output = {
        "complete": True, "task": "turn_switch",
        "scope": "fresh held-out own-5k policy rollouts on seeds 31000-31049; not exact replay of prior stochastic actions",
        "policy_checkpoint_manifest_sha256": "503dcd0e832ca7090725599f1d5f9bf5a1057b180fcbfc92abeeb26cc20796c0",
        "d_metrics_sha256": sha256(args.d_root / "metrics.json"),
        "d_manifest_sha256": sha256(args.d_root / "manifest.json"),
        "r3m_sha256": sha256(args.r3m_asset),
        "r3m_feature_manifest_sha256": sha256(args.feature_manifest),
        "d_method": "five episode-disjoint 30/10/10 OOF critics; fold-calibrated logits averaged on the actual policy H10 suggestion at each policy query",
        "score_direction": "higher means more expert-source-like; it is not a success probability",
        "successes": sum(bool(r["success"]) for r in rows), "failures": sum(not bool(r["success"]) for r in rows),
        "episodes": len(rows), "success_rate": sum(bool(r["success"]) for r in rows) / len(rows),
        "outcome_auc": outcome,
        "query_count_control_auc_success_higher": bootstrap_auc(pos_len, neg_len, 77),
        "per_episode": results,
        "limits": ["The reruns are stochastic and are not the original action traces.",
                   "Fixed-prefix AUC excludes episodes shorter than that prefix; class counts are reported.",
                   "Full-trajectory mean can be confounded by task stage and trajectory length; query-count control is included.",
                   "This measures success/failure association, not HIL event detection or rescue probability."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    print(json.dumps({"successes": output["successes"], "failures": output["failures"],
                      "outcome_auc": outcome,
                      "query_count_control": output["query_count_control_auc_success_higher"]}, indent=2))


if __name__ == "__main__":
    main()
