#!/usr/bin/env python3
"""Score the same saved HIL K4 full-chunk candidates for encoder diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tail_value.cache import atomic_json, read_json, sha256
from tail_value.chunk import ChunkCoverageCritic


@torch.inference_mode()
def score_episode(model, arrays, device, batch_size=128):
    outputs = []
    for start in range(0, len(arrays["state"]), batch_size):
        stop = start + batch_size
        f = torch.as_tensor(arrays["features"][start:stop], dtype=torch.float32, device=device)
        s = torch.as_tensor(arrays["state"][start:stop], dtype=torch.float32, device=device)
        c = torch.as_tensor(arrays["candidates"][start:stop], dtype=torch.float32, device=device)
        outputs.append(model.score_many(f, s, c).cpu().numpy().astype(np.float64))
    return np.concatenate(outputs)


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["representation"] != "absolute_joint_chunk" or payload["config"]["horizon"] != 10:
        raise ValueError("Wrong checkpoint representation")
    model = ChunkCoverageCritic(payload["normalization"], 10, 256)
    model.load_state_dict(payload["model"])
    return model.to(device).eval().requires_grad_(False)


def aggregate(episodes, scale, seed=42):
    if not scale > 1e-12:
        raise ValueError("Degenerate SFT calibration scale")
    bins = {}
    early_late = []
    pre_count = 0
    for episode in episodes:
        rel = np.asarray(episode["relative_frame"], dtype=int)
        mean = np.asarray(episode["candidate_mean"], dtype=float)
        support = np.asarray(episode["candidate_range"], dtype=float)
        # Per-trajectory first ten policy frames define a stable offset, while
        # the *global SFT calibration* supplies the only scale.
        policy = rel < 0
        pre_count += int(policy.sum())
        initial = mean[policy][:10].mean()
        if not np.isfinite(initial):
            raise ValueError("HIL episode has no pre-takeover policy frame")
        normalized = (mean - initial) / scale
        for offset, value, spread in zip(rel, normalized, support):
            if -30 <= offset <= 30:
                bins.setdefault(int(offset), []).append((float(value), float(spread)))
        early = normalized[(rel >= -20) & (rel <= -11)]
        late = normalized[(rel >= -10) & (rel <= -1)]
        if len(early) >= 5 and len(late) >= 5:
            early_late.append(float(late.mean() - early.mean()))
    curve = []
    for offset in sorted(bins):
        values = np.asarray(bins[offset], dtype=float)
        curve.append({"relative_saved_frame": offset, "n_episodes": len(values),
                      "offset_score_mean": float(values[:, 0].mean()),
                      "offset_score_q25": float(np.quantile(values[:, 0], .25)),
                      "offset_score_q75": float(np.quantile(values[:, 0], .75)),
                      "candidate_range_mean": float(values[:, 1].mean())})
    early_late = np.asarray(early_late)
    if len(early_late):
        rng = np.random.default_rng(seed)
        draws = early_late[rng.integers(len(early_late), size=(10000, len(early_late)))].mean(1)
        comparison = {"episodes_with_both_windows": len(early_late),
                      "late_minus_early_mean": float(early_late.mean()),
                      "episode_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist()}
    else:
        comparison = {"episodes_with_both_windows": 0, "late_minus_early_mean": None,
                      "episode_bootstrap_ci95": None}
    return {"full_trajectory_observations": sum(len(ep["relative_frame"]) for ep in episodes),
            "pre_takeover_observations": pre_count, "episodes": len(episodes),
            "curve": curve, "late_vs_early": comparison}


def run(args):
    cache_manifest = read_json(args.cache_dir / "manifest.json")
    if cache_manifest.get("complete") is not True or len(cache_manifest["episodes"]) != 50:
        raise ValueError("HIL encoder cache incomplete")
    eval_metrics = read_json(args.eval_metrics)
    reference_manifest = read_json(args.source_cache / "candidate_manifest.json")
    source_by_id = {int(ep["rollout_index"]): ep for ep in reference_manifest["identity"]["episodes"]}
    cache_by_id = {int(ep["rollout_index"]): ep for ep in cache_manifest["episodes"]}
    if set(cache_by_id) != set(source_by_id):
        raise ValueError("HIL cohort changed")
    results = {"scope": "50 saved HIL episodes; not real coverage labels or success-rate evaluation",
               "source_candidate_manifest_sha256": sha256(args.source_cache / "candidate_manifest.json"),
               "encoder_cache_manifest_sha256": sha256(args.cache_dir / "manifest.json"),
               "eval_metrics_sha256": sha256(args.eval_metrics), "methods": {}}
    for objective in ("td20k", "discriminator10k"):
        checkpoint = args.td_checkpoint if objective == "td20k" else args.disc_checkpoint
        if sha256(checkpoint) != eval_metrics["models"][objective]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint differs from SFT diagnostic: {objective}")
        model = load_model(checkpoint, args.device)
        episodes = []
        for rid in sorted(cache_by_id):
            entry = cache_by_id[rid]
            path = args.cache_dir / entry["path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Changed HIL encoded shard {path}")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            scores = score_episode(model, arrays, args.device)
            relative = arrays["frame_index"].astype(int) - int(source_by_id[rid]["takeover"])
            episodes.append({"rollout_index": rid, "relative_frame": relative.tolist(),
                             "candidate_mean": scores.mean(1).tolist(),
                             "candidate_range": np.ptp(scores, axis=1).tolist()})
        scale = float(eval_metrics["models"][objective]["calibration"]["scale"])
        results["methods"][objective] = {"checkpoint_sha256": sha256(checkpoint),
                                         "sft_calibration_scale": scale,
                                         **aggregate(episodes, scale)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "metrics.json", results)
    atomic_json(args.output_dir / "manifest.json", {"complete": True,
        "metrics_sha256": sha256(args.output_dir / "metrics.json"),
        "encoder_cache_manifest_sha256": results["encoder_cache_manifest_sha256"]})
    print(f"SCORED {args.output_dir / 'metrics.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("cache_dir", "source_cache", "eval_metrics", "td_checkpoint", "disc_checkpoint", "output_dir"):
        p.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
