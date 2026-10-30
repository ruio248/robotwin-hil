#!/usr/bin/env python3
"""Frozen-encoder ablation for full-chunk 32-neighbor expert-action support.

Only the visual encoder changes. The SFT split, [10,14] chunks, candidate
pool, neighbor count, normalization and temperature protocol remain fixed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from coverage_benchmark.common import JOINTS, calibrate, weights
from coverage_benchmark.data import sft_data, subset
from coverage_benchmark.scorers import SupportScorer, act_features, normalization, obs_features
from tail_value.cache import atomic_json, atomic_npz, read_json, sha256


def nearest(query, reference, device, batch=128, exclude_episode=None, reference_episode=None):
    """Exact 32-neighbor search in float64, including original episode exclusion."""
    ref = torch.as_tensor(reference, dtype=torch.float64, device=device)
    q = torch.as_tensor(query, dtype=torch.float64, device=device)
    ref_sq = (ref * ref).sum(1)[None]
    episode = None if reference_episode is None else torch.as_tensor(reference_episode, device=device)
    result, distances = [], []
    for start in range(0, len(q), batch):
        block = q[start:start + batch]
        d = ((block * block).sum(1)[:, None] + ref_sq - 2 * block @ ref.T).clamp_min_(0)
        if exclude_episode is not None:
            ids = torch.as_tensor(exclude_episode[start:start + batch], device=device)
            d.masked_fill_(ids[:, None] == episode[None], float("inf"))
        ix = torch.argsort(d, dim=1, stable=True)[:, :32]
        near_d = torch.gather(d, 1, ix)
        if not torch.isfinite(near_d).all():
            raise ValueError("Insufficient cross-episode expert neighbors")
        result.append(ix.cpu().numpy())
        distances.append(near_d.cpu().numpy())
    return np.concatenate(result), np.concatenate(distances)


def score_rows(z, chunks, support, device, batch=128):
    """Same Gaussian action kernel and 32-neighbor mean as SupportScorer."""
    norm = {k: support[k] for k in ("state_mean", "state_scale", "action_mean", "action_scale")}
    action = act_features(chunks, norm).reshape(len(chunks), -1, 140)
    ix, _ = nearest(z, support["observations"], device, batch=batch)
    expert = np.asarray(support["actions"], dtype=np.float64)
    bw = float(support["action_bandwidth"])
    out = []
    for start in range(0, len(ix), batch):
        stop = start + batch
        candidate = torch.as_tensor(action[start:stop], dtype=torch.float64, device=device)
        neighbors = torch.as_tensor(expert[ix[start:stop]], dtype=torch.float64, device=device)
        d = torch.sum((candidate[:, :, None, :] - neighbors[:, None, :, :]) ** 2, dim=-1)
        out.append(torch.exp(-d / (2 * bw * bw)).mean(2).cpu().numpy())
    return np.concatenate(out)


def rank(expert, policy, episode):
    percentile = ((expert[:, None] > policy).sum(1) + .5 * (expert[:, None] == policy).sum(1)) / policy.shape[1]
    ids = np.unique(episode)
    values = np.asarray([percentile[episode == eid].mean() for eid in ids])
    draws = values[np.random.default_rng(42).integers(len(ids), size=(10000, len(ids)))].mean(1)
    return {"episode_balanced_mean": float(values.mean()),
            "episode_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist(),
            "observation_weighted_mean": float(percentile.mean()),
            "strict_top_fraction": float(np.mean(np.all(expert[:, None] > policy, axis=1))),
            "episodes": {str(int(eid)): float(v) for eid, v in zip(ids, values)}}


def load_support(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def aggregate_hil(episodes, scale):
    """Match coverage_encoder_hil_score.aggregate without importing its model code."""
    if not scale > 1e-12:
        raise ValueError("Degenerate SFT calibration scale")
    bins, early_late, pre_count = {}, [], 0
    for episode in episodes:
        rel = np.asarray(episode["relative_frame"], dtype=int)
        mean = np.asarray(episode["candidate_mean"], dtype=float)
        spread = np.asarray(episode["candidate_range"], dtype=float)
        policy = rel < 0
        pre_count += int(policy.sum())
        initial = mean[policy][:10].mean()
        if not np.isfinite(initial):
            raise ValueError("HIL episode has no pre-takeover policy frame")
        normalized = (mean - initial) / scale
        for offset, value, width in zip(rel, normalized, spread):
            if -30 <= offset <= 30:
                bins.setdefault(int(offset), []).append((float(value), float(width)))
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
    changes = np.asarray(early_late)
    if len(changes):
        draws = changes[np.random.default_rng(42).integers(len(changes), size=(10000, len(changes)))].mean(1)
        comparison = {"episodes_with_both_windows": len(changes),
                      "late_minus_early_mean": float(changes.mean()),
                      "episode_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist()}
    else:
        comparison = {"episodes_with_both_windows": 0, "late_minus_early_mean": None,
                      "episode_bootstrap_ci95": None}
    return {"full_trajectory_observations": sum(len(ep["relative_frame"]) for ep in episodes),
            "pre_takeover_observations": pre_count, "episodes": len(episodes),
            "curve": curve, "late_vs_early": comparison}


def run_sft(args):
    data, provenance = sft_data(args.cache_dir)
    train, cal, diag = (subset(data, split) for split in ("train", "cal", "diag"))
    if (len(train["state"]), len(cal["state"]), len(diag["state"])) != (11461, 430, 844):
        raise ValueError("SFT split changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "sft_manifest.json"
    source_sha = sha256(args.cache_dir / "manifest.json")
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous.get("source_cache_manifest_sha256") != source_sha or not args.resume:
            raise FileExistsError("Existing SFT result requires --resume with identical source")
        for name in ("support.npz", "sft_metrics.json"):
            if sha256(args.output_dir / name) != previous[name + "_sha256"]:
                raise ValueError(f"Changed SFT result: {name}")
        print(f"VERIFIED {manifest_path}", flush=True)
        return
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing nonempty output directory without SFT manifest")
    norm = normalization(train)
    z = obs_features(train["features"], train["state"], norm)
    a = act_features(train["expert_chunks"], norm)
    ix, d = nearest(z, z, args.device, exclude_episode=train["episode"],
                    reference_episode=train["episode"])
    action_distance = np.linalg.norm(a[:, None, :] - a[ix], axis=2)
    bandwidth = max(float(np.median(action_distance)), 1e-8)
    observation_bandwidth = max(float(np.median(np.sqrt(d))), 1e-8)
    support = dict(**norm, observations=z.astype("float32"), actions=a.astype("float32"),
                   action_bandwidth=np.asarray(bandwidth),
                   obs_bandwidth=np.asarray(observation_bandwidth), episode=train["episode"])
    atomic_npz(args.output_dir / "support.npz", support)
    # Score the exact same calibrated/diagnostic cohort as the neural ablation.
    def scores(part):
        obs = obs_features(part["features"], part["state"], norm)
        expert = score_rows(obs, part["expert_chunks"][:, None], support, args.device)[:, 0]
        policy = score_rows(obs, part["policy_chunks"], support, args.device)
        return expert, policy
    ecal, pcal = scores(cal)
    calibration = calibrate(ecal, pcal, beta_max=100.)
    e, p = scores(diag)
    w = np.stack([weights(row, calibration) for row in p])
    ess = 1 / np.sum(w * w, axis=1) / p.shape[1]
    shift = np.einsum("nk,nkhd->nhd", w, diag["policy_chunks"]) - diag["policy_chunks"].mean(1)
    rms = np.sqrt(np.mean(shift[:, :, JOINTS] ** 2, axis=(1, 2)))
    metrics = {"scope": "45 SFT validation episodes split 15/30; internal diagnostic, not independent heldout",
               "encoder_cache_manifest_sha256": source_sha,
               "calibration_anchors": 430, "diagnostic_anchors": 844,
               "calibration_episode_files": provenance["calibration_episodes"],
               "support_sha256": sha256(args.output_dir / "support.npz"),
               "method": "32-neighbor cross-episode expert observation retrieval plus full-chunk Gaussian action kernel",
               "action_bandwidth": bandwidth, "observation_bandwidth": observation_bandwidth,
               "rank": rank(e, p, diag["episode"]), "calibration": calibration,
               "diagnostic": {"expert_mean": float(e.mean()), "policy_mean": float(p.mean()),
                              "score_range_median": float(np.median(np.ptp(p, axis=1))),
                              "ess_fraction_median": float(np.median(ess)),
                              "tilted_mean_joint_rms_median": float(np.median(rms))}}
    atomic_json(args.output_dir / "sft_metrics.json", metrics)
    atomic_json(manifest_path, {"complete": True, "source_cache_manifest_sha256": source_sha,
                                "support.npz_sha256": metrics["support_sha256"],
                                "sft_metrics.json_sha256": sha256(args.output_dir / "sft_metrics.json")})
    print(f"SFT_DONE rank={metrics['rank']['episode_balanced_mean']:.6f} {manifest_path}", flush=True)


def run_hil(args):
    sft_manifest = read_json(args.output_dir / "sft_manifest.json")
    if not sft_manifest.get("complete") or sha256(args.output_dir / "support.npz") != sft_manifest["support.npz_sha256"]:
        raise ValueError("SFT kNN support incomplete or changed")
    cache_manifest = read_json(args.hil_cache / "manifest.json")
    source_manifest = read_json(args.source_cache / "candidate_manifest.json")
    if not cache_manifest.get("complete") or len(cache_manifest["episodes"]) != 50:
        raise ValueError("HIL encoded cohort incomplete")
    if sha256(args.source_cache / "candidate_manifest.json") != cache_manifest["config"]["source_candidate_manifest_sha256"]:
        raise ValueError("HIL source candidate manifest changed")
    source = {int(entry["rollout_index"]): entry for entry in source_manifest["identity"]["episodes"]}
    encoded = {int(entry["rollout_index"]): entry for entry in cache_manifest["episodes"]}
    if set(source) != set(encoded):
        raise ValueError("HIL cohort mismatch")
    result_path = args.output_dir / "hil_metrics.json"
    if result_path.exists():
        raise FileExistsError(f"HIL result exists: {result_path}")
    support = load_support(args.output_dir / "support.npz")
    norm = {k: support[k] for k in ("state_mean", "state_scale", "action_mean", "action_scale")}
    episodes = []
    for rid in sorted(encoded):
        entry = encoded[rid]
        path = args.hil_cache / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Changed HIL encoded shard {path}")
        with np.load(path, allow_pickle=False) as archive:
            frame = archive["frame_index"]
            if archive["candidates"].shape[1:] != (4, 10, 14):
                raise ValueError("HIL candidate shape changed")
            z = obs_features(archive["features"], archive["state"], norm)
            score = score_rows(z, archive["candidates"], support, args.device)
        episodes.append({"rollout_index": rid,
                         "relative_frame": (frame.astype(int) - int(source[rid]["takeover"])).tolist(),
                         "candidate_mean": score.mean(1).tolist(),
                         "candidate_range": np.ptp(score, axis=1).tolist()})
        print(f"HIL_SCORE {len(episodes)}/50 rollout={rid}", flush=True)
    sft_metrics = read_json(args.output_dir / "sft_metrics.json")
    metrics = {"scope": "50 saved HIL trajectories; takeover-aligned retrospective scoring, not intervention rate or success",
               "support_sha256": sft_manifest["support.npz_sha256"],
               "encoder_cache_manifest_sha256": sha256(args.hil_cache / "manifest.json"),
               "source_candidate_manifest_sha256": sha256(args.source_cache / "candidate_manifest.json"),
               "sft_calibration_scale": sft_metrics["calibration"]["scale"],
               **aggregate_hil(episodes, float(sft_metrics["calibration"]["scale"]))}
    if metrics["episodes"] != 50 or metrics["full_trajectory_observations"] != 3988:
        raise ValueError("HIL cohort observation count changed")
    atomic_json(result_path, metrics)
    atomic_json(args.output_dir / "hil_manifest.json", {"complete": True,
        "metrics_sha256": sha256(result_path),
        "encoder_cache_manifest_sha256": metrics["encoder_cache_manifest_sha256"]})
    print(f"HIL_DONE {result_path}", flush=True)


def self_test():
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(42)
    z = rng.normal(size=(100, 18))
    episode = np.repeat(np.arange(10), 10)
    ix, distance = nearest(z, z, "cpu", batch=13, exclude_episode=episode, reference_episode=episode)
    d = cdist(z, z, "sqeuclidean")
    d[episode[:, None] == episode[None]] = np.inf
    expected = np.argsort(d, axis=1)[:, :32]
    np.testing.assert_array_equal(ix, expected)
    np.testing.assert_allclose(distance, np.take_along_axis(d, expected, axis=1), atol=1e-10)
    # Score parity with the repository's original CPU SupportScorer formula.
    action = rng.normal(size=(100, 140)).astype("float32")
    bw = 2.3
    qix, _ = nearest(z[:8], z.astype("float32"), "cpu")
    candidate = rng.normal(size=(8, 4, 140))
    ref = np.exp(-cdist(candidate[0], action[qix[0]], "sqeuclidean") / (2 * bw * bw)).mean(1)
    near = torch.as_tensor(action[qix[0]], dtype=torch.float64)
    test = torch.exp(-torch.sum((torch.as_tensor(candidate[0])[:, None] - near[None]) ** 2, dim=-1) / (2 * bw * bw)).mean(1)
    np.testing.assert_allclose(test.numpy(), ref, atol=1e-12)
    print("SELF_TEST_OK", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("sft", "hil", "self-test"))
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--hil-cache", type=Path)
    p.add_argument("--source-cache", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.stage == "self-test":
        self_test()
    elif args.stage == "sft":
        if args.cache_dir is None or args.output_dir is None: p.error("sft requires --cache-dir and --output-dir")
        run_sft(args)
    else:
        if args.hil_cache is None or args.source_cache is None or args.output_dir is None:
            p.error("hil requires --hil-cache, --source-cache and --output-dir")
        run_hil(args)


if __name__ == "__main__":
    main()
