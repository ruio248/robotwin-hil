#!/usr/bin/env python3
"""Complete retrospective diagnostics for frozen-encoder discriminator ablation.

All labels, observations and K4/K32 policy candidates are pre-existing. This
script only scores cached actions; it never executes a policy or simulator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from coverage_benchmark.common import JOINTS, weights
from coverage_benchmark.scorers import obs_features
from coverage_encoder_knn_ablation import load_support, nearest
from tail_value.cache import atomic_json, atomic_npz, read_json, sha256
from tail_value.chunk import ChunkCoverageCritic


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["representation"] != "absolute_joint_chunk" or payload["config"]["horizon"] != 10 or payload["step"] != 10000:
        raise ValueError("Wrong discriminator checkpoint")
    model = ChunkCoverageCritic(payload["normalization"], 10, 256)
    model.load_state_dict(payload["model"])
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def score(model, features, state, candidates, device, batch=64):
    out = []
    if len(features) != len(state) or len(features) != len(candidates):
        raise ValueError("Scoring arrays are not aligned")
    for start in range(0, len(features), batch):
        stop = start + batch
        f = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
        s = torch.as_tensor(state[start:stop], dtype=torch.float32, device=device)
        a = torch.as_tensor(candidates[start:stop], dtype=torch.float32, device=device)
        out.append(model.score_many(f, s, a).cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def observation_score(features, state, support, device):
    norm = {key: support[key] for key in ("state_mean", "state_scale", "action_mean", "action_scale")}
    z = obs_features(features, state, norm)
    _, d = nearest(z, support["observations"], device)
    bandwidth = float(support["obs_bandwidth"])
    return np.exp(-d / (2 * bandwidth * bandwidth)).mean(1)


def empirical_percentile(values, references, weights=None):
    reference = np.asarray(references, dtype=np.float64)
    order = np.argsort(reference, kind="stable")
    sorted_values = reference[order]
    w = np.ones(len(reference), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)[order]
    prefix = np.r_[0., np.cumsum(w)]
    left = np.searchsorted(sorted_values, values, side="left")
    right = np.searchsorted(sorted_values, values, side="right")
    return 100 * (prefix[left] + prefix[right]) / (2 * prefix[-1])


def bootstrap_mean(values):
    x = np.asarray(values, dtype=np.float64)
    draws = x[np.random.default_rng(42).integers(len(x), size=(10000, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(), "n": len(x)}


def auc(success, failure):
    a = np.asarray(success)[:, None]
    b = np.asarray(failure)[None]
    return float(np.mean((a > b) + .5 * (a == b)))


def historical_summary(episodes, scale):
    positive = [e for e in episodes if e["success"]]
    negative = [e for e in episodes if not e["success"]]
    if (len(positive), len(negative)) != (31, 69):
        raise ValueError("Historical label counts changed")
    result = {}
    for key in ("score_mean", "observation_mean"):
        values = np.asarray([e[key] for e in episodes])
        percentiles = empirical_percentile(values, values)
        pos = np.asarray([i for i, e in enumerate(episodes) if e["success"]])
        neg = np.asarray([i for i, e in enumerate(episodes) if not e["success"]])
        result[key] = {"success_mean": float(values[pos].mean()), "failure_mean": float(values[neg].mean()),
                       "success_minus_failure_raw": float(values[pos].mean() - values[neg].mean()),
                       "success_minus_failure_percentile_pp": float(percentiles[pos].mean() - percentiles[neg].mean()),
                       "auc": auc(values[pos], values[neg])}
        if key == "score_mean":
            result[key]["success_minus_failure_sft_scale"] = result[key]["success_minus_failure_raw"] / scale
    # Pairwise episode-stratified bootstrap retains each method/control score pair.
    v = np.asarray([e["score_mean"] for e in episodes]); b = np.asarray([e["observation_mean"] for e in episodes])
    pos = np.flatnonzero([e["success"] for e in episodes]); neg = np.flatnonzero([not e["success"] for e in episodes])
    rng = np.random.default_rng(42); differences = []
    for _ in range(10000):
        p = rng.choice(pos, len(pos), replace=True); n = rng.choice(neg, len(neg), replace=True)
        differences.append(auc(v[p], v[n]) - auc(b[p], b[n]))
    result["score_minus_observation_auc"] = {"difference": result["score_mean"]["auc"] - result["observation_mean"]["auc"],
        "stratified_episode_bootstrap_ci95": np.quantile(differences, [.025, .975]).tolist()}
    return result


def hil_summary(episodes):
    # A globally comparable rank within EACH method. Each HIL episode gets
    # equal weight in the percentile reference, regardless of its length.
    for key in ("score", "observation"):
        pool = np.concatenate([episode[key] for episode in episodes])
        weight = np.concatenate([np.full(len(episode[key]), 1 / len(episode[key])) for episode in episodes])
        for episode in episodes:
            episode[key + "_percentile"] = empirical_percentile(episode[key], pool, weight)
    rows = []
    for episode in episodes:
        rel = episode["relative_frame"]
        early = (rel >= -15) & (rel <= -11)
        late = (rel >= -5) & (rel <= -1)
        human = (rel >= 30) & (rel <= 39)
        if not early.all() and not (early.sum() == 5 and late.sum() == 5 and human.sum() == 10):
            raise ValueError(f"Missing fixed HIL windows: {episode['rollout_index']}")
        a = episode["score_percentile"]
        b = episode["observation_percentile"]
        score_drop = float(a[late].mean() - a[early].mean())
        obs_drop = float(b[late].mean() - b[early].mean())
        score_rise = float(a[human].mean() - a[late].mean())
        obs_rise = float(b[human].mean() - b[late].mean())
        rows.append({"rollout_index": episode["rollout_index"], "score_drop_pp": score_drop,
                     "observation_drop_pp": obs_drop, "excess_drop_pp": obs_drop - score_drop,
                     "score_human_rise_pp": score_rise, "observation_human_rise_pp": obs_rise,
                     "excess_human_rise_pp": score_rise - obs_rise})
    return {"windows": {"early": [-15, -11], "late": [-5, -1], "human": [30, 39]},
            "percentile_reference": "episode-equal empirical CDF over all saved HIL frames, separately per scorer and observation control",
            "summary": {key: bootstrap_mean([row[key] for row in rows]) for key in rows[0] if key != "rollout_index"},
            "per_episode": rows}


def k32_summary(scores, candidates, calibration):
    scale = float(calibration["scale"])
    ranges = np.ptp(scores, axis=1); std = np.std(scores, axis=1)
    w = np.stack([weights(row, calibration) for row in scores])
    ess = 1 / np.sum(w * w, axis=1) / scores.shape[1]
    shift = np.einsum("nk,nkhd->nhd", w, candidates) - candidates.mean(1)
    rms = np.sqrt(np.mean(shift[:, :, JOINTS] ** 2, axis=(1, 2)))
    rho = []
    for a, score_row in zip(candidates, scores, strict=True):
        distance = pdist(a[:, :, JOINTS].reshape(len(a), -1))
        score_gap = pdist(score_row[:, None])
        if np.ptp(distance) < 1e-12 or np.ptp(score_gap) < 1e-12:
            continue
        value = float(spearmanr(distance, score_gap).statistic)
        if np.isfinite(value): rho.append(value)
    return {"observations": len(scores), "candidate_count": 32,
            "raw_range_median": float(np.median(ranges)),
            "normalized_range_median": float(np.median(ranges / scale)),
            "normalized_std_median": float(np.median(std / scale)),
            "action_gap_spearman_median": float(np.median(rho)) if rho else None,
            "action_gap_spearman_defined": len(rho),
            "ess_fraction_median": float(np.median(ess)),
            "weighted_minus_uniform_12_joint_rms_mrad_median": float(np.median(rms) * 1000),
            "score_min": float(scores.min()), "score_max": float(scores.max())}


def run(args):
    sft = read_json(args.sft_eval)
    info = sft["models"]["discriminator10k"]
    if sha256(args.model) != info["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from frozen SFT evaluation")
    support = load_support(args.support)
    if sha256(args.support) != read_json(args.support_manifest)["support.npz_sha256"]:
        raise ValueError("Observation control support changed")
    source = read_json(args.hil_source_manifest)
    hm = read_json(args.hil_cache / "manifest.json")
    hist = read_json(args.historical_cache / "manifest.json")
    km = read_json(args.k32 / "manifest.json")
    if not hm.get("complete") or len(hm["episodes"]) != 50 or not hist.get("complete") or len(hist["episodes"]) != 100:
        raise ValueError("HIL or historical encoder cache incomplete")
    if not km.get("complete") or len(km["observations"]) != 1260:
        raise ValueError("K32 candidate cache incomplete")
    if sha256(args.hil_source_manifest) != hm["config"]["source_candidate_manifest_sha256"]:
        raise ValueError("HIL source manifest mismatch")
    if sft["encoder_cache_manifest_sha256"] != read_json(args.support_manifest)["source_cache_manifest_sha256"]:
        raise ValueError("SFT scorer/support encoder cohort mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite existing diagnostic")
    model = load_model(args.model, args.device)
    hil_lookup = {}; episodes = []
    takeovers = {int(entry["rollout_index"]): int(entry["takeover"]) for entry in source["identity"]["episodes"]}
    for entry in sorted(hm["episodes"], key=lambda x: x["rollout_index"]):
        rid = int(entry["rollout_index"]); path = args.hil_cache / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Changed HIL encoded shard {path}")
        with np.load(path, allow_pickle=False) as archive:
            a = {key: archive[key] for key in archive.files}
        disc = score(model, a["features"], a["state"], a["candidates"], args.device).mean(1)
        obs = observation_score(a["features"], a["state"], support, args.device)
        frame = a["frame_index"].astype(int)
        for i, f in enumerate(frame):
            hil_lookup[(rid, int(f))] = (a["features"][i], a["state"][i])
        episodes.append({"rollout_index": rid, "relative_frame": frame - takeovers[rid],
                         "score": disc, "observation": obs})
    if len(hil_lookup) != 3988:
        raise ValueError("HIL full observation count changed")
    hil_result = hil_summary(episodes)
    print("HIL_DONE", flush=True)
    k_features, k_state, k_candidates, k_episode, k_frame = [], [], [], [], []
    for row in km["observations"]:
        key = (int(row["rollout_index"]), int(Path(row["frame_path"]).stem))
        if key not in hil_lookup:
            continue
        path = args.k32 / row["cache"]
        if sha256(path) != km["completed"][row["cache"]]:
            raise ValueError(f"Changed K32 candidate shard {path}")
        with np.load(path, allow_pickle=False) as archive:
            candidate = archive["candidates"]
            state = archive["state"]
        f, s = hil_lookup[key]
        if candidate.shape != (32, 10, 14) or not np.allclose(s, state, atol=1e-6, rtol=0):
            raise ValueError(f"K32/HIL feature-state alignment failure {key}")
        k_features.append(f); k_state.append(s); k_candidates.append(candidate)
        k_episode.append(key[0]); k_frame.append(key[1])
    if len(k_features) != 1233:
        raise ValueError("Expected 1233 K32 pre-takeover observations")
    k_candidates = np.stack(k_candidates)
    k_score = score(model, np.stack(k_features), np.stack(k_state), k_candidates, args.device, batch=24)
    k_result = k32_summary(k_score, k_candidates, info["calibration"])
    atomic_npz(args.output_dir / "k32_scores.npz", {"scores": k_score.astype(np.float32),
        "rollout_index": np.asarray(k_episode), "frame_index": np.asarray(k_frame)})
    print("K32_DONE", flush=True)
    historical = []
    for entry in sorted(hist["episodes"], key=lambda x: x["name"]):
        path = args.historical_cache / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Changed historical encoded shard {path}")
        with np.load(path, allow_pickle=False) as archive:
            a = {key: archive[key] for key in archive.files}
        disc = score(model, a["features"], a["state"], a["candidates"], args.device).mean(1)
        obs = observation_score(a["features"], a["state"], support, args.device)
        historical.append({"name": entry["name"], "success": bool(entry["success"]),
                           "frames": len(disc), "score_mean": float(disc.mean()),
                           "observation_mean": float(obs.mean())})
    if sum(e["frames"] for e in historical) != 4834:
        raise ValueError("Historical observation count changed")
    history_result = historical_summary(historical, float(info["calibration"]["scale"]))
    print("HISTORICAL_DONE", flush=True)
    result = {"scope": "Frozen-encoder discriminator; retrospective offline diagnostics only, not a new online success evaluation",
              "source_sha256": {"sft_eval": sha256(args.sft_eval), "model": sha256(args.model),
                                "support": sha256(args.support), "hil_cache_manifest": sha256(args.hil_cache / "manifest.json"),
                                "historical_cache_manifest": sha256(args.historical_cache / "manifest.json"),
                                "hil_source_manifest": sha256(args.hil_source_manifest), "k32_manifest": sha256(args.k32 / "manifest.json")},
              "sft_internal": {"expert_rank": info["rank"], "calibration": info["calibration"],
                               "diagnostic": info["diagnostic"]},
              "hil_k32": k_result, "hil_takeover": hil_result,
              "historical": {"episodes": 100, "observations": 4834,
                             "summary": history_result, "per_episode": historical},
              "caveats": ["HIL human-late window is within recorded human control, not post-handback recovery",
                          "Historical success/failure cohorts differ in trajectory length and phase occupancy",
                          "Observation-only baseline cannot rank actions at the same observation",
                          "No simulator branch test, manual takeover rate or new success rate was measured"]}
    atomic_json(args.output_dir / "metrics.json", result)
    atomic_json(args.output_dir / "manifest.json", {"complete": True,
        "metrics_sha256": sha256(args.output_dir / "metrics.json"),
        "k32_scores_sha256": sha256(args.output_dir / "k32_scores.npz"),
        "source_sha256": result["source_sha256"]})
    print(f"COMPLETE {args.output_dir / 'metrics.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("sft_eval", "model", "support", "support_manifest", "hil_cache",
                "hil_source_manifest", "historical_cache", "k32", "output_dir"):
        p.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
