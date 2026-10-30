#!/usr/bin/env python3
"""Offline chunk-B snapshots on the same rollout/HIL observations.

The old rollout HDF5 contains sparse single actions, not executed chunks.
This script therefore *resamples* full [10,14] policy suggestions at each
saved observation. It never calls a simulator and never labels suggestions as
executed actions. The candidate cache is shared by every B checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tail_value.cache import CAMERAS, PROMPT, atomic_json, atomic_npz, read_json, sha256
from tail_value.chunk import ChunkCoverageCritic
from tail_value.prepare import FrozenEncoder
from tail_value.sources import JOINT_FIELDS, NATIVE_CAMERAS, RAW_CAMERAS, add_xpolicylab_path


def physical_chunk(raw):
    if isinstance(raw, dict) and "actions" in raw:
        raw = raw["actions"]
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], dict):
        value = np.stack([np.concatenate([np.asarray(step[k.removesuffix("s")], dtype=np.float32).reshape(-1)
                                          for k in JOINT_FIELDS]) for step in raw])
    else:
        value = np.asarray(raw, dtype=np.float32)
    if value.shape != (10, 14) or not np.isfinite(value).all():
        raise ValueError(f"Expected finite physical [10,14] policy chunk, got {value.shape}")
    return value


def iter_rollout_frames(path):
    from XPolicyLab.utils.process_data import decode_image_bit
    with h5py.File(path, "r") as handle:
        vectors = [np.asarray(handle[f"state/{k}"], dtype=np.float32).reshape(len(handle[f"state/{k}"]), -1)
                   for k in JOINT_FIELDS]
        states = np.concatenate(vectors, axis=1)
        cameras = [handle[f"vision/{camera}/colors"] for camera in NATIVE_CAMERAS]
        if states.ndim != 2 or states.shape[1] != 14 or any(len(camera) != len(states) for camera in cameras):
            raise ValueError(f"Unaligned HDF5 state/cameras: {path}")
        for i, state in enumerate(states):
            images = {key: decode_image_bit(camera[i]) for key, camera in zip(CAMERAS, cameras, strict=True)}
            yield i, state, images


def iter_hil_frames(path):
    metadata = read_json(path / "episode.json")
    mask = metadata["control_mask"]
    files = sorted((path / "frames").glob("*.pkl"), key=lambda p: int(p.stem))
    if len(files) != len(mask) or [int(p.stem) for p in files] != list(range(len(files))):
        raise ValueError("Unaligned HIL control mask and saved frames")
    if "hil" not in mask or not 0 < mask.index("hil") < len(files) - 1:
        raise ValueError("HIL episode needs a first takeover and post-takeover frames")
    for i, file in enumerate(files):
        with file.open("rb") as stream:
            record = pickle.load(stream)  # only trusted locally collected HIL pickle files
        state = np.asarray(record["joint_action"]["vector"], dtype=np.float32)
        images = {key: np.asarray(record["observation"][camera]["rgb"])
                  for key, camera in zip(CAMERAS, RAW_CAMERAS, strict=True)}
        yield i, state, images


def collect_one(client, encoder, frames, output, *, count, source):
    features, states, candidates, indices = [], [], [], []
    client.call(func_name="reset")
    for i, state, images in frames:
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(f"Invalid state at {source}:{i}")
        client.call(func_name="update_obs", obs={"images": images, "state": state, "instruction": PROMPT})
        raw = client.call(func_name="get_action_candidates", obs=count)
        if not isinstance(raw, (list, tuple)) or len(raw) != count:
            raise ValueError(f"Expected {count} independent policy chunks at {source}:{i}")
        pool = np.stack([physical_chunk(item) for item in raw])
        features.append(encoder(images))
        states.append(state)
        candidates.append(pool)
        indices.append(i)
    data = {"features": np.stack(features).astype(np.float32),
            "state": np.stack(states).astype(np.float32),
            "candidates": np.stack(candidates).astype(np.float32),
            "frame_index": np.asarray(indices, dtype=np.int32)}
    atomic_npz(output, data)
    return len(indices)


def sample(args):
    add_xpolicylab_path()
    from client_server.ws.model_client import WsModelClient
    rollout_files = sorted((args.rollout_root / "data").glob("episode_*.hdf5"))
    records = [json.loads(line) for line in (args.rollout_root / "eval_records.jsonl").read_text().splitlines() if line.strip()]
    by_stem = {Path(r["hdf5_path"]).stem: r for r in records}
    if len(rollout_files) != 100 or len(by_stem) != 100 or any(p.stem not in by_stem for p in rollout_files):
        raise ValueError("Expected the same 100 labeled baseline rollouts")
    if sum(bool(by_stem[p.stem]["success"]) for p in rollout_files) != 31:
        raise ValueError("Old figure cohort must contain exactly 31 successes / 69 failures")
    hil_meta = read_json(args.hil_episode / "episode.json")
    if hil_meta.get("instruction") != PROMPT:
        raise ValueError("HIL task prompt mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "candidate_manifest.json"
    identity = {"schema_version": 1, "rollout_root": str(args.rollout_root.resolve()),
                "hil_episode": str(args.hil_episode.resolve()), "policy_url": args.policy_url,
                "policy_checkpoint": str(args.policy_checkpoint.resolve()), "policy_count": args.num_candidates,
                "action_semantics": "offline same-observation independent full [10,14] policy suggestions; not executed action",
                "encoder": "ImageNet ResNet18, 3 views, long-side 224 + zero pad",
                "source": "same 100 baseline rollouts as old 31/69 figure plus one saved HIL episode"}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("Existing candidate cache has a different protocol")
    else:
        manifest = {"identity": identity, "completed": {}, "complete": False}
        atomic_json(manifest_path, manifest)
    todo = [(p.stem, p, "rollout") for p in rollout_files] + [("hil_event", args.hil_episode, "hil")]
    for name, _, _ in todo:
        cache = args.output_dir / f"{name}.npz"
        if name in manifest["completed"] and (not cache.is_file() or sha256(cache) != manifest["completed"][name]):
            raise ValueError(f"Completed cache missing or changed: {cache}")
    if len(manifest["completed"]) == len(todo):
        manifest["complete"] = True
        atomic_json(manifest_path, manifest)
        return
    if not args.policy_checkpoint.is_dir():
        raise FileNotFoundError(args.policy_checkpoint)
    torch.set_num_threads(4)
    encoder = FrozenEncoder(args.encoder_device, args.encoder_weights)
    with WsModelClient(url=args.policy_url, evaluation_id="tail-chunk-rollout-diagnostic",
                       trial_id="offline-fixed-cohort", request_timeout_s=600,
                       max_connect_attempts=3) as client:
        for name, path, kind in todo:
            if name in manifest["completed"]:
                continue
            frames = iter_rollout_frames(path) if kind == "rollout" else iter_hil_frames(path)
            n = collect_one(client, encoder, frames, args.output_dir / f"{name}.npz",
                            count=args.num_candidates, source=name)
            manifest["completed"][name] = sha256(args.output_dir / f"{name}.npz")
            atomic_json(manifest_path, manifest)
            print(f"SAMPLE {len(manifest['completed'])}/{len(todo)} {name} frames={n}", flush=True)
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    if config.get("bootstrap") != "policy_next_chunk_mean_h10" or config.get("horizon") != 10:
        raise ValueError(f"Not a compatible chunk-aware B checkpoint: {path}")
    model = ChunkCoverageCritic(payload["normalization"], horizon=10, width=int(config["width"]))
    model.load_state_dict(payload["model"])
    return int(payload["step"]), model.eval().to(device).requires_grad_(False)


def score(args):
    manifest = read_json(args.output_dir / "candidate_manifest.json")
    if not manifest.get("complete") or len(manifest["completed"]) != 101:
        raise ValueError("Need all 100 rollouts and one HIL event with the same cached policy candidates")
    torch.set_num_threads(4)
    device = torch.device(args.score_device)
    models = dict(load_model(p, device) for p in args.checkpoint)
    if set(models) != {1000, 10000, 20000}:
        raise ValueError(f"Need exactly checkpoints 1000/10000/20000; found {set(models)}")
    records = [json.loads(line) for line in (args.rollout_root / "eval_records.jsonl").read_text().splitlines() if line.strip()]
    labels = {Path(r["hdf5_path"]).stem: bool(r["success"]) for r in records}
    rows = {step: [] for step in models}
    for name, expected in sorted(manifest["completed"].items()):
        path = args.output_dir / f"{name}.npz"
        if sha256(path) != expected:
            raise ValueError(f"Changed candidate cache {path}")
        with np.load(path, allow_pickle=False) as archive:
            features = torch.from_numpy(archive["features"]).to(device)
            states = torch.from_numpy(archive["state"]).to(device)
            chunks = torch.from_numpy(archive["candidates"]).to(device)
            indices = archive["frame_index"].astype(int)
        if chunks.shape != (len(indices), 4, 10, 14) or features.shape != (len(indices), 1536):
            raise ValueError(f"Invalid cached chunk shapes for {name}")
        for step, model in models.items():
            parts = []
            with torch.inference_mode():
                for start in range(0, len(indices), 128):
                    sl = slice(start, start + 128)
                    parts.append(model.score_many(features[sl], states[sl], chunks[sl]).cpu().numpy())
            values = np.concatenate(parts).astype(float)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Non-finite B scores for {name} at step {step}")
            rows[step].append({"episode": name, "success": labels.get(name),
                               "frame_index": indices.tolist(), "candidate_scores": values.tolist()})
        print(f"SCORE {name} frames={len(indices)}", flush=True)
    atomic_json(args.output_dir / "scores_by_checkpoint.json", {str(k): v for k, v in rows.items()})
    plot(rows, args.output_dir, read_json(args.hil_episode / "episode.json"))


def plot(rows, output, hil_meta):
    takeover = hil_meta["control_mask"].index("hil")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    results = {}
    for step, episodes in sorted(rows.items()):
        rollout = [r for r in episodes if r["episode"] != "hil_event"]
        hil = next(r for r in episodes if r["episode"] == "hil_event")
        if len(rollout) != 100:
            raise ValueError("Missing rollout cohort")
        by_outcome = {}
        fig, ax = plt.subplots(figsize=(9.3, 5.3), layout="constrained")
        for success, color, label in [(True, "#2166ac", "Success (n=31)"),
                                      (False, "#d95f0e", "Failure (n=69)")]:
            selected = [r for r in rollout if r["success"] is success]
            if len(selected) != (31 if success else 69):
                raise ValueError("Changed success/failure labels")
            episode_means = [float(np.asarray(r["candidate_scores"], dtype=float).mean()) for r in selected]
            prefix16 = [float(np.asarray(r["candidate_scores"], dtype=float)[:16].mean()) for r in selected]
            lengths = [len(r["frame_index"]) for r in selected]
            if min(lengths) < 16:
                raise ValueError("Matched-prefix control needs at least 16 saved frames per rollout")
            by_outcome["success" if success else "failure"] = {
                "n": len(selected), "episode_mean_score_mean": float(np.mean(episode_means)),
                "episode_mean_score_median": float(np.median(episode_means)),
                "prefix16_score_mean": float(np.mean(prefix16)),
                "saved_frames_mean": float(np.mean(lengths)),
                "saved_frames_median": float(np.median(lengths))}
            x, mean, low, high = [], [], [], []
            for j in range(10):
                per_episode = []
                for r in selected:
                    values = np.asarray(r["candidate_scores"], dtype=float).mean(axis=1)
                    phase = np.linspace(0, 1, len(values))
                    mask = (phase >= j / 10) & ((phase < (j + 1) / 10) if j < 9 else (phase <= 1))
                    if mask.any():
                        per_episode.append(float(values[mask].mean()))
                x.append((j + .5) / 10)
                mean.append(np.mean(per_episode)); low.append(np.quantile(per_episode, .25)); high.append(np.quantile(per_episode, .75))
            ax.plot(x, mean, color=color, marker="o", linewidth=2, label=label)
            ax.fill_between(x, low, high, color=color, alpha=.15)
        ax.set(title=f"Chunk-aware B at step {step:,}: baseline policy rollouts\n"
                     "Success median 18 saved frames; failure median 60",
               xlabel="Normalized progress within recorded rollout", ylabel="Mean B of 4 resampled policy chunks")
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.savefig(output / f"step_{step:06d}_success_failure.png", dpi=180)
        fig.savefig(output / f"step_{step:06d}_success_failure.pdf")
        plt.close(fig)
        values = np.asarray(hil["candidate_scores"], dtype=float)
        x = np.asarray(hil["frame_index"], dtype=int) - takeover
        if len(x) <= takeover or len(x) != len(hil_meta["control_mask"]):
            raise ValueError("HIL frame count or takeover boundary mismatch")
        fig, ax = plt.subplots(figsize=(9.3, 5.3), layout="constrained")
        ax.axvspan(0, float(x.max()), color="#d95f0e", alpha=.09, label="Human-controlled observations")
        ax.plot(x, values.mean(axis=1), color="#2166ac", linewidth=2, label="Mean of 4 policy suggestions")
        ax.fill_between(x, np.quantile(values, .05, axis=1), np.quantile(values, .95, axis=1),
                        color="#2166ac", alpha=.17, label="Candidate 5–95%")
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1, label="First takeover")
        ax.set(title=f"Chunk-aware B at step {step:,}: one saved HIL takeover",
               xlabel="Saved-frame index relative to first takeover (not seconds)",
               ylabel="B of resampled policy chunks at recorded observations")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=.2)
        fig.savefig(output / f"step_{step:06d}_hil_takeover.png", dpi=180)
        fig.savefig(output / f"step_{step:06d}_hil_takeover.pdf")
        plt.close(fig)
        pre = values[:takeover]
        post = values[takeover:]
        results[str(step)] = {"rollout_by_outcome": by_outcome,
                              "success_minus_failure_episode_mean": float(
                                  by_outcome["success"]["episode_mean_score_mean"]
                                  - by_outcome["failure"]["episode_mean_score_mean"]),
                              "success_minus_failure_matched_prefix16_mean": float(
                                  by_outcome["success"]["prefix16_score_mean"]
                                  - by_outcome["failure"]["prefix16_score_mean"]),
                              "hil_policy_suggestion_mean_pre_takeover": float(pre.mean()),
                              "hil_policy_suggestion_mean_during_human_control": float(post.mean()),
                              "hil_candidate_range_median_pre_takeover": float(np.median(np.ptp(pre, axis=1))),
                              "hil_candidate_range_median_during_human_control": float(np.median(np.ptp(post, axis=1)))}
    atomic_json(output / "comparison_metrics.json", {
        "steps": sorted(rows), "rollouts": 100, "success": 31, "failure": 69,
        "hil_episode": hil_meta["episode_index"], "hil_first_takeover_frame": takeover,
        "checkpoint_results": results,
        "action_semantics": "4 newly resampled full policy chunks at each old recorded observation; none is verified as executed",
        "limitations": ["Rollout outcomes label recorded original trajectories, not resampled counterfactual chunks",
                        "HIL suggested-policy scores are not the actions actually commanded during human control",
                        "SFT critic trained at 15 Hz; HIL control ran at 30 Hz",
                        "Absolute B level is shaped by gamma=.99 and no terminal, not calibrated success probability"]})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("sample", "score"))
    p.add_argument("--rollout-root", type=Path, required=True)
    p.add_argument("--hil-episode", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--policy-url", default="ws://127.0.0.1:18304")
    p.add_argument("--policy-checkpoint", type=Path)
    p.add_argument("--num-candidates", type=int, default=4)
    p.add_argument("--encoder-device", default="cuda")
    p.add_argument("--encoder-weights", type=Path)
    p.add_argument("--score-device", default="cpu")
    p.add_argument("--checkpoint", type=Path, nargs="*")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "sample":
        if args.num_candidates != 4 or not args.policy_checkpoint:
            raise ValueError("Sampling requires K=4 and --policy-checkpoint")
        sample(args)
    else:
        if len(args.checkpoint or []) != 3:
            raise ValueError("Scoring requires three --checkpoint paths")
        score(args)
