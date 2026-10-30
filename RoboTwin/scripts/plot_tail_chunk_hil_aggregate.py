#!/usr/bin/env python3
"""Cohort-level, takeover-aligned chunk-B diagnostic for saved HIL episodes.

The four full policy chunks at each saved observation are counterfactual
suggestions, *not* the actions executed during human control. The same cache
is scored by the 1k/10k/20k critics. Only the common frame window is plotted:
every plotted point includes every episode, with no interpolation or padding.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_tail_chunk_rollout_comparison import collect_one, iter_hil_frames, load_model
from tail_value.cache import PROMPT, atomic_json, read_json, sha256
from tail_value.prepare import FrozenEncoder
from tail_value.sources import add_xpolicylab_path


def cohort(source_manifest: Path):
    source = read_json(source_manifest)
    if not source.get("complete") or len(source["observations"]) != 1260:
        raise ValueError("Expected completed 1260-observation saved-HIL source manifest")
    paths = {Path(row["frame_path"]).parent.parent for row in source["observations"]}
    episodes = []
    for path in paths:
        metadata_path = path / "episode.json"
        meta = read_json(metadata_path)
        mask = meta["control_mask"]
        if "hil" not in mask:
            continue
        if meta.get("instruction") != PROMPT or not 0 < mask.index("hil") < len(mask) - 1:
            raise ValueError(f"Invalid HIL episode metadata: {path}")
        rollout = meta["record"]["rollout_index"]
        episodes.append({"rollout_index": int(rollout), "path": str(path),
                         "metadata_sha256": sha256(metadata_path),
                         "frames": len(mask), "takeover": mask.index("hil"),
                         "label": meta.get("supervisor_label")})
    episodes.sort(key=lambda row: row["rollout_index"])
    if len(episodes) != 50 or len({e["rollout_index"] for e in episodes}) != 50:
        raise ValueError(f"Expected exactly 50 distinct saved HIL episodes, found {len(episodes)}")
    return episodes


def sample(args):
    if args.num_candidates != 4 or args.policy_checkpoint is None:
        raise ValueError("Sampling needs K=4 and a physical baseline policy checkpoint")
    add_xpolicylab_path()
    from client_server.ws.model_client import WsModelClient

    episodes = cohort(args.source_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "candidate_manifest.json"
    identity = {"schema": 1, "source_manifest_sha256": sha256(args.source_manifest),
                "episodes": episodes, "policy_checkpoint": str(args.policy_checkpoint.resolve()),
                "policy_url": args.policy_url, "candidate_count": 4,
                "candidate_shape": [10, 14], "encoder": "frozen ImageNet ResNet18, 3 views",
                "semantics": "offline, newly resampled independent full policy chunks at all saved HIL observations"}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("Existing cache belongs to a different cohort/protocol")
    else:
        manifest = {"identity": identity, "completed": {}, "complete": False}
        atomic_json(manifest_path, manifest)
    for name, digest in manifest["completed"].items():
        path = args.output_dir / f"{name}.npz"
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Completed cache changed or missing: {path}")
    if len(manifest["completed"]) == len(episodes):
        manifest["complete"] = True
        atomic_json(manifest_path, manifest)
        return
    if not args.policy_checkpoint.is_dir():
        raise FileNotFoundError(args.policy_checkpoint)
    torch.set_num_threads(4)
    encoder = FrozenEncoder(args.encoder_device, args.encoder_weights)
    with WsModelClient(url=args.policy_url, evaluation_id="tail-chunk-hil-cohort",
                       trial_id="offline-takeover-aligned", request_timeout_s=600,
                       max_connect_attempts=3) as client:
        for episode in episodes:
            name = f"rollout_{episode['rollout_index']:04d}"
            if name in manifest["completed"]:
                continue
            output = args.output_dir / f"{name}.npz"
            n = collect_one(client, encoder, iter_hil_frames(Path(episode["path"])),
                            output, count=4, source=name)
            if n != episode["frames"]:
                raise ValueError(f"Saved-frame count changed for {name}: {n} vs {episode['frames']}")
            manifest["completed"][name] = sha256(output)
            atomic_json(manifest_path, manifest)
            print(f"SAMPLE {len(manifest['completed'])}/50 {name} frames={n}", flush=True)
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)


def plot_one(output_dir, step, x, raw, centered):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    for kind, data, ylabel in (
        ("raw", raw, "Mean B of 4 resampled full policy chunks"),
        ("delta", centered, "B minus own pre-takeover baseline"),
    ):
        mean = data.mean(axis=0)
        low, high = np.quantile(data, [.25, .75], axis=0)
        fig, ax = plt.subplots(figsize=(9.3, 5.3), layout="constrained")
        ax.axvspan(0, x[-1], color="#d95f0e", alpha=.09,
                   label="Human-controlled observations")
        ax.fill_between(x, low, high, color="#2166ac", alpha=.19,
                        label="Between-trajectory 25–75%")
        ax.plot(x, mean, color="#2166ac", linewidth=2.2,
                label=f"Across-trajectory mean (n={len(data)})")
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.1,
                   label="First takeover")
        if kind == "delta":
            ax.axhline(0, color="#777777", linewidth=.8)
        ax.set(title=f"Chunk-aware B at step {step:,}: 50 saved HIL takeover trajectories",
               xlabel="Saved-frame index relative to first takeover (not seconds)",
               ylabel=ylabel)
        ax.legend(frameon=False, fontsize=9, loc="best")
        ax.grid(alpha=.2)
        fig.savefig(output_dir / f"step_{step:06d}_hil_cohort_{kind}.png", dpi=180)
        fig.savefig(output_dir / f"step_{step:06d}_hil_cohort_{kind}.pdf")
        plt.close(fig)


def score(args):
    manifest = read_json(args.output_dir / "candidate_manifest.json")
    episodes = cohort(args.source_manifest)
    if not manifest.get("complete") or len(manifest["completed"]) != 50:
        raise ValueError("Need a complete 50-episode policy-candidate cache")
    if manifest["identity"]["episodes"] != episodes:
        raise ValueError("Source HIL episodes changed since candidate sampling")
    torch.set_num_threads(4)
    device = torch.device(args.score_device)
    models = dict(load_model(path, device) for path in args.checkpoint)
    if set(models) != {1000, 10000, 20000}:
        raise ValueError(f"Need exactly checkpoints 1000/10000/20000; got {set(models)}")
    first_pre = min(row["takeover"] for row in episodes)
    first_post = min(row["frames"] - row["takeover"] for row in episodes)
    x = np.arange(-first_pre, first_post, dtype=int)
    if first_pre < 10 or first_post < 40:
        raise ValueError("Insufficient common pre/post window for 50 episodes")
    cohort_scores = {step: [] for step in models}
    for episode in episodes:
        name = f"rollout_{episode['rollout_index']:04d}"
        path = args.output_dir / f"{name}.npz"
        if sha256(path) != manifest["completed"][name]:
            raise ValueError(f"Changed candidate cache: {path}")
        with np.load(path, allow_pickle=False) as archive:
            features = torch.from_numpy(archive["features"]).to(device)
            states = torch.from_numpy(archive["state"]).to(device)
            chunks = torch.from_numpy(archive["candidates"]).to(device)
            indices = archive["frame_index"].astype(int)
        n = episode["frames"]
        if features.shape != (n, 1536) or states.shape != (n, 14) or chunks.shape != (n, 4, 10, 14) or not np.array_equal(indices, np.arange(n)):
            raise ValueError(f"Invalid or unaligned cached arrays for {name}")
        for step, model in models.items():
            parts = []
            with torch.inference_mode():
                for start in range(0, n, 128):
                    sl = slice(start, start + 128)
                    parts.append(model.score_many(features[sl], states[sl], chunks[sl]).cpu().numpy())
            values = np.concatenate(parts).astype(float)
            if values.shape != (n, 4) or not np.isfinite(values).all():
                raise FloatingPointError(f"Invalid B scores for {name} at step {step}")
            center = episode["takeover"]
            cohort_scores[step].append(values[center + x].mean(axis=1))
        print(f"SCORE {name} frames={n}", flush=True)
    metrics = {"episodes": 50, "labels": dict(Counter(row["label"] for row in episodes)),
               "source_saved_frames": sum(row["frames"] for row in episodes),
               "common_window": {"first_frame": int(x[0]), "last_frame": int(x[-1]),
                                 "contributing_episodes_per_frame": 50},
               "x_semantics": "saved-frame index aligned to first takeover; not physical seconds",
               "y_semantics": "per-observation mean of 4 independently resampled policy full-chunk B scores; across-trajectory mean and IQR",
               "limitations": ["During HIL, these are counterfactual policy suggestions, not actual human actions",
                               "50 saved HIL episodes are a selected subset, not all attempts or an autonomous success-rate test",
                               "SFT critic trained on 15 Hz data, HIL online control ran at 30 Hz",
                               "B is an uncalibrated critic score, not a probability or coverage ground truth"],
               "checkpoint_results": {}}
    for step, rows in sorted(cohort_scores.items()):
        raw = np.stack(rows)
        baseline = raw[:, (x >= -10) & (x < 0)].mean(axis=1)
        centered = raw - baseline[:, None]
        plot_one(args.output_dir, step, x, raw, centered)
        last_pre = raw[:, (x >= -10) & (x < 0)].mean(axis=1)
        first_post_values = raw[:, (x >= 0) & (x < 10)].mean(axis=1)
        late_post = raw[:, (x >= 30) & (x < 40)].mean(axis=1)
        def describe(value):
            return {"mean": float(np.mean(value)), "median": float(np.median(value)),
                    "q25": float(np.quantile(value, .25)), "q75": float(np.quantile(value, .75))}
        metrics["checkpoint_results"][str(step)] = {
            "last10_pre_takeover": describe(last_pre),
            "first10_human_control": describe(first_post_values),
            "frames30to39_human_control": describe(late_post),
            "paired_first10_minus_last10": describe(first_post_values - last_pre),
            "paired_late_minus_last10": describe(late_post - last_pre)}
    atomic_json(args.output_dir / "hil_cohort_metrics.json", metrics)
    print(json.dumps(metrics["checkpoint_results"], indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("sample", "score"))
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-url", default="ws://127.0.0.1:18304")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--encoder-device", default="cpu")
    parser.add_argument("--encoder-weights", type=Path)
    parser.add_argument("--score-device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, nargs="*")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.phase == "sample":
        sample(arguments)
    else:
        if len(arguments.checkpoint or []) != 3:
            raise ValueError("Scoring requires three --checkpoint paths")
        score(arguments)
