#!/usr/bin/env python3
"""Compare recorded-end terminal-reward critic ablations along demo steps."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import torch

from tail_value.cache import atomic_json, cache_fingerprint, load_episode, load_manifest, sha256
from tail_value.evaluate import score_episode
from tail_value.model import load_checkpoint, models_from_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--expected-steps", type=int, default=10000)
    return parser.parse_args(argv)


def recorded_return(length: int, gamma: float, terminal_reward: float) -> np.ndarray:
    """MC return of the *recorded* path under the assumed boundary; not policy Q."""
    values = np.empty(length - 1, dtype=np.float64)
    continuation = 0.0
    for t in range(length - 2, -1, -1):
        reward = terminal_reward if t == length - 2 else 1.0
        continuation = reward + gamma * continuation
        values[t] = continuation
    return values


def progress_profile(sequences, bins=20):
    per_episode = np.full((len(sequences), bins), np.nan)
    for i, seq in enumerate(sequences):
        progress = np.linspace(0.0, 1.0, len(seq))
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            members = seq[(progress >= lo) & (progress < hi if b < bins - 1 else progress <= hi)]
            if len(members):
                per_episode[i, b] = np.mean(members)
    return (np.arange(bins) + 0.5) / bins, per_episode


def absolute_profile(sequences):
    width = max(map(len, sequences))
    values = np.full((len(sequences), width), np.nan)
    for i, seq in enumerate(sequences):
        values[i, :len(seq)] = seq
    return np.arange(width), values


def profile_summary(values):
    result = {key: np.full(values.shape[1], np.nan) for key in ("median", "p10", "p90")}
    result["count"] = np.isfinite(values).sum(axis=0)
    for column in np.flatnonzero(result["count"]):
        observed = values[np.isfinite(values[:, column]), column]
        result["median"][column] = np.median(observed)
        result["p10"][column], result["p90"][column] = np.quantile(observed, [.1, .9])
    return result


def save_plot(profiles, title, xlabel, output, show_counts=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1.0: "#0072B2", 10.0: "#D55E00", 100.0: "#009E73"}
    fig, (ax, reference) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1]})
    for reward, data in sorted(profiles.items()):
        x = data["x"]
        predicted = profile_summary(data["predicted"])
        mc = profile_summary(data["recorded_mc"])
        color = colors[reward]
        ax.plot(x, predicted["median"], color=color, label=f"terminal reward {reward:g}")
        ax.fill_between(x, predicted["p10"], predicted["p90"], color=color, alpha=.12)
        reference.plot(x, mc["median"], color=color, label=f"R={reward:g}")
        if show_counts:
            # Count is displayed in the legend, not used to scale scores.
            ax.text(.99, .03 + .07 * list(sorted(profiles)).index(reward),
                    f"R={reward:g}: n at final step={int(predicted['count'][-1])}",
                    color=color, transform=ax.transAxes, ha="right", fontsize=8)
    ax.set(ylabel="Predicted B(s, expert action)", title=title)
    ax.legend(loc="best")
    ax.grid(alpha=.2)
    reference.set(xlabel=xlabel, ylabel="Recorded-path MC return")
    reference.grid(alpha=.2)
    reference.legend(loc="best", ncol=3, fontsize=8)
    fig.text(.02, .005, "Bands: 10th–90th percentile across episodes. MC is a constructed expert-path reference, not policy Q.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .025, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + suffix), dpi=180)
    plt.close(fig)


def main(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Report output already contains files: {args.output_dir}")
    torch.set_num_threads(args.cpu_threads)
    manifest = load_manifest(args.cache_dir)
    entries = [e for e in manifest["episodes"] if e["split"] == args.split]
    if not entries:
        raise ValueError(f"No {args.split} episodes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles_abs, profiles_progress, metrics = {}, {}, {}
    expected_fingerprint = cache_fingerprint(manifest)
    initialization = set()
    for reward in (1.0, 10.0, 100.0):
        path = args.runs_root / f"terminal_{int(reward)}" / "last.pt"
        payload = load_checkpoint(path)
        config = payload["config"]
        if (payload["step"] != args.expected_steps or config.get("record_end_terminal_reward") != reward
                or config.get("terminal_semantics") != "assumed_record_end_not_verified_success"
                or payload["cache_fingerprint"] != expected_fingerprint):
            raise ValueError(f"Checkpoint protocol mismatch: {path}")
        initialization.add(config["seed"])
        model, target = models_from_checkpoint(payload, args.device)
        predictions, references, terminal_scores, td_errors, gaps = [], [], [], [], []
        for entry in entries:
            arrays = load_episode(args.cache_dir / entry["file"], "demo", payload["compatibility"]["num_candidates"])
            result = score_episode(model, target, arrays, "demo", config["gamma"], args.batch_size,
                                   args.device, config["score_limit"], config["bootstrap"], reward)
            predicted = result["q_demo"][:-1].astype(np.float64)
            reference = recorded_return(len(arrays["state"]), config["gamma"], reward)
            predictions.append(predicted)
            references.append(reference)
            terminal_scores.append(float(predicted[-1]))
            td_errors.extend(result["td_squared_error"][:-1].tolist())
            gaps.extend((-result["gap"][:-1]).tolist())
        x_abs, pred_abs = absolute_profile(predictions)
        _, mc_abs = absolute_profile(references)
        x_progress, pred_progress = progress_profile(predictions)
        _, mc_progress = progress_profile(references)
        profiles_abs[reward] = {"x": x_abs, "predicted": pred_abs, "recorded_mc": mc_abs}
        profiles_progress[reward] = {"x": x_progress * 100, "predicted": pred_progress, "recorded_mc": mc_progress}
        first = np.asarray([seq[0] for seq in predictions])
        middle = np.asarray([seq[len(seq)//2] for seq in predictions])
        last = np.asarray(terminal_scores)
        metrics[str(int(reward))] = {
            "checkpoint": str(path.resolve()), "checkpoint_sha256": sha256(path),
            "episodes": len(predictions), "transitions": int(sum(map(len, predictions))),
            "predicted_first_median": float(np.median(first)),
            "predicted_middle_median": float(np.median(middle)),
            "predicted_last_median": float(np.median(last)),
            "last_reward_mae": float(np.mean(np.abs(last - reward))),
            "recorded_path_mc_mae": float(np.mean(np.concatenate([np.abs(a-b) for a,b in zip(predictions, references, strict=True)]))),
            "td_mse": float(np.mean(td_errors)),
            "expert_minus_candidate_mean": float(np.mean(gaps)),
        }
    if len(initialization) != 1:
        raise ValueError("Ablation checkpoints use different random seeds")
    save_plot(profiles_abs, f"{args.split} split: B along saved demonstration steps", "Saved transition step index",
              args.output_dir / "b_vs_absolute_step", show_counts=True)
    save_plot(profiles_progress, f"{args.split} split: B vs within-episode progress", "Recorded transition progress (%)",
              args.output_dir / "b_vs_progress")
    atomic_json(args.output_dir / "metrics.json", {
        "protocol": f"Same cache and seed, policy-next-candidate mean bootstrap; {args.expected_steps} optimizer steps each; nonterminal reward 1; final real adjacent recorded transition total reward R and no bootstrap.",
        "terminal_assumption": "Record end treated as terminal for ablation only; LeRobot metadata does not independently verify task success/done.",
        "mc_reference": "Constructed return along recorded expert transitions; not ground-truth policy Q because TD bootstraps next policy candidates.",
        "split": args.split, "cache_fingerprint": expected_fingerprint,
        "rewards": metrics,
    })
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main(parse_args())
