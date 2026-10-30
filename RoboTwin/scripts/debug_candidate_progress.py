#!/usr/bin/env python3
"""Offline same-observation K=4/8/16/32 diagnosis. Never steps a simulator.

Only trusted, locally collected pickle observations may be used as input.
One pool of 32 draws per observation, sampled with fixed RPC microbatches,
is randomly permuted and reused as nested prefixes for paired K comparisons.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np

from tail_value.cache import atomic_json, atomic_npz, json_digest, sha256

KS = (4, 8, 16, 32)
JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
CAMERAS = ("head_camera", "left_camera", "right_camera")
SCHEMA = 1


def observation_schedule(metadata, frame_paths):
    """Map saved pre-HIL observations to policy command counts, failing closed.

    This collection saves frame zero, then every save_freq policy commands.
    We exclude all HIL/post-HIL observations; no mixed frame-rate normalization.
    """
    record = metadata["record"]
    mask = metadata["control_mask"]
    if len(frame_paths) != len(mask) or not frame_paths:
        raise ValueError("Frame/control_mask length mismatch")
    if [int(p.stem) for p in frame_paths] != list(range(len(frame_paths))):
        raise ValueError("Frame IDs must be contiguous from zero")
    if not set(mask) <= {"policy", "hil"}:
        raise ValueError("Unknown control source")
    first_hil = mask.index("hil") if "hil" in mask else len(mask)
    if any(s != "policy" for s in mask[:first_hil]):
        raise ValueError("Not a policy prefix")
    interval = int(metadata["save_freq"])
    if interval <= 0:
        raise ValueError("Invalid saved-frame interval")
    interventions = record.get("interventions", [])
    end = (int(interventions[0]["policy_steps_before_intervention"])
           if first_hil < len(mask) and interventions else int(record["policy_steps"]))
    if first_hil < len(mask) and not interventions:
        raise ValueError("HIL frames without intervention boundary")
    if end <= 0 or first_hil != end // interval + 1:
        raise ValueError(f"Cannot recover exact policy-step mapping: frames={first_hil}, end={end}, interval={interval}")
    if int(record["policy_steps"]) < end:
        raise ValueError("Final policy count precedes first intervention")
    return [{"frame_index": i, "policy_step": i * interval,
             "policy_end_step": end, "progress": i * interval / end,
             "frame_path": str(frame_paths[i])} for i in range(first_hil)]


def load_observation(path):
    with Path(path).open("rb") as handle:
        obs = pickle.load(handle)  # trusted local collection only
    state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Expected finite physical 14D observation state")
    # Verify the same vector assembled by the online adapter (not actual qpos).
    joint = obs["joint_action"]
    assembled = np.concatenate([np.asarray(joint[k]).reshape(-1) for k in
                                ("left_arm", "left_gripper", "right_arm", "right_gripper")])
    np.testing.assert_allclose(assembled, state, rtol=0, atol=1e-6)
    for key in CAMERAS:
        rgb = np.asarray(obs["observation"][key]["rgb"])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected HWC uint8 RGB for {key}")
    return obs


def sample_pool(client, observation, sample_batch, batch_size=4):
    if batch_size not in (2, 4, 8):
        raise ValueError("RPC batch must be 2, 4 or 8")
    chunks = []
    for _ in range(32 // batch_size):
        batch = np.asarray(sample_batch(client, observation, "joint", batch_size), dtype=np.float32)
        if batch.shape != (batch_size, 10, 14) or not np.isfinite(batch).all():
            raise ValueError(f"Invalid candidates {batch.shape}, expected [{batch_size},10,14]")
        chunks.append(batch)
    pool = np.concatenate(chunks)
    # Identical outputs can be a real policy-collapse/quantization observation.
    # Record uniqueness below, but never manufacture diversity or reject it.
    return pool


def paired_indices(seed, rollout, frame):
    return np.random.default_rng(np.random.SeedSequence([seed, rollout, frame])).permutation(32)


def candidate_metrics(chunks, indices, scores=None, beta=10.0):
    chunks = np.asarray(chunks, dtype=np.float64)
    if chunks.shape != (32, 10, 14) or not np.isfinite(chunks).all():
        raise ValueError("Expected finite [32,10,14] pool")
    if sorted(np.asarray(indices).tolist()) != list(range(32)):
        raise ValueError("Invalid nested subset permutation")
    result = {}
    for k in KS:
        selected = indices[:k]
        a = chunks[selected, 0]
        upper = np.triu_indices(k, 1)
        delta = a[:, None, JOINTS] - a[None, :, JOINTS]
        distances = np.sqrt(np.mean(delta ** 2, axis=-1))[upper]
        grip = np.abs(a[:, None, [6, 13]] - a[None, :, [6, 13]])
        item = {"mean_pairwise_rad": float(distances.mean()),
                "max_pairwise_rad": float(distances.max()),
                "unique_first_actions": int(len(np.unique(a, axis=0))),
                "left_gripper_mean_abs": float(grip[..., 0][upper].mean()),
                "right_gripper_mean_abs": float(grip[..., 1][upper].mean())}
        if scores is not None:
            b = np.asarray(scores, dtype=np.float64)[selected]
            if not np.isfinite(b).all():
                raise ValueError("Non-finite coverage score")
            logits = -beta * (b - b.min())
            w = np.exp(logits - logits.max()); w /= w.sum()
            shift = np.sum(w[:, None] * a, axis=0) - a.mean(axis=0)
            item.update({"coverage_mean": float(b.mean()), "coverage_range": float(np.ptp(b)),
                         "coverage_std": float(b.std()), "max_weight": float(w.max()),
                         "ess_fraction": float(1 / np.sum(w ** 2) / k),
                         "tilted_vs_uniform_rad": float(np.sqrt(np.mean(shift[JOINTS] ** 2)))})
        result[str(k)] = item
    return result


def aggregate(rows, metric, bins=20):
    result = []
    for i in range(bins):
        subset = [r for r in rows if min(int(r["progress"] * bins), bins - 1) == i]
        ids = sorted({r["rollout_index"] for r in subset})
        entry = {"progress_lower": i / bins, "progress_upper": (i + 1) / bins,
                 "episodes": len(ids), "observations": len(subset), "k": {}}
        for k in KS:
            values = [np.mean([r["metrics"][str(k)][metric] for r in subset if r["rollout_index"] == ep]) for ep in ids]
            entry["k"][str(k)] = (dict(zip(("q25", "median", "q75"),
                                            map(float, np.quantile(values, [.25, .5, .75])))) if values else None)
        result.append(entry)
    return result


def draw_report(rows, output, complete):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    specs = [("mean_pairwise_rad", "candidate_mean_distance", "平均两两关节距离", "mrad", 1000),
             ("max_pairwise_rad", "candidate_max_distance", "最大两两关节距离", "mrad", 1000)]
    if "coverage_range" in rows[0]["metrics"]["4"]:
        specs += [("tilted_vs_uniform_rad", "tilted_mean_shift", "倾斜加权均值相对普通均值的偏移", "mrad", 1000)]
    summaries = {}
    for metric, filename, title, unit, scale in specs:
        bins = aggregate(rows, metric)
        summaries[metric] = bins
        fig, (ax, counts) = plt.subplots(2, 1, figsize=(10.8, 6.3), sharex=True,
                                       gridspec_kw={"height_ratios": [4, 1]})
        fig.subplots_adjust(top=.80, bottom=.22, left=.105, right=.97, hspace=.10)
        n_episodes = len({r["rollout_index"] for r in rows})
        fig.suptitle(f"候选数量消融：{title}", x=.105, ha="left", y=.97, fontsize=17)
        fig.text(.105, .90, f"{'完整结果' if complete else '小样本调试 / 未完成结果'} · {n_episodes} 条已保存轨迹 · {len(rows)} 个相同观测 · 首动作 / 12 关节", fontsize=11)
        x = np.array([(b["progress_lower"] + b["progress_upper"]) * 50 for b in bins])
        for k, color, marker in zip(KS, ("#277DA8", "#DB8A26", "#3D9567", "#9565B7"), ("o", "s", "^", "D")):
            y = [b["k"][str(k)]["median"] * scale if b["episodes"] else np.nan for b in bins]
            ax.plot(x, y, color=color, marker=marker, ms=4, lw=1.8, label=f"K = {k}")
        ax.set_ylabel(f"{title}（{unit}）")
        ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=.2)
        ax.legend(ncol=4, frameon=False, loc="upper left")
        counts.bar(x, [b["episodes"] for b in bins], width=3.8, color="#B5CEDD")
        counts.set_xlim(0, 100); counts.set_xticks(np.arange(0, 101, 10))
        counts.set_ylabel("轨迹数")
        counts.set_xlabel("策略执行步数占比（%）；不是任务阶段完成度")
        fig.text(.105, .09, "同一观测：固定微批次采样 32 次，再取随机嵌套的 4 / 8 / 16 / 32 个候选。", fontsize=10)
        fig.text(.105, .05, "每 5% 分箱：轨迹内先平均，再取轨迹间中位数；四组使用同一批观测，不补零、不外推。", fontsize=10)
        fig.savefig(output / f"{filename}.png", dpi=190)
        fig.savefig(output / f"{filename}.pdf")
        plt.close(fig)
    return summaries


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--policy-config", type=Path, required=True)
    p.add_argument("--policy-url", default="ws://127.0.0.1:18302")
    p.add_argument("--batch-size", type=int, default=4, choices=(2, 4, 8))
    p.add_argument("--seed", type=int, default=42, help="Nested subset ordering, not server RNG")
    p.add_argument("--critic-checkpoint", type=Path)
    p.add_argument("--encoder-weights", type=Path)
    p.add_argument("--beta", type=float, default=10)
    p.add_argument("--episode-limit", type=int)
    p.add_argument("--frames-per-episode", type=int, help="Uniformly spaced smoke subset; omit for full data")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--report-only", action="store_true")
    return p.parse_args()


def run(args):
    import yaml
    from tail_value.sources import add_xpolicylab_path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        session_path = args.collection_root / "session_20260930_212515.json"
        session = json.loads(session_path.read_text())
        policy = yaml.safe_load(args.policy_config.read_text())
        if policy.get("policy_name") != "Pi_05_RobotTwin" or policy.get("action_type") != "joint":
            raise ValueError("Use the original joint-action Pi0.5 adapter")
        norm = Path(policy["model_path"]) / "assets" / policy["repo_id"] / "norm_stats.json"
        if not norm.is_file():
            raise FileNotFoundError(norm)
        sources = []
        records = {(r["rollout_index"], r["seed"]): r for r in session["records"]}
        episodes = sorted((args.collection_root / "raw").glob("episode_*"))
        if args.episode_limit:
            episodes = episodes[:args.episode_limit]
        for ep in episodes:
            meta_path = ep / "episode.json"
            meta = json.loads(meta_path.read_text())
            r = meta["record"]
            if (r["rollout_index"], r["seed"]) not in records or not r["save_decision"]:
                raise ValueError("Raw episode is not a saved final-session record")
            frames = sorted((ep / "frames").glob("*.pkl"), key=lambda f: int(f.stem))
            schedule = observation_schedule(meta, frames)
            if args.frames_per_episode:
                take = np.unique(np.linspace(0, len(schedule) - 1, min(args.frames_per_episode, len(schedule)), dtype=int))
                schedule = [schedule[i] for i in take]
            for item in schedule:
                f = Path(item["frame_path"]); stat = f.stat()
                item.update({"rollout_index": r["rollout_index"], "seed": r["seed"],
                             "episode_index": r["episode_index"], "instruction": meta["instruction"],
                             "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                             "metadata_sha256": sha256(meta_path)})
                item["cache"] = f"observations/rollout_{r['rollout_index']:04d}_frame_{item['frame_index']:05d}.npz"
                sources.append(item)
        if not sources:
            raise ValueError("No eligible observations")
        config = {"schema": SCHEMA, "source_session_sha256": sha256(session_path),
                  "policy": policy, "policy_config_sha256": sha256(args.policy_config), "norm_stats_sha256": sha256(norm),
                  "batch_size": args.batch_size, "ks": list(KS), "subset_seed": args.seed,
                  "frequency": session["frequency"], "beta": args.beta,
                  "critic_sha256": sha256(args.critic_checkpoint) if args.critic_checkpoint else None,
                  "encoder_sha256": sha256(args.encoder_weights) if args.encoder_weights else None,
                  "script_sha256": sha256(__file__),
                  "selection": "all saved pre-first-HIL policy frames including failures; selected saved subset, not all 87 attempts",
                  "sampling": "fixed RPC microbatches; independent noise; shuffled nested prefixes; server RNG not client-seeded",
                  "progress": "recorded policy command count / first takeover or autonomous end; not semantic phase progress",
                  "metric": "mean unordered-pair RMS over 12 arm joints of chunk FIRST action; grippers separate",
                  "endpoint": "last saved policy frame may precede takeover/end; no synthetic 100% frame",
                  "smoke": bool(args.episode_limit or args.frames_per_episode)}
        identity = json_digest({"config": config, "sources": sources})
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if not (args.resume or args.report_only) or manifest["identity"] != identity:
                raise ValueError("Existing run/config mismatch; use unchanged --resume or a new output dir")
        else:
            if args.report_only:
                raise ValueError("No manifest for report-only")
            if any(p.name != ".lock" for p in args.output_dir.iterdir()):
                raise ValueError("Nonempty output dir without manifest")
            manifest = {"identity": identity, "config": config, "observations": sources,
                        "completed": {}, "complete": False, "created_at": time.time()}
            atomic_json(manifest_path, manifest)
            print(f"PREFLIGHT episodes={len(episodes)} observations={len(sources)} candidates={len(sources)*32} batch={args.batch_size}", flush=True)
        pending = []
        for s in sources:
            path = args.output_dir / s["cache"]
            if s["cache"] in manifest["completed"]:
                if not path.is_file() or sha256(path) != manifest["completed"][s["cache"]]:
                    raise ValueError(f"Corrupt cache: {path}")
            else:
                pending.append(s)
        t_start = time.monotonic()
        finished_this_run = 0
        if pending and not args.report_only:
            import torch
            torch.set_num_threads(2)
            # The existing adapter imports RoboTwin asset metadata using CWD.
            # Import it in its normal launch directory; no task/env is created.
            previous_cwd = Path.cwd()
            try:
                os.chdir(Path(__file__).resolve().parents[1])
                from eval_policy_xpolicylab import robotwin_obs_to_xpolicylab, sample_absolute_joint_candidates
            finally:
                os.chdir(previous_cwd)
            add_xpolicylab_path()
            from client_server.ws.model_client import WsModelClient
            scorer = None
            if args.critic_checkpoint:
                from coverage_sampling.critic import OnlineCoverage
                scorer = OnlineCoverage(args.critic_checkpoint, args.encoder_weights, device="cpu")
            with WsModelClient(url=args.policy_url, evaluation_id="candidate-progress", trial_id="offline-debug",
                               request_timeout_s=600, max_connect_attempts=3) as client:
                client.call(func_name="reset")
                for s in pending:
                    frame_start = time.monotonic()
                    obs = load_observation(s["frame_path"])
                    policy_obs = robotwin_obs_to_xpolicylab(obs, instruction=s["instruction"], frequency=session["frequency"])
                    client.call(func_name="update_obs", obs=policy_obs)
                    pool = sample_pool(client, obs, sample_absolute_joint_candidates, args.batch_size)
                    indices = paired_indices(args.seed, s["rollout_index"], s["frame_index"])
                    arrays = {"candidates": pool, "subset_order": indices,
                              "state": np.asarray(obs["joint_action"]["vector"], dtype=np.float32),
                              "input_sha256": np.asarray(sha256(s["frame_path"]))}
                    if scorer:
                        arrays["coverage_scores"] = scorer.score_actions(obs, pool[:, 0])
                    metrics = candidate_metrics(pool, indices, arrays.get("coverage_scores"), args.beta)
                    path = args.output_dir / s["cache"]
                    atomic_npz(path, arrays)
                    manifest["completed"][s["cache"]] = sha256(path)
                    atomic_json(manifest_path, manifest)
                    finished_this_run += 1
                    elapsed = time.monotonic() - t_start
                    done = len(manifest["completed"])
                    status = {"status": "sampling", "pid": os.getpid(), "done": done, "total": len(sources),
                              "rollout_index": s["rollout_index"], "frame_index": s["frame_index"],
                              "last_observation_seconds": time.monotonic() - frame_start,
                              "elapsed_seconds": elapsed,
                              "eta_seconds": elapsed / finished_this_run * (len(sources) - done),
                              "last_metrics": metrics, "updated_at": time.time()}
                    atomic_json(args.output_dir / "status.json", status)
                    print(f"SAMPLE {done}/{len(sources)} rollout={s['rollout_index']} frame={s['frame_index']} p={s['progress']:.3f} seconds={status['last_observation_seconds']:.2f} eta_min={status['eta_seconds']/60:.1f}", flush=True)
        rows = []
        for s in sources:
            if s["cache"] not in manifest["completed"]:
                continue
            with np.load(args.output_dir / s["cache"], allow_pickle=False) as z:
                metrics = candidate_metrics(z["candidates"], z["subset_order"], z["coverage_scores"] if "coverage_scores" in z else None, args.beta)
            rows.append({**s, "metrics": metrics})
        if not rows:
            raise ValueError("No completed observations")
        complete = len(rows) == len(sources)
        summaries = draw_report(rows, args.output_dir, complete and not config["smoke"])
        atomic_json(args.output_dir / "per_observation_metrics.json", rows)
        atomic_json(args.output_dir / "progress_metrics.json", summaries)
        manifest["complete"] = complete
        atomic_json(manifest_path, manifest)
        atomic_json(args.output_dir / "status.json", {"status": "complete" if complete else "partial_report",
                    "done": len(rows), "total": len(sources), "smoke": config["smoke"], "updated_at": time.time()})
        print(f"REPORT complete={complete} observations={len(rows)} output={args.output_dir}", flush=True)
    finally:
        lock.close()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments)
    except Exception:
        # Do not overwrite another live run's status when lock acquisition fails.
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        raise SystemExit(1)
