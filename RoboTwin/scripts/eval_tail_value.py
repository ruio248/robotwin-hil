#!/usr/bin/env python3
"""Offline critic diagnostics; does not connect to a policy or simulator."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tail_value.cache import (assert_compatible, atomic_json, cache_fingerprint,
                              load_episode, load_manifest, sha256)
from tail_value.evaluate import plot_episode, plot_summary, report_rows, score_episode, summarize
from tail_value.model import load_checkpoint, models_from_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True, help="Trusted checkpoints to compare")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("val", "heldout", "hil", "all"), default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-plots", type=int, default=10, help="Per checkpoint/suite; metrics always include all frames")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args(argv)


def run(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if args.batch_size < 1 or args.max_plots < 0 or args.cpu_threads < 1:
        raise ValueError("Invalid batch size/plot count/thread count")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a new report directory; existing reports are immutable")
    torch.set_num_threads(args.cpu_threads)
    suites = {"val", "heldout", "hil"} if args.suite == "all" else {args.suite}
    caches = [(root, load_manifest(root)) for root in args.cache_dir]
    available = {e["split"] for _, m in caches for e in m["episodes"]}
    if suites - available:
        raise ValueError(f"Missing requested suite caches: {sorted(suites - available)}")
    if len({p.resolve() for p in args.cache_dir}) != len(args.cache_dir):
        raise ValueError("Duplicate cache directories")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, all_rows = {}, []
    for ckpt_index, path in enumerate(args.checkpoint):
        payload = load_checkpoint(path)
        model, target = models_from_checkpoint(payload, args.device)
        label = f"{ckpt_index:02d}_expert_bootstrap_step{payload['step']}"
        checkpoint_metrics = {"path": str(path.resolve()), "sha256": sha256(path), "config": payload["config"], "suites": {}}
        for suite in sorted(suites):
            output = args.output_dir / label / suite
            output.mkdir(parents=True, exist_ok=True)
            results, identities = [], set()
            for root, manifest in caches:
                entries = [e for e in manifest["episodes"] if e["split"] == suite]
                if not entries:
                    continue
                assert_compatible(payload["compatibility"], manifest["config"]["compatibility"])
                if suite == "val" and cache_fingerprint(manifest) != payload["cache_fingerprint"]:
                    raise ValueError("Validation cache must match checkpoint training cache/split exactly")
                for entry in entries:
                    identity = (suite, entry["id"])
                    if identity in identities:
                        raise ValueError(f"Duplicate evaluation episode: {identity}")
                    identities.add(identity)
                    arrays = load_episode(root / entry["file"], entry["kind"], payload["compatibility"]["num_candidates"])
                    result = score_episode(model, target, arrays, entry["kind"], payload["config"]["gamma"],
                                           args.batch_size, args.device, payload["config"]["score_limit"])
                    all_rows.extend(report_rows(result, entry, suite, label))
                    if len(results) < args.max_plots:
                        plot_episode(result, entry, output / f"{entry['id']}.png")
                    results.append(result)
            checkpoint_metrics["suites"][suite] = summarize(results)
            plot_summary(results, output)
        metrics[label] = checkpoint_metrics
    # Explicit schema preserves nullable demo-only fields even when HIL rows come first.
    scalar_float = ("q_pi_mean", "q_pi_p05", "q_pi_p95", "q_pi_min", "q_pi_max", "candidate_std_mean",
                    "q_demo", "farthest_distance", "q_farthest", "td_target", "td_squared_error", "gap")
    schema = pa.schema([(k, pa.string()) for k in ("checkpoint", "suite", "episode", "kind", "supervisor_label")]
                       + [(k, pa.int64()) for k in ("frame_index", "farthest_index", "control_source")]
                       + [(k, pa.float64()) for k in scalar_float]
                       + [(k, pa.list_(pa.float64())) for k in ("candidate_scores", "candidate_distances")])
    pq.write_table(pa.Table.from_pylist(all_rows, schema=schema), args.output_dir / "scores.parquet")
    atomic_json(args.output_dir / "metrics.json", {"checkpoints": metrics,
                "interpretation": "HIL scores evaluate policy suggestions at logged observations, not executed actions. No HIL TD or coverage labels.",
                "limitations": ["approximate demonstration transitions", "not success probabilities", "not an online recovery benchmark"]})
    print(f"Wrote {len(all_rows)} scored frames to {args.output_dir}", flush=True)
    return metrics


if __name__ == "__main__":
    run(parse_args())
