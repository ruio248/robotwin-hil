#!/usr/bin/env python3
"""Train a frozen-feature support critic, not the robot policy."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from tail_value.cache import atomic_json, cache_fingerprint, load_manifest
from tail_value.model import (POLICY_BOOTSTRAP, CoverageCritic, TransitionTable, atomic_checkpoint, critic_loss, finite_guard,
                              load_checkpoint, make_target, model_digest, models_from_checkpoint, restore_rng,
                              rng_state, seed_all, soft_update)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Trusted local checkpoint; repeat its hyperparameters")
    parser.add_argument("--steps", type=int, default=10000, help="Total steps including resumed steps")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--alpha", type=float, default=.01)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--target-tau", type=float, default=.005)
    parser.add_argument("--score-limit", type=float, default=1e4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args(argv)


def objective_kwargs(config):
    return {"gamma": config["gamma"], "alpha": config["alpha"], "score_limit": config["score_limit"]}


def score_summary(scores):
    values = scores.flatten().float().cpu()
    return {"mean": values.mean().item(), "std": values.std(unbiased=False).item(),
            "quantiles": torch.quantile(values, torch.tensor([0., .05, .5, .95, 1.])).tolist()}


@torch.no_grad()
def validate(model, target, table, device, config):
    model.eval()
    sums, scores = {}, {key: [] for key in ("q_demo", "q_pi", "q_next_pi", "target", "expert_minus_policy")}
    for start in range(0, len(table), config["batch_size"]):
        batch = table.batch(slice(start, start + config["batch_size"]), device)
        loss, components, values = critic_loss(model, target, batch, **objective_kwargs(config))
        n = len(batch["state"])
        for key, value in {"loss": loss, **components}.items():
            sums[key] = sums.get(key, 0.) + value.item() * n
        for key in scores:
            scores[key].append(values[key].cpu())
    result = {key: value / len(table) for key, value in sums.items()}
    result.update({key: score_summary(torch.cat(value)) for key, value in scores.items()})
    model.train()
    return result


def run(args):
    for name in ("steps", "batch_size", "width", "eval_every", "log_every", "cpu_threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.gamma < 1 or not 0 < args.target_tau <= 1:
        raise ValueError("Require 0<=gamma<1 and 0<target-tau<=1")
    for name in ("lr", "grad_clip", "score_limit"):
        if not math.isfinite(getattr(args, name)) or not float(getattr(args, name)) > 0:
            raise ValueError(f"{name} must be positive")
    for name in ("alpha", "weight_decay"):
        if not math.isfinite(getattr(args, name)) or not float(getattr(args, name)) >= 0:
            raise ValueError(f"{name} must be nonnegative")
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    device = torch.device(args.device)
    manifest = load_manifest(args.cache_dir)
    if manifest["config"]["source_format"] != "lerobot":
        raise ValueError("Training is restricted to the original LeRobot train/val cache")
    fingerprint = cache_fingerprint(manifest)
    config = {key: value for key, value in vars(args).items()
              if key not in ("cache_dir", "output_dir", "resume", "steps", "device", "eval_every", "log_every", "cpu_threads")}
    config.update({"bootstrap": POLICY_BOOTSTRAP,
                   "candidate_use": "current_conservative_and_next_target_mean"})
    train = TransitionTable(args.cache_dir, manifest, "train")
    val = TransitionTable(args.cache_dir, manifest, "val")
    generator = torch.Generator().manual_seed(args.seed)
    start_step = 0
    payload = None
    if args.resume:
        payload = load_checkpoint(args.resume)
        if payload["config"] != config or payload["cache_fingerprint"] != fingerprint:
            raise ValueError("Resume hyperparameters or source cache changed; repeat original arguments")
        model, target = models_from_checkpoint(payload, device)
        normalization = payload["normalization"]
        start_step = payload["step"]
        if args.steps <= start_step:
            raise ValueError("steps must exceed checkpoint step")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("Refusing to overwrite a training run; choose a new directory or --resume")
        normalization = train.normalization()
        model = CoverageCritic(normalization, args.width).to(device)
        target = make_target(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if payload:
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"], generator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "config.json", {**config, "steps": args.steps, "cache_fingerprint": fingerprint,
                "train_transitions": len(train), "val_transitions": len(val), "compatibility": manifest["config"]["compatibility"],
                "limitations": [manifest["config"]["transition_semantics"], "not a success probability or online recovery evaluation"]})
    log_path = args.output_dir / "metrics.jsonl"
    def log(record):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
    log({"event": "resume" if payload else "start", "step": start_step,
         "bootstrap": config["bootstrap"], "candidate_use": config["candidate_use"],
         "alpha": args.alpha, "model_sha256": model_digest(model),
         "train_transitions": len(train), "val_transitions": len(val)})
    started = time.monotonic()
    step = start_step
    try:
        model.train()
        for step in range(start_step + 1, args.steps + 1):
            positions = torch.randint(len(train), (args.batch_size,), generator=generator)
            batch = train.batch(positions, device)
            optimizer.zero_grad(set_to_none=True)
            loss, components, values = critic_loss(model, target, batch, **objective_kwargs(config))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            finite_guard(dict(model.named_parameters()))
            soft_update(target, model, args.target_tau)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                log({"event": "train", "step": step, "loss": loss.item(),
                     **{k: v.item() for k, v in components.items()}, "grad_norm": norm.item(),
                     **{key: score_summary(value) for key, value in values.items()},
                     "elapsed_seconds": time.monotonic() - started})
            if step % args.eval_every == 0 or step == args.steps:
                validation = validate(model, target, val, device, {**config, "batch_size": args.batch_size})
                log({"event": "val", "step": step, **validation})
                checkpoint = {"schema_version": 1, "step": step, "config": config, "normalization": normalization,
                              "model": model.state_dict(), "target_model": target.state_dict(), "optimizer": optimizer.state_dict(),
                              "rng": rng_state(generator), "cache_fingerprint": fingerprint,
                              "compatibility": manifest["config"]["compatibility"], "validation": validation}
                atomic_checkpoint(args.output_dir / f"step_{step:07d}.pt", checkpoint)
                atomic_checkpoint(args.output_dir / "last.pt", checkpoint)
    except (FloatingPointError, RuntimeError) as exc:
        atomic_json(args.output_dir / "failure.json", {"step": step, "error": str(exc),
                    "note": "Last valid checkpoint retained; scores were not silently clipped."})
        raise
    return args.output_dir / "last.pt"


if __name__ == "__main__":
    run(parse_args())
