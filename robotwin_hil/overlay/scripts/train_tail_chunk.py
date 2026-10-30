#!/usr/bin/env python3
"""Train a fresh 10-action coverage critic on episode-safe macro transitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from tail_value.cache import atomic_json, json_digest
from tail_value.chunk import ChunkCoverageCritic, ChunkTransitionTable, chunk_critic_loss, load_chunk_manifest
from tail_value.model import (atomic_checkpoint, finite_guard, load_checkpoint, make_target,
                              restore_rng, rng_state, seed_all, soft_update)


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--alpha", type=float, default=1.)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--target-tau", type=float, default=.005)
    p.add_argument("--score-limit", type=float, default=1e4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-threads", type=int, default=4)
    return p.parse_args(argv)


def score_summary(value):
    value = value.flatten().float().cpu()
    return {"mean": value.mean().item(), "std": value.std(unbiased=False).item(),
            "quantiles": torch.quantile(value, torch.tensor([0., .05, .5, .95, 1.])).tolist()}


@torch.no_grad()
def validate(model, target, table, device, args):
    model.eval()
    sums = {}
    collected = {key: [] for key in ("q_expert", "q_policy", "q_next", "td_target", "expert_minus_policy")}
    for start in range(0, len(table), args.batch_size):
        batch = table.batch(slice(start, start + args.batch_size), device)
        loss, components, values = chunk_critic_loss(model, target, batch, gamma=args.gamma,
                                                      alpha=args.alpha, score_limit=args.score_limit)
        n = len(batch["state"])
        for key, value in {"loss": loss, **components}.items():
            sums[key] = sums.get(key, 0.) + value.item() * n
        for key in collected:
            collected[key].append(values[key].cpu())
    result = {key: value / len(table) for key, value in sums.items()}
    result.update({key: score_summary(torch.cat(parts)) for key, parts in collected.items()})
    model.train()
    return result


def run(args):
    for name in ("steps", "batch_size", "width", "eval_every", "log_every", "cpu_threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.gamma < 1 or not 0 < args.target_tau <= 1:
        raise ValueError("Invalid gamma/target-tau")
    for name in ("lr", "grad_clip", "score_limit"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("alpha", "weight_decay"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    device = torch.device(args.device)
    manifest = load_chunk_manifest(args.cache_dir)
    if manifest["config"]["horizon"] != 10:
        raise ValueError("This experiment requires H=10")
    fingerprint = json_digest({"config": manifest["config"], "episodes": manifest["episodes"]})
    config = {key: value for key, value in vars(args).items()
              if key not in ("cache_dir", "output_dir", "resume", "steps", "device", "eval_every", "log_every", "cpu_threads")}
    config.update({"horizon": 10, "bootstrap": "policy_next_chunk_mean_h10",
                   "candidate_use": "current_conservative_and_next_macro_target_mean",
                   "reward": "sum(gamma**i for i in range(10))",
                   "terminal": "none_inferred"})
    train = ChunkTransitionTable(args.cache_dir, manifest, "train")
    val = ChunkTransitionTable(args.cache_dir, manifest, "val")
    generator = torch.Generator().manual_seed(args.seed)
    payload = None
    if args.resume:
        payload = load_checkpoint(args.resume)
        if (payload["config"] != config or payload["cache_fingerprint"] != fingerprint
                or payload.get("representation") != "absolute_joint_chunk"):
            raise ValueError("Checkpoint parameters or chunk cache changed")
        if args.steps <= payload["step"]:
            raise ValueError("Resume step must increase")
        normalization = payload["normalization"]
        model = ChunkCoverageCritic(normalization, horizon=10, width=args.width).to(device)
        model.load_state_dict(payload["model"])
        target = make_target(model)
        target.load_state_dict(payload["target_model"])
        start_step = payload["step"]
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("Refusing to overwrite run directory")
        normalization = train.normalization()
        model = ChunkCoverageCritic(normalization, horizon=10, width=args.width).to(device)
        target = make_target(model)
        start_step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if payload:
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"], generator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "config.json", {**config, "steps": args.steps,
        "cache_fingerprint": fingerprint, "train_transitions": len(train), "val_transitions": len(val),
        "limitations": ["SFT at 15 Hz, online execution often 30 Hz", "no verified terminal label",
                        "next-recorded-joint-target expert action approximation", "not a success probability"]})
    def log(record):
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
    log({"event": "resume" if payload else "start", "step": start_step, "alpha": args.alpha,
         "bootstrap": config["bootstrap"], "model_sha256": model_digest(model),
         "train_transitions": len(train), "val_transitions": len(val)})
    started = time.monotonic()
    step = start_step
    try:
        model.train()
        for step in range(start_step + 1, args.steps + 1):
            positions = torch.randint(len(train), (args.batch_size,), generator=generator)
            batch = train.batch(positions, device)
            optimizer.zero_grad(set_to_none=True)
            loss, components, values = chunk_critic_loss(model, target, batch, gamma=args.gamma,
                                                          alpha=args.alpha, score_limit=args.score_limit)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            finite_guard(dict(model.named_parameters()))
            soft_update(target, model, args.target_tau)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                log({"event": "train", "step": step, "loss": loss.item(),
                     **{k: v.item() for k, v in components.items()}, "grad_norm": grad.item(),
                     **{key: score_summary(value) for key, value in values.items()},
                     "elapsed_seconds": time.monotonic() - started})
            if step % args.eval_every == 0 or step == args.steps:
                result = validate(model, target, val, device, args)
                log({"event": "val", "step": step, **result})
                checkpoint = {"schema_version": 1, "representation": "absolute_joint_chunk",
                    "step": step, "config": config, "normalization": normalization,
                    "model": model.state_dict(), "target_model": target.state_dict(),
                    "optimizer": optimizer.state_dict(), "rng": rng_state(generator),
                    "cache_fingerprint": fingerprint, "validation": result}
                atomic_checkpoint(args.output_dir / f"step_{step:07d}.pt", checkpoint)
                atomic_checkpoint(args.output_dir / "last.pt", checkpoint)
                inference = {"schema_version": 1, "representation": "absolute_joint_chunk",
                    "step": step, "config": config, "normalization": normalization,
                    "model": model.state_dict(), "cache_fingerprint": fingerprint}
                atomic_checkpoint(args.output_dir / "inference.pt", inference)
    except (FloatingPointError, RuntimeError) as exc:
        atomic_json(args.output_dir / "failure.json", {"step": step, "error": str(exc)})
        raise
    return args.output_dir / "last.pt"


if __name__ == "__main__":
    run(parse_args())
