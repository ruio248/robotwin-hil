#!/usr/bin/env python3
"""Train one H50 source discriminator from 50 same-observation expert/policy episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tail_value.chunk import ChunkCoverageCritic


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_episode(meta_path: Path, task: str) -> dict:
    meta = json.loads(meta_path.read_text())
    if meta.get("task") != task or meta.get("expert_success") is not True or meta.get("chunk_horizon") != 50:
        raise ValueError(f"invalid H50 expert episode: {meta_path}")
    folder = meta_path.parent
    with np.load(folder / "pairs.npz", allow_pickle=False) as source:
        state = source["state"].astype(np.float32)
        expert = source["expert_chunks"].astype(np.float32)
        policy = source["policy_chunks"].astype(np.float32)
    with np.load(folder / "features_r3m18.npz", allow_pickle=False) as source:
        features = source["features"].astype(np.float32)
    n = len(state)
    if n < 1 or state.shape != (n, 14) or expert.shape != (n, 50, 14) or policy.shape != (n, 50, 14) or features.shape != (n, 1536):
        raise ValueError(f"invalid paired shapes: {folder}")
    if meta.get("complete_pairs") != n or not all(np.isfinite(x).all() for x in (state, expert, policy, features)):
        raise ValueError(f"incomplete or nonfinite episode: {folder}")
    return {"scene_seed": int(meta["scene_seed"]), "state": state, "expert": expert,
            "policy": policy, "features": features,
            "pair_sha256": digest(folder / "pairs.npz"),
            "feature_sha256": digest(folder / "features_r3m18.npz")}


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    episodes = []
    for root in args.task_root:
        state = json.loads((root / "state.json").read_text())
        if state.get("status") != "complete" or state.get("expert_chunk_horizon") != 50:
            raise ValueError(f"incomplete H50 collection: {root}")
        episodes.extend(load_episode(path, args.task) for path in sorted(root.glob("episode_*_scene_*/metadata.json")))
    if len(episodes) != 50 or len({e["scene_seed"] for e in episodes}) != 50:
        raise ValueError(f"need exactly 50 unique successful expert scenes, got {len(episodes)}")
    train_scenes = sorted(e["scene_seed"] for e in episodes)
    if args.heldout_scene_list:
        heldout = set(json.loads(args.heldout_scene_list.read_text()))
        overlap = heldout.intersection(train_scenes)
        if overlap:
            raise ValueError(f"train/test scene overlap: {sorted(overlap)}")
    states = np.concatenate([e["state"] for e in episodes])
    actions = np.concatenate([e["expert"].reshape(-1, 14) for e in episodes])
    norm = {"state_mean": torch.from_numpy(states.mean(0)),
            "state_scale": torch.from_numpy(np.maximum(states.std(0), .05)),
            "action_mean": torch.from_numpy(actions.mean(0)),
            "action_scale": torch.from_numpy(np.maximum(actions.std(0), .05))}
    device = torch.device(args.device)
    model = ChunkCoverageCritic(norm, horizon=50, width=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "protocol.json", {
        "task": args.task, "status": "training", "horizon": 50,
        "expert_episodes": 50, "scene_seeds": train_scenes,
        "pair_semantics": "same stored RGB/state/prompt; expert future H50 versus policy predicted H50",
        "task_roots": [str(root.resolve()) for root in args.task_root],
        "training_steps": args.steps, "batch_size": args.batch_size, "seed": args.seed,
        "source_sha256": {str(e["scene_seed"]): {"pairs": e["pair_sha256"], "features": e["feature_sha256"]} for e in episodes},
    })
    sizes = np.asarray([len(e["state"]) for e in episodes])
    for step in range(1, args.steps + 1):
        ids = rng.integers(len(episodes), size=args.batch_size)
        rows = [int(rng.integers(sizes[i])) for i in ids]
        picked = [episodes[i] for i in ids]
        features = torch.as_tensor(np.stack([e["features"][j] for e, j in zip(picked, rows)]), device=device)
        state = torch.as_tensor(np.stack([e["state"][j] for e, j in zip(picked, rows)]), device=device)
        expert = torch.as_tensor(np.stack([e["expert"][j] for e, j in zip(picked, rows)]), device=device)
        policy = torch.as_tensor(np.stack([e["policy"][j] for e, j in zip(picked, rows)]), device=device)
        optimizer.zero_grad(set_to_none=True)
        expert_logit = model(features, state, expert)
        policy_logit = model(features, state, policy)
        loss = .5 * (F.softplus(-expert_logit).mean() + F.softplus(policy_logit).mean())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 500 == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "expert_logit": float(expert_logit.detach().mean()),
                              "policy_logit": float(policy_logit.detach().mean())}), flush=True)
    checkpoint = args.output / "model.pt"
    temporary = args.output / "model.pt.tmp"
    torch.save({"protocol": "h50_same_observation_source_bce_v1", "task": args.task,
                "horizon": 50, "steps": args.steps, "model": model.cpu().state_dict(),
                "normalization": norm, "scene_seeds": train_scenes}, temporary)
    os.replace(temporary, checkpoint)
    protocol = json.loads((args.output / "protocol.json").read_text())
    protocol.update(status="complete", checkpoint_sha256=digest(checkpoint), final_loss=float(loss.detach()))
    save_json(args.output / "protocol.json", protocol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-scene-list", type=Path)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
