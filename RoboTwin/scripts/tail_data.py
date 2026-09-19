#!/usr/bin/env python3
"""Cache frozen-policy candidates and features; never execute simulator actions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

from tail_value.cache import (APPROXIMATION, PROMPT, SCHEMA_VERSION, atomic_json, atomic_npz,
                              json_digest, load_episode, read_json, sha256)
from tail_value.sources import add_xpolicylab_path, discover


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-format", choices=("lerobot", "native", "hil"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--policy-url", default="ws://127.0.0.1:18301")
    parser.add_argument("--policy-config", type=Path, required=True, help="Declared checkpoint identity; server must be exclusively owned by this run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Episode split seed, not server RNG seed")
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--encoder-weights", type=Path, help="Optional local torchvision ResNet18 IMAGENET1K_V1 state dict")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args):
    import yaml
    import numpy as np
    from tail_value.prepare import ENCODER_SPEC, FrozenEncoder, generate_episode
    if args.num_candidates < 1:
        raise ValueError("num-candidates must be positive")
    sources = discover(args.dataset_root, args.source_format, args.episode_limit, args.seed)
    policy = yaml.safe_load(args.policy_config.read_text())
    if policy.get("policy_name") != "Pi_05_RobotTwin" or policy.get("action_type") != "joint":
        raise ValueError("v1 supports the physical-joint Pi_05_RobotTwin adapter only")
    identity = {k: v for k, v in policy.items() if k not in ("host", "port", "protocol", "ws_ping_interval_s", "ws_ping_timeout_s")}
    encoder = dict(ENCODER_SPEC)
    # Record exact local weights when supplied; the same file is required across suites.
    encoder["local_weights_sha256"] = sha256(args.encoder_weights) if args.encoder_weights else None
    compatibility = {"prompt": PROMPT, "encoder": encoder, "action_semantics": "absolute-joint-target-first-chunk-step-14d",
                     "num_candidates": args.num_candidates, "policy_identity": identity}
    entries = [{"id": s.id, "split": s.split, "kind": s.kind, "file": f"episodes/{s.id}.npz",
                "source_fingerprint": s.fingerprint(), "metadata": s.metadata} for s in sources]
    config = {"source_format": args.source_format, "dataset_root": str(args.dataset_root.resolve()), "seed": args.seed,
              "episode_limit": args.episode_limit, "policy_url": args.policy_url, "compatibility": compatibility,
              "transition_semantics": APPROXIMATION if args.source_format != "hil" else "observations only; policy suggestions, not executed actions",
              "terminal_handling": "no synthetic terminal; no transition from final recorded frame",
              "sampling_reproducibility": "server RNG not controlled; cached draws are authoritative"}
    manifest_path = args.output_dir / "manifest.json"
    config_id = json_digest({"config": config, "sources": entries})
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Cache exists; use --resume with the same inputs")
        manifest = read_json(manifest_path)
        if manifest.get("config_id") != config_id:
            raise ValueError("Resume inputs/config changed; use a new output directory")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("Refusing to use nonempty directory without a cache manifest")
        manifest = {"schema_version": SCHEMA_VERSION, "config_id": config_id, "config": config, "episodes": entries,
                    "created_at": datetime.now(timezone.utc).isoformat(), "complete": False}
        atomic_json(manifest_path, manifest)
    pending = []
    for source, entry in zip(sources, manifest["episodes"], strict=True):
        path = args.output_dir / entry["file"]
        if entry.get("sha256"):
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError(f"Corrupt completed shard: {path}")
            load_episode(path, source.kind, args.num_candidates)
            print(f"resume: verified {source.id}", flush=True)
        else:
            pending.append((source, entry))
    if pending:
        add_xpolicylab_path()
        from client_server.ws.model_client import WsModelClient
        frozen_encoder = FrozenEncoder(args.device, args.encoder_weights)
        with WsModelClient(url=args.policy_url, evaluation_id="tail-" + uuid.uuid4().hex,
                           trial_id="offline-tail", request_timeout_s=300, max_connect_attempts=2) as client:
            for source, entry in pending:
                print(f"prepare {source.id} ({source.split})", flush=True)
                arrays = generate_episode(source, client, frozen_encoder, args.num_candidates,
                                          lambda n: print(f"{source.id}: {n} frames", flush=True))
                path = args.output_dir / entry["file"]
                atomic_npz(path, arrays)
                entry.update({"sha256": sha256(path), "frames": len(arrays["state"]),
                              "candidate_std_mean": float(arrays["a_pi"].std(axis=1).mean()),
                              "all_candidates_identical_fraction": float(np.all(arrays["a_pi"] == arrays["a_pi"][:, :1], axis=(1, 2)).mean())})
                atomic_json(manifest_path, manifest)
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)
    print(f"complete: {len(manifest['episodes'])} episodes -> {manifest_path}", flush=True)


if __name__ == "__main__":
    run(parse_args())
