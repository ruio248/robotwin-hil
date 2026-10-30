#!/usr/bin/env python3
"""Cache real same-observation policy chunks and aligned expert SFT windows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

import numpy as np

from tail_value.cache import (ACTION_DIM, PROMPT, atomic_json, atomic_npz,
                              cache_fingerprint, json_digest, load_episode,
                              load_manifest, read_json, sha256)
from tail_value.chunk import validate_chunk_episode
from tail_value.sources import JOINT_FIELDS, add_xpolicylab_path, discover, iter_episode


def policy_chunk(raw, horizon):
    if isinstance(raw, dict) and "actions" in raw:
        raw = raw["actions"]
    if isinstance(raw, np.ndarray) and raw.ndim == 2:
        array = np.asarray(raw, dtype=np.float32)
    elif isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], dict):
        array = np.stack([np.concatenate([
            np.asarray(step[key.removesuffix("s")], dtype=np.float32).reshape(-1)
            for key in JOINT_FIELDS]) for step in raw]).astype(np.float32)
    else:
        array = np.asarray(raw, dtype=np.float32)
    if array.shape != (horizon, ACTION_DIM) or not np.isfinite(array).all():
        raise ValueError(f"Policy did not return finite physical [{horizon},14] chunk: {array.shape}")
    return array.copy()


def anchor_indices(length, horizon):
    anchors = np.arange(0, length - horizon + 1, horizon, dtype=np.int64)
    if len(anchors) < 2:
        raise ValueError("Need at least two complete expert chunks in the same episode")
    return anchors


def build_episode(source, base, client, *, horizon, candidates, progress=None):
    n = len(base["state"])
    anchors = anchor_indices(n, horizon)
    anchor_set = set(anchors.tolist())
    expert = np.stack([base["a_demo"][t:t + horizon] for t in anchors]).astype(np.float32)
    pool = []
    client.call(func_name="reset")
    for frame in iter_episode(source):
        t = int(frame["frame_index"])
        if t >= n or not np.allclose(frame["state"], base["state"][t], atol=1e-5, rtol=1e-5):
            raise ValueError(f"Source/cache state mismatch: {source.id} frame {t}")
        if not np.allclose(frame["a_demo"], base["a_demo"][t], atol=1e-5, rtol=1e-5):
            raise ValueError(f"Source/cache expert action mismatch: {source.id} frame {t}")
        if t not in anchor_set:
            continue
        client.call(func_name="update_obs", obs={"images": frame["images"],
                                                  "state": frame["state"], "instruction": PROMPT})
        raw = client.call(func_name="get_action_candidates", obs=candidates)
        if not isinstance(raw, (list, tuple)) or len(raw) != candidates:
            raise ValueError("Policy did not return K independent chunks")
        pool.append(np.stack([policy_chunk(candidate, horizon) for candidate in raw]))
        if progress and len(pool) % 10 == 0:
            progress(len(pool), len(anchors))
    if len(pool) != len(anchors):
        raise ValueError("Source video/parquet ended before all macro anchors")
    data = {"features": base["features"][anchors].astype(np.float32),
            "state": base["state"][anchors].astype(np.float32),
            "expert_chunks": expert,
            "policy_chunks": np.stack(pool).astype(np.float32),
            "frame_index": anchors,
            "valid_transition": np.arange(len(anchors)) < len(anchors) - 1}
    validate_chunk_episode(data, horizon=horizon, candidates=candidates)
    return data


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-cache", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--policy-url", default="ws://127.0.0.1:18303")
    p.add_argument("--policy-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-candidates", type=int, default=4)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--train-limit", type=int, help="Pilot only: first N original-train episodes")
    p.add_argument("--val-limit", type=int, help="Pilot only: first N original-val episodes")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def run(args):
    if args.horizon != 10 or not 2 <= args.num_candidates <= 8:
        raise ValueError("Require H=10 and 2<=K<=8 for batched Pi0.5 sampling")
    if (args.train_limit is None) != (args.val_limit is None):
        raise ValueError("Both pilot limits must be specified together")
    if args.train_limit is not None and (args.train_limit < 1 or args.val_limit < 1):
        raise ValueError("Pilot limits must be positive")
    if not args.policy_checkpoint.is_dir():
        raise FileNotFoundError(args.policy_checkpoint)
    base_manifest = load_manifest(args.base_cache)
    if base_manifest["config"]["source_format"] != "lerobot":
        raise ValueError("Only the original SFT demonstration cache is allowed")
    source_by_id = {s.id: s for s in discover(args.dataset_root, "lerobot", seed=42)}
    if any(source.metadata["fps"] != 15 for source in source_by_id.values()):
        raise ValueError("Expected the audited 15 Hz SFT dataset")
    entries = list(base_manifest["episodes"])
    if len(entries) != 450 or {e["id"] for e in entries} != set(source_by_id):
        raise ValueError("Expected the same complete 450 SFT episodes")
    if args.train_limit is not None:
        train = [e for e in entries if e["split"] == "train"][:args.train_limit]
        val = [e for e in entries if e["split"] == "val"][:args.val_limit]
        entries = train + val
    config = {"representation": "absolute_joint_chunk", "horizon": args.horizon,
              "num_candidates": args.num_candidates, "base_cache_fingerprint": cache_fingerprint(base_manifest),
              "source_dataset": str(args.dataset_root.resolve()),
              "source_fps": 15, "online_frequency_warning": "policy rollout uses 30 Hz in current HIL",
              "policy_checkpoint": str(args.policy_checkpoint.resolve()),
              "policy_url": args.policy_url,
              "anchor_semantics": "0,H,2H... complete SFT windows; next anchor is H recorded frames later",
              "action_semantics": "14D absolute joint targets; expert action[t] equals next recorded state",
              "transition_semantics": "H-step approximate demonstrated transition; no terminal fabricated",
              "sampling_reproducibility": "Server RNG not client-seeded; saved candidate cache is authoritative"}
    selected = [{"id": e["id"], "split": e["split"], "kind": "demo",
                 "file": f"episodes/{e['id']}.npz", "source_sha256": e["sha256"],
                 "source_fingerprint": e.get("source_fingerprint")}
                for e in entries]
    manifest_path = args.output_dir / "manifest.json"
    config_id = json_digest({"config": config, "episodes": selected})
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Cache exists; pass --resume")
        manifest = read_json(manifest_path)
        if manifest["config_id"] != config_id:
            raise ValueError("Cache identity changed")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("Choose an empty chunk cache directory")
        manifest = {"schema_version": 1, "config_id": config_id, "config": config,
                    "episodes": selected, "created_at": datetime.now(timezone.utc).isoformat(),
                    "complete": False}
        atomic_json(manifest_path, manifest)
    pending = []
    for entry in manifest["episodes"]:
        path = args.output_dir / entry["file"]
        if entry.get("sha256"):
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError(f"Corrupt completed chunk cache {path}")
            with np.load(path, allow_pickle=False) as archive:
                validate_chunk_episode({key: archive[key] for key in archive.files},
                                       horizon=args.horizon, candidates=args.num_candidates)
        else:
            pending.append(entry)
    if pending:
        add_xpolicylab_path()
        from client_server.ws.model_client import WsModelClient
        with WsModelClient(url=args.policy_url, evaluation_id="tail-chunk-" + uuid.uuid4().hex,
                           trial_id="offline-tail-chunk", request_timeout_s=600,
                           max_connect_attempts=3) as client:
            for entry in pending:
                print(f"prepare {entry['id']} ({entry['split']})", flush=True)
                source = source_by_id[entry["id"]]
                base_path = args.base_cache / next(e["file"] for e in base_manifest["episodes"] if e["id"] == entry["id"])
                base = load_episode(base_path, "demo", base_manifest["config"]["compatibility"]["num_candidates"])
                if len(base["state"]) != source.metadata["length"]:
                    raise ValueError(f"Episode length changed: {entry['id']}")
                data = build_episode(source, base, client, horizon=args.horizon,
                                     candidates=args.num_candidates,
                                     progress=lambda n, total: print(f"{entry['id']}: {n}/{total} anchors", flush=True))
                path = args.output_dir / entry["file"]
                atomic_npz(path, data)
                entry.update({"sha256": sha256(path), "anchors": len(data["state"]),
                              "transitions": int(data["valid_transition"].sum())})
                atomic_json(manifest_path, manifest)
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)
    print(f"complete: {len(manifest['episodes'])} episodes -> {manifest_path}", flush=True)


if __name__ == "__main__":
    run(parse_args())
