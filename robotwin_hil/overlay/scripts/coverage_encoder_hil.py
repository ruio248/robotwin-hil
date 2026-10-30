#!/usr/bin/env python3
"""Re-encode fixed HIL candidate observations without executing any policy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from coverage_encoder_ablation import FrozenVisualEncoder, normalize_views
from tail_value.cache import atomic_json, atomic_npz, json_digest, read_json, sha256
from tail_value.sources import CAMERAS, discover, iter_episode


def run(args):
    source_manifest = read_json(args.source_cache / "candidate_manifest.json")
    if source_manifest.get("complete") is not True or len(source_manifest["identity"]["episodes"]) != 50:
        raise ValueError("Expected complete 50-episode HIL candidate cohort")
    asset_file = args.asset / "model.safetensors" if args.name == "dinov2_small" else args.asset
    if not asset_file.is_file():
        raise FileNotFoundError(asset_file)
    config = {"encoder": args.name, "asset_sha256": sha256(asset_file),
              "source_candidate_manifest_sha256": sha256(args.source_cache / "candidate_manifest.json"),
              "preprocess": "original saved RGB; long-side=224; centered zero pad; ImageNet mean/std",
              "feature_transform": "three view-wise unit-L2 512D blocks; DINOv2-S padded 384->512",
              "candidate_semantics": "unchanged saved K4 full 10-action chunks",
              "cohort": "50 saved HIL episodes; includes full trajectories, not a success-rate sample"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Use --resume")
        manifest = read_json(manifest_path)
        if manifest["config_id"] != json_digest(config):
            raise ValueError("HIL encoder-cache identity changed")
    else:
        if any(args.output_dir.iterdir()):
            raise FileExistsError("Output directory is not empty")
        manifest = {"schema_version": 1, "config_id": json_digest(config), "config": config,
                    "episodes": [], "complete": False,
                    "created_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(manifest_path, manifest)
    encoder = None if args.name == "imagenet_resnet18" else FrozenVisualEncoder(args.name, args.asset, args.device)
    done = {entry["rollout_index"]: entry for entry in manifest["episodes"]}
    for episode in source_manifest["identity"]["episodes"]:
        rid = int(episode["rollout_index"])
        out_path = args.output_dir / f"rollout_{rid:04d}.npz"
        if rid in done:
            if not out_path.is_file() or sha256(out_path) != done[rid]["sha256"]:
                raise ValueError(f"Changed completed HIL encoder shard: {out_path}")
            continue
        raw_path = Path(episode["path"])
        metadata_path = raw_path / "episode.json"
        if sha256(metadata_path) != episode["metadata_sha256"]:
            raise ValueError(f"Changed raw HIL metadata: {raw_path}")
        source_npz = args.source_cache / f"rollout_{rid:04d}.npz"
        expected_sha = source_manifest["completed"].get(f"rollout_{rid:04d}")
        if not source_npz.is_file():
            raise FileNotFoundError(source_npz)
        if not expected_sha or sha256(source_npz) != expected_sha:
            raise ValueError(f"Changed HIL candidate shard: {source_npz}")
        with np.load(source_npz, allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        frame_index = original["frame_index"]
        if original["candidates"].shape[1:] != (4, 10, 14):
            raise ValueError("Expected K4 full-chunk candidate cache")
        if encoder is None:
            features = normalize_views(original["features"], 512)
        else:
            source = discover(raw_path, "hil")[0]
            images = {view: [] for view in CAMERAS}
            wanted = {int(i): pos for pos, i in enumerate(frame_index)}
            recorded = np.zeros(len(frame_index), dtype=bool)
            for frame in iter_episode(source):
                pos = wanted.get(int(frame["frame_index"]))
                if pos is None:
                    continue
                if recorded[pos] or not np.allclose(frame["state"], original["state"][pos], atol=1e-4, rtol=1e-4):
                    raise ValueError(f"HIL RGB/state misalignment rollout {rid} frame {frame['frame_index']}")
                for view in CAMERAS:
                    images[view].append((pos, frame["images"][view].copy()))
                recorded[pos] = True
            if not recorded.all():
                raise ValueError(f"Missing HIL RGB frames: {rid}")
            view_features = []
            for view in CAMERAS:
                ordered = [image for _, image in sorted(images[view])]
                view_features.append(encoder.encode(ordered, args.batch_size))
            features = normalize_views(np.concatenate(view_features, axis=1), encoder.width)
        data = {key: value.copy() for key, value in original.items()}
        data["features"] = features
        atomic_npz(out_path, data)
        manifest["episodes"].append({"rollout_index": rid, "path": out_path.name,
            "sha256": sha256(out_path), "source_sha256": sha256(source_npz),
            "raw_metadata_sha256": episode["metadata_sha256"], "frames": len(frame_index)})
        atomic_json(manifest_path, manifest)
        print(f"HIL_ENCODED {len(manifest['episodes'])}/50 rollout={rid} frames={len(frame_index)}", flush=True)
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)
    print(f"COMPLETE {manifest_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--name", required=True, choices=("imagenet_resnet18", "r3m_resnet18", "dinov2_small"))
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
