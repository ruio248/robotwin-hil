#!/usr/bin/env python3
"""Encode the three stored RGB views with the project's frozen R3M-ResNet18."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_tensor(image: np.ndarray) -> torch.Tensor:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC uint8 RGB, got {image.dtype} {image.shape}")
    value = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
    height, width = image.shape[:2]
    scale = 224 / max(height, width)
    resized_h, resized_w = max(1, round(height * scale)), max(1, round(width * scale))
    value = F.interpolate(value[None], size=(resized_h, resized_w), mode="bilinear",
                          align_corners=False, antialias=True)[0]
    top, left = (224 - resized_h) // 2, (224 - resized_w) // 2
    value = F.pad(value, (left, 224 - resized_w - left, top, 224 - resized_h - top))
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    return (value - mean) / std


def r3m_state(checkpoint: dict, expected: dict) -> dict:
    container = checkpoint.get("r3m", checkpoint.get("state_dict", checkpoint))
    if not isinstance(container, dict):
        raise ValueError("R3M checkpoint contains no state dict")
    prefixes = ("module.convnet.", "convnet.", "module.module.convnet.", "module.")
    for prefix in prefixes:
        found = {key[len(prefix):]: value for key, value in container.items()
                 if key.startswith(prefix) and key[len(prefix):] in expected}
        if (set(found) == set(expected)
                and all(tuple(found[k].shape) == tuple(expected[k].shape) for k in expected)):
            return found
    raise ValueError(f"Cannot map a complete compatible R3M ResNet18 state; keys={list(container)[:8]}")


def atomic_npz(path: Path, **arrays) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def encode_episode(model, episode_dir: Path, device: torch.device, batch_size: int):
    source = episode_dir / "pairs.npz"
    candidates = episode_dir / "policy_candidates_k4.npz"
    target = episode_dir / "features_r3m18.npz"
    if not source.is_file() or not candidates.is_file():
        raise FileNotFoundError(f"Need raw pairs and K4 candidates in {episode_dir}")
    with np.load(source, allow_pickle=False) as raw, np.load(candidates, allow_pickle=False) as policy:
        state = raw["state"].astype(np.float32)
        expert = raw["expert_chunks"].astype(np.float32)
        actions = policy["policy_chunks"].astype(np.float32)
        views = [raw[name] for name in ("images_head", "images_left", "images_right")]
        n = len(state)
        if expert.shape != (n, 10, 14) or actions.shape != (n, 4, 10, 14):
            raise ValueError(f"invalid action shapes in {episode_dir}: {expert.shape}, {actions.shape}")
        if any(value.shape[0] != n or value.dtype != np.uint8 for value in views):
            raise ValueError(f"image/state anchor mismatch in {episode_dir}")
        all_features = []
        for images in views:
            chunks = []
            for start in range(0, n, batch_size):
                tensor = torch.stack([image_tensor(im) for im in images[start:start + batch_size]]).to(device)
                with torch.inference_mode():
                    chunks.append(model(tensor).float().cpu().numpy())
            all_features.append(np.concatenate(chunks, axis=0))
        features = np.concatenate(all_features, axis=1).astype(np.float32)
        if features.shape != (n, 1536) or not np.isfinite(features).all():
            raise ValueError(f"invalid R3M feature array {features.shape}")
        atomic_npz(target, features=features)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1 or not args.asset.is_file():
        parser.error("batch-size must be positive and the R3M asset must exist")
    torch.set_num_threads(4)
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = torch.nn.Identity()
    checkpoint = torch.load(args.asset, map_location="cpu", weights_only=True)
    model.load_state_dict(r3m_state(checkpoint, model.state_dict()), strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval().requires_grad_(False)
    episodes = sorted(args.task_root.glob("episode_*_scene_*"))
    if not episodes:
        raise FileNotFoundError(f"No episode directories under {args.task_root}")
    files = {}
    for index, episode_dir in enumerate(episodes, 1):
        target = encode_episode(model, episode_dir, device, args.batch_size)
        files[str(target.relative_to(args.task_root))] = sha256(target)
        print(f"ENCODED {index}/{len(episodes)} {episode_dir.name}", flush=True)
    task_name = args.task_root.name
    for suffix in ("_seed0_n30", "_seed1_n30"):
        if task_name.endswith(suffix):
            task_name = task_name.removesuffix(suffix)
            break
    manifest = {
        "complete": True,
        "task": task_name,
        "encoder": "R3M-ResNet18 frozen, 512D per view",
        "encoder_sha256": sha256(args.asset),
        "preprocess": "RGB; long-side resize to 224; centered zero pad; ImageNet mean/std; three camera blocks kept in HWC source order",
        "feature_shape_per_anchor": [1536],
        "action_space": "absolute_joint qpos",
        "candidate_count": 4,
        "episodes": len(episodes),
        "files": files,
    }
    path = args.task_root / "r3m_feature_manifest.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    print(f"COMPLETE {path}", flush=True)


if __name__ == "__main__":
    main()
