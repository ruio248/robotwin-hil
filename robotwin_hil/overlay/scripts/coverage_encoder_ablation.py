#!/usr/bin/env python3
"""Re-encode the *same* SFT macro anchors for a frozen-encoder ablation.

Only visual features change. States, expert chunks, policy candidates, episode
splits and transition masks are copied byte-for-byte from the audited source
cache. All encoders produce three unit-L2 visual blocks of 512 dimensions;
DINOv2-S's 384-D block is right-zero-padded after normalization. No policy
service or simulator is used.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from tail_value.cache import CAMERAS, atomic_json, atomic_npz, json_digest, read_json, sha256
from tail_value.chunk import load_chunk_manifest, validate_chunk_episode
from tail_value.prepare import image_tensor
from tail_value.sources import discover


def normalize_views(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 * width or not np.isfinite(values).all():
        raise ValueError(f"Expected finite [N,{3 * width}] visual features")
    views = values.reshape(-1, 3, width)
    norms = np.linalg.norm(views.astype(np.float64), axis=2, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero-length encoder feature; cannot L2-normalize")
    views = views / norms.astype(np.float32)
    if width < 512:
        views = np.pad(views, ((0, 0), (0, 0), (0, 512 - width)))
    return views.reshape(-1, 1536).astype(np.float32)


@contextmanager
def selected_video_frames(path: Path, height: int, width: int, horizon: int):
    """FFmpeg presentation frames 0,H,2H,..., without FPS resampling."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path),
             "-map", "0:v:0", "-vf", f"select=not(mod(n\\,{horizon}))", "-vsync", "0",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            stdout=subprocess.PIPE, stderr=errors,
        )
        def frames():
            size = height * width * 3
            while True:
                payload = process.stdout.read(size)
                if not payload:
                    break
                if len(payload) != size:
                    raise ValueError(f"Truncated selected RGB frame: {path}")
                yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
            if process.wait(timeout=30):
                errors.seek(0)
                raise ValueError(f"FFmpeg selection failed: {errors.read().decode(errors='replace')}")
        try:
            yield frames()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            process.stdout.close()


class FrozenVisualEncoder:
    def __init__(self, name: str, asset: Path, device: str):
        self.name = name
        self.device = torch.device(device)
        if name in ("imagenet_resnet18", "r3m_resnet18"):
            from torchvision.models import resnet18
            model = resnet18(weights=None)
            model.fc = torch.nn.Identity()
            if name == "imagenet_resnet18":
                state = {k: v for k, v in torch.load(asset, map_location="cpu", weights_only=True).items()
                         if k in model.state_dict()}
            else:
                checkpoint = torch.load(asset, map_location="cpu", weights_only=True)
                state = r3m_convnet_state(checkpoint, model.state_dict())
            model.load_state_dict(state, strict=True)
            self.width = 512
        elif name == "dinov2_small":
            from transformers import AutoModel
            model = AutoModel.from_pretrained(asset, local_files_only=True, trust_remote_code=False)
            self.width = int(model.config.hidden_size)
            if self.width != 384:
                raise ValueError(f"Unexpected DINOv2-S width {self.width}")
        else:
            raise ValueError(name)
        self.model = model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, images: list[np.ndarray], batch_size: int) -> np.ndarray:
        parts = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([image_tensor(im) for im in images[start:start + batch_size]]).to(self.device)
            output = self.model(batch)
            if self.name == "dinov2_small":
                output = output.last_hidden_state[:, 0]
            parts.append(output.float().cpu().numpy())
        return np.concatenate(parts, axis=0)


def r3m_convnet_state(checkpoint: dict, expected: dict) -> dict:
    """Accept only complete, shape-compatible official R3M convnet weights."""
    if not isinstance(checkpoint, dict):
        raise ValueError("R3M checkpoint is not a dictionary")
    container = checkpoint.get("r3m", checkpoint.get("state_dict", checkpoint))
    if not isinstance(container, dict):
        raise ValueError("R3M checkpoint has no state dictionary")
    prefixes = ("module.convnet.", "convnet.", "module.module.convnet.", "module.")
    for prefix in prefixes:
        found = {key[len(prefix):]: value for key, value in container.items()
                 if key.startswith(prefix) and key[len(prefix):] in expected}
        if set(found) == set(expected) and all(tuple(found[key].shape) == tuple(expected[key].shape) for key in expected):
            return found
    raise ValueError(f"Cannot map complete R3M convnet state; sample keys={list(container)[:8]}")


def load_source(root: Path, manifest: dict) -> dict:
    sources = {source.id: source for source in discover(root, "lerobot", seed=42)}
    if len(sources) != 450 or {e["id"] for e in manifest["episodes"]} != set(sources):
        raise ValueError("SFT source and audited chunk cache disagree")
    if any(s.metadata["fps"] != 15 for s in sources.values()):
        raise ValueError("Expected original SFT metadata at 15 Hz")
    return sources


def reencode_episode(entry: dict, source, old_path: Path, encoder: FrozenVisualEncoder,
                     batch_size: int, parity_encoder: FrozenVisualEncoder | None = None) -> dict:
    if sha256(old_path) != entry["original_sha256"]:
        raise ValueError(f"Changed original candidate shard {old_path}")
    with np.load(old_path, allow_pickle=False) as archive:
        old = {key: archive[key] for key in archive.files}
    validate_chunk_episode(old, horizon=10, candidates=4)
    anchors = old["frame_index"].astype(int)
    if not np.array_equal(anchors, np.arange(len(anchors)) * 10):
        raise ValueError(f"Changed macro anchor schedule: {entry['id']}")
    expected_fingerprints = entry.get("source_fingerprint") or []
    by_suffix = {Path(f["path"]).name + "/" + Path(f["path"]).parent.name: f["sha256"]
                 for f in expected_fingerprints}
    frames_by_view = []
    with ExitStack() as stack:
        streams = []
        for video, shape in zip(source.files[1:4], source.metadata["camera_shapes"], strict=True):
            key = video.name + "/" + video.parent.name
            if by_suffix.get(key) != sha256(video):
                raise ValueError(f"Source video differs from audited cache: {video}")
            streams.append(stack.enter_context(selected_video_frames(video, shape[1], shape[2], 10)))
        for camera, stream in zip(CAMERAS, streams, strict=True):
            frames = [next(stream) for _ in anchors]
            if len(frames) != len(anchors):
                raise ValueError(f"Video too short for {camera}: {entry['id']}")
            # The source video may have a final short, incomplete 10-frame window.
            trailing = list(stream)
            if len(trailing) > 1:
                raise ValueError(f"Extra complete macro windows in {camera}: {entry['id']}")
            frames_by_view.append(frames)
    if parity_encoder is not None:
        reference = np.concatenate([parity_encoder.encode(frames[:min(3, len(frames))], batch_size)
                                    for frames in frames_by_view], axis=1)
        prior = old["features"][:len(reference)]
        diff = float(np.max(np.abs(reference - prior)))
        cosine = np.sum(reference * prior, axis=1) / (
            np.linalg.norm(reference, axis=1) * np.linalg.norm(prior, axis=1))
        # Old features were computed with an older CUDA/torchvision runtime;
        # RGB bytes are checked separately above. Tolerate tiny kernel drift.
        if diff > .03 or float(cosine.min()) < .99999:
            raise ValueError(f"RGB/anchor parity failure {entry['id']}: max abs={diff:.6g}, cosine={cosine.min():.8f}")
        print(f"PARITY {entry['id']} max_abs={diff:.6g} min_cosine={cosine.min():.8f}", flush=True)
    values = np.concatenate([encoder.encode(frames, batch_size) for frames in frames_by_view], axis=1)
    data = {key: value.copy() for key, value in old.items()}
    data["features"] = normalize_views(values, encoder.width)
    validate_chunk_episode(data, horizon=10, candidates=4)
    for key in ("state", "expert_chunks", "policy_chunks", "frame_index", "valid_transition"):
        if not np.array_equal(old[key], data[key]):
            raise AssertionError(f"Changed non-visual field {key}")
    return data


def run(args):
    manifest = load_chunk_manifest(args.source_cache)
    if not manifest.get("complete") or len(manifest["episodes"]) != 450:
        raise ValueError("Need the complete 450-episode audited source cache")
    if args.name == "dinov2_small":
        asset_file = args.asset / "model.safetensors"
    else:
        asset_file = args.asset
    if not asset_file.is_file():
        raise FileNotFoundError(asset_file)
    config = {
        "encoder": args.name, "asset_sha256": sha256(asset_file),
        "source_cache_manifest_sha256": sha256(args.source_cache / "manifest.json"),
        "source_dataset": str(args.dataset_root.resolve()),
        "preprocess": "original RGB; long-side=224; centered zero pad; ImageNet mean/std; FFmpeg presentation frame 0,10,20,...",
        "output": "three view-wise unit-L2 blocks, 512 dimensions each; DINOv2-S 384->512 right-zero-padding",
        "features": 1536, "horizon": 10, "policy_candidate_source": "unchanged original cache",
        "source_fps_metadata": 15,
    }
    config_id = json_digest(config)
    dest = args.output_dir
    dest_manifest = dest / "manifest.json"
    if dest_manifest.exists():
        if not args.resume:
            raise FileExistsError("Use --resume for an existing encoder cache")
        state = read_json(dest_manifest)
        if state["config_id"] != config_id:
            raise ValueError("Encoder cache identity changed")
    else:
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError("Output directory is not empty")
        state = {"schema_version": 1, "config_id": config_id,
                 "config": {**manifest["config"], "encoder_ablation": config},
                 "episodes": [{**e, "sha256": None, "original_sha256": e["sha256"]}
                              for e in manifest["episodes"]],
                 "complete": False, "created_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(dest_manifest, state)
    sources = None if args.name == "imagenet_resnet18" else load_source(args.dataset_root, manifest)
    encoder = None if sources is None else FrozenVisualEncoder(args.name, args.asset, args.device)
    parity = None
    if sources is not None and args.parity_asset is not None:
        parity = FrozenVisualEncoder("imagenet_resnet18", args.parity_asset, args.device)
    for index, entry in enumerate(state["episodes"]):
        path = dest / entry["file"]
        if entry.get("sha256"):
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError(f"Completed ablation shard changed: {path}")
            continue
        old_path = args.source_cache / entry["file"]
        if sha256(old_path) != entry["original_sha256"]:
            raise ValueError(f"Original shard changed: {old_path}")
        if encoder is None:
            with np.load(old_path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            validate_chunk_episode(data, horizon=10, candidates=4)
            data["features"] = normalize_views(data["features"], 512)
        else:
            data = reencode_episode(entry, sources[entry["id"]], old_path, encoder,
                                    args.batch_size, parity if index < 2 else None)
        atomic_npz(path, data)
        entry["sha256"] = sha256(path)
        atomic_json(dest_manifest, state)
        print(f"ENCODED {index + 1}/450 {entry['id']} {len(data['state'])} anchors", flush=True)
    state["complete"] = True
    atomic_json(dest_manifest, state)
    print(f"COMPLETE {dest_manifest}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cache", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--name", required=True, choices=("imagenet_resnet18", "r3m_resnet18", "dinov2_small"))
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--parity-asset", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
