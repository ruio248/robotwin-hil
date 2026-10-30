#!/usr/bin/env python3
"""Re-encode the exact 100 historical rollout observations for encoder ablation.

No policy inference or simulation. Candidate chunks, state and frame indices are
copied from the audited K4 cache; only frozen visual features are replaced.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from coverage_encoder_ablation import FrozenVisualEncoder, normalize_views
from tail_value.cache import CAMERAS, atomic_json, atomic_npz, json_digest, read_json, sha256
from tail_value.sources import JOINT_FIELDS, NATIVE_CAMERAS, add_xpolicylab_path


def run(args):
    source_manifest_path = args.source_cache / "candidate_manifest.json"
    source_manifest = read_json(source_manifest_path)
    if source_manifest.get("complete") is not True:
        raise ValueError("Historical K4 cache incomplete")
    records = [read_json_line for read_json_line in args.records.read_text().splitlines() if read_json_line.strip()]
    import json
    records = [json.loads(line) for line in records]
    if len(records) != 100 or sum(bool(r["success"]) for r in records) != 31:
        raise ValueError("Expected 100 historical rollouts with 31 successes")
    names = [Path(record["hdf5_path"]).stem for record in records]
    if len(set(names)) != 100 or not all(name in source_manifest["completed"] for name in names):
        raise ValueError("Historical label/cache identity mismatch")
    asset_file = args.asset / "model.safetensors" if args.name == "dinov2_small" else args.asset
    if args.name != "imagenet_resnet18" and not asset_file.is_file():
        raise FileNotFoundError(asset_file)
    config = {"encoder": args.name,
              "encoder_asset_sha256": sha256(asset_file) if args.name != "imagenet_resnet18" else None,
              "source_candidate_manifest_sha256": sha256(source_manifest_path),
              "records_sha256": sha256(args.records), "candidate_count": 4,
              "cohort": "100 historical baseline policy rollouts; 31 success / 69 failure",
              "preprocess": "same 224 long-side centered-pad ImageNet-normalized RGB; view-wise L2; DINO 384->512 pad"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Existing output requires --resume")
        manifest = read_json(manifest_path)
        if manifest["config_id"] != json_digest(config):
            raise ValueError("Historical cache identity changed")
    else:
        if any(args.output_dir.iterdir()):
            raise FileExistsError("Refusing nonempty output directory")
        manifest = {"config_id": json_digest(config), "config": config,
                    "episodes": [], "complete": False}
        atomic_json(manifest_path, manifest)
    done = {entry["name"]: entry for entry in manifest["episodes"]}
    encoder = None if args.name == "imagenet_resnet18" else FrozenVisualEncoder(args.name, args.asset, args.device)
    if encoder is not None and args.rgb_helper_python is None:
        import h5py
        add_xpolicylab_path()
        from XPolicyLab.utils.process_data import decode_image_bit
    for record, name in zip(records, names, strict=True):
        dest = args.output_dir / f"{name}.npz"
        if name in done:
            if not dest.is_file() or sha256(dest) != done[name]["sha256"]:
                raise ValueError(f"Changed completed historical episode {dest}")
            continue
        original = args.source_cache / f"{name}.npz"
        if sha256(original) != source_manifest["completed"][name]:
            raise ValueError(f"Changed K4 candidate cache {original}")
        with np.load(original, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        n = len(arrays["state"])
        if arrays["features"].shape != (n, 1536) or arrays["state"].shape != (n, 14) or arrays["candidates"].shape != (n, 4, 10, 14):
            raise ValueError(f"Changed historical candidate shape {name}")
        if encoder is None:
            features = normalize_views(arrays["features"], 512)
            hdf5_sha = None
        else:
            raw = args.rollout_root / record["hdf5_path"]
            if not raw.is_file():
                raise FileNotFoundError(raw)
            if args.rgb_helper_python is not None:
                if args.rgb_helper_script is None:
                    raise ValueError("RGB helper script required")
                with tempfile.TemporaryDirectory() as temporary:
                    rgb_path = Path(temporary) / "frames.npz"
                    subprocess.run([str(args.rgb_helper_python), str(args.rgb_helper_script),
                                    "--hdf5", str(raw), "--candidate-cache", str(original),
                                    "--output", str(rgb_path)], check=True)
                    with np.load(rgb_path, allow_pickle=False) as archive:
                        per_view = [encoder.encode(list(archive[camera]), args.batch_size) for camera in CAMERAS]
            else:
                with h5py.File(raw, "r") as handle:
                    state = np.concatenate([np.asarray(handle[f"state/{field}"]).reshape(len(handle[f"state/{field}"]), -1)
                                            for field in JOINT_FIELDS], axis=1).astype(np.float32)
                    frame = arrays["frame_index"].astype(int)
                    if np.any(frame < 0) or np.any(frame >= len(state)) or not np.allclose(state[frame], arrays["state"], atol=1e-5, rtol=1e-5):
                        raise ValueError(f"Historical RGB/state misalignment {name}")
                    per_view = []
                    for camera in NATIVE_CAMERAS:
                        stream = handle[f"vision/{camera}/colors"]
                        images = [decode_image_bit(stream[int(i)]) for i in frame]
                        if any(np.asarray(im).ndim != 3 or np.asarray(im).shape[-1] != 3 for im in images):
                            raise ValueError(f"Invalid historical RGB frame {name}/{camera}")
                        per_view.append(encoder.encode(images, args.batch_size))
            features = normalize_views(np.concatenate(per_view, axis=1), encoder.width)
            hdf5_sha = sha256(raw)
        arrays["features"] = features
        atomic_npz(dest, arrays)
        entry = {"name": name, "path": dest.name, "sha256": sha256(dest),
                 "source_candidate_sha256": source_manifest["completed"][name],
                 "source_hdf5_sha256": hdf5_sha, "frames": n, "success": bool(record["success"])}
        manifest["episodes"].append(entry)
        atomic_json(manifest_path, manifest)
        print(f"ENCODED {len(manifest['episodes'])}/100 {name} frames={n}", flush=True)
    if len(manifest["episodes"]) != 100 or sum(e["frames"] for e in manifest["episodes"]) != 4834:
        raise ValueError("Historical cohort changed")
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)
    print(f"COMPLETE {manifest_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cache", required=True, type=Path)
    p.add_argument("--rollout-root", required=True, type=Path)
    p.add_argument("--records", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--name", required=True, choices=("imagenet_resnet18", "r3m_resnet18", "dinov2_small"))
    p.add_argument("--asset", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--rgb-helper-python", type=Path)
    p.add_argument("--rgb-helper-script", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or (args.name != "imagenet_resnet18" and args.asset is None):
        p.error("Invalid batch size or missing encoder asset")
    run(args)


if __name__ == "__main__":
    main()
