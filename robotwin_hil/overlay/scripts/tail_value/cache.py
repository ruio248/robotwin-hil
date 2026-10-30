"""Versioned, non-pickle episode caches and reproducible episode splits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA_VERSION = 1
PROMPT = "Pass the red bar from the left arm to the right arm and place it in the blue tray."
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ACTION_DIM = 14
FEATURE_DIM = 1536
# The policy adapter returns this representation after OpenPI's internal
# delta-to-absolute output transform.  Coverage distances must use this
# physical representation for both policy candidates and recorded actions.
ACTION_SPACE = "absolute_joint_qpos"
ACTION_ORDER = (
    "left_arm_joint[6]",
    "left_gripper[1]",
    "right_arm_joint[6]",
    "right_gripper[1]",
)
APPROXIMATION = "next-recorded-joint-target; adjacent saved observations, not verified executed-command transitions"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def episode_split(ids, seed=42):
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError("Training cache requires at least two unique complete episodes")
    ordered = np.random.default_rng(seed).permutation(sorted(ids)).tolist()
    n_val = max(1, int(round(len(ids) * 0.1)))
    val = set(ordered[:n_val])
    return {str(i): "val" if i in val else "train" for i in ids}


def transition_mask(length, demo):
    mask = np.zeros(length, dtype=bool)
    if demo:
        mask[:-1] = True
    return mask


def validate_episode(arrays, kind, candidates=None):
    required = {"features", "state", "a_pi", "frame_index", "valid_transition"}
    if required - arrays.keys():
        raise ValueError(f"Missing cache fields: {required - arrays.keys()}")
    n = len(arrays["state"])
    k = arrays["a_pi"].shape[1] if arrays["a_pi"].ndim == 3 else 0
    if n < (2 if kind == "demo" else 1) or k < 1:
        raise ValueError("Empty/short episode or invalid candidate shape")
    shapes = {"features": (n, FEATURE_DIM), "state": (n, ACTION_DIM),
              "a_pi": (n, k, ACTION_DIM), "frame_index": (n,), "valid_transition": (n,)}
    if kind == "demo":
        shapes["a_demo"] = (n, ACTION_DIM)
    elif kind != "hil" or "a_demo" in arrays:
        raise ValueError("HIL must not contain inferred expert actions")
    for key, shape in shapes.items():
        if key not in arrays or arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid/nonfinite {key}; expected {shape}")
    if candidates is not None and k != candidates:
        raise ValueError("Candidate count differs from manifest")
    if not np.issubdtype(arrays["frame_index"].dtype, np.integer):
        raise ValueError("frame_index must be integer")
    if n > 1 and not np.all(np.diff(arrays["frame_index"]) == 1):
        raise ValueError("Frames must be adjacent and increasing")
    if arrays["valid_transition"].dtype != np.bool_ or not np.array_equal(
        arrays["valid_transition"], transition_mask(n, kind == "demo")
    ):
        raise ValueError("Invalid transition mask: no cross-episode or fabricated terminal transitions")
    if kind == "demo" and not np.allclose(arrays["a_demo"][:-1], arrays["state"][1:], atol=1e-4, rtol=1e-4):
        raise ValueError("Demo action is not the declared next-recorded-state target")
    if kind == "hil":
        if arrays.get("control_source", np.empty(0)).shape != (n,):
            raise ValueError("HIL control mask/frame count mismatch")
        if not np.isin(arrays["control_source"], [0, 1]).all():
            raise ValueError("HIL control_source must be 0=policy or 1=HIL")


def load_episode(path, kind, candidates=None):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    validate_episode(arrays, kind, candidates)
    return arrays


def load_manifest(root, complete=True):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported cache schema")
    if complete and not manifest.get("complete"):
        raise ValueError("Cache is incomplete; resume tail_data.py first")
    ids = [entry["id"] for entry in manifest["episodes"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate cache episode IDs")
    for entry in manifest["episodes"]:
        path = (root / entry["file"]).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Cache episode path escapes cache directory")
        if complete and (not path.is_file() or sha256(path) != entry.get("sha256")):
            raise ValueError(f"Missing/corrupt cache shard: {path}")
    return manifest


def cache_fingerprint(manifest):
    return json_digest({"config": manifest["config"], "episodes": manifest["episodes"]})


def assert_compatible(expected, actual):
    if expected != actual:
        keys = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
        raise ValueError(f"Incompatible critic/cache representation: {keys}")
