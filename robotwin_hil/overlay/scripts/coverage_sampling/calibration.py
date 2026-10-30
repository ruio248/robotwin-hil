"""Load frozen benchmark calibration with model/artifact identity checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_discriminator_calibration(path, key, checkpoint):
    """Require the completed fit manifest; reject mixed models/calibrations."""
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    digest = file_sha256(path)
    if manifest.get("complete") is not True or digest != manifest.get("calibration_sha256"):
        raise ValueError("Incomplete fit or calibration SHA mismatch")
    model = manifest.get("models", {}).get(key, {})
    if model.get("kind") != "neural" or model.get("sha256") != file_sha256(checkpoint):
        raise ValueError("Calibration key/checkpoint SHA mismatch")
    calibration = json.loads(path.read_text())[key]
    beta, scale = float(calibration["beta"]), float(calibration["scale"])
    degenerate = calibration["degenerate"]
    if not isinstance(degenerate, bool):
        raise ValueError("Invalid degenerate calibration flag")
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(scale) or scale < 0:
        raise ValueError("Invalid frozen calibration values")
    if scale == 0 and not degenerate:
        raise ValueError("Zero calibration scale must be marked degenerate")
    return calibration, digest
