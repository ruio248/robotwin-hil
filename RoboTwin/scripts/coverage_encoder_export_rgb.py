#!/usr/bin/env python3
"""Export one audited historical rollout's RGB observations for a separate encoder runtime."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from tail_value.cache import atomic_npz
from tail_value.sources import CAMERAS, JOINT_FIELDS, NATIVE_CAMERAS, add_xpolicylab_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True, type=Path)
    p.add_argument("--candidate-cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    add_xpolicylab_path()
    from XPolicyLab.utils.process_data import decode_image_bit
    with np.load(args.candidate_cache, allow_pickle=False) as archive:
        frame = archive["frame_index"].astype(int)
        expected_state = archive["state"]
    with h5py.File(args.hdf5, "r") as handle:
        state = np.concatenate([np.asarray(handle[f"state/{field}"]).reshape(len(handle[f"state/{field}"]), -1)
                                for field in JOINT_FIELDS], axis=1).astype(np.float32)
        if np.any(frame < 0) or np.any(frame >= len(state)) or not np.allclose(state[frame], expected_state, atol=1e-5, rtol=1e-5):
            raise ValueError("Historical HDF5 and candidate-cache states/frames disagree")
        images = {}
        for camera, native in zip(CAMERAS, NATIVE_CAMERAS, strict=True):
            stream = handle[f"vision/{native}/colors"]
            images[camera] = np.stack([decode_image_bit(stream[int(i)]) for i in frame])
            if images[camera].dtype != np.uint8 or images[camera].ndim != 4 or images[camera].shape[-1] != 3:
                raise ValueError(f"Invalid RGB {native}")
    atomic_npz(args.output, images)


if __name__ == "__main__":
    main()
