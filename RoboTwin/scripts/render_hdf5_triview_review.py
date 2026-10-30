#!/usr/bin/env python3
"""Render RoboTwin diagnostic HDF5 camera streams into a 10 fps review video."""

from __future__ import annotations

import argparse
import subprocess

import cv2
import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5")
    parser.add_argument("output")
    parser.add_argument("--fps", type=int, default=10)
    cli = parser.parse_args()
    camera_names = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    with h5py.File(cli.hdf5, "r") as source:
        datasets = [source[f"vision/{name}/colors"] for name in camera_names]
        lengths = [len(dataset) for dataset in datasets]
        if min(lengths) < 2 or len(set(lengths)) != 1:
            raise RuntimeError(f"camera streams incomplete or misaligned: {lengths}")
        shapes = [tuple(int(value) for value in source[f"vision/{name}/shape"][:])
                  for name in camera_names]
        if len(set(shapes)) != 1 or shapes[0][-1] != 3:
            raise RuntimeError(f"unexpected camera shapes: {shapes}")
        height, width, _ = shapes[0]
        ffmpeg = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width * 3}x{height}",
            "-r", str(cli.fps), "-i", "-", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "24", "-f", "mp4", cli.output,
        ], stdin=subprocess.PIPE)
        try:
            for index in range(lengths[0]):
                frames = []
                for dataset in datasets:
                    encoded = np.frombuffer(bytes(dataset[index]), dtype=np.uint8)
                    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if decoded is None or decoded.shape != (height, width, 3):
                        raise RuntimeError(f"invalid image at frame {index}")
                    frames.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
                combined = np.ascontiguousarray(np.concatenate(frames, axis=1))
                ffmpeg.stdin.write(combined.tobytes())
        finally:
            ffmpeg.stdin.close()
            if ffmpeg.wait(timeout=60) != 0:
                raise RuntimeError("ffmpeg failed")


if __name__ == "__main__":
    main()
