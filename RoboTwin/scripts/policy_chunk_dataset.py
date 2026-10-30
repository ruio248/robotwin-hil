"""Read audited one-stage policy observation/chunk pairs for OpenPI training.

Unlike LeRobot's temporal-indexed episodes, every row already carries its
real executed ten-action label.  This adapter deliberately does not infer a
chunk by indexing the next ten *saved* images, which are 15 commands apart.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset


PROMPT = "Pass the red bar from the left arm to the right arm and place it in the blue tray."
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class PolicyChunkDataset(Dataset):
    """Return LeRobot-like raw samples accepted by OpenPI's RoboTwin transforms."""

    def __init__(self, path: str | Path):
        self.path = str(Path(path).expanduser().resolve())
        self._reader: h5py.File | None = None
        self._pid: int | None = None
        with h5py.File(self.path, "r") as data:
            if data.attrs.get("source") != "executed_policy_actions_not_expert":
                raise ValueError("not an audited policy chunk file")
            if int(data.attrs.get("horizon", -1)) != 10 or int(data.attrs.get("label_action_stride", -1)) != 2:
                raise ValueError("wrong H=10/30-to-15-Hz policy chunk contract")
            self._length = int(data["action.chunk"].shape[0])
            if data["action.chunk"].shape != (self._length, 10, 14):
                raise ValueError("action.chunk must have shape [N,10,14]")
            if data["observation.state"].shape != (self._length, 14):
                raise ValueError("observation.state must have shape [N,14]")
            for camera in CAMERAS:
                if data[f"observation.images.{camera}"].shape[0] != self._length:
                    raise ValueError(f"camera row count mismatch: {camera}")

    def __len__(self) -> int:
        return self._length

    def _open(self) -> h5py.File:
        pid = os.getpid()
        if self._reader is None or self._pid != pid:
            if self._reader is not None:
                self._reader.close()
            self._reader = h5py.File(self.path, "r")
            self._pid = pid
        return self._reader

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self._length:
            raise IndexError(index)
        data = self._open()
        return {
            "observation.state": np.asarray(data["observation.state"][index], dtype=np.float32),
            "observation.images.cam_high": np.asarray(data["observation.images.cam_high"][index]),
            "observation.images.cam_left_wrist": np.asarray(data["observation.images.cam_left_wrist"][index]),
            "observation.images.cam_right_wrist": np.asarray(data["observation.images.cam_right_wrist"][index]),
            "action": np.asarray(data["action.chunk"][index], dtype=np.float32),
            "prompt": PROMPT,
        }

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_reader"] = None
        state["_pid"] = None
        return state

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
            self._pid = None
