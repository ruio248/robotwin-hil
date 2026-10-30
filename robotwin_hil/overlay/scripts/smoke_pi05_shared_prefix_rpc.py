#!/usr/bin/env python3
"""Exercise K=32 shared-prefix candidates through a dedicated local WS server."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np

from client_server.ws import WsModelClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18305)
    args = parser.parse_args()
    if args.port != 18305:
        raise ValueError("smoke test is restricted to diagnostic port 18305")
    manifest = json.loads((args.candidate_cache / "manifest.json").read_text())
    if manifest.get("complete") is not True:
        raise ValueError("candidate cache is incomplete")
    frame = Path(manifest["observations"][0]["frame_path"])
    with frame.open("rb") as stream:
        observation = pickle.load(stream)  # trusted project-owned cache
    encoded_observation = {
        "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        "images": {key: observation["observation"][camera]["rgb"] for key, camera in zip(
            ("cam_high", "cam_left_wrist", "cam_right_wrist"),
            ("head_camera", "left_camera", "right_camera"), strict=True)},
        "instruction": "Pass the red bar from the left arm to the right arm and place it in the blue tray.",
    }
    client = WsModelClient(
        url=f"ws://127.0.0.1:{args.port}", evaluation_id="shared-prefix-smoke", trial_id="saved-frame",
        max_connect_attempts=3, request_timeout_s=30,
    )
    try:
        capability = client.call(func_name="candidate_sampling_info")
        if not capability.get("shared_prefix_enabled") or capability.get("max_batch_size", 0) < 32:
            raise AssertionError(f"server does not advertise shared-prefix K=32: {capability}")
        client.call(func_name="update_obs", obs=encoded_observation)
        start = time.perf_counter()
        raw = client.call(func_name="get_action_candidates", obs=32)
        if not isinstance(raw, list) or len(raw) != 32 or any(len(chunk) != 10 for chunk in raw):
            raise AssertionError("server must return 32 ten-step chunks")
        chunks = np.stack([
            np.stack([np.concatenate([np.asarray(part).reshape(-1) for part in action.values()])
                      for action in chunk])
            for chunk in raw
        ])
        elapsed = time.perf_counter() - start
        if chunks.shape != (32, 10, 14) or not np.isfinite(chunks).all():
            raise AssertionError(f"invalid returned chunks: {chunks.shape}")
        print(json.dumps({"complete": True, "capability": capability, "shape": list(chunks.shape),
                          "distinct_chunks": int(len(np.unique(chunks.reshape(32, -1), axis=0))),
                          "rpc_seconds": elapsed}), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
