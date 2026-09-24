"""Online frozen scorer using exactly the tail training visual preprocessing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tail_value.cache import CAMERAS, sha256
from tail_value.model import CoverageCritic, finite_guard
from tail_value.prepare import ENCODER_SPEC, FrozenEncoder


class OnlineCoverage:
    def __init__(self, checkpoint, encoder_weights, device="cpu"):
        checkpoint, encoder_weights = Path(checkpoint), Path(encoder_weights)
        # Explicit local artifacts prevent accidental model downloads in eval.
        if not checkpoint.is_file() or not encoder_weights.is_file():
            raise FileNotFoundError("Provide a trusted critic checkpoint and local ResNet18 weights")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload["config"]
        self.device = torch.device(device)
        self.model = CoverageCritic(payload["normalization"], width=int(config["width"]))
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).requires_grad_(False).eval()
        self.encoder = FrozenEncoder(self.device, weights_path=encoder_weights)
        self.limit = float(config.get("score_limit", 1e4))
        self.metadata = {"critic_checkpoint": str(checkpoint.resolve()), "critic_sha256": sha256(checkpoint),
                         "critic_step": payload.get("step"), "critic_alpha": config.get("alpha"),
                         "critic_bootstrap": config.get("bootstrap"), "encoder": ENCODER_SPEC,
                         "encoder_weights": str(encoder_weights.resolve()), "encoder_sha256": sha256(encoder_weights),
                         "action_semantics": "executed absolute joint target [left6,gripper,right6,gripper]",
                         "state_semantics": "RoboTwin joint_action.vector (same drive-target state as training)"}

    @torch.inference_mode()
    def __call__(self, observation, action):
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.shape != (14,) or action.shape != (14,):
            raise ValueError("Coverage critic requires 14D state and absolute joint action")
        cameras = ("head_camera", "left_camera", "right_camera")
        images = {key: observation["observation"][camera]["rgb"] for key, camera in zip(CAMERAS, cameras)}
        features = self.encoder(images)
        tensors = [torch.as_tensor(x, device=self.device)[None] for x in (features, state, action)]
        finite_guard(dict(zip(("features", "state", "action"), tensors)))
        score = self.model(*tensors)
        finite_guard({"coverage": score}, self.limit)
        return float(score.item())
