"""Frozen features and repeated same-observation policy sampling."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .cache import ACTION_DIM, ACTION_SPACE, CAMERAS, FEATURE_DIM, PROMPT, transition_mask, validate_episode
from .sources import JOINT_FIELDS, iter_episode, rgb_image

ENCODER_SPEC = {"name": "resnet18", "weights": "IMAGENET1K_V1", "features": FEATURE_DIM,
                "resize": "bilinear-align_corners_false-antialias; long-side=224; centered-zero-pad-before-imagenet-normalize",
                "cameras": list(CAMERAS)}


def image_tensor(image):
    image = rgb_image(image)
    value = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
    height, width = image.shape[:2]
    scale = 224 / max(height, width)
    h, w = max(1, round(height * scale)), max(1, round(width * scale))
    value = F.interpolate(value[None], size=(h, w), mode="bilinear", align_corners=False, antialias=True)[0]
    top, left = (224 - h) // 2, (224 - w) // 2
    value = F.pad(value, (left, 224 - w - left, top, 224 - h - top))
    return (value - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


class FrozenEncoder:
    def __init__(self, device, weights_path=None):
        from torchvision.models import resnet18, ResNet18_Weights
        self.device = torch.device(device)
        model = resnet18(weights=None if weights_path else ResNet18_Weights.IMAGENET1K_V1)
        if weights_path:
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model.fc = torch.nn.Identity()
        self.model = model.to(self.device).requires_grad_(False).eval()

    @torch.no_grad()
    def __call__(self, images):
        batch = torch.stack([image_tensor(images[c]) for c in CAMERAS]).to(self.device)
        return self.model(batch).reshape(-1).cpu().numpy().astype(np.float32)


def first_action(chunk):
    """Extract the first physical absolute-joint action from a policy chunk.

    OpenPI may train the arm dimensions in delta space, but its output
    transform restores the current state before the RoboTwin adapter receives
    the action.  The cache deliberately stores that final absolute qpos
    representation; it is the same space as the recorded 14D action.
    """
    if isinstance(chunk, dict) and "actions" in chunk:
        chunk = chunk["actions"]
    if not isinstance(chunk, (list, tuple, np.ndarray)) or len(chunk) == 0:
        raise ValueError("Policy must return a nonempty action chunk")
    if isinstance(chunk[0], dict):
        # Runtime adapter uses singular *_joint_state; HDF5 uses *_joint_states.
        action = np.concatenate([np.asarray(chunk[0][key.removesuffix("s")], dtype=np.float32).reshape(-1) for key in JOINT_FIELDS])
    else:
        array = np.asarray(chunk, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != ACTION_DIM:
            raise ValueError(f"Expected physical [horizon,{ACTION_DIM}] {ACTION_SPACE} policy actions, got {array.shape}")
        action = array[0]
    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError(f"Policy first action is not finite physical {ACTION_DIM}D {ACTION_SPACE}")
    return action.copy()


def sample_candidates(client, frame, count):
    observation = {"images": frame["images"], "state": frame["state"], "instruction": PROMPT}
    client.call(func_name="update_obs", obs=observation)
    # Do not reset RNG or use later time positions as independent samples.
    return np.stack([first_action(client.call(func_name="get_action")) for _ in range(count)])


def generate_episode(source, client, encoder, count, progress=None):
    client.call(func_name="reset")
    data = {"features": [], "state": [], "a_pi": [], "frame_index": []}
    data["a_demo" if source.kind == "demo" else "control_source"] = []
    for i, frame in enumerate(iter_episode(source)):
        if frame["state"].shape != (14,) or not np.isfinite(frame["state"]).all():
            raise ValueError("Invalid observed robot state")
        data["features"].append(encoder(frame["images"]))
        data["state"].append(frame["state"])
        data["a_pi"].append(sample_candidates(client, frame, count))
        data["frame_index"].append(frame["frame_index"])
        key = "a_demo" if source.kind == "demo" else "control_source"
        data[key].append(frame[key])
        if progress and (i + 1) % 25 == 0:
            progress(i + 1)
    arrays = {key: np.asarray(value, dtype=np.int64 if key == "frame_index" else np.int8 if key == "control_source" else np.float32)
              for key, value in data.items()}
    arrays["valid_transition"] = transition_mask(len(arrays["state"]), source.kind == "demo")
    validate_episode(arrays, source.kind, count)
    return arrays
