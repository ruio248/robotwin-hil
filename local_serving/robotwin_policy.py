"""OpenPI transforms for the 14-DoF RoboTwin ALOHA-AgileX embodiment."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboTwinInputs(transforms.DataTransformFn):
    """Map three-view RoboTwin LeRobot observations into OpenPI inputs."""

    model_type: _model.ModelType
    has_left_wrist: bool = True
    has_right_wrist: bool = True

    def __call__(self, data: dict) -> dict:
        # Dataset samples arrive after RepackTransform with flat
        # ``observation/*`` keys, while the XPolicyLab runtime adapter sends
        # the already-packed ``images``/``state`` structure. Accept both so
        # training and websocket inference share the same model transform.
        if "images" in data and "state" in data:
            in_images = data["images"]
            head = _parse_image(in_images["cam_high"])
            left_wrist = (
                _parse_image(in_images["cam_left_wrist"])
                if self.has_left_wrist and "cam_left_wrist" in in_images
                else np.zeros_like(head)
            )
            right_wrist = (
                _parse_image(in_images["cam_right_wrist"])
                if self.has_right_wrist and "cam_right_wrist" in in_images
                else np.zeros_like(head)
            )
            state = data["state"]
        else:
            head = _parse_image(data["observation/image"])
            left_wrist = (
                _parse_image(data["observation/left_wrist_image"])
                if self.has_left_wrist and "observation/left_wrist_image" in data
                else np.zeros_like(head)
            )
            right_wrist = (
                _parse_image(data["observation/right_wrist_image"])
                if self.has_right_wrist and "observation/right_wrist_image" in data
                else np.zeros_like(head)
            )
            state = data["observation/state"]
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.has_left_wrist else np.False_,
                "right_wrist_0_rgb": np.True_ if self.has_right_wrist else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboTwinOutputs(transforms.DataTransformFn):
    """Trim PI0.5's padded action samples back to 14 robot controls."""

    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
