#!/usr/bin/env python3
"""Dedicated batch-sampling policy server without modifying the shared adapter."""
from __future__ import annotations

import asyncio
import numpy as np

from XPolicyLab.setup_policy_server import parse_args_and_config
from XPolicyLab.policy.Pi_05_RobotTwin import model as adapter
from XPolicyLab.utils.process_data import unpack_robot_state
from client_server.ws.model_server import PolicyServer, PolicyServerConfig


def enable_legacy_openpi_batch_images():
    """Handle batched CHW images in older OpenPI RoboTwin transforms.

    Some deployed OpenPI revisions implement batched ``policy.infer`` but
    ``RoboTwinInputs`` only transposes a single CHW image. This changes the
    dedicated diagnostic process only; it never edits the shared policy.
    """
    from openpi.policies import robotwin_policy

    original_parse = robotwin_policy._parse_image
    original_inputs = robotwin_policy.RoboTwinInputs.__call__

    def parse_image(image):
        value = np.asarray(image)
        if value.ndim == 4 and value.shape[1] == 3:
            if np.issubdtype(value.dtype, np.floating):
                value = (255 * value).astype(np.uint8)
            return np.transpose(value, (0, 2, 3, 1))
        return original_parse(image)

    def outputs(self, data):
        actions = np.asarray(data["actions"])
        return {"actions": actions[..., :self.action_dim]}

    def inputs(self, data):
        result = original_inputs(self, data)
        state = np.asarray(result["state"])
        if state.ndim == 2:
            result["image_mask"] = {key: np.full(state.shape[0], bool(value), dtype=np.bool_)
                                    for key, value in result["image_mask"].items()}
        return result

    robotwin_policy._parse_image = parse_image
    robotwin_policy.RoboTwinInputs.__call__ = inputs
    robotwin_policy.RoboTwinOutputs.__call__ = outputs


def get_action_candidates(self, num_candidates):
    if (isinstance(num_candidates, bool) or not isinstance(num_candidates, int)
            or not 2 <= num_candidates <= 8):
        raise ValueError("num_candidates must be an integer between 2 and 8")
    if self.observation_window is None or self.observation_window["state"].shape[0] != 1:
        raise ValueError("Batch sampling requires exactly one updated observation")
    one = adapter.slice_stacked_obs(self.observation_window, 0)
    batched = adapter.stack_obs([one] * num_candidates)
    actions = np.asarray(self.policy.infer(batched)["actions"])
    if (actions.ndim != 3 or actions.shape[0] != num_candidates
            or actions.shape[1:] != (10, 14) or not np.isfinite(actions).all()):
        raise ValueError(f"Expected finite [{num_candidates},10,14] policy result, got {actions.shape}")
    if self.robot_action_dim_info is None:
        return [actions[i] for i in range(num_candidates)]
    return [unpack_robot_state(actions[i], self.action_type, self.robot_action_dim_info,
                               source_type="obs") for i in range(num_candidates)]


def main():
    config = parse_args_and_config()
    if config.get("policy_name") != "Pi_05_RobotTwin" or config.get("action_type") != "joint":
        raise ValueError("This dedicated server is only for the physical-joint Pi0.5 baseline")
    enable_legacy_openpi_batch_images()
    adapter.Model.get_action_candidates = get_action_candidates
    model = adapter.Model(config)
    server = PolicyServer(model, PolicyServerConfig(host=config["host"], port=int(config["port"]),
                                               ws_ping_interval_s=config.get("ws_ping_interval_s", 20.),
                                               ws_ping_timeout_s=config.get("ws_ping_timeout_s", 20.)))
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
