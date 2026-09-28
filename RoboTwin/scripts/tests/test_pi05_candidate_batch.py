"""CPU-only contract checks for the Pi0.5 same-observation candidate RPC."""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace
import unittest
from typing import Any

import numpy as np
import einops


MODEL_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "XPolicyLab/policy/Pi_05_RobotTwin/model.py"
)
TRANSFORM_SOURCE = MODEL_SOURCE.parent / "openpi/src/openpi/policies/robotwin_policy.py"


def load_model_class():
    """Compile just the adapter so these tests need no checkpoint or GPU."""
    tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Model"
        or isinstance(node, ast.FunctionDef) and node.name in {
            "stack_obs", "slice_stacked_obs", "encode_obs", "ensure_chw_uint8"
        }
    ]
    namespace = {"Any": Any, "np": np, "ModelTemplate": object}
    exec(compile(tree, str(MODEL_SOURCE), "exec"), namespace)
    return namespace["Model"]


def load_robotwin_transforms():
    tree = ast.parse(TRANSFORM_SOURCE.read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_parse_image", "_image_mask"}
        or isinstance(node, ast.ClassDef) and node.name in {"RoboTwinInputs", "RoboTwinOutputs"}
    ]
    namespace = {
        "__name__": __name__, "np": np, "einops": einops, "dataclasses": dataclasses,
        "transforms": SimpleNamespace(DataTransformFn=object),
        "_model": SimpleNamespace(ModelType=object),
    }
    exec(compile(tree, str(TRANSFORM_SOURCE), "exec"), namespace)
    return namespace["RoboTwinInputs"], namespace["RoboTwinOutputs"]


def observation(value: float = 0.0):
    return {
        "state": np.full(14, value, dtype=np.float32),
        "images": {name: np.full((3, 4, 4), index, dtype=np.uint8)
                   for index, name in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist"))},
        "instruction": "Pass the red bar",
    }


class FakePolicy:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def infer(self, obs):
        self.calls.append(obs)
        if self.result is not None:
            return {"actions": self.result}
        count = obs["state"].shape[0]
        return {"actions": np.stack([
            np.full((10, 14), index + 1, dtype=np.float32)
            for index in range(count)
        ])}


class Pi05CandidateBatchTests(unittest.TestCase):
    def make_model(self, policy=None):
        model_class = load_model_class()
        model = model_class.__new__(model_class)
        model.policy = policy or FakePolicy()
        model.observation_window = None
        model._latest_env_idx_list = [0]
        model.robot_action_dim_info = None
        model.action_type = "joint"
        return model

    def test_one_inference_for_four_distinct_noise_rows(self):
        model = self.make_model()
        model.update_obs_batch([observation(0.25)])
        result = model.get_action_candidates(4)
        self.assertEqual(len(model.policy.calls), 1)
        self.assertEqual(len(result), 4)
        self.assertEqual([float(item[0, 0]) for item in result], [1, 2, 3, 4])
        batched = model.policy.calls[0]
        self.assertEqual(batched["state"].shape, (4, 14))
        np.testing.assert_array_equal(batched["state"], np.full((4, 14), 0.25))
        self.assertEqual(batched["images"]["cam_high"].shape, (4, 3, 4, 4))
        self.assertEqual(batched["prompt"], ["Pass the red bar"] * 4)
        self.assertEqual(model.observation_window["state"].shape, (1, 14))

    def test_invalid_request_or_policy_output_fails_closed(self):
        model = self.make_model()
        for count in (True, 1, 9, 4.0, "4"):
            with self.assertRaises(ValueError):
                model.get_action_candidates(count)
        with self.assertRaises(AssertionError):
            model.get_action_candidates(4)
        model.update_obs_batch([observation(), observation(1)])
        with self.assertRaises(ValueError):
            model.get_action_candidates(4)

        for output in (np.zeros((10, 14)), np.zeros((3, 10, 14)),
                       np.zeros((4, 10, 13)),
                       np.full((4, 10, 14), np.nan)):
            policy = FakePolicy(output)
            model = self.make_model(policy)
            model.update_obs_batch([observation()])
            with self.assertRaises(ValueError):
                model.get_action_candidates(4)
            self.assertEqual(len(policy.calls), 1)

    def test_robotwin_transforms_preserve_batch_axis_and_action_dim(self):
        inputs_class, outputs_class = load_robotwin_transforms()
        batched = {
            "state": np.zeros((4, 14), dtype=np.float32),
            "images": {name: np.zeros((4, 3, 4, 4), dtype=np.uint8)
                       for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
            "prompt": ["task"] * 4,
        }
        parsed = inputs_class(model_type=None)(batched)
        self.assertEqual(parsed["image"]["base_0_rgb"].shape, (4, 4, 4, 3))
        np.testing.assert_array_equal(parsed["image_mask"]["base_0_rgb"], np.ones(4, dtype=bool))
        np.testing.assert_array_equal(parsed["image_mask"]["left_wrist_0_rgb"], np.ones(4, dtype=bool))
        single = inputs_class(model_type=None, has_left_wrist=False)({
            "state": np.zeros(14),
            "images": {"cam_high": np.zeros((3, 4, 4), dtype=np.uint8)},
        })
        self.assertEqual(single["image"]["base_0_rgb"].shape, (4, 4, 3))
        self.assertEqual(single["image_mask"]["base_0_rgb"].shape, ())
        np.testing.assert_array_equal(single["image_mask"]["left_wrist_0_rgb"], np.bool_(False))
        output = outputs_class()({"actions": np.zeros((4, 10, 32))})["actions"]
        self.assertEqual(output.shape, (4, 10, 14))
        self.assertEqual(outputs_class()({"actions": np.zeros((10, 32))})["actions"].shape, (10, 14))


if __name__ == "__main__":
    unittest.main()
