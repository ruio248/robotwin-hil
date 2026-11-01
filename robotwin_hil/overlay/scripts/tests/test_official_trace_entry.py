"""The one-pass D archive must leave the official eval control flow unchanged."""

import ast
import json
import sys
import tempfile
import traceback
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
OFFICIAL = Path(__file__).resolve().parents[4] / "RoboTwin/scripts/eval_policy_xpolicylab.py"
TRACED = SCRIPTS / "eval_policy_xpolicylab_official_trace.py"
sys.path.insert(0, str(SCRIPTS))
from official_policy_trace import OfficialPolicyTrace, TraceRecordingError


def load_eval(source, namespace):
    tree = ast.parse(source.read_text())
    names = {"eval_remote_policy", "normalize_action_chunk", "absolute_joint_chunk_from_response"}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["eval_remote_policy"]


class OfficialTraceEntryTests(unittest.TestCase):
    def test_only_eval_entry_and_recording_helper_differ_from_official(self):
        official = ast.parse(OFFICIAL.read_text())
        traced = ast.parse(TRACED.read_text())
        modified = {"main", "main_batch", "eval_remote_policy", "parse_args"}
        original_functions = {node.name: ast.dump(node, include_attributes=False)
                              for node in official.body if isinstance(node, ast.FunctionDef)}
        traced_functions = {node.name: ast.dump(node, include_attributes=False)
                            for node in traced.body if isinstance(node, ast.FunctionDef)}
        for name, original in original_functions.items():
            if name not in modified:
                self.assertEqual(traced_functions[name], original, name)
        self.assertEqual(set(traced_functions) - set(original_functions),
                         {"absolute_joint_chunk_from_response"})

    def test_same_scene_instruction_actions_and_outcome_with_recording(self):
        def obs():
            return {
                "vision": {name: {"color": np.zeros((3, 4, 3), dtype=np.uint8)} for name in
                           ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                "state": {"left_arm_joint_state": np.zeros(6), "left_ee_joint_state": np.zeros(1),
                          "right_arm_joint_state": np.zeros(6), "right_ee_joint_state": np.zeros(1)},
            }

        class Env:
            def __init__(self):
                self.events = []
                self.eval_video_path = None
                self.render_freq = 0
                self.step_lim = 2
                self.suc = 0

            def setup_demo(self, **kwargs):
                self.seed = kwargs["seed"]
                self.take_action_cnt = 0
                self.eval_success = False
                self.events.append(("setup", self.seed))

            def play_once(self):
                self.plan_success = self.seed != 100000
                return {"info": {}}

            def check_success(self):
                return self.plan_success

            def close_env(self, **kwargs):
                pass

            def set_instruction(self, *, instruction):
                self.instruction = instruction
                self.events.append(("instruction", self.seed, instruction))

            def get_instruction(self):
                return self.instruction

            def get_obs(self):
                return obs()

            def take_action(self, action, *, action_type):
                self.events.append(("action", self.seed, tuple(action), action_type))
                self.take_action_cnt += 1
                if self.seed == 100001:
                    self.eval_success = True

        class Client:
            def __init__(self):
                self.calls = []

            def call(self, func_name, **kwargs):
                self.calls.append(func_name)
                if func_name == "get_action":
                    return np.ones((2, 14), dtype=np.float32)
                return None

        namespace = {
            "Any": object, "Mapping": Mapping, "Sequence": Sequence,
            "np": np, "traceback": traceback,
            "OfficialPolicyTrace": OfficialPolicyTrace, "TraceRecordingError": TraceRecordingError,
            "UnStableError": type("UnStableError", (Exception,), {}),
            "parse_bool": bool, "safe_close_env": lambda env, **kwargs: None,
            "normalize_robotwin_action_type": lambda value: "qpos",
            "build_instruction": lambda *args: "the same seen instruction",
            "prepare_policy_case": lambda *args: None, "reset_policy": lambda *args: None,
            "notify_trial_end": lambda *args: None,
            "robotwin_obs_to_xpolicylab": lambda observation, **kwargs: observation,
            "xpolicylab_action_to_robotwin": lambda action, **kwargs: (np.asarray(action), "qpos"),
            "is_episode_end": lambda env: env.eval_success or env.take_action_cnt >= env.step_lim,
        }
        official_eval = load_eval(OFFICIAL, dict(namespace))
        traced_eval = load_eval(TRACED, dict(namespace))
        args = {"task_name": "handover_block", "policy_name": "test", "render_freq": 0,
                "clear_cache_freq": 100, "task_config": "demo_clean", "ckpt_setting": "test"}
        usr = {"expert_check": True, "frequency": 50, "action_type": "joint"}
        plain_env, plain_client = Env(), Client()
        plain_result = official_eval("handover_block", plain_env, dict(args), plain_client, 100000,
                                     usr_args=usr, test_num=2, instruction_type="seen")
        with tempfile.TemporaryDirectory() as temp:
            trace = OfficialPolicyTrace(Path(temp) / "trace", run_info={})
            traced_env, traced_client = Env(), Client()
            traced_result = traced_eval("handover_block", traced_env, dict(args), traced_client, 100000,
                                        usr_args=usr, test_num=2, instruction_type="seen", trace=trace)
            trace.finish_run(expected_count=2, official_success_count=traced_result[1])
            manifest = json.loads((trace.root / "manifest.json").read_text())
            self.assertEqual(plain_result, traced_result)
            self.assertEqual(plain_env.events, traced_env.events)
            self.assertEqual(plain_client.calls, traced_client.calls)
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["d_analysis_ready"])
            self.assertEqual([ep["scene_seed"] for ep in manifest["episodes"]], [100001, 100002])
            self.assertEqual([ep["success"] for ep in manifest["episodes"]], [True, False])
            self.assertEqual([ep["status"] for ep in manifest["episodes"]],
                             ["success", "policy_failure"])
            self.assertEqual(manifest["successful_episodes"], 1)
            self.assertEqual(manifest["policy_failure_episodes"], 1)
            self.assertEqual(manifest["query_covered_official_episodes"], 2)
            self.assertEqual(manifest["skipped_scenes"][0]["stage"], "expert_rejected")
            self.assertEqual([ep["rollout_steps"] for ep in manifest["episodes"]], [1, 2])

            class BrokenEnv(Env):
                def take_action(self, action, *, action_type):
                    if self.take_action_cnt == 1:
                        raise RuntimeError("simulator lost after one executed action")
                    super().take_action(action, action_type=action_type)

            broken_trace = OfficialPolicyTrace(Path(temp) / "broken_trace", run_info={})
            _, broken_success = traced_eval(
                "handover_block", BrokenEnv(), dict(args), Client(), 100002,
                usr_args={**usr, "expert_check": False}, test_num=1,
                instruction_type="seen", trace=broken_trace,
            )
            broken_trace.finish_run(expected_count=1, official_success_count=broken_success)
            broken = json.loads((broken_trace.root / "manifest.json").read_text())
            self.assertEqual(broken["status"], "infra_error")
            self.assertFalse(broken["d_analysis_ready"])
            self.assertIsNone(broken["episodes"][0]["success"])
            self.assertEqual(broken["episodes"][0]["rollout_steps"], 1)

            def failed_reset(*args):
                raise RuntimeError("policy websocket reset failed")

            init_eval = load_eval(TRACED, {**namespace, "reset_policy": failed_reset})
            init_trace = OfficialPolicyTrace(Path(temp) / "initialization_trace", run_info={})
            with self.assertRaisesRegex(RuntimeError, "policy websocket reset failed"):
                init_eval("handover_block", Env(), dict(args), Client(), 100002,
                          usr_args={**usr, "expert_check": False}, test_num=1,
                          instruction_type="seen", trace=init_trace)
            init_trace.finish_run(expected_count=1, official_success_count=None,
                                  error="policy websocket reset failed")
            init_manifest = json.loads((init_trace.root / "manifest.json").read_text())
            self.assertEqual(init_manifest["status"], "aborted")
            self.assertEqual(init_manifest["episodes"][0]["status"], "infra_error")
            self.assertIsNone(init_manifest["episodes"][0]["success"])
            self.assertIn("policy websocket reset failed", init_manifest["episodes"][0]["exception"])


if __name__ == "__main__":
    unittest.main()
