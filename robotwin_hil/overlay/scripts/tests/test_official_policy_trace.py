"""Query archives must preserve alignment and never turn infra errors into D labels."""

import ast
import json
import sys
import tempfile
import traceback
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official_policy_trace import OfficialPolicyTrace, TraceRecordingError


def observation():
    return {
        "vision": {
            name: {"color": np.full((4, 5, 3), index, dtype=np.uint8)}
            for index, name in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist"))
        },
        "state": {
            "left_arm_joint_state": np.zeros(6),
            "left_ee_joint_state": np.zeros(1),
            "right_arm_joint_state": np.ones(6),
            "right_ee_joint_state": np.ones(1),
        },
    }


class OfficialPolicyTraceTests(unittest.TestCase):
    def test_successful_two_query_episode_and_exact_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace = OfficialPolicyTrace(root, run_info={"task_name": "handover_block"})
            trace.begin_episode(scene_seed=100000, instruction="hand over block", official_index=0)
            trace.record_query(xpl_obs=observation(), absolute_joint_chunk=np.ones((50, 14)),
                               policy_steps_before_query=0)
            trace.action_executed()
            trace.action_executed()
            trace.record_query(xpl_obs=observation(), absolute_joint_chunk=np.zeros((50, 14)),
                               policy_steps_before_query=2)
            trace.action_executed()
            trace.finish_episode(success=True, rollout_failed=False, exception=None,
                                 rollout_steps=3, counted_in_official_result=True)
            trace.finish_run(expected_count=1, official_success_count=1)

            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["d_analysis_ready"])
            episode = manifest["episodes"][0]
            self.assertEqual([q["executed_actions"] for q in episode["queries"]], [2, 1])
            self.assertEqual([q["policy_steps_before_query"] for q in episode["queries"]], [0, 2])
            path = root / episode["directory"] / episode["queries"][0]["file"]
            with np.load(path, allow_pickle=False) as arrays:
                self.assertEqual(arrays["absolute_joint_chunk"].shape, (50, 14))
                self.assertEqual(arrays["rgb_cam_head"].shape, (4, 5, 3))
                self.assertEqual(arrays["state_left_arm_joint_state"].shape, (6,))
            with self.assertRaises(TraceRecordingError):
                OfficialPolicyTrace(root, run_info={})

    def test_post_action_exception_excluded_even_if_official_counts_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace = OfficialPolicyTrace(root, run_info={})
            trace.begin_episode(scene_seed=100000, instruction="test", official_index=0)
            trace.record_query(xpl_obs=observation(), absolute_joint_chunk=np.zeros((10, 14)),
                               policy_steps_before_query=0)
            trace.action_executed()
            trace.finish_episode(success=False, rollout_failed=True, exception="simulator error",
                                 rollout_steps=1, counted_in_official_result=True)
            trace.finish_run(expected_count=1, official_success_count=0)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "infra_error")
            self.assertFalse(manifest["d_analysis_ready"])
            self.assertFalse(manifest["episodes"][0]["d_eligible"])
            self.assertIsNone(manifest["episodes"][0]["success"])
            self.assertEqual(manifest["episodes"][0]["exception"], "simulator error")

    def test_actual_eval_loop_excludes_partial_rollout_exception(self):
        source = Path(__file__).resolve().parents[1] / "eval_policy_xpolicylab_official_trace.py"
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in
                     {"eval_remote_policy", "normalize_action_chunk"}]

        class UnStableError(Exception):
            pass

        namespace = {
            "Any": object, "Mapping": Mapping, "Sequence": Sequence,
            "Path": Path, "np": np, "traceback": traceback,
            "OfficialPolicyTrace": OfficialPolicyTrace,
            "TraceRecordingError": TraceRecordingError,
            "UnStableError": UnStableError,
            "parse_bool": bool,
            "normalize_robotwin_action_type": lambda _: "qpos",
            "build_instruction": lambda *args, **kwargs: "the exact instruction",
            "prepare_policy_case": lambda *args: None,
            "reset_policy": lambda *args: None,
            "notify_trial_end": lambda *args: None,
            "robotwin_obs_to_xpolicylab": lambda obs, **kwargs: obs,
            "absolute_joint_chunk_from_response": lambda raw, *args: np.asarray(raw),
            "xpolicylab_action_to_robotwin": lambda action, **kwargs: (np.asarray(action), "qpos"),
            "is_episode_end": lambda env: env.eval_success or env.take_action_cnt >= env.step_lim,
        }
        exec(compile(tree, str(source), "exec"), namespace)

        class Client:
            policy_calls = 0

            def call(self, func_name, **kwargs):
                if func_name == "get_action":
                    self.policy_calls += 1
                    return np.ones((2, 14), dtype=np.float32)
                return None

        class Env:
            def __init__(self):
                self.eval_video_path = None
                self.eval_success = False
                self.take_action_cnt = 0
                self.step_lim = 2
                self.render_freq = 0

            def setup_demo(self, **kwargs):
                pass

            def set_instruction(self, instruction):
                self.instruction = instruction

            def get_instruction(self):
                return self.instruction

            def get_obs(self):
                return observation()

            def take_action(self, action, *, action_type):
                if self.take_action_cnt == 1:
                    raise RuntimeError("renderer lost after first action")
                self.take_action_cnt += 1

            def close_env(self, **kwargs):
                pass

        args = {
            "task_name": "handover_block", "policy_name": "test", "render_freq": 0,
            "clear_cache_freq": 100, "task_config": "demo_clean", "ckpt_setting": "test",
        }
        usr_args = {"expert_check": False, "frequency": 50, "action_type": "joint"}
        with tempfile.TemporaryDirectory() as temp:
            trace = OfficialPolicyTrace(Path(temp) / "trace", run_info={})
            env, client = Env(), Client()
            _, successes = namespace["eval_remote_policy"](
                "handover_block", env, args, client, 100000, usr_args=usr_args,
                test_num=1, instruction_type="seen", trace=trace,
            )
            self.assertEqual(successes, 0)  # Preserve upstream accounting.
            self.assertEqual(client.policy_calls, 1)
            trace.finish_run(expected_count=1, official_success_count=successes)
            manifest = json.loads((Path(temp) / "trace" / "manifest.json").read_text())
            episode = manifest["episodes"][0]
            self.assertEqual(episode["rollout_steps"], 1)
            self.assertEqual(episode["queries"][0]["executed_actions"], 1)
            self.assertTrue(episode["rollout_failed"])
            self.assertIn("renderer lost", episode["exception"])
            self.assertTrue(episode["counted_in_official_result"])
            self.assertFalse(episode["d_eligible"])
            self.assertEqual(manifest["status"], "infra_error")

    def test_pre_action_exception_not_counted_and_missing_inputs_abort(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace = OfficialPolicyTrace(root, run_info={})
            trace.begin_episode(scene_seed=100000, instruction="test", official_index=0)
            bad = observation()
            del bad["vision"]["cam_left_wrist"]
            with self.assertRaises(TraceRecordingError):
                trace.record_query(xpl_obs=bad, absolute_joint_chunk=np.zeros((10, 14)),
                                   policy_steps_before_query=0)
            trace.finish_run(expected_count=1, official_success_count=None,
                             error="TraceRecordingError: missing camera")
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "aborted")
            self.assertFalse(manifest["d_analysis_ready"])

    def test_infra_setup_skip_pauses_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace = OfficialPolicyTrace(root, run_info={})
            trace.skip_scene(scene_seed=100000, stage="eval_setup_unstable", reason="UnStableError")
            trace.begin_episode(scene_seed=100001, instruction="test", official_index=0)
            trace.finish_episode(success=False, rollout_failed=False, exception=None,
                                 rollout_steps=0, counted_in_official_result=True)
            trace.finish_run(expected_count=1, official_success_count=0)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["infra_error_skipped_scenes"], 1)
            self.assertFalse(manifest["d_analysis_ready"])

    def test_no_policy_query_is_not_a_d_ready_success_or_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = OfficialPolicyTrace(Path(temp) / "trace", run_info={})
            trace.begin_episode(scene_seed=100000, instruction="test", official_index=0)
            trace.finish_episode(success=False, rollout_failed=False, exception=None,
                                 rollout_steps=0, counted_in_official_result=True)
            trace.finish_run(expected_count=1, official_success_count=0)
            manifest = json.loads((trace.root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "aborted")
            self.assertFalse(manifest["d_analysis_ready"])
            self.assertEqual(manifest["query_covered_official_episodes"], 0)


if __name__ == "__main__":
    unittest.main()
