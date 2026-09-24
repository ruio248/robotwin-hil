"""CPU tests for sampling, future-state scoring and transactional restoration."""
from __future__ import annotations

import copy
import ast
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import types
import traceback
import unittest
from unittest.mock import patch
from collections.abc import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coverage_sampling.core import DecisionSampler, EpisodeLog, SamplingConfig, coverage_weights
from coverage_sampling.options import config_from_args
from coverage_sampling.robotwin import RobotwinRollout, SceneSnapshot


class FakeJoint:
    def __init__(self):
        self.position, self.velocity = np.array([3.0]), np.array([4.0])
    def get_drive_target(self): return self.position.copy()
    def get_drive_velocity_target(self): return self.velocity.copy()
    def set_drive_target(self, value): self.position = np.array(value, copy=True)
    def set_drive_velocity_target(self, value): self.velocity = np.array(value, copy=True)


class FakeArticulation:
    def __init__(self):
        self.joint = FakeJoint()
        self.qf, self.qacc = np.array([5.0]), np.array([6.0])
    def get_active_joints(self): return [self.joint]
    def get_qf(self): return self.qf.copy()
    def set_qf(self, value): self.qf = np.array(value, copy=True)
    def get_qacc(self): return self.qacc.copy()
    def set_qacc(self, value): self.qacc = np.array(value, copy=True)
    def get_root_pose(self): return SimpleNamespace(p=np.zeros(3), q=np.array([1, 0, 0, 0]))
    def get_qpos(self): return self.joint.position
    def get_qvel(self): return self.joint.velocity


class FakeEnv:
    def __init__(self):
        self.x = 0.0
        self.articulation = FakeArticulation()
        self.scene = SimpleNamespace(
            physx_system=SimpleNamespace(pack=lambda: self.x, unpack=self._unpack),
            get_all_articulations=lambda: [self.articulation],
            get_all_actors=lambda: [SimpleNamespace(get_pose=lambda: SimpleNamespace(
                p=np.array([self.x, 0, 0]), q=np.array([1, 0, 0, 0])))])
        self.robot = SimpleNamespace(left_gripper_val=.2, right_gripper_val=.3)
        self.take_action_cnt, self.step_lim = 0, 20
        self.eval_success, self.save_data, self.render_freq = False, True, 1
        self.eval_video_path = "real_episode_video"
        self.now_obs = {"nested": [7]}
        self.external_writes = 0
        self.stop_at = float("inf")
    def _unpack(self, x): self.x = x
    def _update_render(self): pass
    def get_obs(self):
        self.now_obs = {"joint_action": {"vector": np.full(14, self.x, dtype=np.float32)}}
        return copy.deepcopy(self.now_obs)
    def take_action(self, action, action_type):
        if self.eval_video_path or self.save_data or self.render_freq:
            self.external_writes += 1
        self.x += float(action[0])
        self.take_action_cnt += 1
        self.articulation.joint.position[:] = self.x
        self.articulation.joint.velocity[:] = 99
        self.articulation.qf[:] = 98
        self.articulation.qacc[:] = 97
        self.robot.left_gripper_val = .9
        self.now_obs = {"mutated": True}
        self.eval_success = self.x >= self.stop_at
        random.random()
        np.random.random()


def score(observation, action):
    return float(observation["joint_action"]["vector"][0] + action[0])


def chunk(value, horizon=2):
    return np.full((horizon, 14), value, dtype=np.float32)


class SamplingTests(unittest.TestCase):
    def test_gibbs_ratio_shift_and_limits(self):
        scores = np.array([40.1, 40.2, 40.4])
        weights = coverage_weights(scores, 3)
        self.assertAlmostEqual(weights[0] / weights[1], np.exp(.3))
        np.testing.assert_allclose(weights, coverage_weights(scores + 1000, 3))
        np.testing.assert_allclose(coverage_weights(scores, 0), [1/3]*3)
        np.testing.assert_allclose(coverage_weights([2, 2, 2], 10), [1/3]*3)
        np.testing.assert_array_equal(coverage_weights([-1e300, 1e300], 1e300), [1, 0])

    def test_bad_scores_and_config(self):
        for values in ([], [[1, 2]], [np.nan], [np.inf]):
            with self.assertRaises(ValueError): coverage_weights(values, 1)
        for beta in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError): coverage_weights([1, 2], beta)
        for kwargs in ({"window_start": -1}, {"window_end": 0}, {"num_candidates": 1},
                       {"horizon": 0}, {"seed": -1}, {"replay_atol": np.nan}):
            params = dict(mode="enhanced", window_start=1, window_end=2)
            params.update(kwargs)
            with self.assertRaises(ValueError): SamplingConfig(**params)

    def test_inclusive_window_and_same_candidate_generation(self):
        calls, verifies = [], []
        class Backend:
            def evaluate(self, candidates, verify, atol):
                verifies.append(verify)
                return [{"coverage": [float(c[0, 0]), 99]} for c in candidates], None
        config = SamplingConfig("enhanced", 1, 2, num_candidates=3, horizon=2)
        sampler = DecisionSampler(config, Backend(), 7)
        def sample():
            calls.append(len(calls))
            return chunk(len(calls))
        for decision, expected in enumerate([1, 3, 3, 1]):
            before = len(calls)
            selected, record = sampler.select(decision, sample)
            self.assertEqual(len(calls)-before, expected)
            self.assertEqual(selected.shape, (2, 14))
            self.assertEqual(record["num_candidates"], expected)
        self.assertEqual(verifies, [True, False])

    def test_stochastic_selection_not_argmin(self):
        class Backend:
            def evaluate(self, candidates, **kwargs):
                return [{"coverage": [x]} for x in (1., 1.2, 1.4, 1.6)], None
        cfg = SamplingConfig("enhanced", 0, 0, horizon=2, beta=2)
        sampler = DecisionSampler(cfg, Backend(), 123)
        selected = [sampler.select(0, lambda: chunk(0))[1]["selected"] for _ in range(3000)]
        observed = np.bincount(selected, minlength=4)/len(selected)
        np.testing.assert_allclose(observed, coverage_weights([1, 1.2, 1.4, 1.6], 2), atol=.03)
        self.assertTrue((observed > 0).all())

    def test_vanilla_uses_same_rollouts_and_uniform_selection(self):
        config = SamplingConfig("vanilla", 0, 0, horizon=2)
        env = FakeEnv()
        sampler = DecisionSampler(config, RobotwinRollout(env, score), 7)
        _, record = sampler.select(0, lambda: chunk(1))
        self.assertEqual(len(record["branches"]), 4)
        self.assertEqual(record["weights"], [.25]*4)
        self.assertEqual(env.take_action_cnt, 0)

    def test_wrong_horizon_nonfinite_actions_and_no_silent_truncation(self):
        cfg = SamplingConfig("enhanced", 1, 2, horizon=2)
        sampler = DecisionSampler(cfg, None, 7)
        for candidate in (chunk(0, 10), np.zeros((2, 13)), chunk(np.nan)):
            with self.assertRaises(ValueError): sampler.select(0, lambda: candidate)

    def test_cli_rejects_unsupported_or_missing_inputs(self):
        self.assertIsNone(config_from_args({"es_mode": "off"}))
        valid = {"es_mode": "enhanced", "task_name": "handover_to_tray", "es_critic": "critic.pt",
                 "es_encoder_weights": "encoder.pt", "es_window_start": 1, "es_window_end": 2}
        self.assertEqual(config_from_args(valid).horizon, 10)
        for changes in ({"eval_batch": "true"}, {"action_type": "ee"}, {"es_critic": None},
                        {"es_window_end": None}, {"task_name": "other_task"}):
            with self.assertRaises(ValueError): config_from_args({**valid, **changes})


class RolloutTests(unittest.TestCase):
    def test_real_future_states_pre_action_and_restoration(self):
        env = FakeEnv()
        snapshot = SceneSnapshot(env)
        branches, replay = RobotwinRollout(env, score).evaluate(
            np.stack([chunk(1), chunk(2)]), verify=True, atol=1e-6)
        self.assertEqual(branches[0]["coverage"], [1, 2])
        self.assertEqual(branches[1]["coverage"], [2, 4])
        self.assertEqual(replay["state_max_abs_error"], 0)
        self.assertEqual(env.x, 0)
        self.assertEqual(env.take_action_cnt, 0)
        self.assertFalse(env.eval_success)
        self.assertEqual(env.now_obs, snapshot.env_state["now_obs"])
        self.assertEqual(env.robot.left_gripper_val, .2)
        np.testing.assert_array_equal(env.articulation.joint.position, [3])
        np.testing.assert_array_equal(env.articulation.joint.velocity, [4])
        np.testing.assert_array_equal(env.articulation.qf, [5])
        np.testing.assert_array_equal(env.articulation.qacc, [6])
        self.assertEqual(random.getstate(), snapshot.python_rng)
        np.testing.assert_array_equal(np.random.get_state()[1], snapshot.numpy_rng[1])
        self.assertEqual(env.external_writes, 0)
        self.assertEqual(env.eval_video_path, "real_episode_video")
        self.assertTrue(env.save_data)

    def test_exception_restores_scene_and_recording(self):
        env = FakeEnv()
        def broken(obs, action):
            if env.take_action_cnt == 1:
                raise RuntimeError("scorer failed")
            return score(obs, action)
        with self.assertRaisesRegex(RuntimeError, "scorer failed"):
            RobotwinRollout(env, broken).evaluate([chunk(1)], verify=False, atol=1e-6)
        self.assertEqual((env.x, env.take_action_cnt, env.robot.left_gripper_val), (0, 0, .2))
        self.assertTrue(env.save_data)

    def test_early_terminal_and_remaining_step_budget(self):
        env = FakeEnv()
        env.stop_at = 1
        branches, _ = RobotwinRollout(env, score).evaluate([chunk(1)], verify=True, atol=1e-6)
        self.assertEqual(branches[0]["coverage"], [1])
        self.assertTrue(branches[0]["success"])
        self.assertFalse(env.eval_success)
        env.stop_at, env.step_lim = 100, 1
        branches, _ = RobotwinRollout(env, score).evaluate([chunk(1)], verify=False, atol=1e-6)
        self.assertTrue(branches[0]["step_limit"])
        self.assertEqual(branches[0]["scored_steps"], 1)

    def test_replay_check_detects_hidden_state_leak(self):
        env = FakeEnv()
        counter = [0]
        def leaking(obs, action):
            counter[0] += 1
            return float(counter[0])
        with self.assertRaisesRegex(RuntimeError, "replay mismatch"):
            RobotwinRollout(env, leaking).evaluate([chunk(1), chunk(2)], verify=True, atol=1e-6)
        self.assertEqual(env.x, 0)

    def test_branch_order_independence(self):
        env = FakeEnv()
        backend = RobotwinRollout(env, score)
        forward, _ = backend.evaluate([chunk(1), chunk(2)], verify=True, atol=1e-6)
        reverse, _ = backend.evaluate([chunk(2), chunk(1)], verify=True, atol=1e-6)
        self.assertEqual(forward, reverse[::-1])

    def test_log_reports_unreached_window_and_refuses_overwrite(self):
        config = SamplingConfig("enhanced", 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"episode.jsonl"
            log = EpisodeLog(path, config, {"episode_seed": 123})
            log.finish(False, None, 0)
            result = json.loads(path.read_text().splitlines()[-1])
            self.assertFalse(result["window_reached"])
            self.assertIsNone(result["active_min"])
            with self.assertRaises(FileExistsError): EpisodeLog(path, config, {})


class EvaluatorIntegrationTests(unittest.TestCase):
    """Run the actual evaluator function, with the simulator/server injected.

    Extracting functions avoids importing SAPIEN's GPU renderer in CPU tests;
    the evaluator loop and action adapters are compiled unchanged from source.
    """
    def _run(self, mode, directory, bad_chunk=False):
        from coverage_sampling.core import SamplingConfig
        source = Path(__file__).resolve().parents[1] / "eval_policy_xpolicylab.py"
        tree = ast.parse(source.read_text())
        names = {"eval_remote_policy", "normalize_action_chunk", "xpolicylab_action_to_robotwin",
                 "normalize_robotwin_action_type", "is_episode_end"}
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        ns = {"np": np, "Path": Path, "Mapping": Mapping, "Sequence": Sequence, "Any": object,
              "traceback": traceback, "parse_bool": bool, "build_instruction": lambda *a, **kw: "instruction",
              "prepare_policy_case": lambda *a: None, "reset_policy": lambda client: client.call("reset"),
              "notify_trial_end": lambda *a: None, "safe_close_env": lambda *a, **kw: None,
              "robotwin_obs_to_xpolicylab": lambda observation, **kw: observation}
        exec(compile(tree, str(source), "exec"), ns)

        class Env(FakeEnv):
            def setup_demo(self, **kwargs):
                self.eval_video_path, self.render_freq, self.step_lim = None, 0, 4
            def set_instruction(self, instruction): self.instruction = instruction
            def get_instruction(self): return self.instruction
            def close_env(self, **kwargs): pass
        class Client:
            def __init__(self): self.calls, self.observations = [], []
            def call(self, func_name, **kwargs):
                self.calls.append(func_name)
                if func_name == "update_obs":
                    self.observations.append(kwargs["obs"]["joint_action"]["vector"][0])
                if func_name == "get_action": return chunk(1, 1 if bad_chunk else 2)
        class Scorer:
            metadata = {"critic_sha256": "test", "encoder_sha256": "test_encoder"}
            def __init__(self, *args, **kwargs): pass
            def __call__(self, observation, action): return score(observation, action)
        fake_critic = types.ModuleType("coverage_sampling.critic")
        fake_critic.OnlineCoverage = Scorer
        env, client = Env(), Client()
        args = {"task_name": "handover_to_tray", "policy_name": "Pi_05_RobotTwin", "clear_cache_freq": 10,
                "render_freq": 0, "ckpt_setting": "test", "task_config": "test"}
        usr_args = {"task_name": "handover_to_tray", "expert_check": False, "es_mode": mode,
                    "es_critic": "test", "es_encoder_weights": "test", "es_window_start": 1,
                    "es_window_end": 1, "es_num_candidates": 3, "es_horizon": 2, "es_log_dir": directory}
        with patch.dict(sys.modules, {"coverage_sampling.critic": fake_critic}), contextlib.redirect_stdout(io.StringIO()):
            ns["eval_remote_policy"]("handover_to_tray", env, args, client, 7,
                                     usr_args=usr_args, test_num=1, seed_list=[7])
        return env, client

    def test_opt_in_loop_no_policy_calls_in_branches_and_real_step_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            env, client = self._run("enhanced", directory)
            self.assertEqual(client.calls.count("reset"), 1)
            self.assertEqual(client.calls.count("get_action"), 4)  # one outside, three inside
            self.assertEqual(client.observations, [0, 1, 2, 3])  # only real execution, never lookahead
            self.assertEqual(env.take_action_cnt, 4)
            self.assertEqual(env.external_writes, 4)
            records = [json.loads(line) for line in next(Path(directory).glob("*.jsonl")).read_text().splitlines()]
            executed = [r for r in records if r["event"] == "executed"]
            self.assertEqual([r["coverage"] for r in executed], [1, 2, 3, 4])
            self.assertEqual([r["prediction_error"] for r in executed], [None, None, 0, 0])
            self.assertTrue(records[-1]["window_reached"])
            self.assertEqual(records[-1]["active_executed_scores"], 2)

    def test_off_path_preserves_single_policy_call_per_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            env, client = self._run("off", directory)
            self.assertEqual(client.calls.count("get_action"), 2)
            self.assertEqual(env.take_action_cnt, 4)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_bad_candidate_aborts_instead_of_counting_as_policy_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "evaluation aborted"):
                self._run("enhanced", directory, bad_chunk=True)
            records = [json.loads(line) for line in next(Path(directory).glob("*.jsonl")).read_text().splitlines()]
            self.assertIn("Expected finite", records[-1]["error"])
            self.assertEqual(records[-1]["rollout_steps"], 0)

    def test_comparison_uses_real_execution_and_rejects_unmatched_critics(self):
        from summarize_enhanced_sampling import compare, read_suite
        with tempfile.TemporaryDirectory() as directory:
            vanilla_dir, enhanced_dir = Path(directory)/"vanilla", Path(directory)/"enhanced"
            self._run("vanilla", vanilla_dir)
            self._run("enhanced", enhanced_dir)
            vanilla, enhanced = read_suite(vanilla_dir, "vanilla"), read_suite(enhanced_dir, "enhanced")
            result = compare(vanilla, enhanced)
            self.assertEqual(result["paired_window_reached"], 1)
            self.assertEqual(result["paired_enhanced_minus_vanilla_active_min"], 0)
            self.assertEqual(result["per_seed"][0]["enhanced"]["max_execution_prediction_error"], 0)
            enhanced[7]["head"]["critic_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "critic_sha256"): compare(vanilla, enhanced)

    def test_comparison_rejects_incomplete_log(self):
        from summarize_enhanced_sampling import read_suite
        with tempfile.TemporaryDirectory() as directory:
            log = EpisodeLog(Path(directory)/"episode_0000.jsonl", SamplingConfig("enhanced", 0, 1), {})
            log.stream.close()
            with self.assertRaisesRegex(ValueError, "Incomplete"): read_suite(directory, "enhanced")


if __name__ == "__main__":
    unittest.main()
