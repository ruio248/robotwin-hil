"""Chunk scoring, frozen calibration and execution wiring; no simulator required."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coverage_sampling.calibration import file_sha256, load_discriminator_calibration
from coverage_sampling.core import DecisionSampler, EpisodeLog, SamplingConfig, config_record, coverage_weights
from coverage_sampling.options import add_arguments, config_from_args


class ChunkSamplingTests(unittest.TestCase):
    def candidates(self):
        a = np.zeros((4, 10, 14), dtype=np.float32)
        a[:, 1:, :] = np.arange(4)[:, None, None]
        return a

    def test_entire_chunk_scored_and_one_weight_applied_to_all_steps(self):
        candidates = self.candidates()
        calls = []
        def scoring(chunks):
            calls.append(chunks.copy())
            return chunks[:, -1, :].mean(1) ** 2
        cfg = SamplingConfig("enhanced", 0, 0, score_representation="full_chunk", beta=2, score_scale=3)
        result, row = DecisionSampler(cfg, None, 1).select(
            0, lambda: None, sample_candidates=lambda k: candidates, score_candidates=scoring)
        np.testing.assert_array_equal(calls[0], candidates)
        np.testing.assert_array_equal(calls[1], result[None])
        self.assertEqual(len(calls), 2)
        w = coverage_weights(np.arange(4) ** 2, 2, scale=3)
        np.testing.assert_allclose(result, np.einsum("k,khj->hj", w, candidates), atol=1e-7)
        self.assertGreater(w[0], w[-1])
        self.assertNotAlmostEqual(row["blended_chunk_score"], row["weighted_candidate_score"])
        self.assertEqual(row["score_basis"], "current_observation_full_chunk")
        self.assertIsNone(row["score_action_index"])
        self.assertIsNone(row["selected"])
        self.assertEqual(row["branches"], [])
        self.assertGreater(row["joint_rms_shift_from_uniform"], 0)

    def test_permutation_and_off_path_and_uniform(self):
        a = self.candidates()
        cfg = SamplingConfig("enhanced", 1, 1, score_representation="full_chunk")
        sampler = DecisionSampler(cfg, None, 42)
        forbidden = lambda *args: self.fail("Inactive decision must not sample K or score")
        single, row = sampler.select(0, lambda: a[2], sample_candidates=forbidden, score_candidates=forbidden)
        np.testing.assert_array_equal(single, a[2])
        self.assertEqual(row["policy_inference_requests"], 1)
        scoring = lambda x: x[:, -1, 0]
        result, _ = sampler.select(1, lambda: None, sample_candidates=lambda k: a, score_candidates=scoring)
        perm = [2, 0, 3, 1]
        reordered, _ = sampler.select(1, lambda: None, sample_candidates=lambda k: a[perm], score_candidates=scoring)
        np.testing.assert_allclose(result, reordered, atol=1e-7)
        for changed, score_fn in ((dataclasses.replace(cfg, mode="vanilla"), scoring),
                                  (dataclasses.replace(cfg, uniform_weights=True, score_scale=0), scoring),
                                  (cfg, lambda x: np.ones(len(x)))):
            result, row = DecisionSampler(changed, None, 42).select(
                1, lambda: None, sample_candidates=lambda k: a, score_candidates=score_fn)
            np.testing.assert_array_equal(result, a.mean(0))
            self.assertEqual(row["ess_fraction"], 1)
            self.assertEqual(row["candidate_weight_kl_to_uniform"], 0)

    def test_fail_closed_and_legacy_resume_format(self):
        with self.assertRaises(ValueError):
            SamplingConfig("enhanced", 0, 0, score_representation="full_chunk", horizon=2)
        cfg = SamplingConfig("enhanced", 0, 0, score_representation="full_chunk")
        with self.assertRaisesRegex(ValueError, "direct scorer"):
            DecisionSampler(cfg, None, 1).select(0, lambda: None, sample_candidates=lambda k: self.candidates())
        for values in (np.ones(3), np.full(4, np.nan)):
            with self.assertRaises(ValueError):
                DecisionSampler(cfg, None, 1).select(0, lambda: None,
                    sample_candidates=lambda k: self.candidates(), score_candidates=lambda a: values)
        old = config_record(SamplingConfig("enhanced", 0, 0))
        self.assertNotIn("score_representation", old)
        self.assertNotIn("calibration_sha256", old)
        self.assertIn("calibration_sha256", config_record(cfg))
        k32 = SamplingConfig("enhanced", 0, 9, num_candidates=32, score_representation="full_chunk")
        self.assertEqual(config_record(k32)["policy_microbatch"], 4)

    def test_chunk_execution_log_labels_score_as_chunk_not_first_action(self):
        cfg = SamplingConfig("enhanced", 0, 0, score_representation="full_chunk")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.jsonl"
            log = EpisodeLog(path, cfg, {"selection_score_basis": cfg.score_basis})
            log.executed(0, 0, 2.0, True, np.zeros(14), coverage_basis=cfg.score_basis)
            log.executed(0, 1, None, True, np.zeros(14), coverage_basis=cfg.score_basis)
            log.finish(False, None, 2)
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(rows[1]["coverage_basis"], "current_observation_full_chunk")
            self.assertEqual(rows[-1]["active_executed_scores"], 1)


class CalibrationTests(unittest.TestCase):
    def fixture(self, root):
        checkpoint = root / "disc.pt"
        checkpoint.write_bytes(b"fixture-model")
        path = root / "calibration.json"
        path.write_text(json.dumps({"disc_s42": {"beta": 100., "scale": 1.6, "degenerate": False}}))
        manifest = {"complete": True, "calibration_sha256": file_sha256(path),
                    "models": {"disc_s42": {"kind": "neural", "sha256": file_sha256(checkpoint)}}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        args = dict(es_mode="enhanced", task_name="handover_to_tray", es_window_start=0, es_window_end=9,
                    es_scorer="discriminator_chunk", es_critic=str(checkpoint), es_encoder_weights="encoder.pt",
                    es_calibration=str(path), es_calibration_key="disc_s42")
        return args

    def test_frozen_beta_scale_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            cfg = config_from_args(args)
            self.assertEqual((cfg.beta, cfg.score_scale, cfg.score_representation), (100., 1.6, "full_chunk"))
            self.assertEqual(config_from_args({**args, "es_beta": 0}).beta, 0)
            scores = np.array([1., 1.1, 1.2, 1.3])
            expected = np.exp(-100 * (scores - scores.min()) / 1.6)
            np.testing.assert_allclose(coverage_weights(scores, cfg.beta, scale=cfg.score_scale), expected / expected.sum())
            Path(args["es_critic"]).write_bytes(b"wrong model")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
                config_from_args(args)

    def test_reject_missing_corrupt_and_wrong_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            for changes in ({"es_calibration": None}, {"es_horizon": 9}, {"es_scorer": "td_first_action"}):
                with self.assertRaises(ValueError): config_from_args({**args, **changes})
            Path(args["es_calibration"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "calibration SHA"):
                config_from_args(args)
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        self.assertIsNone(parser.parse_args([]).es_beta)

    def test_launcher_preserves_frequency_and_does_not_force_legacy_beta(self):
        repo = Path(__file__).resolve().parents[3]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("HIL_", "MANUAL_"))}
        env["HIL_DRY_RUN"] = "1"
        command = subprocess.check_output(["bash", str(repo / "local_serving/run_hg_dagger_discriminator.sh")],
                                          env=env, text=True)
        for part in ("--frequency 30", "--es-horizon 10", "--es-num-candidates 32",
                     "--es-policy-microbatch 8", "--es-scorer discriminator_chunk",
                     "--es-activation manual", "disc_s42.pt", "--es-calibration ", "--ckpt-name v2_promptfix_9999"):
            self.assertIn(part, command)
        self.assertNotIn("--es-beta", command)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch tests require the robotwin_hil environment")
class NeuralAdapterTests(unittest.TestCase):
    def test_real_head_parity_repeat_tail_sensitivity_and_encoding_once(self):
        from tail_value.chunk import ChunkCoverageCritic
        from coverage_sampling.critic import OnlineChunkDiscriminator
        from coverage_benchmark.scorers import NeuralScorer
        torch.manual_seed(42)
        norm = {k: np.zeros(14, np.float32) if k.endswith("mean") else np.ones(14, np.float32)
                for k in ("state_mean", "state_scale", "action_mean", "action_scale")}
        model = ChunkCoverageCritic(norm, 10, 256)
        features = np.random.default_rng(42).normal(size=1536).astype(np.float32)
        calls = []
        class Encoder:
            def __init__(self, *a, **kw): pass
            def __call__(self, images): calls.append(images); return features
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "disc.pt"; encoder = root / "encoder.pt"
            encoder.write_bytes(b"fixture-encoder")
            torch.save(dict(representation="absolute_joint_chunk", step=10000,
                            config={"horizon": 10, "width": 256, "objective": "balanced_source_bce", "seed": 42},
                            normalization=norm, model=model.state_dict()), path)
            with patch("coverage_sampling.critic.FrozenEncoder", Encoder):
                online = OnlineChunkDiscriminator(path, encoder)
            offline = NeuralScorer(path)
            obs = {"joint_action": {"vector": np.zeros(14, np.float32)}, "observation": {
                cam: {"rgb": np.zeros((8, 8, 3), np.uint8)} for cam in ("head_camera", "left_camera", "right_camera")}}
            bound = online.bind_observation(obs)
            self.assertEqual(len(calls), 0)
            chunks = np.zeros((4, 10, 14), np.float32)
            chunks[:, -1, :] = np.arange(4)[:, None]  # first actions identical
            expected = offline.score(features, obs["joint_action"]["vector"], chunks)
            actual = bound(chunks)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(bound(chunks), actual)
            np.testing.assert_allclose(bound(chunks[::-1]), actual[::-1], atol=1e-6)
            self.assertGreater(np.ptp(actual), 1e-6)
            bound(chunks.mean(0, keepdims=True))
            self.assertEqual(len(calls), 1)
            for invalid in (chunks[:, 0], chunks[:, :9], np.full_like(chunks, np.nan)):
                with self.assertRaises(ValueError): bound(invalid)


if __name__ == "__main__":
    unittest.main()
