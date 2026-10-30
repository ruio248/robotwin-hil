"""CPU unit tests for offline paired sample-count diagnostics."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debug_candidate_progress import (aggregate, candidate_metrics, observation_schedule,
                                      paired_indices, sample_pool)


class CandidateProgressTests(unittest.TestCase):
    def test_exact_policy_timeline_excludes_hil(self):
        meta = {"save_freq": 15, "control_mask": ["policy"] * 3 + ["hil"] * 2,
                "record": {"policy_steps": 37, "interventions": [{"policy_steps_before_intervention": 37}]}}
        rows = observation_schedule(meta, [Path(f"{i}.pkl") for i in range(5)])
        self.assertEqual([r["policy_step"] for r in rows], [0, 15, 30])
        self.assertAlmostEqual(rows[-1]["progress"], 30/37)
        self.assertLess(rows[-1]["progress"], 1)
        meta["control_mask"] = ["policy"] * 2 + ["hil"] * 3
        with self.assertRaises(ValueError):
            observation_schedule(meta, [Path(f"{i}.pkl") for i in range(5)])

    def test_missing_frame_fails_and_post_hil_is_excluded(self):
        meta = {"save_freq": 15, "control_mask": ["policy"] * 2,
                "record": {"policy_steps": 20}}
        with self.assertRaises(ValueError):
            observation_schedule(meta, [Path("0.pkl"), Path("2.pkl")])
        meta["control_mask"] = ["policy", "policy", "hil", "policy"]
        meta["record"]["interventions"] = [{"policy_steps_before_intervention": 16}]
        rows = observation_schedule(meta, [Path(f"{i}.pkl") for i in range(4)])
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[-1]['progress'], 15/16)
        self.assertEqual(rows[-1]['policy_end_step'], 16)

    def test_fixed_microbatch_same_observation_no_reset(self):
        calls = []; obs = object(); client = object()
        def sample(c, o, action_type, n):
            self.assertIs(c, client); self.assertIs(o, obs)
            self.assertEqual(action_type, "joint"); self.assertEqual(n, 4)
            offset = len(calls) * 4
            calls.append(1)
            return np.broadcast_to(np.arange(offset, offset+4)[:, None, None], (4, 10, 14)).copy()
        pool = sample_pool(client, obs, sample)
        self.assertEqual(len(calls), 8)
        np.testing.assert_array_equal(pool[:, 0, 0], np.arange(32))
        same = sample_pool(client, obs, lambda *a: np.zeros((4, 10, 14)))
        self.assertEqual(candidate_metrics(same, np.arange(32))['32']['unique_first_actions'], 1)
        with self.assertRaises(ValueError):
            sample_pool(client, obs, lambda *a: np.full((4, 10, 14), np.nan))

    def test_metric_first_action_only_and_grippers_separate(self):
        a = np.zeros((32, 10, 14))
        a[:, 1:] = 1000  # later chunk positions must not affect current-action metric
        a[:, 0, 6] = np.arange(32)
        m = candidate_metrics(a, np.arange(32))
        for value in m.values():
            self.assertEqual(value["mean_pairwise_rad"], 0)
            self.assertGreater(value["left_gripper_mean_abs"], 0)
        a[:, 0, 0] = np.arange(32) * np.sqrt(12)
        m = candidate_metrics(a, np.arange(32))
        self.assertAlmostEqual(m['4']['mean_pairwise_rad'], 10/6)
        self.assertAlmostEqual(m['4']['max_pairwise_rad'], 3)
        self.assertAlmostEqual(m['32']['max_pairwise_rad'], 31)

    def test_prefixes_reproducible_max_monotonic_mean_not_required(self):
        indices = paired_indices(42, 7, 3)
        np.testing.assert_array_equal(indices, paired_indices(42, 7, 3))
        self.assertFalse(np.array_equal(indices, paired_indices(42, 7, 4)))
        a = np.random.default_rng(1).normal(size=(32, 10, 14))
        m = candidate_metrics(a, indices, scores=np.full(32, 40.0))
        maxima = [m[str(k)]['max_pairwise_rad'] for k in (4, 8, 16, 32)]
        self.assertEqual(maxima, sorted(maxima))
        for value in m.values():
            self.assertAlmostEqual(value['ess_fraction'], 1)
            self.assertLess(value['tilted_vs_uniform_rad'], 1e-15)
        reverse = candidate_metrics(a, indices[::-1])
        self.assertAlmostEqual(m['32']['mean_pairwise_rad'], reverse['32']['mean_pairwise_rad'])

    def test_equal_episode_weight_and_missing_bins_not_zero(self):
        def row(ep, value):
            return {'rollout_index': ep, 'progress': .25,
                    'metrics': {str(k): {'x': value} for k in (4, 8, 16, 32)}}
        bins = aggregate([row(0, 0)] * 10 + [row(1, 10)], 'x')
        self.assertEqual(bins[5]['k']['4']['median'], 5)
        self.assertEqual(bins[5]['episodes'], 2)
        self.assertIsNone(bins[0]['k']['4'])


if __name__ == '__main__':
    unittest.main()
