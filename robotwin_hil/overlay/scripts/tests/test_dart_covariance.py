import unittest

import numpy as np

from dart_covariance import fit_covariance, noisy_expert_action


class DartCovarianceTest(unittest.TestCase):
    def test_covariance_and_trace(self):
        expert = np.zeros((4, 12))
        learner = np.zeros_like(expert)
        learner[:, 0] = [1, -1, 1, -1]
        cov = fit_covariance(learner, expert, alpha_per_step=0.04)
        self.assertAlmostEqual(np.trace(cov), 0.04)
        self.assertAlmostEqual(cov[0, 0], 0.04)

    def test_episode_balancing(self):
        expert = np.zeros((3, 12))
        learner = np.zeros_like(expert)
        learner[:, 0] = [1, 2, 2]
        pooled = fit_covariance(learner, expert)
        balanced = fit_covariance(learner, expert, episode_offsets=[0, 1, 3])
        self.assertAlmostEqual(pooled[0, 0], 3)
        self.assertAlmostEqual(balanced[0, 0], 2.5)

    def test_grippers_unchanged_and_seeded(self):
        clean = np.arange(14, dtype=float)
        covariance = np.eye(12) * 0.01
        first = noisy_expert_action(clean, covariance, np.random.default_rng(42))
        second = noisy_expert_action(clean, covariance, np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[[6, 13]], clean[[6, 13]])
        self.assertNotEqual(first[0], clean[0])

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            fit_covariance(np.zeros((2, 12)), np.zeros((2, 12)), alpha_per_step=0.1)
        with self.assertRaises(ValueError):
            noisy_expert_action(np.zeros(14), -np.eye(12), np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
