import unittest

import numpy as np

from recompute_iwr_norm_stats_episode_safe import action_window_delta, build_mixed_samples


class EpisodeSafeNormStatsTest(unittest.TestCase):
    def test_action_window_never_crosses_episode_and_clamps_inside_episode(self):
        first = {
            "episode_index": 4,
            "state": np.zeros((2, 14), dtype=np.float32),
            "action": np.stack([np.full(14, 0.1), np.full(14, 2.7)]).astype(np.float32),
        }
        second = {
            "episode_index": 5,
            "state": np.zeros((2, 14), dtype=np.float32),
            "action": np.zeros((2, 14), dtype=np.float32),
        }
        window = action_window_delta(first, anchor=1, horizon=4)
        np.testing.assert_allclose(window[:, 0], [2.7, 2.7, 2.7, 2.7])
        np.testing.assert_allclose(window[:, 8], [2.7, 2.7, 2.7, 2.7])
        # Gripper dimensions remain absolute, even though arm dimensions are deltas.
        np.testing.assert_allclose(window[:, 6], [2.7, 2.7, 2.7, 2.7])

    def test_delta_is_relative_to_anchor_state_for_each_arm_joint(self):
        episode = {
            "episode_index": 0,
            "state": np.full((3, 14), 1.0, dtype=np.float32),
            "action": np.full((3, 14), 1.4, dtype=np.float32),
        }
        window = action_window_delta(episode, anchor=0, horizon=2)
        np.testing.assert_allclose(window[:, :6], 0.4)
        np.testing.assert_allclose(window[:, 7:13], 0.4)
        np.testing.assert_allclose(window[:, [6, 13]], 1.4)

    def test_equal_anchor_count_per_source(self):
        def ep(episode_index, length, offset):
            return {
                "episode_index": episode_index,
                "state": np.zeros((length, 14), dtype=np.float32),
                "action": np.full((length, 14), offset, dtype=np.float32),
            }

        sources = {"baseline": [ep(0, 20, 0.1)], "dagger": [ep(1, 5, 0.2)]}
        states, actions, counts = build_mixed_samples(sources, horizon=3, budget=100)
        self.assertEqual(counts["baseline"]["sampled_anchor_frames"], 5)
        self.assertEqual(counts["dagger"]["sampled_anchor_frames"], 5)
        self.assertEqual(states.shape, (10, 14))
        self.assertEqual(actions.shape, (30, 14))


if __name__ == "__main__":
    unittest.main()
