import unittest

import numpy as np

from analyze_coverage_expert_rank import METHODS, analyze, paired_percentile


class TestExpertRank(unittest.TestCase):
    def test_same_observation_rank_and_ties(self):
        self.assertEqual(paired_percentile(.35, [.1, .2, .3, .4]), .75)
        self.assertEqual(paired_percentile(.2, [.1, .2, .2, .3]), .5)
        self.assertEqual(paired_percentile(1, [1, 1, 1, 1]), .5)
        with self.assertRaisesRegex(ValueError, "four finite"):
            paired_percentile(0, [1, 2, 3])

    def test_episode_balanced_and_observation_only(self):
        def row(expert, scores, frame):
            return {"frame_index": frame,
                    "expert_scores": {method: expert if method != "observation" else 0
                                      for method, _ in METHODS},
                    "scores": {method: scores if method != "observation" else [0] * 4
                               for method, _ in METHODS}}
        episodes = [
            {"metadata": {"episode": 1}, "rows": [row(1, [0]*4, 0)]},
            {"metadata": {"episode": 2}, "rows": [row(-1, [0]*4, 0), row(-1, [0]*4, 1)]},
        ]
        observations, summary = analyze(episodes, bootstrap_repeats=100)
        self.assertEqual(len(observations), 3)
        self.assertEqual(summary["knn"]["episode_balanced_mean"], .5)
        self.assertAlmostEqual(summary["knn"]["observation_weighted_mean"], 1/3)
        self.assertEqual(summary["observation"]["episode_balanced_mean"], .5)
        self.assertEqual(summary["observation"]["tied_pair_fraction"], 1)


if __name__ == "__main__":
    unittest.main()
