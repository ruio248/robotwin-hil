"""Fast synthetic checks for chunk identity and episode-safe H-step TD."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tail_chunk_data import anchor_indices, policy_chunk
from eval_tail_chunk_hil import (exact_step_mean_ranges, exact_step_median_ranges,
                                 exact_step_max_ranges,
                                 exact_step_min_median_max_ranges)
from tail_value.cache import atomic_npz, sha256
from tail_value.chunk import (ChunkCoverageCritic, ChunkTransitionTable, chunk_critic_loss,
                              macro_reward, validate_chunk_episode)


class SumCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.horizon = 10
        self.scale = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, features, state, chunk):
        return self.scale * chunk.sum((-1, -2))

    def score_many(self, features, state, chunks):
        return self.scale * chunks.sum((-1, -2))


class TailChunkTests(unittest.TestCase):
    def test_exact_step_average_does_not_merge_adjacent_observations(self):
        def row(rollout, step, spread):
            return {"rollout_index": rollout, "policy_step": step,
                    "k": {str(k): {"b_range": spread * k / 4} for k in (4, 8, 16, 32)}}

        rows = [row(0, 0, 1.), row(1, 0, 3.), row(0, 15, 9.)]
        result = exact_step_mean_ranges(rows)
        self.assertEqual([item["policy_step"] for item in result], [0, 15])
        self.assertEqual([item["trajectories"] for item in result], [2, 1])
        self.assertEqual(result[0]["mean_by_k"]["4"], 2.)
        self.assertEqual(result[0]["mean_by_k"]["32"], 16.)
        self.assertEqual(result[1]["mean_by_k"]["4"], 9.)
        median = exact_step_median_ranges(rows + [row(2, 0, 100.)])
        self.assertEqual(median[0]["median_by_k"]["4"], 3.)
        self.assertEqual(median[0]["median_by_k"]["32"], 24.)
        maximum = exact_step_max_ranges(rows + [row(2, 0, 100.)])
        self.assertEqual(maximum[0]["max_by_k"]["4"], 100.)
        self.assertEqual(maximum[0]["max_by_k"]["32"], 800.)
        self.assertEqual(maximum[1]["max_by_k"]["4"], 9.)
        envelope = exact_step_min_median_max_ranges(rows + [row(2, 0, 100.)])
        self.assertEqual(envelope[0]["min_by_k"]["4"], 1.)
        self.assertEqual(envelope[0]["median_by_k"]["4"], 3.)
        self.assertEqual(envelope[0]["max_by_k"]["4"], 100.)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            exact_step_mean_ranges(rows + [row(0, 0, 5.)])

    def test_full_chunk_conversion_keeps_late_actions(self):
        raw = [{"left_arm_joint_state": np.full(6, i), "left_ee_joint_state": [i],
                "right_arm_joint_state": np.full(6, i + 1), "right_ee_joint_state": [i + 1]}
               for i in range(10)]
        result = policy_chunk(raw, 10)
        self.assertEqual(result.shape, (10, 14))
        self.assertEqual(result[9, 0], 9)
        self.assertEqual(result[9, 13], 10)
        self.assertEqual(anchor_indices(35, 10).tolist(), [0, 10, 20])

    def test_macro_target_uses_h_step_reward_and_next_chunk_only(self):
        online, target = SumCritic(), SumCritic()
        target.scale.requires_grad_(False)
        batch = {"features": torch.zeros(2, 1536), "state": torch.zeros(2, 14),
                 "expert_chunks": torch.zeros(2, 10, 14),
                 "policy_chunks": torch.ones(2, 4, 10, 14),
                 "next_features": torch.zeros(2, 1536), "next_state": torch.zeros(2, 14),
                 "next_policy_chunks": torch.full((2, 4, 10, 14), 2.)}
        loss, components, values = chunk_critic_loss(online, target, batch, gamma=.99, alpha=1.)
        expected = macro_reward(.99) + .99 ** 10 * 280
        self.assertAlmostEqual(values["td_target"][0].item(), expected, places=4)
        self.assertAlmostEqual(components["conservative"].item(), 140., places=4)
        self.assertAlmostEqual(loss.item(), expected ** 2 + 140, delta=.02)
        loss.backward()
        self.assertIsNone(target.scale.grad)
        self.assertIsNotNone(online.scale.grad)

    def test_episode_boundary_never_becomes_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for episode in range(2):
                data = {"features": np.full((3, 1536), episode, dtype=np.float32),
                        "state": np.full((3, 14), episode, dtype=np.float32),
                        "expert_chunks": np.zeros((3, 10, 14), dtype=np.float32),
                        "policy_chunks": np.zeros((3, 4, 10, 14), dtype=np.float32),
                        "frame_index": np.array([0, 10, 20], dtype=np.int64),
                        "valid_transition": np.array([True, True, False])}
                validate_chunk_episode(data, horizon=10, candidates=4)
                relative = f"episodes/{episode}.npz"
                atomic_npz(root / relative, data)
                entries.append({"file": relative, "sha256": sha256(root / relative), "split": "train"})
            table = ChunkTransitionTable(root, {"config": {"horizon": 10,
                                                           "num_candidates": 4}, "episodes": entries}, "train")
            self.assertEqual(len(table), 4)
            self.assertEqual(table.indices.tolist(), [0, 1, 3, 4])
            result = table.batch(torch.tensor([1, 2]), "cpu")
            self.assertEqual(result["next_state"][:, 0].tolist(), [0., 1.])
            model = ChunkCoverageCritic(table.normalization(), horizon=10, width=8)
            self.assertEqual(tuple(model(result["features"], result["state"],
                                         result["expert_chunks"]).shape), (2,))


if __name__ == "__main__":
    unittest.main()
