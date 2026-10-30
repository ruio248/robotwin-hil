from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "coverage_benchmark" / "annotate_pair_stages.py"
SPEC = importlib.util.spec_from_file_location("annotate_pair_stages", MODULE_PATH)
ANNOTATOR = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(ANNOTATOR)


class PairStageDtwTests(unittest.TestCase):
    def test_dtw_recovers_monotone_stretched_trajectory(self):
        t = np.linspace(0.0, 1.0, 80, dtype=np.float32)
        saved = np.stack([t, t * t, np.sin(t * 3.0)], axis=1)
        replay_t = np.linspace(0.0, 1.0, 87, dtype=np.float32)
        replay = np.stack([replay_t, replay_t * replay_t, np.sin(replay_t * 3.0)], axis=1)
        mapped, matches, cost, _band = ANNOTATOR.monotone_qpos_dtw(saved, replay)

        self.assertTrue(np.isfinite(cost))
        self.assertEqual(mapped.shape, (len(saved),))
        self.assertTrue(np.all(np.diff(mapped) >= 0))
        self.assertTrue(all(matches))
        self.assertLess(float(np.max(np.abs(saved - replay[mapped]))), 0.03)

    def test_reconstructs_overlapping_horizon_chunks(self):
        states = np.arange(16 * 14, dtype=np.float32).reshape(16, 14)
        anchors = np.asarray([0, 5], dtype=np.int64)
        chunks = np.stack([states[1:11], states[6:16]])
        with tempfile.TemporaryDirectory() as temp:
            pair_path = Path(temp) / "pairs.npz"
            np.savez_compressed(
                pair_path,
                state=states[anchors],
                sampled_state_index=anchors,
                expert_chunks=chunks,
            )
            reconstructed = ANNOTATOR.reconstruct_saved_qpos(pair_path)
        np.testing.assert_array_equal(reconstructed, states)


if __name__ == "__main__":
    unittest.main()
