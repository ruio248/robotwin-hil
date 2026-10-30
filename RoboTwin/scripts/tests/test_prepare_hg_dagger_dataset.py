import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "prepare_hg_dagger_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_hg_dagger_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareHGDaggerDatasetTests(unittest.TestCase):
    def test_segments_are_split_at_controller_boundaries(self):
        self.assertEqual(
            list(MODULE._segment_runs(("policy", "policy", "hil", "hil", "policy"))),
            [(0, 2, "policy"), (2, 4, "hil"), (4, 5, "policy")],
        )

    def test_equal_stats_budget_is_source_balanced(self):
        result = MODULE._equal_sample_indices([3, 4, 5], 6)
        self.assertEqual(sum(len(item) for item in result), 6)
        self.assertTrue(all(np.all(item >= 0) for item in result))
        self.assertTrue(all(np.all(item < length) for item, length in zip(result, [3, 4, 5], strict=True)))

    def test_joint_action_is_exactly_14d(self):
        vector = np.arange(14, dtype=np.float32)
        self.assertTrue(np.array_equal(MODULE._vector({"joint_action": {"vector": vector}}, "test"), vector))
        with self.assertRaises(ValueError):
            MODULE._vector({"joint_action": {"vector": np.zeros(13)}}, "test")


if __name__ == "__main__":
    unittest.main()
