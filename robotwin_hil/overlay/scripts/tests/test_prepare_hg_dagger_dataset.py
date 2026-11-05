import importlib.util
import json
import pickle
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).parents[1] / "prepare_hg_dagger_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_hg_dagger_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareHGDaggerDatasetTests(unittest.TestCase):
    def test_default_minimum_means_transitions_and_old_flag_remains_alias(self):
        required = ["prepare", "--raw-root", "/unused/raw", "--output-dir", "/unused/output"]
        with patch.object(sys, "argv", required):
            self.assertEqual(MODULE.parse_args().min_segment_transitions,
                             MODULE.MIN_HIL_SEGMENT_TRANSITIONS)
        with patch.object(sys, "argv", [*required, "--min-segment-frames", "3"]):
            self.assertEqual(MODULE.parse_args().min_segment_transitions, 3)

    def test_two_hil_segments_create_two_episodes_without_cross_boundary_action(self):
        import tempfile

        class Dataset:
            def __init__(self):
                self.rows = []
                self.saved = 0

            def add_frame(self, row, task=None):
                self.rows.append((row, task))

            def save_episode(self, task=None):
                self.saved += 1

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            episode = raw / "episode_0000007"
            frames = episode / "frames"
            frames.mkdir(parents=True)
            mask = ["policy", "hil", "hil", "hil", "policy", "hil", "hil", "hil"]
            for index in range(len(mask)):
                (frames / f"{index}.pkl").write_bytes(pickle.dumps({
                    "joint_action": {"vector": np.full(14, index, dtype=np.float32)},
                }))
            (episode / "episode.json").write_text(json.dumps({
                "episode_index": 7, "instruction": "stack", "control_mask": mask,
                "record": {"accepted_for_training": True},
            }))
            dataset = Dataset()
            with patch.object(MODULE, "CAMERA_MAP", {}):
                counts = MODULE.extract_source(MODULE.discover_episodes([raw], None),
                                               {"dagger": dataset}, image_size=(1, 1),
                                               min_segment_transitions=MODULE.MIN_HIL_SEGMENT_TRANSITIONS)
            self.assertEqual(dataset.saved, 2)
            self.assertEqual([int(row["action"][0]) for row, _ in dataset.rows], [2, 3, 6, 7])
            self.assertEqual(counts["dagger"], 4)

    def test_discovery_uses_same_strict_training_gate_as_hdf5_export(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            for index, record in enumerate((
                {"accepted_for_training": True},
                {"accepted_for_training": False},
                {"accepted_for_training": True, "diagnostic_only": True},
            )):
                episode = raw / f"episode_{index:07d}"
                frames = episode / "frames"
                frames.mkdir(parents=True)
                for frame in range(2):
                    (frames / f"{frame}.pkl").write_bytes(b"frame")
                (episode / "episode.json").write_text(json.dumps({
                    "episode_index": index, "instruction": "stack", "control_mask": ["hil", "hil"],
                    "record": record,
                }))
            episodes = MODULE.discover_episodes([raw], None)
            self.assertEqual([episode.episode_index for episode in episodes], [0])

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
