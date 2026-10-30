"""The optional policy archive must not mix frames across human control."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_rollout_archive import archive_policy_rollout, policy_runs


class PolicyRolloutArchiveTests(unittest.TestCase):
    def test_splits_at_hil_gap(self):
        self.assertEqual(
            policy_runs(["policy", "policy", "hil", "policy", "policy"]),
            [[0, 1], [3, 4]],
        )

    def test_archives_only_policy_segments_with_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            cache.mkdir()
            for index in range(6):
                (cache / f"{index}.pkl").write_bytes(str(index).encode())
            archive = archive_policy_rollout(
                cache, root, rollout_index=7, seed=100007,
                control_mask=["policy", "policy", "hil", "policy", "policy", "hil"],
                save_freq=15, sampling_mode="gaussian", autonomous_success=False,
            )
            self.assertIsNotNone(archive)
            metadata = json.loads((archive / "rollout.json").read_text())
            self.assertEqual(metadata["source"], "policy_rollout_not_expert_or_hil")
            self.assertEqual([s["source_frame_indices"] for s in metadata["segments"]],
                             [[0, 1], [3, 4]])
            self.assertEqual((archive / "segment_001/frames/1.pkl").read_bytes(), b"4")
            with self.assertRaises(FileExistsError):
                archive_policy_rollout(
                    cache, root, rollout_index=7, seed=100007,
                    control_mask=["policy"] * 6, save_freq=15,
                    sampling_mode="gaussian", autonomous_success=False,
                )


if __name__ == "__main__":
    unittest.main()
