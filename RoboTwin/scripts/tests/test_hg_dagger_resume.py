"""Recovery checks for raw-first HG-DAgger collection."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hg_dagger_resume import inspect_collection, preserve_incomplete_cache


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ResumeTests(unittest.TestCase):
    def make_collection(self, root):
        for episode_index, rollout_index, seed in ((0, 0, 41000), (1, 2, 41002)):
            raw = root / "raw" / f"episode_{episode_index:07d}"
            (raw / "frames").mkdir(parents=True)
            (raw / "frames" / "0.pkl").write_bytes(b"frame")
            save_json(raw / "episode.json", {
                "episode_index": episode_index, "seed": seed,
                "control_mask": ["policy", "hil"],
                "record": {"rollout_index": rollout_index, "seed": seed,
                           "save_decision": True, "hil_frames": 1,
                           "expert_result": {"success": True}},
            })
        (root / "episodes.jsonl").write_text(
            "".join(json.dumps({"episode_index": i, "saved": True}) + "\n" for i in (0, 1)),
            encoding="utf-8")
        save_json(root / "session_20260930_164713.json", {
            "sampling_mode": "enhanced", "target_mode": "hil", "saved_episodes": 2,
            "saved_hil_episodes": 2, "total_seconds": 20.0, "aborted_rollouts": 0,
            "records": [
                {"rollout_index": 0, "seed": 41000, "save_decision": True},
                {"rollout_index": 1, "seed": 41001, "save_decision": False},
                {"rollout_index": 2, "seed": 41002, "save_decision": True},
            ],
        })
        for index, seed, status in ((0, 41000, "completed"), (1, 41001, "completed"),
                                    (2, 41002, "completed"), (3, 41003, "quit")):
            log = root / "sampling" / f"episode_{index:04d}_seed_{seed}.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps({"event": "hil_outcome", "status": status}) + "\n",
                           encoding="utf-8")

    def test_resume_preserves_counts_and_reserves_interrupted_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_collection(root)
            cache = root / ".cache" / "episode2"
            cache.mkdir(parents=True)
            (cache / "0.pkl").write_bytes(b"unfinished")
            state = inspect_collection(root, target_mode="hil", expected={"sampling_mode": "enhanced"},
                                       resume=True)
            self.assertEqual((state.next_rollout_index, state.next_episode_index,
                              state.saved_valid_hil), (4, 2, 2))
            self.assertEqual(state.used_seeds, {41000, 41001, 41002, 41003})
            self.assertEqual(len(state.records), 3)
            self.assertEqual(state.prior_seconds, 20.0)
            moved = preserve_incomplete_cache(root)
            self.assertEqual(len(moved), 1)
            self.assertEqual((moved[0] / "0.pkl").read_bytes(), b"unfinished")
            self.assertFalse(cache.exists())
            self.assertTrue((root / "raw" / "episode_0000000" / "frames" / "0.pkl").exists())

    def test_explicit_resume_required_and_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = inspect_collection(root, target_mode="hil")
            self.assertEqual(empty.next_rollout_index, 0)
            self.make_collection(root)
            with self.assertRaisesRegex(ValueError, "use --resume"):
                inspect_collection(root, target_mode="hil")
            with self.assertRaisesRegex(ValueError, "sampling_mode"):
                inspect_collection(root, target_mode="hil", expected={"sampling_mode": "off"},
                                   resume=True)
            (root / "raw" / "episode_0000001" / "episode.json").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete raw episode"):
                inspect_collection(root, target_mode="hil", resume=True)

    def test_saved_index_must_match_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_collection(root)
            (root / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "saved": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "disagree"):
                inspect_collection(root, target_mode="hil", resume=True)


if __name__ == "__main__":
    unittest.main()
