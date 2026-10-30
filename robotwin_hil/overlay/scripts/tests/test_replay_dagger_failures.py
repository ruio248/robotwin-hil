import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_dagger_failures import EARLY_STEPS, aggregate, save, state_differences


class FailureReplayTests(unittest.TestCase):
    def test_fixed_intervention_steps_are_not_seed_selected(self):
        self.assertEqual(EARLY_STEPS, {"handover_block": 150, "move_can_pot": 50, "stack_blocks_two": 100})

    def test_exact_snapshot_matches(self):
        a = {"p": [0.1, 0.2, 0.3], "held": True, "branch": None}
        self.assertEqual(state_differences(a, a), [])

    def test_small_float_error_tolerated_but_not_contact_change(self):
        self.assertEqual(state_differences([1.0], [1.00001]), [])
        self.assertTrue(state_differences({"held": True}, {"held": False}))

    def test_missing_and_nonfinite_states_rejected(self):
        self.assertTrue(state_differences({"p": [0, 0, 1]}, {"p": [0, 0]}))
        self.assertTrue(state_differences([1.0], [float("nan")]))
        self.assertTrue(state_differences([0.0], [0.01]))

    def test_aggregate_does_not_count_control_success_as_rescue(self):
        with tempfile.TemporaryDirectory() as folder:
            for seed, failed in [(1, True), (2, False)]:
                root = Path(folder) / "move_can_pot" / f"seed_{seed}"
                save(root / "control.json", {"failure_reproduced": failed,
                     "terminal_recovery": {"rescued": failed}})
                save(root / "early.json", {"early_recovery": {"rescued": True, "status": "rescued"}})
            result = aggregate("move_can_pot", [{"seed": 1}, {"seed": 2}], folder)
            self.assertTrue(result["complete"])
            self.assertEqual(result["failure_reproduced"], 1)
            self.assertEqual(result["control_succeeded"], 1)
            self.assertEqual(result["terminal_rescued"], 1)
            self.assertEqual(result["early_rescued_on_reproduced_failures"], 1)
            self.assertFalse(result["accepted_for_training"])

    def test_partial_case_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            save(Path(folder) / "handover_block/seed_1/control.json", {})
            result = aggregate("handover_block", [{"seed": 1}], folder)
            self.assertFalse(result["complete"])
            self.assertEqual(result["completed"], 0)


if __name__ == "__main__":
    unittest.main()
