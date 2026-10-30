"""One-stage pairing and eligibility checks without simulator dependencies."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from one_stage_dagger import (
    baseline_seeds, handoff_max_abs_difference, handoff_pair_differences,
    handoff_pair_ready, handoff_record, is_valid_hil,
)


class OneStageTests(unittest.TestCase):
    def test_handoff_requires_completed_stage_five_and_live_receiver_grasp(self):
        state = {
            "receiver_holds": True, "source_holds": False, "placed": False,
            "recoverable": True, "stage_id": 6,
        }
        env = SimpleNamespace(plan_success=True, infer_recovery_state=lambda: state)
        self.assertTrue(handoff_record(env)["ready"])
        for changed in ({"source_holds": True}, {"receiver_holds": False},
                        {"placed": True}, {"stage_id": 5}, {"recoverable": False}):
            env.infer_recovery_state = lambda changed=changed: {**state, **changed}
            self.assertFalse(handoff_record(env)["ready"])
        env.plan_success = False
        env.infer_recovery_state = lambda: state
        self.assertFalse(handoff_record(env)["ready"])

    def test_valid_hil_requires_recovery_and_legal_recorded_intervention(self):
        kwargs = dict(interventions=1, hil_frames=2, expert_result={"success": True},
                      final_success=True, joints_legal=True)
        self.assertTrue(is_valid_hil(**kwargs))
        for changed in (dict(interventions=0), dict(hil_frames=0),
                        dict(expert_result={"success": False}),
                        dict(final_success=False), dict(joints_legal=False)):
            self.assertFalse(is_valid_hil(**{**kwargs, **changed}))

    def test_handoff_signatures_reject_unmatched_initial_states(self):
        signature = {name: [0.0, 1.0] for name in
                     ("left_joint", "right_joint", "bar_p", "bar_q", "tray_p", "tray_q")}
        original = {"signature": signature}
        matched = {"signature": {name: values.copy() for name, values in signature.items()}}
        self.assertEqual(handoff_max_abs_difference(original, matched), 0.0)
        matched["signature"]["bar_p"][0] = 0.001
        self.assertAlmostEqual(handoff_max_abs_difference(original, matched), 0.001)
        self.assertTrue(handoff_pair_ready(handoff_pair_differences(original, matched)))
        matched["signature"]["right_joint"][0] = 0.03
        self.assertFalse(handoff_pair_ready(handoff_pair_differences(original, matched)))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            handoff_max_abs_difference(original, {})

    def test_paired_arm_replays_every_baseline_seed(self):
        report = {
            "one_stage": True, "sampling_mode": "off", "aborted": False,
            "checkpoint_name": "v2_promptfix_9999", "task_config": "handover_to_tray_v2_promptfix",
            "frequency": 30, "rollouts": 3,
            "records": [
                {"seed": 100000, "rollout_status": "completed"},
                {"seed": 100001, "rollout_status": "prefix_failed"},
                {"seed": 100002, "rollout_status": "completed"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            args = dict(checkpoint="v2_promptfix_9999",
                        task_config="handover_to_tray_v2_promptfix", frequency=30)
            self.assertEqual(baseline_seeds(path, **args), [100000, 100002])
            report["sampling_mode"] = "enhanced"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baseline report"):
                baseline_seeds(path, **args)


if __name__ == "__main__":
    unittest.main()
