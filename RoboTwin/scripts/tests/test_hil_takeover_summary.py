"""Paired human takeover rates use actual i-key requests from started episodes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_hil_takeover import summarize


def session(mode, requests, steps=(3, 3, 3)):
    config = {"mode": mode, "window_start": 2, "window_end": 4,
              "num_candidates": 4, "horizon": 10, "beta": 10, "seed": 42, "replay_atol": 1e-4}
    return {
        "sampling_mode": mode, "sampling_config": None if mode == "off" else config,
        "sampling_window": [2, 4], "critic_sha256": None if mode == "off" else "same-critic",
        "policy_name": "Pi_05_RobotTwin", "policy_host": "127.0.0.1", "policy_port": 18311,
        "checkpoint_name": "same-checkpoint",
        "task_config": "same-task", "instruction": "same prompt", "frequency": 30,
        "step_limit": None, "seed_mode": "sequential",
        "records": [
            {"seed": seed, "policy_steps": step, "manual_takeover_requests": requested,
             "intervention_count": int(requested > 0 and seed != 101),
             "rollout_status": "aborted" if seed == 102 else "completed"}
            for seed, step, requested in zip((100, 101, 102), steps, requests)
        ],
    }


class TakeoverSummaryTests(unittest.TestCase):
    def test_paired_rates_include_cancelled_i_and_valid_aborts(self):
        sessions = {
            "off": session("off", (0, 1, 0)),
            "vanilla": session("vanilla", (1, 1, 0), steps=(3, 3, 0)),
            "enhanced": session("enhanced", (1, 1, 1)),
        }
        result = summarize(sessions)
        self.assertEqual(result["paired_seeds"], [100, 101])
        self.assertEqual(result["excluded_from_paired"], [102])
        self.assertEqual(result["modes"]["off"]["paired_manual_request_probability"], .5)
        self.assertEqual(result["modes"]["vanilla"]["paired_manual_request_probability"], 1)
        self.assertEqual(result["modes"]["enhanced"]["paired_manual_request_probability"], 1)
        self.assertEqual(result["paired_probability_differences"]["enhanced_minus_off"], .5)
        self.assertEqual(result["modes"]["off"]["recovery_trigger_episodes"], 0)
        self.assertEqual(result["modes"]["enhanced"]["aborted_valid_rollouts"], 1)

    def test_rejects_mismatched_seeds_and_critics(self):
        sessions = {mode: session(mode, (0, 0, 0)) for mode in ("off", "vanilla", "enhanced")}
        changed = deepcopy(sessions)
        changed["enhanced"]["records"][1]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "seed list"):
            summarize(changed)
        changed = deepcopy(sessions)
        changed["enhanced"]["critic_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "same critic"):
            summarize(changed)


if __name__ == "__main__":
    unittest.main()
