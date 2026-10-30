from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from hg_dagger_stack_bowls_three import parse_args  # noqa: E402


class StackBowlsCollectorCliTests(unittest.TestCase):
    def parse_smoke(self, *extra: str):
        argv = [
            "hg_dagger_stack_bowls_three.py",
            "--output-dir", "/tmp/stack-bowls-cli-test",
            "--port", "18465",
            "--headless-smoke",
            "--episodes", "1",
            "--target-saved", "1",
            *extra,
        ]
        with patch.object(sys, "argv", argv):
            return parse_args()

    def test_automatic_takeover_can_run_expert_to_completion(self) -> None:
        args = self.parse_smoke("--auto-intervene-step", "300")
        self.assertEqual(args.auto_intervene_step, 300)
        self.assertEqual(args.auto_handback_stage_actions, -1)

    def test_automatic_handback_requires_automatic_takeover(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.parse_smoke("--auto-handback-stage-actions", "2")
        self.assertIn("automatic r requires", stderr.getvalue())

    def test_automatic_takeover_and_handback_are_supported(self) -> None:
        args = self.parse_smoke(
            "--auto-intervene-step", "300",
            "--auto-handback-stage-actions", "2",
        )
        self.assertEqual(args.auto_intervene_step, 300)
        self.assertEqual(args.auto_handback_stage_actions, 2)

    def test_automatic_abort_can_be_used_alone_in_smoke(self) -> None:
        args = self.parse_smoke("--auto-abort-step", "10")
        self.assertEqual(args.auto_abort_step, 10)


if __name__ == "__main__":
    unittest.main()
