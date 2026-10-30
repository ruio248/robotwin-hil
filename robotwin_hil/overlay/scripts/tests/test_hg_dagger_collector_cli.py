from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from hg_dagger_handover_block import parse_args as parse_handover_args  # noqa: E402
from hg_dagger_move_can_pot import parse_args as parse_move_args  # noqa: E402


class CollectorCliTests(unittest.TestCase):
    def parse_smoke(self, parser, *extra: str):
        argv = [
            "collector.py",
            "--output-dir", "/tmp/collector-cli-test",
            "--port", "18465",
            "--headless-smoke",
            "--episodes", "1",
            "--target-saved", "1",
            *extra,
        ]
        with patch.object(sys, "argv", argv):
            return parser()

    def test_x_cancellation_can_be_smoked_without_i_or_r(self) -> None:
        for parser in (parse_move_args, parse_handover_args):
            with self.subTest(parser=parser.__module__):
                args = self.parse_smoke(parser, "--auto-abort-step", "5")
                self.assertEqual(args.auto_abort_step, 5)
                self.assertEqual(args.auto_intervene_step, -1)
                self.assertEqual(args.auto_handback_stage_actions, -1)

    def test_automatic_i_can_continue_under_expert_control(self) -> None:
        for parser in (parse_move_args, parse_handover_args):
            with self.subTest(parser=parser.__module__):
                args = self.parse_smoke(parser, "--auto-intervene-step", "50")
                self.assertEqual(args.auto_intervene_step, 50)
                self.assertEqual(args.auto_handback_stage_actions, -1)

    def test_automatic_r_requires_automatic_i(self) -> None:
        for parser in (parse_move_args, parse_handover_args):
            with self.subTest(parser=parser.__module__):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                    self.parse_smoke(parser, "--auto-handback-stage-actions", "2")
                self.assertIn("automatic r requires", stderr.getvalue())

    def test_automatic_i_and_r_are_supported_together(self) -> None:
        for parser in (parse_move_args, parse_handover_args):
            with self.subTest(parser=parser.__module__):
                args = self.parse_smoke(
                    parser,
                    "--auto-intervene-step", "50",
                    "--auto-handback-stage-actions", "2",
                )
                self.assertEqual(args.auto_handback_stage_actions, 2)


if __name__ == "__main__":
    unittest.main()
