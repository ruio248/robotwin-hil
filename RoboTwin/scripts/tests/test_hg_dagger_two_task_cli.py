import contextlib
import io
import sys
import unittest
from unittest.mock import patch

SCRIPT_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from hg_dagger_task_collector import _auto_trigger_due, parse_args


class TwoTaskCollectorCliTests(unittest.TestCase):
    def parse_smoke(self, task, *extra):
        argv = ["collector.py", "--output-dir", "/tmp/hg-task-test", "--port", "18465",
                "--headless-smoke", "--episodes", "1", "--target-saved", "1", *extra]
        with patch.object(sys, "argv", argv):
            return parse_args(task)

    def test_abort_only_smoke_allowed_for_both_tasks(self):
        for task in ("place_can_basket", "rotate_qrcode"):
            with self.subTest(task=task):
                args = self.parse_smoke(task, "--auto-abort-step", "5")
                self.assertEqual(args.auto_abort_step, 5)

    def test_handback_requires_takeover_trigger(self):
        with patch.object(sys, "argv", ["collector.py", "--output-dir", "/tmp/hg-task-test",
                                         "--port", "18465", "--headless-smoke",
                                         "--episodes", "1", "--target-saved", "1",
                                         "--auto-handback-stage-actions", "2"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args("rotate_qrcode")

    def test_production_rejects_automatic_operator_keys(self):
        with patch.object(sys, "argv", ["collector.py", "--output-dir", "/tmp/hg-task-test",
                                         "--port", "18465", "--auto-intervene-step", "50"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args("place_can_basket")

    def test_automatic_smoke_trigger_fires_once(self):
        self.assertTrue(_auto_trigger_due(50, 50, False))
        self.assertFalse(_auto_trigger_due(51, 50, True))
        self.assertFalse(_auto_trigger_due(100, -1, False))


if __name__ == "__main__":
    unittest.main()
