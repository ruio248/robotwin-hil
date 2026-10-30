import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from hg_dagger_task_collector import parse_args


class ExtraTaskCollectorCliTests(unittest.TestCase):
    TASKS = {
        "pick_dual_bottles": 400,
        "put_bottles_dustbin": 1700,
        "stack_blocks_two": 800,
    }

    def test_each_task_has_isolated_smoke_cli_and_default_limit(self):
        for task, expected_limit in self.TASKS.items():
            argv = ["collector.py", "--output-dir", "/tmp/hg-extra-test", "--port", "18465",
                    "--headless-smoke", "--episodes", "1", "--target-saved", "1"]
            with self.subTest(task=task), patch.object(sys, "argv", argv):
                args = parse_args(task)
                self.assertEqual(args.step_limit, expected_limit)
                self.assertEqual(args.seed_start, 40000)

    def test_exact_instruction_override_is_available_for_scene_reproduction(self):
        prompt = "Move red block and green block to the center and stack green on red."
        argv = ["collector.py", "--output-dir", "/tmp/hg-extra-test", "--port", "18465",
                "--headless-smoke", "--episodes", "1", "--target-saved", "1",
                "--instruction", f"  {prompt}  "]
        with patch.object(sys, "argv", argv):
            args = parse_args("stack_blocks_two")
        self.assertEqual(args.instruction, prompt)

    def test_production_seed_and_smoke_only_hooks_are_enforced(self):
        for task in self.TASKS:
            argv = ["collector.py", "--output-dir", "/tmp/hg-extra-test", "--port", "18465",
                    "--auto-intervene-step", "20"]
            with self.subTest(task=task), patch.object(sys, "argv", argv), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(task)


if __name__ == "__main__":
    unittest.main()
