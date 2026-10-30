"""Guard the upstream/source separation used by the compatibility builder."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from robotwin_hil import prepare_runtime


class PrepareRuntimeTest(unittest.TestCase):
    def test_overlay_wins_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "RoboTwin"
            overlay = base / "overlay"
            policy = base / "compat" / "XPolicyLab"
            target = base / "outputs" / "RoboTwin"
            (source / "envs").mkdir(parents=True)
            (overlay / "envs").mkdir(parents=True)
            (policy / "policy" / "Pi_05_RobotTwin").mkdir(parents=True)
            (source / "envs" / "_base_task.py").write_text("official\n")
            (overlay / "envs" / "_base_task.py").write_text("extended\n")
            (overlay / "envs" / "handover_to_tray.py").write_text("custom task\n")
            (policy / "policy" / "Pi_05_RobotTwin" / "adapter.py").write_text("adapter\n")
            (source / "XPolicyLab").mkdir()
            (source / "XPolicyLab" / "official.py").write_text("upstream\n")

            with mock.patch.object(prepare_runtime, "SOURCE", source), \
                 mock.patch.object(prepare_runtime, "OVERLAY", overlay), \
                 mock.patch.object(prepare_runtime, "POLICY_COMPAT", policy), \
                 mock.patch.object(prepare_runtime, "_git", side_effect=lambda *args: prepare_runtime.UPSTREAM_COMMIT if args[0] == "rev-parse" else ""):
                prepare_runtime.build(target)
                self.assertEqual((target / "envs" / "_base_task.py").read_text(), "extended\n")
                self.assertEqual((source / "envs" / "_base_task.py").read_text(), "official\n")
                self.assertFalse((target / "XPolicyLab" / "official.py").exists())
                self.assertTrue((target / "XPolicyLab" / "policy" / "Pi_05_RobotTwin" / "adapter.py").is_file())
                metadata = json.loads((target / prepare_runtime.METADATA_NAME).read_text())
                self.assertEqual(metadata["overlay_files"], 2)
                prepare_runtime.build(target)  # idempotent
                (overlay / "envs" / "_base_task.py").write_text("updated\n")
                with self.assertRaisesRegex(RuntimeError, "--rebuild"):
                    prepare_runtime.build(target)
                prepare_runtime.build(target, rebuild=True)
                self.assertEqual((target / "envs" / "_base_task.py").read_text(), "updated\n")
                self.assertEqual(len(list(target.parent.glob("RoboTwin.backup-*"))), 1)


if __name__ == "__main__":
    unittest.main()
