from dataclasses import FrozenInstanceError
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from hg_dagger_task_adapters import Adapter, get_adapter


class TaskAdapterTests(unittest.TestCase):
    def test_all_registered_collectors_return_matching_adapters(self):
        for task_name in (
            "place_can_basket", "rotate_qrcode", "pick_dual_bottles",
            "put_bottles_dustbin", "stack_blocks_two",
        ):
            with self.subTest(task_name=task_name):
                adapter = get_adapter(task_name)
                self.assertEqual(adapter.task_name, task_name)
                self.assertTrue(callable(adapter.snapshot))
                self.assertTrue(callable(adapter.classify))
                self.assertTrue(callable(adapter.recovery))
                self.assertTrue(callable(adapter.quality))

    def test_compatibility_exports_use_the_frozen_legacy_contract(self):
        from dagger.adapters.legacy import Adapter as LegacyAdapter

        self.assertIs(Adapter, LegacyAdapter)
        for task_name in (
            "place_can_basket", "rotate_qrcode", "pick_dual_bottles",
            "put_bottles_dustbin", "stack_blocks_two",
        ):
            with self.subTest(task_name=task_name):
                module = importlib.import_module(f"dagger.adapters.{task_name}")
                direct = module.get_adapter()
                compat = get_adapter(task_name)
                self.assertIsInstance(compat, LegacyAdapter)
                self.assertEqual(direct.task_name, compat.task_name)
                self.assertEqual(direct.stage_labels, compat.stage_labels)
                self.assertIs(direct.recovery, compat.recovery)
                self.assertEqual(compat.quality.__module__, module.__name__)
                with self.assertRaises(FrozenInstanceError):
                    compat.task_name = "other_task"

    def test_unknown_task_keeps_legacy_error(self):
        with self.assertRaisesRegex(
            ValueError, "^no HG-DAgger adapter registered for 'unknown_task'$",
        ):
            get_adapter("unknown_task")

    def test_source_and_assembled_imports_find_ancestor_and_defer_recovery(self):
        repo_root = next(
            parent for parent in SCRIPT_DIR.parents if (parent / "dagger").is_dir()
        )
        recovery_modules = (
            "place_can_basket_recovery", "rotate_qrcode_recovery",
            "hg_dagger_extra_task_recovery",
        )
        probe = textwrap.dedent("""
            import sys
            sys.path.insert(0, sys.argv[1])
            import hg_dagger_task_adapters as compat
            from dagger.adapters.legacy import Adapter
            assert sys.argv[2] in sys.path
            assert compat.Adapter is Adapter
            for name in sys.argv[3:]:
                assert name not in sys.modules, name
            for task in (
                "place_can_basket", "rotate_qrcode", "pick_dual_bottles",
                "put_bottles_dustbin", "stack_blocks_two",
            ):
                adapter = compat.get_adapter(task)
                assert isinstance(adapter, Adapter)
                assert adapter.task_name == task
        """)
        with tempfile.TemporaryDirectory() as temporary:
            isolated_root = Path(temporary).resolve() / "repo"
            isolated_root.mkdir()
            (isolated_root / "dagger").symlink_to(repo_root / "dagger", target_is_directory=True)
            for layout in (
                "robotwin_hil/overlay/scripts",
                "outputs/robotwin_hil_runtime/RoboTwin/scripts",
            ):
                with self.subTest(layout=layout):
                    scripts = isolated_root / layout
                    scripts.mkdir(parents=True)
                    for name in ("hg_dagger_task_adapters", *recovery_modules):
                        shutil.copy2(SCRIPT_DIR / f"{name}.py", scripts / f"{name}.py")
                    result = subprocess.run(
                        [sys.executable, "-I", "-c", probe, str(scripts),
                         str(isolated_root), *recovery_modules],
                        cwd=temporary, capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_place_can_table_regrasp_requires_settled_two_snapshot_state(self):
        adapter = get_adapter("place_can_basket")
        current = {
            "can_p": [-0.2, 0.0, 0.8], "basket_p": [0.0, 0.0, 0.8],
            "can_table_contact": True, "can_basket_contact": False,
            "can_lift_m": 0.0, "source_gripper_open": True,
            "receiver_closed": True, "receiver_basket_contact": True,
            "receiver_ee_to_basket_m": 0.03, "task_success": False,
            "basket_axis_z": 1.0, "can_basket_l1_m": 0.2,
            "basket_lift_m": 0.0, "source_arm": "left", "receiver_arm": "right",
        }
        self.assertEqual(adapter.classify(current, None)["reason"],
                         "no_previous_pose_for_table_can")
        previous = dict(current, can_p=[-0.205, 0.0, 0.8])
        decision = adapter.classify(current, previous)
        self.assertTrue(decision["ready"])
        self.assertEqual(decision["branch"], "regrasp_place_can")
        moving = dict(current, can_p=[-0.16, 0.0, 0.8])
        self.assertEqual(adapter.classify(moving, previous)["branch"], "wait_for_settle")
