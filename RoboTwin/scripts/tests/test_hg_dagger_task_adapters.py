from pathlib import Path
import sys
import unittest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from hg_dagger_task_adapters import get_adapter


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
