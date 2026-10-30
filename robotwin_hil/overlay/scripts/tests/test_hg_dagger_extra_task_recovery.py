from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from hg_dagger_extra_task_recovery import (
    _actor_gripper_contact, _run_stage,
    classify_dual_bottles_state, classify_dustbin_state,
    classify_stack_blocks_two_state,
)
from hg_dagger_task_adapters import get_adapter


def gripper(*, held=False, open_state=True):
    return {
        "left": {"closed": bool(held), "open": bool(open_state and not held),
                 "contact": bool(held), "distance_m": 0.02 if held else 0.4},
        "right": {"closed": False, "open": True, "contact": False, "distance_m": 0.4},
    }


class ExtraTaskRecoveryTests(unittest.TestCase):
    def test_gripper_contact_uses_actor_identity_not_shared_model_name(self):
        class Entity:
            def __init__(self, name, entity_id):
                self.name = name
                self.entity_id = entity_id

            def get_id(self):
                return self.entity_id

            def get_name(self):
                return self.name

        class Body:
            def __init__(self, entity):
                self.entity = entity

        class ActorWrapper:
            def __init__(self, entity):
                self.actor = entity

            def get_name(self):
                return self.actor.get_name()

        bottle1 = Entity("001_bottle", 1)
        bottle2 = Entity("001_bottle", 2)
        finger = Entity("left_finger", 3)
        contact = SimpleNamespace(bodies=(Body(bottle2), Body(finger)))
        task_env = SimpleNamespace(
            robot=SimpleNamespace(left_fix_gripper_name=["left_finger"],
                                  right_fix_gripper_name=[], left_gripper=[], right_gripper=[]),
            scene=SimpleNamespace(get_contacts=lambda: [contact]),
        )
        self.assertFalse(_actor_gripper_contact(task_env, ActorWrapper(bottle1), "left"))
        self.assertTrue(_actor_gripper_contact(task_env, ActorWrapper(bottle2), "left"))

    def test_combined_arm_motion_is_passed_as_two_native_move_arguments(self):
        class FakeEnv:
            plan_success = False
            current_stage_id = 0
            current_control_source = "policy"

            def __init__(self):
                self.moves = []

            def move(self, *motions):
                self.moves.append(motions)

        env = FakeEnv()
        event = _run_stage(
            env, {"executed_stage_ids": []}, 1, "paired_clear",
            lambda: (("right", ["grasp"]), ("left", ["back_to_origin"])),
        )
        self.assertTrue(event["plan_success"])
        self.assertEqual(env.moves, [(("right", ["grasp"]), ("left", ["back_to_origin"]))])

    def test_single_arm_recovery_stage_builder_returns_motion_without_executing_it(self):
        class FakeEnv:
            plan_success = False
            current_stage_id = 0
            current_control_source = "policy"

            def __init__(self):
                self.moves = []

            def move(self, motion):
                self.moves.append(motion)
                self.plan_success = True
                return True

        env = FakeEnv()
        motion = ("left", ["move_to_drop_pose"])
        event = _run_stage(env, {"executed_stage_ids": []}, 5, "move_over_bin",
                           lambda: motion)
        self.assertTrue(event["plan_success"])
        self.assertEqual(env.moves, [motion])

    def test_adapters_registered_and_prompts_use_correct_task_ids(self):
        from hg_dagger_task_collector import TASKS
        for name in ("pick_dual_bottles", "put_bottles_dustbin", "stack_blocks_two"):
            with self.subTest(task=name):
                self.assertEqual(get_adapter(name).task_name, name)
                self.assertGreater(TASKS[name]["step_limit"], 0)

    def test_dual_bottles_repairs_only_unfinished_supported_object(self):
        bottle1 = {"pose": [-0.06, -0.105, 0.95],
                   "table_contact": False, "grippers": gripper(held=True)}
        open_grippers = {
            "left": {"closed": True, "open": False, "contact": False, "distance_m": 0.4},
            "right": {"closed": False, "open": True, "contact": False, "distance_m": 0.4},
        }
        bottle2 = {"pose": [0.15, 0.1, 0.76],
                   "table_contact": True, "grippers": open_grippers}
        current = {
            "bottles": {"bottle1": bottle1, "bottle2": bottle2},
            "target_and_hold_ok": {"bottle1": True, "bottle2": False},
            "strict_physical_quality": False,
        }
        previous = {"bottles": {"bottle1": bottle1, "bottle2": bottle2}}
        decision = classify_dual_bottles_state(current, previous)
        self.assertTrue(decision["ready"])
        self.assertEqual(decision["pending"], [{"name": "bottle2", "arm": "right", "held": False}])
        self.assertEqual(classify_dual_bottles_state(current, None)["branch"], "wait_for_settle")

    def test_dual_bottles_rejects_wrong_arm_and_unstable_object(self):
        bottles = {
            "bottle1": {"pose": [-0.06, -0.105, 0.95], "table_contact": False,
                        "grippers": gripper(held=True)},
            "bottle2": {"pose": [0.15, 0.1, 0.76], "table_contact": True,
                        "grippers": gripper()},
        }
        current = {"bottles": bottles, "target_and_hold_ok": {"bottle1": False, "bottle2": False},
                   "strict_physical_quality": False}
        prev_same = {"bottles": {name: dict(value) for name, value in bottles.items()}}
        bottles["bottle1"]["grippers"] = {
            "left": {"closed": False, "open": True, "contact": False, "distance_m": 0.4},
            "right": {"closed": True, "open": False, "contact": True, "distance_m": 0.02},
        }
        self.assertEqual(classify_dual_bottles_state(current, prev_same)["reason"],
                         "bottle1_held_by_wrong_arm")

    def test_dual_bottles_can_safely_open_free_closed_grippers_before_regrasp(self):
        closed_free = {
            "left": {"closed": True, "open": False, "contact": False, "distance_m": 0.30},
            "right": {"closed": True, "open": False, "contact": False, "distance_m": 0.32},
        }
        bottles = {
            "bottle1": {"pose": [-0.2, 0.1, 0.76], "table_contact": True,
                        "grippers": closed_free},
            "bottle2": {"pose": [0.2, 0.1, 0.76], "table_contact": True,
                        "grippers": closed_free},
        }
        current = {"bottles": bottles,
                   "target_and_hold_ok": {"bottle1": False, "bottle2": False},
                   "strict_physical_quality": False}
        previous = {"bottles": {name: dict(state) for name, state in bottles.items()}}
        decision = classify_dual_bottles_state(current, previous)
        self.assertTrue(decision["ready"])
        self.assertEqual(decision["open_grippers"], ["left", "right"])
        self.assertEqual(decision["reposition_arms"], ["left", "right"])
        self.assertEqual(len(decision["pending"]), 2)

    def test_dual_bottles_rejects_closed_gripper_near_bottle_without_contact(self):
        grippers = {
            "left": {"closed": True, "open": False, "contact": False, "distance_m": 0.10},
            "right": {"closed": False, "open": True, "contact": False, "distance_m": 0.40},
        }
        bottles = {
            "bottle1": {"pose": [-0.2, 0.1, 0.76], "table_contact": True,
                        "grippers": grippers},
            "bottle2": {"pose": [0.2, 0.1, 0.76], "table_contact": True,
                        "grippers": {"left": dict(grippers["left"], distance_m=0.40),
                                     "right": grippers["right"]}},
        }
        current = {"bottles": bottles,
                   "target_and_hold_ok": {"bottle1": False, "bottle2": False},
                   "strict_physical_quality": False}
        previous = {"bottles": {name: dict(state) for name, state in bottles.items()}}
        decision = classify_dual_bottles_state(current, previous)
        self.assertFalse(decision["ready"])
        self.assertEqual(decision["reason"], "bottle1_left_near_object_without_contact")

    def test_dustbin_preserves_completed_bottles_and_rejects_motion(self):
        bottles = [
            {"pose": [-0.45, 0.0, 0.4], "in_bin": True, "table_contact": False,
             "dustbin_contact": True, "grippers": gripper()},
            {"pose": [0.12, 0.1, 0.76], "in_bin": False, "table_contact": True,
             "dustbin_contact": False, "grippers": gripper()},
            {"pose": [-0.16, 0.08, 0.76], "in_bin": False, "table_contact": True,
             "dustbin_contact": False, "grippers": gripper()},
        ]
        current = {"bottles": bottles, "strict_physical_quality": False}
        previous = {"bottles": [dict(item) for item in bottles]}
        decision = classify_dustbin_state(current, previous)
        self.assertEqual([item["index"] for item in decision["pending"]], [2, 1])
        self.assertEqual([item["arm"] for item in decision["pending"]], ["left", "right"])
        bottles[1]["pose"] = [0.20, 0.1, 0.76]
        self.assertEqual(classify_dustbin_state(current, previous)["branch"], "wait_for_settle")

    def test_dustbin_can_safely_open_a_free_closed_gripper_before_regrasp(self):
        closed_far = {
            "left": {"closed": True, "open": False, "contact": False, "distance_m": 0.40},
            "right": {"closed": False, "open": True, "contact": False, "distance_m": 0.40},
        }
        bottles = [
            {"pose": [-0.45, 0.0, 0.4], "in_bin": True, "table_contact": False,
             "dustbin_contact": True, "grippers": closed_far},
            {"pose": [0.12, 0.1, 0.76], "in_bin": False, "table_contact": True,
             "dustbin_contact": False, "grippers": closed_far},
            {"pose": [-0.16, 0.08, 0.76], "in_bin": False, "table_contact": True,
             "dustbin_contact": False, "grippers": closed_far},
        ]
        current = {"bottles": bottles, "strict_physical_quality": False}
        previous = {"bottles": [dict(item) for item in bottles]}
        decision = classify_dustbin_state(current, previous)
        self.assertTrue(decision["ready"])
        self.assertEqual(decision["open_grippers"], ["left"])

    def test_dustbin_rejects_a_closed_free_gripper_near_any_bottle(self):
        far = {
            "left": {"closed": True, "open": False, "contact": False, "distance_m": 0.40},
            "right": {"closed": False, "open": True, "contact": False, "distance_m": 0.40},
        }
        near = {**far, "left": {"closed": True, "open": False,
                                 "contact": False, "distance_m": 0.08}}
        bottles = [
            {"pose": [0.12, 0.1, 0.76], "in_bin": False, "table_contact": True,
             "dustbin_contact": False, "grippers": far},
            {"pose": [-0.16, 0.08, 0.76], "in_bin": False, "table_contact": True,
             "dustbin_contact": False, "grippers": near},
            {"pose": [0.25, 0.2, 0.76], "in_bin": False, "table_contact": True,
             "dustbin_contact": False, "grippers": far},
        ]
        current = {"bottles": bottles, "strict_physical_quality": False}
        previous = {"bottles": [dict(item) for item in bottles]}
        decision = classify_dustbin_state(current, previous)
        self.assertFalse(decision["ready"])
        self.assertEqual(decision["reason"], "bottle_1_left_near_object_without_contact")

    def test_stack_blocks_repairs_top_only_when_base_is_verified(self):
        opened = gripper()
        positions = [[0.0, -0.13, 0.775], [0.18, 0.02, 0.766]]
        current = {
            "block_p": positions,
            "block_grippers": [opened, opened],
            "base_target": [0.0, -0.13, 0.75],
            "table_contact": [True, True],
            "block2_on_block1": False,
            "base_at_target": True,
            "strict_physical_quality": False,
        }
        previous = {"block_p": [list(p) for p in positions]}
        decision = classify_stack_blocks_two_state(current, previous)
        self.assertTrue(decision["ready"])
        self.assertEqual(decision["pending"], [{"name": "block2", "index": 1,
                                                "held": False, "arm": "right"}])

    def test_stack_blocks_keeps_verified_base_grasp_and_repairs_top_with_other_arm(self):
        base_held_left = {
            "left": {"closed": True, "open": False, "contact": True,
                     "distance_m": 0.02},
            "right": {"closed": False, "open": True, "contact": False,
                      "distance_m": 0.40},
        }
        top_signals = {
            "left": {"closed": True, "open": False, "contact": False,
                     "distance_m": 0.3346},
            "right": {"closed": False, "open": True, "contact": False,
                      "distance_m": 0.3795},
        }
        positions = [[-0.0076, -0.1010, 0.8171], [0.2394, 0.0177, 0.7650]]
        current = {
            "block_p": positions,
            "block_grippers": [base_held_left, top_signals],
            "base_target": [0.0, -0.13, 0.75],
            "table_contact": [True, True],
            "block2_on_block1": False,
            "base_at_target": True,
            "strict_physical_quality": False,
        }
        previous = {"block_p": [list(point) for point in positions]}
        decision = classify_stack_blocks_two_state(current, previous)
        self.assertTrue(decision["ready"])
        self.assertEqual(decision["pending"], [{"name": "block2", "index": 1,
                                                "held": False, "arm": "right"}])

    def test_stack_blocks_still_rejects_unassociated_closed_gripper(self):
        opened = gripper()
        closed_free = {
            "left": {"closed": True, "open": False, "contact": False,
                     "distance_m": 0.34},
            "right": {"closed": False, "open": True, "contact": False,
                      "distance_m": 0.38},
        }
        positions = [[0.0, -0.13, 0.775], [0.18, 0.02, 0.766]]
        current = {
            "block_p": positions,
            "block_grippers": [opened, closed_free],
            "base_target": [0.0, -0.13, 0.75],
            "table_contact": [True, True],
            "block2_on_block1": False,
            "base_at_target": True,
            "strict_physical_quality": False,
        }
        previous = {"block_p": [list(point) for point in positions]}
        decision = classify_stack_blocks_two_state(current, previous)
        self.assertFalse(decision["ready"])
        self.assertEqual(decision["reason"], "block2_unheld_with_closed_gripper")

    def test_stack_blocks_rejects_unsupported_block(self):
        opened = gripper()
        current = {
            "block_p": [[0.0, -0.13, 0.775], [0.18, 0.02, 0.9]],
            "block_grippers": [opened, opened], "base_target": [0.0, -0.13, 0.75],
            "table_contact": [True, False], "block2_on_block1": False,
            "base_at_target": True, "strict_physical_quality": False,
        }
        previous = {"block_p": current["block_p"]}
        self.assertEqual(classify_stack_blocks_two_state(current, previous)["reason"],
                         "block2_unsupported")


if __name__ == "__main__":
    unittest.main()
