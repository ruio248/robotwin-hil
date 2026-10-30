import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_policy_xpolicylab import build_instruction


class FakeTaskEnv:
    source_arm_tag = "left"
    receiver_arm_tag = "right"


def fake_descriptions(_task_name, episode_info_list, _max_descriptions):
    params = episode_info_list[0]
    templates = [
        "Hand over from {a} to {b}",
        "Move the object from {a} toward {b}",
    ]
    random.shuffle(templates)
    return [
        {
            "seen": [
                template.format(a=params["{a}"], b=params["{b}"])
                for template in templates
            ]
        }
    ]


class BuildInstructionTest(unittest.TestCase):
    def test_scene_tag_instruction_is_repeatable_and_ignores_expert_result(self):
        args = {"task_name": "handover_to_tray"}
        env = FakeTaskEnv()
        with patch(
            "eval_policy_xpolicylab.generate_episode_descriptions",
            side_effect=fake_descriptions,
        ):
            first = build_instruction(
                args,
                {"info": {"{a}": "expert-source", "{b}": "expert-receiver"}},
                "seen",
                100,
                task_env=env,
                instruction_source="scene_tags",
                instruction_seed=31001,
            )
            second = build_instruction(
                args,
                {"info": {}},
                "seen",
                100,
                task_env=env,
                instruction_source="scene_tags",
                instruction_seed=31001,
            )

        self.assertEqual(first, second)
        self.assertIn("left", first)
        self.assertIn("right", first)
        self.assertNotIn("expert-", first)

    def test_instruction_generation_restores_process_random_states(self):
        args = {"task_name": "handover_to_tray"}
        random.seed(7)
        np.random.seed(7)
        python_state = random.getstate()
        numpy_state = np.random.get_state()

        with patch(
            "eval_policy_xpolicylab.generate_episode_descriptions",
            side_effect=fake_descriptions,
        ):
            build_instruction(
                args,
                {"info": {}},
                "seen",
                100,
                task_env=FakeTaskEnv(),
                instruction_source="scene_tags",
                instruction_seed=31001,
            )

        self.assertEqual(random.getstate(), python_state)
        restored_numpy_state = np.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(restored_numpy_state[1], numpy_state[1])
        self.assertEqual(restored_numpy_state[2:], numpy_state[2:])


if __name__ == "__main__":
    unittest.main()
