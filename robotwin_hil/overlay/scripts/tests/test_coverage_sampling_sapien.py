"""Optional real SAPIEN 3 CPU test: articulation drives and contact replay.

No renderer, policy server, robot assets or GPU required. This tests the
snapshot backend, not the full visual handover task or policy performance.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coverage_sampling.robotwin import RobotwinRollout, SceneSnapshot, physical_signature


@unittest.skipUnless(importlib.util.find_spec("sapien"), "SAPIEN not installed")
class SapienSnapshotTests(unittest.TestCase):
    def test_articulation_contact_and_selected_execution(self):
        import sapien

        class Env:
            def __init__(self):
                self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
                self.scene.set_timestep(.004)
                self.scene.add_ground(0, render=False)
                builder = self.scene.create_articulation_builder()
                base = builder.create_link_builder()
                base.set_name("base")
                slider = builder.create_link_builder(base)
                slider.set_name("slider")
                slider.set_joint_name("slide")
                slider.set_joint_properties("prismatic", limits=[[-1., 1.]],
                                            pose_in_parent=sapien.Pose([0, 0, .05]), pose_in_child=sapien.Pose())
                slider.add_box_collision(half_size=[.04, .04, .04])
                self.arm = builder.build(fix_root_link=True)
                self.joint = self.arm.get_active_joints()[0]
                self.joint.set_drive_properties(1000, 100)
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[.04, .04, .04])
                self.box = builder.build(name="pushed_box")
                self.box.set_pose(sapien.Pose([.13, 0, .04]))
                for _ in range(20): self.scene.step()
                self.robot = SimpleNamespace(left_gripper_val=0., right_gripper_val=0.)
                self.take_action_cnt, self.step_lim = 0, 100
                self.eval_success, self.save_data, self.render_freq = False, True, 1
                self.eval_video_path, self.now_obs = "video", {}
                self.writes = 0
            def _update_render(self): pass
            def get_obs(self):
                self.now_obs = {"joint_action": {"vector": np.full(14, self.arm.get_qpos()[0], dtype=np.float32)}}
                return self.now_obs
            def take_action(self, action, action_type):
                self.writes += int(bool(self.eval_video_path or self.save_data or self.render_freq))
                self.take_action_cnt += 1
                self.joint.set_drive_target(float(action[0]))
                self.joint.set_drive_velocity_target(.01)
                self.arm.set_qf(np.array([.1]))
                self.robot.left_gripper_val = float(action[6])
                for _ in range(25): self.scene.step()

        env = Env()
        original = physical_signature(env).copy()
        initial_target = env.joint.get_drive_target().copy()
        score = lambda obs, action: float(obs["joint_action"]["vector"][0] + action[0])
        chunks = np.zeros((2, 3, 14), dtype=np.float32)
        chunks[0, :, 0] = [.12, .20, .25]
        chunks[1, :, 0] = [-.1, -.2, -.3]
        backend = RobotwinRollout(env, score)
        branches, replay = backend.evaluate(chunks, verify=True, atol=1e-4)
        np.testing.assert_allclose(physical_signature(env), original, atol=1e-7)
        np.testing.assert_array_equal(env.joint.get_drive_target(), initial_target)
        self.assertEqual(env.take_action_cnt, 0)
        self.assertEqual(env.writes, 0)
        self.assertLessEqual(replay["state_max_abs_error"], 1e-4)
        self.assertNotEqual(branches[0]["coverage"][0], branches[0]["coverage"][1])
        actual = []
        for action in chunks[0]:
            actual.append(score(env.get_obs(), action))
            env.take_action(action, "qpos")
        np.testing.assert_allclose(actual, branches[0]["coverage"], atol=1e-4)
        self.assertEqual(env.take_action_cnt, 3)
        self.assertEqual(env.writes, 3)


if __name__ == "__main__":
    unittest.main()
