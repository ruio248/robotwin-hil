"""Transactional SAPIEN CPU lookahead for the handover_to_tray qpos executor.

Physics is restored between branches, along with controller targets and the
Python fields touched by Base_Task.take_action/get_obs. No candidate frames
are sent to the real policy server or written to its episode video.
"""
from __future__ import annotations

import contextlib
import copy
import io
import random

import numpy as np


class RestoreError(RuntimeError):
    """A live scene cannot safely continue after a failed restoration."""


class _Discard(io.TextIOBase):
    def write(self, text):
        return len(text)


class SceneSnapshot:
    ENV_FIELDS = ("take_action_cnt", "eval_success", "now_obs", "current_stage_id",
                  "plan_success", "left_js", "right_js", "eval_video_path", "save_data", "render_freq",
                  "FRAME_IDX", "current_control_source", "control_mask")
    ROBOT_FIELDS = ("left_gripper_val", "right_gripper_val", "left_js", "right_js")

    def __init__(self, env):
        self.env = env
        self.system = env.scene.physx_system
        if not callable(getattr(self.system, "pack", None)) or not callable(getattr(self.system, "unpack", None)):
            raise RuntimeError("Enhanced sampling requires SAPIEN CPU PhysX pack/unpack")
        if getattr(env, "crazy_random_light", False):
            raise ValueError("Changing lights are not supported by the branch snapshot")
        self.physics = self.system.pack()
        self.articulations = []
        for articulation in env.scene.get_all_articulations():
            joints = [(joint, np.array(joint.get_drive_target(), copy=True),
                       np.array(joint.get_drive_velocity_target(), copy=True))
                      for joint in articulation.get_active_joints()]
            self.articulations.append((articulation, articulation.get_qf().copy(),
                                       articulation.get_qacc().copy(), joints))
        self.env_state = {key: copy.deepcopy(getattr(env, key)) for key in self.ENV_FIELDS if hasattr(env, key)}
        self.robot_state = {key: copy.deepcopy(getattr(env.robot, key))
                            for key in self.ROBOT_FIELDS if hasattr(env.robot, key)}
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()

    def restore(self):
        try:
            self.system.unpack(self.physics)
            for articulation, qf, qacc, joints in self.articulations:
                articulation.set_qf(qf)
                articulation.set_qacc(qacc)
                for joint, position, velocity in joints:
                    joint.set_drive_target(position)
                    joint.set_drive_velocity_target(velocity)
            for key, value in self.env_state.items():
                setattr(self.env, key, copy.deepcopy(value))
            for key, value in self.robot_state.items():
                setattr(self.env.robot, key, copy.deepcopy(value))
            random.setstate(self.python_rng)
            np.random.set_state(self.numpy_rng)
            self.env._update_render()
        except Exception as exc:
            raise RestoreError("Failed to restore lookahead snapshot; abort this evaluation") from exc


def physical_signature(env):
    """Compare replay endpoints without relying only on a nearly flat critic."""
    values = []
    for actor in env.scene.get_all_actors():
        pose = actor.get_pose()
        values.extend((pose.p, pose.q))
        for component in getattr(actor, "components", ()):
            if hasattr(component, "linear_velocity"):
                values.extend((component.linear_velocity, component.angular_velocity))
    for articulation in env.scene.get_all_articulations():
        pose = articulation.get_root_pose()
        values.extend((pose.p, pose.q, articulation.get_qpos(), articulation.get_qvel()))
        if hasattr(articulation, "get_root_linear_velocity"):
            values.extend((articulation.get_root_linear_velocity(), articulation.get_root_angular_velocity()))
    return np.concatenate([np.asarray(value).reshape(-1) for value in values])


class RobotwinRollout:
    def __init__(self, env, scorer, on_restored=None):
        self.env, self.scorer = env, scorer
        self.on_restored = on_restored

    def _branch(self, actions):
        env = self.env
        env.eval_video_path, env.save_data, env.render_freq = None, False, 0
        scores = []
        for action in actions:
            if env.eval_success or env.take_action_cnt >= env.step_lim:
                break
            observation = env.get_obs()
            # Pair action_h with observation BEFORE executing action_h.
            scores.append(float(self.scorer(observation, action)))
            env.take_action(action, action_type="qpos")
        if not scores or not np.isfinite(scores).all():
            raise ValueError("Branch has no finite pre-action coverage scores")
        return {"coverage": scores, "scored_steps": len(scores),
                "success": bool(env.eval_success),
                "step_limit": bool(env.take_action_cnt >= env.step_lim)}, physical_signature(env)

    def evaluate(self, candidates, *, verify, atol):
        snapshot = SceneSnapshot(self.env)
        branches, signatures = [], []
        replay = None
        try:
            with contextlib.redirect_stdout(_Discard()):
                for chunk in candidates:
                    snapshot.restore()
                    if self.on_restored is not None:
                        self.on_restored()
                    branch, signature = self._branch(chunk)
                    branches.append(branch)
                    signatures.append(signature)
                # Check the same candidate after all other branches at the first
                # active decision of EVERY episode (including the vanilla arm).
                if verify:
                    snapshot.restore()
                    if self.on_restored is not None:
                        self.on_restored()
                    repeated, signature = self._branch(candidates[0])
                    original = branches[0]
                    same = (repeated["scored_steps"] == original["scored_steps"]
                            and repeated["success"] == original["success"]
                            and repeated["step_limit"] == original["step_limit"])
                    state_error = float(np.max(np.abs(signature - signatures[0])))
                    score_error = float(np.max(np.abs(np.asarray(repeated["coverage"]) - original["coverage"]))) if same else float("inf")
                    if not same or not np.isfinite([state_error, score_error]).all() or max(state_error, score_error) > atol:
                        raise RuntimeError(f"Branch replay mismatch: state={state_error}, score={score_error}, atol={atol}")
                    replay = {"state_max_abs_error": state_error, "coverage_max_abs_error": score_error, "atol": atol}
        finally:
            snapshot.restore()
        return branches, replay
