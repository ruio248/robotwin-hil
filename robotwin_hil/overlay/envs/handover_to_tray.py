"""Long-horizon bimanual handover followed by guided tray placement.

The task keeps final placement forgiving enough for a scripted expert to be
reliable. Stage records and physical-state snapshots are for later tail-state
reset and intervention experiments; they are never policy inputs.
"""

from __future__ import annotations

import math

import numpy as np
import sapien
import transforms3d as t3d

from dagger.adapters.handover_to_tray import HandoverRecoveryMixin

from ._base_task import Base_Task
from .utils import ArmTag, create_box, rand_pose


class handover_to_tray(HandoverRecoveryMixin, Base_Task):
    """Pass a vertical red bar to the opposite arm and park it in a blue tray."""


    def setup_demo(self, **kwargs):
        source_side = str(kwargs.get("source_side", "random")).lower()
        if source_side not in {"left", "right", "random"}:
            raise ValueError(f"source_side must be left, right, or random, got {source_side!r}")
        self.source_side = source_side
        self.geometry_profile = str(kwargs.get("geometry_profile", "v1")).lower()
        if self.geometry_profile not in {"v1", "v2_interior"}:
            raise ValueError(f"unsupported geometry_profile: {self.geometry_profile!r}")
        super()._init_task_env_(**kwargs)
        self.episode_seed = int(kwargs.get("seed", 0))

    def load_actors(self):
        source_arm = self.source_side
        if source_arm == "random":
            source_arm = "left" if np.random.rand() < 0.5 else "right"
        self.source_arm_tag = ArmTag(source_arm)
        self.receiver_arm_tag = self.source_arm_tag.opposite

        if self.geometry_profile == "v2_interior":
            # Stay inside the v1 collection support while avoiding the
            # reachability boundary found in the initial expert-seed sweep.
            source_x = [-0.22, -0.14] if self.source_arm_tag == "left" else [0.14, 0.22]
            source_y = [0.00, 0.11]
            source_yaw = 0.10
        else:
            source_x = [-0.25, -0.11] if self.source_arm_tag == "left" else [0.11, 0.25]
            source_y = [-0.03, 0.15]
            source_yaw = 0.20
        self.bar = create_box(
            scene=self,
            pose=rand_pose(
                xlim=source_x,
                ylim=source_y,
                zlim=[0.842],
                qpos=[0.981, 0, 0, 0.195],
                rotate_rand=True,
                rotate_lim=[0, 0, source_yaw],
            ),
            half_size=(0.03, 0.03, 0.10),
            color=(0.90, 0.08, 0.08),
            name="handover_bar",
            boxtype="long",
        )

        if self.geometry_profile == "v2_interior" and self.receiver_arm_tag == "right":
            target_x = [0.14, 0.21]
            target_y = [0.14, 0.19]
            target_yaw = 0.06
        elif self.geometry_profile == "v2_interior":
            target_x = [-0.21, -0.14]
            target_y = [0.14, 0.19]
            target_yaw = 0.06
        elif self.receiver_arm_tag == "right":
            target_x = [0.12, 0.23]
            target_y = [0.12, 0.20]
            target_yaw = 0.12
        else:
            # Keep the mirrored target at the same forward distance as the
            # right-arm target. The earlier near-center range caused the left
            # receiver to approach the tray from an awkward self-collision
            # configuration for a subset of seeds.
            target_x = [-0.23, -0.12]
            target_y = [0.12, 0.20]
            target_yaw = 0.12
        tray_pose = rand_pose(
            xlim=target_x,
            ylim=target_y,
            zlim=[0.741],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, target_yaw],
        )
        self.tray_floor = create_box(
            scene=self,
            pose=tray_pose,
            half_size=(0.075, 0.060, 0.005),
            color=(0.05, 0.18, 0.90),
            name="blue_tray_floor",
            is_static=True,
        )

        # The rails are wide enough for reliable planning but make orientation
        # errors visible and produce a recoverable final-stage tail state.
        rotation = t3d.quaternions.quat2mat(np.asarray(tray_pose.q, dtype=np.float64))
        for suffix, local_offset in (("left", [0.0, 0.060, 0.030]), ("right", [0.0, -0.060, 0.030])):
            rail_position = np.asarray(tray_pose.p, dtype=np.float64) + rotation @ np.asarray(local_offset)
            create_box(
                scene=self,
                pose=sapien.Pose(rail_position, tray_pose.q),
                half_size=(0.075, 0.007, 0.030),
                color=(0.08, 0.28, 0.98),
                name=f"blue_tray_rail_{suffix}",
                is_static=True,
            )

        self.add_prohibit_area(self.bar, padding=0.10)
        self.handover_middle_pose = [0.0, 0.0, 0.90, 0.0, 1.0, 0.0, 0.0]
        self.current_stage_id = 0
        self.stage_snapshots = []
        self.episode_metadata = {
            "task": "handover_to_tray",
            "seed": self.episode_seed,
            "source_arm": str(self.source_arm_tag),
            "receiver_arm": str(self.receiver_arm_tag),
            "stage_names": self._STAGE_NAMES,
        }


    def play_once(self):
        self.play_prefix_until_placement()
        # Keep the native handover_block synchronization: the source arm
        # retreats while the receiver places the bar, preventing a transient
        # one-arm hold from disturbing the object during the final transfer.
        self.current_stage_id = 6
        if self.plan_success:
            self.move(
                self.back_to_origin(self.source_arm_tag),
                self.place_actor(
                    self.bar,
                    target_pose=self._tray_placement_target(),
                    arm_tag=self.receiver_arm_tag,
                    functional_point_id=0,
                    pre_dis=0.05,
                    dis=0.0,
                    constrain="align",
                    pre_dis_axis="fp",
                ),
            )
        self._record_stage(6)

        success = bool(self.plan_success and self.check_success())
        self.episode_metadata["success"] = success
        self.episode_metadata["plan_success"] = bool(self.plan_success)
        self.episode_metadata["stage_count"] = len(self.stage_snapshots)
        self.info["info"] = {
            "{a}": str(self.source_arm_tag),
            "{b}": str(self.receiver_arm_tag),
        }
        self.info["task_metadata"] = self.episode_metadata
        self.info["stage_snapshots"] = self.stage_snapshots
        return self.info

    def success_metrics(self) -> dict[str, float | bool]:
        bar_pose = self.bar.get_functional_point(0, "pose")
        tray_pose = self.tray_floor.get_functional_point(1, "pose")
        position_error = np.abs(np.asarray(bar_pose.p) - np.asarray(tray_pose.p))

        bar_axis = self.bar.get_pose().to_transformation_matrix()[:3, 0]
        tray_axis = self.tray_floor.get_pose().to_transformation_matrix()[:3, 0]
        yaw_dot = abs(float(np.dot(bar_axis[:2], tray_axis[:2])))
        yaw_error_deg = math.degrees(math.acos(np.clip(yaw_dot, -1.0, 1.0)))
        return {
            "position_error_x": float(position_error[0]),
            "position_error_y": float(position_error[1]),
            "position_error_z": float(position_error[2]),
            "yaw_error_deg": float(yaw_error_deg),
            "left_gripper_open": bool(self.is_left_gripper_open()),
            "right_gripper_open": bool(self.is_right_gripper_open()),
        }

    def check_success(self):
        metrics = self.success_metrics()
        return bool(
            metrics["position_error_x"] < 0.040
            and metrics["position_error_y"] < 0.040
            and metrics["position_error_z"] < 0.025
            and metrics["yaw_error_deg"] <= 15.0
            and metrics["left_gripper_open"]
            and metrics["right_gripper_open"]
        )
