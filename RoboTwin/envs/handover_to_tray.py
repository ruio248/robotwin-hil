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

from ._base_task import Base_Task
from .utils import ArmTag, create_box, rand_pose


class handover_to_tray(Base_Task):
    """Pass a vertical red bar to the opposite arm and park it in a blue tray."""

    _STAGE_NAMES = {
        1: "source_grasp",
        2: "source_lift",
        3: "handover_pose",
        4: "receiver_grasp",
        5: "source_release_and_retreat",
        6: "guided_tray_placement",
    }

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

    @staticmethod
    def _pose_dict(pose: sapien.Pose) -> dict[str, list[float]]:
        return {
            "position": np.asarray(pose.p, dtype=np.float64).round(8).tolist(),
            "quaternion_wxyz": np.asarray(pose.q, dtype=np.float64).round(8).tolist(),
        }

    def _record_stage(self, stage_id: int) -> None:
        self.current_stage_id = stage_id
        self.stage_snapshots.append(
            {
                "stage_id": stage_id,
                "stage_name": self._STAGE_NAMES[stage_id],
                "seed": self.episode_seed,
                "plan_success": bool(self.plan_success),
                "left_joint_state": np.asarray(self.robot.get_left_arm_jointState(), dtype=np.float64).round(8).tolist(),
                "right_joint_state": np.asarray(self.robot.get_right_arm_jointState(), dtype=np.float64).round(8).tolist(),
                "bar_pose": self._pose_dict(self.bar.get_pose()),
                "tray_pose": self._pose_dict(self.tray_floor.get_pose()),
            }
        )

    def _tray_placement_target(self) -> sapien.Pose:
        target = self.tray_floor.get_functional_point(1, "pose")
        if self.receiver_arm_tag != "left":
            return target

        return target

    def _gripper_link_names(self, arm_tag: ArmTag) -> set[str]:
        """Return the physics link names belonging to one arm's gripper."""
        side = str(arm_tag)
        names = set(getattr(self.robot, f"{side}_fix_gripper_name", []) or [])
        for item in getattr(self.robot, f"{side}_gripper", []) or []:
            if item is None:
                continue
            try:
                names.add(item[0].child_link.get_name())
            except (AttributeError, IndexError, TypeError):
                continue
        return names

    def _arm_contacts_bar(self, arm_tag: ArmTag) -> bool:
        bar_name = self.bar.get_name()
        gripper_names = self._gripper_link_names(arm_tag)
        for contact in self.scene.get_contacts():
            try:
                name_a = contact.bodies[0].entity.name
                name_b = contact.bodies[1].entity.name
            except (AttributeError, IndexError):
                continue
            if (name_a == bar_name and name_b in gripper_names) or (
                name_b == bar_name and name_a in gripper_names
            ):
                return True
        return False

    def _arm_near_bar(self, arm_tag: ArmTag, threshold: float = 0.17) -> bool:
        """Closed-gripper proximity fallback for momentary contact dropouts."""
        ee_pose = np.asarray(self.get_arm_pose(arm_tag), dtype=np.float64)
        distances = []
        for point_id in range(8):
            try:
                point_pose = self.bar.get_contact_point(point_id, "pose")
                distances.append(float(np.linalg.norm(ee_pose[:3] - np.asarray(point_pose.p))))
            except (IndexError, TypeError, ValueError, AttributeError):
                continue
        return bool(distances and min(distances) <= threshold)

    def arm_holds_bar(self, arm_tag: ArmTag) -> bool:
        """Best-effort grasp predicate used only by the privileged expert."""
        arm_tag = ArmTag(arm_tag)
        gripper_closed = (
            self.is_left_gripper_close()
            if arm_tag == "left"
            else self.is_right_gripper_close()
        )
        if not gripper_closed:
            return False
        return self._arm_contacts_bar(arm_tag) or self._arm_near_bar(arm_tag)

    def infer_recovery_state(self) -> dict[str, object]:
        """Infer the recovery branch from live privileged simulator state."""
        source_holds = self.arm_holds_bar(self.source_arm_tag)
        receiver_holds = self.arm_holds_bar(self.receiver_arm_tag)
        metrics = self.success_metrics()
        placed = bool(
            metrics["position_error_x"] < 0.040
            and metrics["position_error_y"] < 0.040
            and metrics["position_error_z"] < 0.025
            and metrics["yaw_error_deg"] <= 15.0
        )
        bar_position = np.asarray(self.bar.get_pose().p, dtype=np.float64)
        recoverable = bool(
            np.isfinite(bar_position).all()
            and -0.55 <= bar_position[0] <= 0.55
            and -0.30 <= bar_position[1] <= 0.40
            and 0.70 <= bar_position[2] <= 1.25
        )

        if placed:
            branch = "release_at_tray"
            stage_id = 6
        elif receiver_holds:
            branch = "receiver_place"
            stage_id = 5 if source_holds else 6
        elif source_holds:
            branch = "resume_handover"
            stage_id = 3
        else:
            branch = "restart_source_grasp"
            stage_id = 1

        return {
            "branch": branch,
            "stage_id": stage_id,
            "source_holds": bool(source_holds),
            "receiver_holds": bool(receiver_holds),
            "placed": bool(placed),
            "recoverable": bool(recoverable or source_holds or receiver_holds),
            "bar_position": bar_position.round(8).tolist(),
            "success_metrics": metrics,
        }

    def recover_from_current_state(self) -> dict[str, object]:
        """Run the scripted expert from a policy-visited live simulator state.

        This method deliberately does not reset the scene.  It selects the
        shortest valid suffix of the original six-stage expert using grasp and
        placement predicates, then replans every motion from current qpos.
        """
        initial = self.infer_recovery_state()
        result: dict[str, object] = {
            "initial": initial,
            "branch": initial["branch"],
            "executed_stage_ids": [],
            "plan_success": False,
            "success": False,
        }
        if not initial["recoverable"]:
            result["reason"] = "bar_outside_recovery_workspace"
            return result

        self.plan_success = True
        executed_stage_ids: list[int] = result["executed_stage_ids"]  # type: ignore[assignment]

        def run(stage_id: int, *actions) -> bool:
            executed_stage_ids.append(stage_id)
            return self._run_stage(stage_id, *actions)

        source_holds = bool(initial["source_holds"])
        receiver_holds = bool(initial["receiver_holds"])
        placed = bool(initial["placed"])

        if placed:
            # A policy may reach the tray but forget to release one gripper.
            run(
                6,
                self.open_gripper(self.source_arm_tag),
                self.open_gripper(self.receiver_arm_tag),
            )
        else:
            if not source_holds and not receiver_holds:
                # A half-closed gripper cannot execute the normal grasp suffix.
                run(
                    1,
                    self.open_gripper(self.source_arm_tag),
                    self.open_gripper(self.receiver_arm_tag),
                )
                run(
                    1,
                    self.grasp_actor(
                        self.bar,
                        arm_tag=self.source_arm_tag,
                        pre_grasp_dis=0.07,
                        grasp_dis=0.0,
                        contact_point_id=[0, 1, 2, 3],
                    ),
                )
                source_holds = bool(
                    self.plan_success and self.arm_holds_bar(self.source_arm_tag)
                )
                if self.plan_success and not source_holds:
                    self.plan_success = False
                    result["reason"] = "source_grasp_did_not_attach"
                if self.plan_success:
                    run(2, self.move_by_displacement(self.source_arm_tag, z=0.10))

            if source_holds and not receiver_holds and self.plan_success:
                # Ensure the receiving gripper can approach before moving the
                # object back to the canonical handover pose.
                if (
                    self.is_left_gripper_close()
                    if self.receiver_arm_tag == "left"
                    else self.is_right_gripper_close()
                ):
                    run(3, self.open_gripper(self.receiver_arm_tag))
                run(
                    3,
                    self.place_actor(
                        self.bar,
                        target_pose=self.handover_middle_pose,
                        arm_tag=self.source_arm_tag,
                        functional_point_id=0,
                        pre_dis=0.0,
                        dis=0.0,
                        is_open=False,
                        constrain="free",
                    ),
                )
                run(
                    4,
                    self.grasp_actor(
                        self.bar,
                        arm_tag=self.receiver_arm_tag,
                        pre_grasp_dis=0.07,
                        grasp_dis=0.0,
                        contact_point_id=[4, 5, 6, 7],
                    ),
                )
                receiver_holds = bool(
                    self.plan_success and self.arm_holds_bar(self.receiver_arm_tag)
                )
                if self.plan_success and not receiver_holds:
                    self.plan_success = False
                    result["reason"] = "receiver_grasp_did_not_attach"

            if source_holds and receiver_holds and self.plan_success:
                run(5, self.open_gripper(self.source_arm_tag))
                run(5, self.move_by_displacement(self.source_arm_tag, z=0.10, move_axis="arm"))
                source_holds = False

            if receiver_holds and self.plan_success:
                run(
                    6,
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

        success = bool(self.plan_success and self.check_success())
        self.eval_success = success
        result["plan_success"] = bool(self.plan_success)
        result["success"] = success
        result["final"] = self.infer_recovery_state()
        self.episode_metadata["success"] = success
        self.episode_metadata["plan_success"] = bool(self.plan_success)
        self.episode_metadata["stage_count"] = len(self.stage_snapshots)
        self.episode_metadata["recovery"] = result
        self.info["info"] = {
            "{a}": str(self.source_arm_tag),
            "{b}": str(self.receiver_arm_tag),
        }
        self.info["task_metadata"] = self.episode_metadata
        self.info["stage_snapshots"] = self.stage_snapshots
        self.info["recovery"] = result
        return result

    def _run_stage(self, stage_id: int, *actions) -> bool:
        self.current_stage_id = stage_id
        if self.plan_success:
            self.move(*actions)
        self._record_stage(stage_id)
        return bool(self.plan_success)

    def play_once(self):
        self._run_stage(
            1,
            self.grasp_actor(
                self.bar,
                arm_tag=self.source_arm_tag,
                pre_grasp_dis=0.07,
                grasp_dis=0.0,
                contact_point_id=[0, 1, 2, 3],
            ),
        )
        self._run_stage(2, self.move_by_displacement(self.source_arm_tag, z=0.10))
        self._run_stage(
            3,
            self.place_actor(
                self.bar,
                target_pose=self.handover_middle_pose,
                arm_tag=self.source_arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
                constrain="free",
            ),
        )
        self._run_stage(
            4,
            self.grasp_actor(
                self.bar,
                arm_tag=self.receiver_arm_tag,
                pre_grasp_dis=0.07,
                grasp_dis=0.0,
                contact_point_id=[4, 5, 6, 7],
            ),
        )
        # Base_Task.move can execute one action stream per arm. Opening and
        # retreating use the same source arm, so they remain sequential inside
        # the single semantic release-and-retreat stage.
        self.current_stage_id = 5
        if self.plan_success:
            self.move(self.open_gripper(self.source_arm_tag))
            self.move(self.move_by_displacement(self.source_arm_tag, z=0.10, move_axis="arm"))
        self._record_stage(5)
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
