"""handover_to_tray stages and live-state expert recovery.

Recovery methods are moved verbatim from the existing custom task. The
environment inherits the mixin so historical callers keep the same methods.
"""
from __future__ import annotations

import traceback
import numpy as np

from .base import TaskAdapter

PROMPT = "Pass the red bar from the left arm to the right arm and place it in the blue tray."
RESERVED_SEED_RANGES = (
    (1, 579, "demonstration (1-579)"),
    (30000, 30022, "development (30000-30022)"),
    (31000, 31103, "frozen test (31000-31103)"),
)
STAGE_LABELS = {
    1: "source_grasp               左臂抓取红杆",
    2: "source_lift               左臂抬起",
    3: "handover_pose             移向交接位姿",
    4: "receiver_grasp            右臂抓取（含悬空直接放）",
    5: "source_release_and_retreat 左臂松开并后撤",
    6: "guided_tray_placement     放入蓝托盘",
}


class HandoverRecoveryMixin:
    _STAGE_NAMES = {
        1: "source_grasp",
        2: "source_lift",
        3: "handover_pose",
        4: "receiver_grasp",
        5: "source_release_and_retreat",
        6: "guided_tray_placement",
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


    def _arm_min_bar_distance(self, arm_tag: ArmTag) -> float:
        """Smallest distance from the arm's end effector to the bar contact
        points, in metres. Returns NaN when no contact point can be read."""
        ee_pose = np.asarray(self.get_arm_pose(arm_tag), dtype=np.float64)
        distances = []
        for point_id in range(8):
            try:
                point_pose = self.bar.get_contact_point(point_id, "pose")
                distances.append(float(np.linalg.norm(ee_pose[:3] - np.asarray(point_pose.p))))
            except (IndexError, TypeError, ValueError, AttributeError):
                continue
        return float(min(distances)) if distances else float("nan")


    def _arm_near_bar(self, arm_tag: ArmTag, threshold: float = 0.17) -> bool:
        """Closed-gripper proximity fallback for momentary contact dropouts."""
        distance = self._arm_min_bar_distance(arm_tag)
        return bool(np.isfinite(distance) and distance <= threshold)


    def arm_holds_bar(self, arm_tag: ArmTag) -> bool:
        """Best-effort grasp predicate used only by the privileged expert."""
        from envs.utils import ArmTag

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
        elif self._bar_near_tray(bar_position):
            # The policy dropped the bar above/near the tray with both grippers
            # free. Grabbing it again with the source would be the wrong plan:
            # the receiver can grasp it in place and finish the placement.
            branch = "guided_place_from_air"
            stage_id = 4
        else:
            branch = "restart_source_grasp"
            stage_id = 1

        return {
            "branch": branch,
            "stage_id": stage_id,
            "source_holds": bool(source_holds),
            "receiver_holds": bool(receiver_holds),
            "placed": bool(placed),
            "near_tray": bool(self._bar_near_tray(bar_position)),
            "source_ee_to_bar": round(self._arm_min_bar_distance(self.source_arm_tag), 4),
            "receiver_ee_to_bar": round(self._arm_min_bar_distance(self.receiver_arm_tag), 4),
            "recoverable": bool(recoverable or source_holds or receiver_holds),
            "bar_position": bar_position.round(8).tolist(),
            "success_metrics": metrics,
        }


    def _bar_near_tray(self, bar_position: np.ndarray | None = None) -> bool:
        """True when the bar is hovering close enough to the tray to be
        recovered by a direct receiver grasp + placement instead of a full
        source-side restart."""
        if bar_position is None:
            bar_position = np.asarray(self.bar.get_pose().p, dtype=np.float64)
        if not np.isfinite(bar_position).all():
            return False
        tray_position = np.asarray(
            self.tray_floor.get_functional_point(1, "pose").p,
            dtype=np.float64,
        )
        return bool(
            abs(bar_position[0] - tray_position[0]) < 0.28
            and abs(bar_position[1] - tray_position[1]) < 0.28
            and 0.70 <= bar_position[2] <= 1.25
        )


    @staticmethod
    def _forced_recovery_state(
        initial: dict[str, object], stage_id: int
    ) -> dict[str, object]:
        """Override the inferred branch so recovery starts from a human-chosen
        stage (1-6) instead of the privileged auto-inference."""
        overrides = {
            1: ("restart_source_grasp", 1, False, False, False, False),
            2: ("resume_handover", 3, True, False, False, False),
            3: ("resume_handover", 3, True, False, False, False),
            4: ("guided_place_from_air", 4, False, False, False, True),
            5: ("receiver_place", 5, True, True, False, False),
            6: ("receiver_place", 6, False, True, False, False),
        }
        if stage_id not in overrides:
            raise ValueError(f"unsupported recovery start stage: {stage_id}")
        (
            branch,
            entry_stage,
            source_holds,
            receiver_holds,
            placed,
            near_tray,
        ) = overrides[stage_id]
        forced = dict(initial)
        forced["branch"] = branch
        forced["stage_id"] = entry_stage
        forced["source_holds"] = source_holds
        forced["receiver_holds"] = receiver_holds
        forced["placed"] = placed
        forced["near_tray"] = near_tray
        forced["forced_stage_id"] = stage_id
        return forced


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
        air_place = bool(
            initial.get("near_tray")
            and not source_holds
            and not receiver_holds
            and not placed
        )

        if placed:
            # A policy may reach the tray but forget to release one gripper.
            run(
                6,
                self.open_gripper(self.source_arm_tag),
                self.open_gripper(self.receiver_arm_tag),
            )
        elif air_place:
            # Bar is already hovering near the tray with no gripper holding it.
            # Grasp it directly with the receiver and finish the placement.
            run(
                4,
                self.open_gripper(self.source_arm_tag),
                self.open_gripper(self.receiver_arm_tag),
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
                # The policy usually stops with the receiver hovering right
                # next to the bar. Nudge it back a little along its approach
                # axis (back to roughly the pre-grasp clearance) instead of
                # sending it all the way home, then re-approach. Preparation
                # only: a failed retreat falls back to the original flow.
                if not run(
                    4,
                    self.move_by_displacement(
                        self.receiver_arm_tag, z=0.10, move_axis="arm"
                    ),
                ):
                    self.plan_success = True
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


    def recovery_stage_iterator(self, start_stage_id: int | None = None):
        """Yield the scripted expert's recovery stages one at a time.

        The supervisor can hand control back to the policy after any yielded
        stage. Each ``next()`` executes one semantic stage (or opens the
        final grippers) and yields a progress dict. The final yield has
        ``phase == "done"``.

        ``start_stage_id`` optionally forces the recovery entry stage (1-6)
        chosen by a human, overriding the privileged auto-inference.
        """
        initial = self.infer_recovery_state()
        if start_stage_id is not None:
            initial = self._forced_recovery_state(initial, int(start_stage_id))
        if not initial["recoverable"]:
            yield {
                "phase": "done",
                "success": False,
                "plan_success": False,
                "reason": "bar_outside_recovery_workspace",
                "initial": initial,
                "executed_stage_ids": [],
            }
            return

        self.plan_success = True
        executed_stage_ids: list[int] = []
        source_holds = bool(initial["source_holds"])
        receiver_holds = bool(initial["receiver_holds"])
        placed = bool(initial["placed"])
        air_place = bool(
            initial.get("near_tray")
            and not source_holds
            and not receiver_holds
            and not placed
        )
        reason = None

        def progress():
            return {
                "phase": "stage",
                "plan_success": bool(self.plan_success),
                "executed_stage_ids": list(executed_stage_ids),
                "source_holds": bool(source_holds),
                "receiver_holds": bool(receiver_holds),
            }

        if placed:
            self._run_stage(
                6,
                self.open_gripper(self.source_arm_tag),
                self.open_gripper(self.receiver_arm_tag),
            )
            executed_stage_ids.append(6)
            yield progress()
        elif air_place:
            self._run_stage(
                4,
                self.open_gripper(self.source_arm_tag),
                self.open_gripper(self.receiver_arm_tag),
            )
            executed_stage_ids.append(4)
            yield progress()

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
            executed_stage_ids.append(4)
            yield progress()

            receiver_holds = bool(
                self.plan_success and self.arm_holds_bar(self.receiver_arm_tag)
            )
            if self.plan_success and not receiver_holds:
                self.plan_success = False
                reason = "receiver_grasp_did_not_attach"
            if receiver_holds and self.plan_success:
                self._run_stage(
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
                executed_stage_ids.append(6)
                yield progress()
        else:
            if not source_holds and not receiver_holds:
                self._run_stage(
                    1,
                    self.open_gripper(self.source_arm_tag),
                    self.open_gripper(self.receiver_arm_tag),
                )
                executed_stage_ids.append(1)
                yield progress()

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
                executed_stage_ids.append(1)
                yield progress()

                source_holds = bool(
                    self.plan_success and self.arm_holds_bar(self.source_arm_tag)
                )
                if self.plan_success and not source_holds:
                    self.plan_success = False
                    reason = "source_grasp_did_not_attach"
                if self.plan_success:
                    self._run_stage(
                        2,
                        self.move_by_displacement(self.source_arm_tag, z=0.10),
                    )
                    executed_stage_ids.append(2)
                    yield progress()

            if source_holds and not receiver_holds and self.plan_success:
                if (
                    self.is_left_gripper_close()
                    if self.receiver_arm_tag == "left"
                    else self.is_right_gripper_close()
                ):
                    self._run_stage(3, self.open_gripper(self.receiver_arm_tag))
                    executed_stage_ids.append(3)
                    yield progress()

                # Nudge the receiver back along its approach axis (small
                # clearance, not the home pose) before re-centring the bar.
                # Preparation only: if the retreat cannot be planned, keep the
                # original re-grasp attempt.
                retreated = self._run_stage(
                    4,
                    self.move_by_displacement(
                        self.receiver_arm_tag, z=0.10, move_axis="arm"
                    ),
                )
                executed_stage_ids.append(4)
                yield progress()
                if not retreated:
                    self.plan_success = True

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
                executed_stage_ids.append(3)
                yield progress()

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
                executed_stage_ids.append(4)
                yield progress()

                receiver_holds = bool(
                    self.plan_success and self.arm_holds_bar(self.receiver_arm_tag)
                )
                if self.plan_success and not receiver_holds:
                    self.plan_success = False
                    reason = "receiver_grasp_did_not_attach"

            if source_holds and receiver_holds and self.plan_success:
                self._run_stage(5, self.open_gripper(self.source_arm_tag))
                executed_stage_ids.append(5)
                yield progress()

                self._run_stage(
                    5,
                    self.move_by_displacement(
                        self.source_arm_tag,
                        z=0.10,
                        move_axis="arm",
                    ),
                )
                executed_stage_ids.append(5)
                source_holds = False
                yield progress()

            if receiver_holds and self.plan_success:
                self._run_stage(
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
                executed_stage_ids.append(6)
                yield progress()

        success = bool(self.plan_success and self.check_success())
        self.eval_success = success
        result = {
            "initial": initial,
            "branch": initial["branch"],
            "executed_stage_ids": executed_stage_ids,
            "plan_success": bool(self.plan_success),
            "success": success,
            "reason": reason,
            "final": self.infer_recovery_state(),
        }
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
        yield {"phase": "done", "result": result}


    def _run_stage(self, stage_id: int, *actions) -> bool:
        self.current_stage_id = stage_id
        if self.plan_success:
            try:
                self.move(*actions)
            except Exception as exc:
                # A planner/motion exception from an extreme policy state must
                # fail this stage, not kill the whole DAgger session.
                self.plan_success = False
                self.episode_metadata["expert_stage_error"] = {
                    "stage_id": int(stage_id),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                print(
                    f"\033[91m[EXPERT] stage {stage_id} planning failed: {exc}\033[0m",
                    flush=True,
                )
        self._record_stage(stage_id)
        return bool(self.plan_success)


    def play_prefix_until_placement(self):
        """Run the native scripted expert through stage 5, without placing."""
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
        return bool(self.plan_success)


class HandoverToTrayAdapter(TaskAdapter):
    name = "handover_to_tray"
    prompt = PROMPT
    stage_labels = STAGE_LABELS
    default_task_config = "handover_to_tray_v2_promptfix"
    default_checkpoint_name = "v2_promptfix_9999"
    default_frequency = 30
    supports_one_stage = True

    def infer_recovery_state(self, env):
        return env.infer_recovery_state()

    def recovery_stage_iterator(self, env, start_stage_id=None):
        return env.recovery_stage_iterator(start_stage_id=start_stage_id)

    def success_metrics(self, env):
        return env.success_metrics()

    def play_prefix_until_placement(self, env):
        return env.play_prefix_until_placement()

    def handoff_record(self, env):
        from one_stage_dagger import handoff_record
        return handoff_record(env)

    def describe_recovery_state(self, state):
        return [
            f"branch={state['branch']}  entry_stage={state['stage_id']}",
            f"source_holds={state['source_holds']} receiver_holds={state['receiver_holds']} "
            f"placed={state['placed']} near_tray={state.get('near_tray', False)}",
            f"末端到杆距离: source={state.get('source_ee_to_bar')} "
            f"receiver={state.get('receiver_ee_to_bar')} (米)",
            f"bar_pos={state.get('bar_position')}",
        ]
