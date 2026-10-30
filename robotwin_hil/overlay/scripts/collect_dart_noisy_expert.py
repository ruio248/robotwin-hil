#!/usr/bin/env python3
"""Collect noisy scripted-supervisor rollouts with clean intended action labels.

The supervisor plans absolute joint-target trajectories.  We perturb only the
executed 12 arm-joint targets, while saving the unperturbed next target beside
each sampled observation.  This makes the action label explicit and prevents
the usual next-observed-qpos label from silently becoming the noisy command.

Because RoboTwin's scripted supervisor replans at semantic stage boundaries,
this is a stage-plan DART adaptation, not a fully reactive expert queried at
every simulator frame.  The metadata records that limitation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def make_action(task, control_seq: dict, index: int) -> np.ndarray:
    """Return clean absolute qpos target [left arm+grip, right arm+grip]."""
    values = []
    for side in ("left", "right"):
        arm = control_seq[f"{side}_arm"]
        if arm is None:
            qpos = np.asarray(getattr(task.robot, f"get_{side}_arm_jointState")(), dtype=np.float64)[:-1]
        else:
            positions = np.asarray(arm["position"], dtype=np.float64)
            qpos = positions[min(index, len(positions) - 1)]
        grip = control_seq[f"{side}_gripper"]
        if grip is None:
            grip_value = float(getattr(task.robot, f"get_{side}_gripper_val")())
        else:
            results = np.asarray(grip["result"], dtype=np.float64).reshape(-1)
            grip_value = float(results[min(index, len(results) - 1)])
        values.extend(qpos.tolist())
        values.append(grip_value)
    return np.asarray(values, dtype=np.float32)


def install_collector(task, covariance: np.ndarray, scale: float, seed: int) -> dict:
    rng = np.random.default_rng(seed + 880_301)
    stats = {"noisy_control_steps": 0, "saved_frames": 0,
             "noise_sq_sum": 0.0, "noise_values": 0, "noise_abs_max": 0.0,
             "noisy_joint_targets_outside_pi": 0}
    original_get_obs = task.get_obs
    original_take_picture = task._take_picture
    original_take_dense_action = task.take_dense_action
    task._dart_label = np.zeros(14, dtype=np.float32)

    def get_obs_with_intended_action():
        obs = original_get_obs()
        obs["dart_intended_action"] = task._dart_label.copy()
        return obs

    def take_picture_with_count():
        before = task.FRAME_IDX
        original_take_picture()
        if task.FRAME_IDX > before:
            stats["saved_frames"] += task.FRAME_IDX - before

    # Keep the label attached to every saved frame in the whole rollout,
    # including task-level snapshots taken between dense-action calls.
    task.get_obs = get_obs_with_intended_action
    task._take_picture = take_picture_with_count

    def noisy_dense_action(control_seq, save_freq=-1):
        left_arm, left_grip = control_seq["left_arm"], control_seq["left_gripper"]
        right_arm, right_grip = control_seq["right_arm"], control_seq["right_gripper"]
        arms = [left_arm, right_arm]
        max_len = max(
            [len(arm["position"]) for arm in arms if arm is not None]
            + [int(grip["num_step"]) for grip in (left_grip, right_grip) if grip is not None]
            + [0]
        )
        if max_len == 0:
            return original_take_dense_action(control_seq, save_freq=save_freq)

        save_freq = task.save_freq if save_freq == -1 else save_freq
        noise = (rng.multivariate_normal(np.zeros(12), covariance * scale,
                                        size=max_len, check_valid="raise")
                 if scale > 0 else np.zeros((max_len, 12), dtype=np.float64))
        modified = dict(control_seq)
        for name, section in (("left_arm", slice(0, 6)), ("right_arm", slice(6, 12))):
            arm = control_seq[name]
            if arm is None:
                continue
            clean = np.asarray(arm["position"], dtype=np.float64)
            draw = noise[:len(clean), section]
            changed = dict(arm)
            changed["position"] = clean + draw
            modified[name] = changed
            stats["noise_sq_sum"] += float(np.square(draw).sum())
            stats["noise_values"] += int(draw.size)
            stats["noise_abs_max"] = max(stats["noise_abs_max"], float(np.abs(draw).max(initial=0)))
            stats["noisy_joint_targets_outside_pi"] += int(np.count_nonzero(np.abs(changed["position"]) > math.pi))

        # Absolute joint targets make the following clean plan point a
        # restoring target after the preceding noisy command.
        if save_freq is not None:
            task._dart_label = make_action(task, control_seq, 0)
            task._take_picture()
        for idx in range(max_len):
            if left_arm is not None and idx < len(left_arm["position"]):
                task.robot.set_arm_joints(modified["left_arm"]["position"][idx],
                                          modified["left_arm"]["velocity"][idx], "left")
            if left_grip is not None and idx < left_grip["num_step"]:
                task.robot.set_gripper(left_grip["result"][idx], "left", left_grip["per_step"])
            if right_arm is not None and idx < len(right_arm["position"]):
                task.robot.set_arm_joints(modified["right_arm"]["position"][idx],
                                          modified["right_arm"]["velocity"][idx], "right")
            if right_grip is not None and idx < right_grip["num_step"]:
                task.robot.set_gripper(right_grip["result"][idx], "right", right_grip["per_step"])
            task.scene.step()
            if task.render_freq and idx % task.render_freq == 0:
                task._update_render()
                task._render_viewer_if_available()
            if save_freq is not None and idx % save_freq == 0:
                task._dart_label = make_action(task, control_seq, min(idx + 1, max_len - 1))
                task._take_picture()
        if save_freq is not None:
            task._dart_label = make_action(task, control_seq, max_len - 1)
            task._take_picture()
        stats["noisy_control_steps"] += max_len
        return True

    task.take_dense_action = noisy_dense_action
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--covariance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="handover_to_tray")
    parser.add_argument("--config", default="handover_to_tray_v2_promptfix")
    parser.add_argument("--seed-start", type=int, default=130000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()
    if cli.episodes < 1 or not cli.scales or any(value < 0 for value in cli.scales):
        raise ValueError("episodes and scales must be nonempty/nonnegative")

    root = cli.robotwin_root.resolve()
    sys.path.insert(0, str(root))
    from scripts.collect_data import class_decorator
    from scripts.smoke_handover_to_tray import load_args
    from envs.utils.pkl2hdf5 import load_pkl_file
    import h5py

    with np.load(cli.covariance, allow_pickle=False) as data:
        covariance = np.asarray(data["covariance"], dtype=np.float64)
    if covariance.shape != (12, 12) or not np.isfinite(covariance).all():
        raise ValueError("covariance must be finite [12,12]")
    if not np.allclose(covariance, covariance.T, atol=1e-10) or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")

    cli.output_dir = cli.output_dir.resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    config_record = {
        "algorithm": "DART_style_stage_plan_noise_with_clean_absolute_target_labels",
        "task": cli.task,
        "task_config": cli.config,
        "seed_start": cli.seed_start,
        "episodes_per_scale": cli.episodes,
        "scales": cli.scales,
        "covariance_trace_rad2": float(np.trace(covariance)),
        "execution_noise": "correlated Gaussian on executed 12 arm-joint targets; grippers unchanged",
        "label": "unperturbed absolute plan target at each sampled observation",
        "expert_feedback": "scripted planner replans at stage boundaries; not queried each simulator frame",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    config_path = cli.output_dir / "collector_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config_record and not cli.resume:
        raise FileExistsError(f"existing config differs: {config_path}")
    if not config_path.exists():
        config_path.write_text(json.dumps(config_record, indent=2) + "\n")

    base_args = load_args(cli.task, cli.config)
    for scale in cli.scales:
        scale_name = f"scale_{str(scale).replace('.', 'p')}"
        scale_dir = cli.output_dir / scale_name
        scale_dir.mkdir(parents=True, exist_ok=True)
        records_path = scale_dir / "episodes.jsonl"
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()] if records_path.exists() else []
        done = {int(row["seed"]) for row in records}
        if len(done) != len(records):
            raise ValueError(f"duplicate seed in {records_path}")
        for index in range(cli.episodes):
            seed = cli.seed_start + index
            if seed in done:
                if not cli.resume:
                    raise FileExistsError(f"seed {seed} already exists; use --resume")
                continue
            episode_dir = scale_dir / f"episode_{index:04d}"
            episode_dir.mkdir(parents=True, exist_ok=False)
            args = deepcopy(base_args)
            args.update(seed=seed, now_ep_num=index, need_plan=True,
                        save_data=True, save_video=False, collect_data=False,
                        render_freq=0, save_path=str(episode_dir), save_freq=15)
            task = class_decorator(cli.task)
            row = {"seed": seed, "scale": scale, "status": "started"}
            try:
                task.setup_demo(**args)
                stats = install_collector(task, covariance, scale, seed)
                info = task.play_once()
                legal, joint_absmax = task.planned_joints_legal()
                frame_count = int(task.FRAME_IDX)
                if frame_count >= 2:
                    task.merge_pkl_to_hdf5_video(instructions=["Move the bar from one arm to the other and place it in the tray."])
                    h5_path = episode_dir / "data" / f"episode_{index:07d}.hdf5"
                    pkls = [load_pkl_file(episode_dir / ".cache" / f"episode{index}" / f"{j}.pkl")
                            for j in range(frame_count)]
                    labels = np.stack([np.asarray(item["dart_intended_action"], dtype=np.float32) for item in pkls])
                    if labels.shape != (frame_count, 14) or not np.isfinite(labels).all():
                        raise ValueError(f"invalid clean label matrix {labels.shape}")
                    with h5py.File(h5_path, "r+") as h5:
                        action = h5["action"]
                        fields = {
                            "left_arm_joint_states": labels[:-1, :6],
                            "left_ee_joint_states": labels[:-1, 6:7],
                            "right_arm_joint_states": labels[:-1, 7:13],
                            "right_ee_joint_states": labels[:-1, 13:14],
                        }
                        for name, values in fields.items():
                            if name in action:
                                del action[name]
                            action.create_dataset(name, data=values)
                        h5["additional_info"].create_dataset("dart_scale", data=np.full(frame_count - 1, scale, np.float32))
                    row.update(status="completed", frames=frame_count,
                               plan_success=bool(task.plan_success),
                               task_success=bool(task.plan_success and task.check_success()),
                               joints_legal=bool(legal), joint_absmax=float(joint_absmax),
                               hdf5=str(h5_path), **stats)
                else:
                    row.update(status="no_training_frames", frames=frame_count, **stats)
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc(limit=8))
            finally:
                try:
                    task.close_env(clear_cache=False)
                except Exception as exc:
                    row["close_error"] = f"{type(exc).__name__}: {exc}"
            with records_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            print(json.dumps({"seed": seed, "scale": scale, "status": row["status"],
                              "task_success": row.get("task_success"), "frames": row.get("frames")}), flush=True)
            if row["status"] == "error":
                raise RuntimeError(f"DART data collection failed at seed={seed}, scale={scale}: {row['error']}")


if __name__ == "__main__":
    main()
