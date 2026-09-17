#!/usr/bin/env python3
"""Human-gated DAgger collection for RoboTwin handover_to_tray.

The policy controls the live scene until the supervisor presses ``i``.  The
remaining policy action chunk is discarded immediately, and the privileged
scripted expert replans from the current simulator state.  Only successful,
joint-legal expert recovery suffixes are written to the native RoboTwin HDF5
dataset.  Policy actions are never written as expert labels.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import traceback
import tty
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))
if str(ROBOTWIN_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT / "scripts"))

from eval_policy_xpolicylab import (  # noqa: E402
    build_policy_client,
    class_decorator,
    close_policy_client,
    is_episode_end,
    load_task_args,
    normalize_action_chunk,
    notify_trial_end,
    prepare_policy_case,
    reset_policy,
    robotwin_obs_to_xpolicylab,
    safe_close_env,
    xpolicylab_action_to_robotwin,
)


PROMPT = "Pass the red bar from the left arm to the right arm and place it in the blue tray."


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class HumanInterventionInput:
    """Poll SAPIEN's focused window and, optionally, the launching terminal."""

    def __init__(self, task_env, intervention_key: str = "i", quit_key: str = "q"):
        self.task_env = task_env
        self.intervention_key = intervention_key.lower()
        self.quit_key = quit_key.lower()
        self._fd: int | None = None
        self._term_attrs = None

    def __enter__(self):
        try:
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._term_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
        except (OSError, termios.error):
            self._fd = None
            self._term_attrs = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None and self._term_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._term_attrs)
            except (OSError, termios.error):
                pass

    def _poll_viewer(self) -> str | None:
        viewer = getattr(self.task_env, "viewer", None)
        window = getattr(viewer, "window", None)
        if window is None:
            return None
        for key, event in (
            (self.intervention_key, "intervene"),
            (self.quit_key, "quit"),
        ):
            try:
                if bool(window.key_press(key)) or bool(window.key_down(key)):
                    return event
            except (AttributeError, RuntimeError):
                continue
        return None

    def _poll_terminal(self) -> str | None:
        if self._fd is None:
            return None
        try:
            readable, _, _ = select.select([self._fd], [], [], 0.0)
            if not readable:
                return None
            data = os.read(self._fd, 64).decode("utf-8", errors="ignore").lower()
        except OSError:
            return None
        if self.quit_key in data:
            return "quit"
        if self.intervention_key in data:
            return "intervene"
        return None

    def poll(self) -> str | None:
        return self._poll_viewer() or self._poll_terminal()


def next_episode_index(output_dir: Path) -> int:
    indexes = []
    for path in (output_dir / "data").glob("episode_*.hdf5"):
        try:
            indexes.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(indexes, default=-1) + 1


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def save_successful_recovery(
    task_env,
    output_dir: Path,
    episode_index: int,
    instruction: str,
    record: dict[str, Any],
) -> Path:
    if int(task_env.FRAME_IDX) < 2:
        raise RuntimeError("Expert recovery produced fewer than two recorded frames.")
    task_env.merge_pkl_to_hdf5_video(instructions=[instruction])
    hdf5_path = output_dir / "data" / f"episode_{episode_index:07d}.hdf5"
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)

    instruction_payload = {
        "episode_index": episode_index,
        "seen": [instruction],
        "unseen": [instruction],
    }
    write_json(
        output_dir / "instruction" / f"episode_{episode_index:07d}.json",
        instruction_payload,
    )

    scene_info_path = output_dir / "scene_info.json"
    scene_info = read_json_object(scene_info_path)
    scene_info[f"episode_{episode_index}"] = task_env.info
    write_json(scene_info_path, scene_info)
    append_jsonl(output_dir / "interventions.jsonl", record)
    task_env.remove_data_cache()
    return hdf5_path


def discard_recovery_cache(task_env) -> None:
    cache = getattr(task_env, "folder_path", {}).get("cache")
    if cache and Path(cache).exists():
        task_env.remove_data_cache()


def render_initial_frame(task_env) -> None:
    viewer = getattr(task_env, "viewer", None)
    if viewer is None:
        raise RuntimeError(
            "Human-gated mode needs a SAPIEN viewer. Set --render-freq to a positive value "
            "and launch inside the 5090 desktop session."
        )
    task_env._update_render()
    viewer.render()


def run_intervention_expert(
    task_env,
    *,
    save_data: bool,
    output_dir: Path,
    episode_index: int,
    instruction: str,
    seed: int,
    policy_steps: int,
) -> tuple[dict[str, Any], Path | None]:
    inferred = task_env.infer_recovery_state()
    task_env.current_stage_id = int(inferred["stage_id"])
    task_env.save_data = bool(save_data)
    task_env.save_dir = str(output_dir)
    task_env.save_video = False
    task_env.FRAME_IDX = 0
    task_env.folder_path = {}
    task_env.ep_num = int(episode_index)

    print("\n\033[93m[HG-DAGGER] i detected: policy chunk discarded.\033[0m")
    print(
        "\033[93m[HG-DAGGER] Expert takeover: "
        f"branch={inferred['branch']} stage={inferred['stage_id']} "
        f"source_holds={inferred['source_holds']} receiver_holds={inferred['receiver_holds']}\033[0m"
    )

    if save_data:
        task_env._take_picture()
    recovery = task_env.recover_from_current_state()
    if save_data:
        task_env.current_stage_id = max(
            1,
            min(6, int(task_env.current_stage_id)),
        )
        task_env._take_picture()

    joints_legal, joint_absmax = task_env.planned_joints_legal()
    accepted = bool(recovery.get("success") and recovery.get("plan_success") and joints_legal)
    record = {
        "episode_index": int(episode_index),
        "seed": int(seed),
        "instruction": instruction,
        "policy_steps_before_intervention": int(policy_steps),
        "intervention_key": "i",
        "intervened": True,
        "policy_chunk_discarded": True,
        "expert_invoked": True,
        "expert_recovery": recovery,
        "joints_legal": bool(joints_legal),
        "joint_absmax": float(joint_absmax),
        "accepted_for_training": accepted,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    task_env.episode_metadata["hg_dagger"] = record
    task_env.info["task_metadata"] = task_env.episode_metadata
    task_env.info["hg_dagger"] = record

    hdf5_path = None
    if save_data and accepted:
        hdf5_path = save_successful_recovery(
            task_env,
            output_dir,
            episode_index,
            instruction,
            record,
        )
    elif save_data:
        discard_recovery_cache(task_env)
        append_jsonl(output_dir / "rejected_interventions.jsonl", record)
    return record, hdf5_path


def build_runtime_args(cli: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    user_args: dict[str, Any] = {
        "task_name": "handover_to_tray",
        "task_config": cli.task_config,
        "policy_name": cli.policy_name,
        "host": cli.host,
        "port": int(cli.port),
        "protocol": "ws",
        "xpolicylab_root": str(ROBOTWIN_ROOT / "XPolicyLab"),
        "ckpt_name": cli.ckpt_name,
        "ckpt_setting": cli.ckpt_name,
        "action_type": "joint",
        "evaluation_id": f"hg-dagger-handover-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "trial_id": "hg-dagger-handover",
        "action_case_id": "hg-dagger-handover-v2",
    }
    args, _ = load_task_args(user_args)
    args["eval_instruction"] = "seen"
    args["eval_mode"] = True
    args["render_freq"] = int(cli.render_freq)
    args["need_plan"] = True
    args["save_data"] = False
    args["save_video"] = False
    args["save_path"] = str(cli.output_dir)
    args["save_freq"] = int(cli.save_freq)
    args["render_max_num_materials"] = int(cli.render_max_num_materials)
    args["render_max_num_textures"] = int(cli.render_max_num_textures)
    user_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    user_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    return user_args, args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Human-gated DAgger for RoboTwin handover_to_tray (press i to intervene)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18300)
    parser.add_argument("--policy-name", default="Pi_05")
    parser.add_argument("--ckpt-name", default="v2_promptfix_9999")
    parser.add_argument("--task-config", default="handover_to_tray_v2_promptfix")
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--render-freq", type=int, default=5)
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument(
        "--render-max-num-materials",
        type=int,
        default=128,
        help="SAPIEN material capacity; the upstream SAPIEN default is ample here.",
    )
    parser.add_argument(
        "--render-max-num-textures",
        type=int,
        default=512,
        help="SAPIEN texture capacity; the upstream SAPIEN default is ample here.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROBOTWIN_ROOT
        / "data"
        / "handover_to_tray_hg_dagger_r1"
        / "handover_to_tray"
        / "aloha_agilex",
    )
    parser.add_argument("--save-data", default="true")
    parser.add_argument(
        "--acceptance",
        action="store_true",
        help="Run one episode and exit nonzero unless i is detected and expert recovery succeeds.",
    )
    parser.add_argument(
        "--auto-intervene-step",
        type=int,
        default=-1,
        help="Test-only hook; -1 requires a real i key, 0 intervenes before the first policy action.",
    )
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    if cli.render_freq <= 0:
        raise ValueError("--render-freq must be positive for human supervision")
    if cli.acceptance:
        cli.episodes = 1
        cli.save_data = "false"
    requested_seeds = range(int(cli.seed_start), int(cli.seed_start) + int(cli.episodes))
    if any(31000 <= seed <= 31103 for seed in requested_seeds):
        raise ValueError(
            "Refusing to collect or tune on the frozen 31000--31103 test range. "
            "Use collection seeds such as 40000+."
        )
    cli.output_dir = cli.output_dir.expanduser().resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    user_args, args = build_runtime_args(cli)
    task_env = class_decorator("handover_to_tray")
    model_client = build_policy_client(user_args)
    save_data = parse_bool(cli.save_data)
    data_episode_index = next_episode_index(cli.output_dir)
    records: list[dict[str, Any]] = []
    aborted = False

    print("\n" + "=" * 72)
    print("RoboTwin handover_to_tray human-gated DAgger")
    print("Focus the SAPIEN window or this terminal, then press i to intervene.")
    print("Press q to abort the current session.")
    print(f"Prompt: {PROMPT}")
    print(f"Output: {cli.output_dir}")
    print("=" * 72 + "\n")

    try:
        for rollout_index in range(int(cli.episodes)):
            seed = int(cli.seed_start) + rollout_index
            args["save_data"] = False
            args["need_plan"] = True
            args["eval_mode"] = True
            task_env.setup_demo(
                now_ep_num=data_episode_index,
                seed=seed,
                is_test=False,
                **args,
            )
            task_env.set_instruction(PROMPT)
            render_initial_frame(task_env)
            prepare_policy_case(model_client, "handover_to_tray", seed, PROMPT, "joint")
            reset_policy(model_client)

            policy_steps = 0
            intervention_requested = cli.auto_intervene_step == 0
            quit_requested = False
            autonomous_success = False

            print(
                f"\n\033[96m[ROLLOUT {rollout_index + 1}/{cli.episodes}] seed={seed}; "
                "press i before an unrecoverable failure.\033[0m"
            )

            with HumanInterventionInput(task_env) as keyboard:
                while not is_episode_end(task_env) and not intervention_requested:
                    event = keyboard.poll()
                    if event == "quit":
                        quit_requested = True
                        break
                    if event == "intervene":
                        intervention_requested = True
                        break

                    observation = task_env.get_obs()
                    xpl_obs = robotwin_obs_to_xpolicylab(
                        observation,
                        instruction=PROMPT,
                        env_idx=0,
                        frequency=int(cli.frequency),
                        task_env=task_env,
                    )
                    model_client.call(func_name="update_obs", obs=xpl_obs)
                    action_chunk = normalize_action_chunk(model_client.call(func_name="get_action"))
                    if not action_chunk:
                        raise RuntimeError("Policy returned an empty action chunk.")

                    for action in action_chunk:
                        event = keyboard.poll()
                        if event == "quit":
                            quit_requested = True
                            break
                        if event == "intervene":
                            intervention_requested = True
                            break

                        flat_action, robotwin_action_type = xpolicylab_action_to_robotwin(
                            action,
                            action_type="joint",
                            current_observation=observation,
                        )
                        task_env.take_action(flat_action, action_type=robotwin_action_type)
                        policy_steps += 1

                        if task_env.eval_success:
                            autonomous_success = True
                            break
                        if is_episode_end(task_env):
                            break
                        if (
                            cli.auto_intervene_step >= 0
                            and policy_steps >= cli.auto_intervene_step
                        ):
                            intervention_requested = True
                            break

                        event = keyboard.poll()
                        if event == "quit":
                            quit_requested = True
                            break
                        if event == "intervene":
                            intervention_requested = True
                            break

                        observation = task_env.get_obs()
                        xpl_obs = robotwin_obs_to_xpolicylab(
                            observation,
                            instruction=PROMPT,
                            env_idx=0,
                            frequency=int(cli.frequency),
                            task_env=task_env,
                        )
                        model_client.call(func_name="update_obs", obs=xpl_obs)

                    if quit_requested or intervention_requested or autonomous_success:
                        break

            if quit_requested:
                aborted = True
                notify_trial_end(model_client, "handover_to_tray", seed, False)
                safe_close_env(task_env)
                break

            if intervention_requested:
                record, hdf5_path = run_intervention_expert(
                    task_env,
                    save_data=save_data,
                    output_dir=cli.output_dir,
                    episode_index=data_episode_index,
                    instruction=PROMPT,
                    seed=seed,
                    policy_steps=policy_steps,
                )
                record["rollout_index"] = rollout_index
                record["autonomous_success"] = False
                records.append(record)
                accepted = bool(record["accepted_for_training"])
                notify_trial_end(model_client, "handover_to_tray", seed, accepted)
                if accepted:
                    print("\033[92m[HG-DAGGER] Expert recovery SUCCESS.\033[0m")
                    if hdf5_path is not None:
                        print(f"\033[92m[HG-DAGGER] Saved {hdf5_path}\033[0m")
                        data_episode_index += 1
                else:
                    print("\033[91m[HG-DAGGER] Expert recovery FAILED; data rejected.\033[0m")
            else:
                record = {
                    "rollout_index": rollout_index,
                    "seed": seed,
                    "instruction": PROMPT,
                    "policy_steps_before_intervention": policy_steps,
                    "intervened": False,
                    "autonomous_success": bool(autonomous_success),
                    "accepted_for_training": False,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                records.append(record)
                notify_trial_end(
                    model_client,
                    "handover_to_tray",
                    seed,
                    autonomous_success,
                )
                label = "SUCCESS" if autonomous_success else "FAIL"
                print(f"[POLICY] Autonomous rollout {label}; no expert data saved.")

            safe_close_env(task_env)

            if cli.acceptance:
                last_record = records[-1]
                expert_recovery = last_record.get("expert_recovery", {})
                checks = {
                    "intervention_detected": bool(intervention_requested),
                    "policy_chunk_discarded": bool(
                        last_record.get("policy_chunk_discarded")
                    ),
                    "expert_invoked": bool(last_record.get("expert_invoked")),
                    "plan_success": bool(expert_recovery.get("plan_success")),
                    "task_success": bool(expert_recovery.get("success")),
                    "joints_legal": bool(last_record.get("joints_legal")),
                }
                passed = all(checks.values())
                report = {
                    "acceptance_passed": passed,
                    "criterion": "all takeover and expert-recovery checks are true",
                    "checks": checks,
                    "record": last_record,
                }
                report_path = cli.output_dir / "acceptance_report.json"
                write_json(report_path, report)
                if passed:
                    print("\n\033[92m========== HG-DAGGER ACCEPTANCE: PASS ==========\033[0m")
                    print(f"report={report_path}")
                    return 0
                print("\n\033[91m========== HG-DAGGER ACCEPTANCE: FAIL ==========\033[0m")
                print(f"report={report_path}")
                return 2
    except Exception:
        print("\n\033[91mHG-DAgger session error:\033[0m")
        print(traceback.format_exc())
        safe_close_env(task_env, clear_cache=True)
        return 3
    finally:
        close_policy_client(model_client)

    session_report = {
        "aborted": aborted,
        "rollouts": len(records),
        "interventions": sum("expert_recovery" in item for item in records),
        "accepted_recoveries": sum(bool(item.get("accepted_for_training")) for item in records),
        "autonomous_successes": sum(bool(item.get("autonomous_success")) for item in records),
        "records": records,
    }
    report_path = cli.output_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(report_path, session_report)
    print(f"session_report={report_path}")
    return 130 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
