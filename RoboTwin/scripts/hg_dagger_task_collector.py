#!/usr/bin/env python3
"""Shared, fail-closed HG-DAgger runtime for task-specific adapters.

Adapters supply only the task prompt, live-state snapshot/classifier, expert
suffix, and quality predicate. This runtime owns policy chunk interruption,
human i/r/x/q controls, segment/frame labels and raw episode persistence.
Automatic keys are diagnostic-only and can never be accepted as training data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("ROBOTWIN_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

TASKS = {
    "place_can_basket": {
        "prompt": "use the arm to grasp the can and put it in the basket, then lift the basket with the other arm",
        "step_limit": 700,
        "adapter_module": "hg_dagger_task_adapters",
    },
    "rotate_qrcode": {
        "prompt": "Use arm to catch the qrcode board on the table, pick it up and rotate to let the qrcode face towards you",
        "step_limit": 400,
        "adapter_module": "hg_dagger_task_adapters",
    },
    "pick_dual_bottles": {
        "prompt": "Use both arms to pick up bottle1 and bottle2 simultaneously and move them to the front target locations, bottle1 on the left and bottle2 on the right. Keep holding both bottles.",
        "step_limit": 400,
        "adapter_module": "hg_dagger_task_adapters",
    },
    "put_bottles_dustbin": {
        "prompt": "Place all three bottles into the dustbin to the left of the table.",
        "step_limit": 1700,
        "adapter_module": "hg_dagger_task_adapters",
    },
    "stack_blocks_two": {
        "prompt": "Move block1 to the target position, then place block2 on top of block1.",
        "step_limit": 800,
        "adapter_module": "hg_dagger_task_adapters",
    },
}


def parse_args(task_name: str, argv: list[str] | None = None) -> argparse.Namespace:
    if task_name not in TASKS:
        raise ValueError(f"unsupported task: {task_name}")
    parser = argparse.ArgumentParser(description=f"HG-DAgger collector for {task_name}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--target-saved", type=int, default=30)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--step-limit", type=int, default=TASKS[task_name]["step_limit"])
    parser.add_argument("--frequency", type=int, default=50)
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument("--render-freq", type=int, default=5)
    parser.add_argument(
        "--instruction", default=None,
        help="optional exact task instruction (useful to reproduce a benchmark scene)",
    )
    parser.add_argument("--headless-smoke", action="store_true",
                        help="one frozen-seed diagnostic rollout; never training data")
    parser.add_argument("--auto-intervene-step", type=int, default=-1)
    parser.add_argument("--auto-handback-stage-actions", type=int, default=-1)
    parser.add_argument("--auto-abort-step", type=int, default=-1)
    cli = parser.parse_args(argv)
    if cli.instruction is not None:
        cli.instruction = cli.instruction.strip()
        if not cli.instruction:
            parser.error("instruction must not be empty")
    if min(cli.episodes, cli.target_saved, cli.step_limit, cli.frequency, cli.save_freq) < 1:
        parser.error("episodes, target-saved, step-limit, frequency and save-freq must be positive")
    if cli.seed_start < 40000 and not cli.headless_smoke:
        parser.error("production collection must use fresh seeds >=40000")
    auto = (cli.auto_intervene_step, cli.auto_handback_stage_actions, cli.auto_abort_step)
    if any(value >= 0 for value in auto) and not cli.headless_smoke:
        parser.error("automatic i/r/x hooks are restricted to --headless-smoke")
    if any(value == 0 or value < -1 for value in auto):
        parser.error("automatic i/r/x trigger steps must be positive or disabled (-1)")
    if cli.auto_handback_stage_actions > 0 and cli.auto_intervene_step < 1:
        parser.error("automatic r requires a positive automatic i trigger")
    if cli.headless_smoke and (cli.episodes != 1 or cli.target_saved != 1):
        parser.error("headless smoke is exactly one diagnostic rollout")
    return cli


def _load_adapter(task_name: str):
    from hg_dagger_task_adapters import get_adapter
    return get_adapter(task_name)


def _auto_trigger_due(step: int, trigger_step: int, already_fired: bool) -> bool:
    """Return whether a smoke-only automatic key should fire exactly once."""
    return bool(trigger_step > 0 and not already_fired and step >= trigger_step)


def _build_runtime(task_name: str, cli: argparse.Namespace, adapter):
    from eval_policy_xpolicylab import load_task_args

    user = {
        "task_name": task_name, "task_config": "demo_clean",
        "policy_name": "Pi_05_RobotTwin", "host": cli.host, "port": cli.port,
        "protocol": "ws", "xpolicylab_root": str(ROOT / "XPolicyLab"),
        "ckpt_name": f"public_pi05_{task_name}",
        "ckpt_setting": f"public_pi05_{task_name}", "action_type": "joint",
        "evaluation_id": f"hg-dagger-{task_name}",
        "trial_id": f"hg-dagger-{task_name}",
        "action_case_id": f"hg-dagger-{task_name}",
    }
    args, _ = load_task_args(user)
    args.update({
        "eval_instruction": "seen", "eval_mode": True,
        "render_freq": 0 if cli.headless_smoke else cli.render_freq,
        "need_plan": True, "save_data": False, "save_video": False,
        "save_path": str(cli.output_dir), "save_freq": cli.save_freq,
    })
    user["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    user["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    return user, args


def openpi_observation_from_robotwin(
    observation: dict[str, Any], instruction: str,
) -> dict[str, Any]:
    """Encode observations exactly as the fixed-H50 OpenPI evaluator does."""
    import numpy as np

    cameras = observation["observation"]
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"expected a finite 14D joint state, got {state.shape}")
    images = {
        "cam_high": np.asarray(cameras["head_camera"]["rgb"], dtype=np.uint8).transpose(2, 0, 1),
        "cam_left_wrist": np.asarray(cameras["left_camera"]["rgb"], dtype=np.uint8).transpose(2, 0, 1),
        "cam_right_wrist": np.asarray(cameras["right_camera"]["rgb"], dtype=np.uint8).transpose(2, 0, 1),
    }
    if any(image.ndim != 3 or image.shape[0] != 3 for image in images.values()):
        raise ValueError("expected three CHW RGB camera images")
    return {"state": state, "images": images, "prompt": instruction}


def build_openpi_policy_client(host: str, port: int, task_name: str):
    """Use native OpenPI websocket transport and verify task/action metadata."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(host, port)
    try:
        metadata = client.get_server_metadata()
        expected = {
            "task": task_name,
            "config": f"pi05_robotwin_{task_name}",
            "action_horizon": 50,
            "action_dim": 14,
            "action_space": "absolute_joint",
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"OpenPI server metadata mismatch: {mismatches}")
        return client, metadata
    except BaseException:
        close = getattr(client, "close", None)
        if callable(close):
            close()
        raise


def close_openpi_policy_client(client) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def run_episode(task_name: str, cli: argparse.Namespace, adapter, client,
                policy_metadata: dict[str, Any], args: dict, seed: int,
                rollout_index: int, saved_index: int) -> tuple[dict[str, Any], bool]:
    from eval_policy_xpolicylab import (
        class_decorator, is_episode_end, normalize_action_chunk,
        safe_close_env, xpolicylab_action_to_robotwin,
    )
    from hg_dagger_handover import (
        HumanInterventionInput, append_jsonl, append_segment, finalize_episode,
        render_initial_frame,
    )

    task_env = class_decorator(task_name)
    saved = False
    started = time.time()
    try:
        task_env.setup_demo(now_ep_num=saved_index, seed=seed, is_test=False, **args)
        task_env.step_lim = cli.step_limit
        prompt = cli.instruction or TASKS[task_name]["prompt"]
        task_env.set_instruction(prompt)
        if not cli.headless_smoke:
            if getattr(task_env, "viewer", None) is None and not sys.stdin.isatty():
                raise RuntimeError("interactive HIL needs a focused viewer or terminal")
            if getattr(task_env, "viewer", None) is not None:
                render_initial_frame(task_env)
        task_env.save_data = True
        task_env.save_video = False
        task_env.save_dir = str(cli.output_dir)
        task_env.save_freq = cli.save_freq
        task_env.FRAME_IDX = 0
        task_env.folder_path = {}
        task_env.control_mask = []
        task_env.current_control_source = "policy"
        task_env.current_stage_id = 0
        task_env.episode_metadata = {
            "task": task_name, "seed": seed, "instruction": prompt,
            "policy": "published task-specific pi0.5", "expert_prefilter": False,
            "action_type": "absolute_joint", "frequency_field": cli.frequency,
            "generated_horizon": 50, "execute_horizon": 50,
            "save_freq": cli.save_freq,
            "policy_server_metadata": policy_metadata,
        }
        task_env._take_picture()

        source, source_start = "policy", int(task_env.FRAME_IDX)
        segments: list[dict] = []
        requests: list[dict] = []
        interventions: list[dict] = []
        policy_steps = 0
        expert_actions = 0
        mode = "policy"
        recovery = None
        previous_snapshot = adapter.snapshot(task_env, policy_steps)
        aborted = quit_requested = expert_failed = False
        auto_intervene_fired = False
        auto_handback_fired = False
        auto_abort_fired = False

        def switch_source(new_source: str) -> None:
            nonlocal source, source_start
            append_segment(segments, source, source_start, int(task_env.FRAME_IDX))
            source, source_start = new_source, int(task_env.FRAME_IDX)

        def request_takeover() -> None:
            nonlocal mode, recovery, expert_actions, previous_snapshot
            current = adapter.snapshot(task_env, policy_steps)
            decision = adapter.classify(current, previous_snapshot)
            requests.append({"policy_steps": policy_steps, "snapshot": current,
                             "decision": decision})
            if not decision.get("ready"):
                print(f"[HIL] takeover rejected: {decision.get('reason')}", flush=True)
                previous_snapshot = current
                return
            print(f"[HG-DAgger] stage map: {adapter.stage_labels}", flush=True)
            recovery = adapter.recovery(task_env, decision)
            expert_actions = 0
            interventions.append({
                "policy_steps_at_takeover": policy_steps,
                "decision": decision, "snapshot": current,
                "stage_events": [], "handback": False,
            })
            switch_source("hil")
            mode = "expert"
            print(f"[HIL] i accepted: {decision['branch']}; current chunk discarded", flush=True)

        with HumanInterventionInput(task_env, {
            "i": "intervene", "r": "handback", "x": "abort", "q": "quit",
        }) as keyboard:
            while not is_episode_end(task_env):
                event = keyboard.poll()
                if event == "quit":
                    quit_requested = True
                    break
                if event == "abort":
                    aborted = True
                    break

                if mode == "policy":
                    if _auto_trigger_due(policy_steps, cli.auto_intervene_step,
                                         auto_intervene_fired):
                        event = "intervene"
                        auto_intervene_fired = True
                    if event == "intervene":
                        request_takeover()
                        if mode == "expert":
                            continue
                    observation = task_env.get_obs()
                    chunk = normalize_action_chunk(client.infer(
                        openpi_observation_from_robotwin(observation, prompt)
                    ))
                    if len(chunk) != 50:
                        raise RuntimeError(f"expected a 50-action chunk, got {len(chunk)}")
                    interrupted = False
                    for action in chunk:
                        event = keyboard.poll()
                        if _auto_trigger_due(policy_steps, cli.auto_intervene_step,
                                             auto_intervene_fired):
                            event = "intervene"
                            auto_intervene_fired = True
                        if _auto_trigger_due(policy_steps, cli.auto_abort_step,
                                             auto_abort_fired):
                            event = "abort"
                            auto_abort_fired = True
                        if event in {"quit", "abort", "intervene"}:
                            if event == "quit":
                                quit_requested = True
                            elif event == "abort":
                                aborted = True
                            else:
                                request_takeover()
                            interrupted = True
                            break
                        flat, action_type = xpolicylab_action_to_robotwin(
                            action, action_type="joint", current_observation=observation,
                        )
                        task_env.current_control_source = "policy"
                        task_env.current_stage_id = 0
                        task_env.take_action(flat, action_type=action_type)
                        policy_steps += 1
                        if policy_steps % cli.save_freq == 0:
                            task_env._take_picture()
                        if is_episode_end(task_env):
                            break
                        observation = task_env.get_obs()
                    if quit_requested or aborted:
                        break
                    if not interrupted:
                        previous_snapshot = adapter.snapshot(task_env, policy_steps)
                else:
                    if _auto_trigger_due(expert_actions, cli.auto_handback_stage_actions,
                                         auto_handback_fired):
                        event = "handback"
                        auto_handback_fired = True
                    if event == "handback":
                        current = adapter.snapshot(task_env, policy_steps)
                        decision = adapter.classify(current, previous_snapshot)
                        if (decision.get("ready") and task_env.plan_success
                                and not adapter.quality(task_env, current)["strict_success"]):
                            interventions[-1]["handback"] = True
                            interventions[-1]["post_expert_snapshot"] = current
                            interventions[-1]["post_expert_decision"] = decision
                            recovery.close()
                            switch_source("policy")
                            task_env.current_control_source = "policy"
                            task_env.current_stage_id = 0
                            mode = "policy"
                            previous_snapshot = current
                            continue
                        print(f"[HIL] handback denied: {decision.get('reason', 'state_not_safe')}",
                              flush=True)
                    event = next(recovery, None)
                    if event is None:
                        expert_failed = True
                        break
                    interventions[-1]["stage_events"].append(event)
                    if event["phase"] == "done":
                        interventions[-1]["expert_result"] = event["result"]
                        expert_failed = not bool(event["result"].get("success"))
                        if expert_failed:
                            break
                        # A completed task is terminal; otherwise wait for r or
                        # continue the remaining staged suffix as designed.
                        break
                    expert_actions += 1
                    if not event.get("plan_success"):
                        expert_failed = True
                        break

        append_segment(segments, source, source_start, int(task_env.FRAME_IDX))
        final_snapshot = adapter.snapshot(task_env, policy_steps)
        quality = adapter.quality(task_env, final_snapshot)
        official_success = bool(task_env.check_success())
        joints_legal, joint_absmax = task_env.planned_joints_legal()
        hil_frames = task_env.control_mask.count("hil")
        valid_hil = bool(
            not aborted and not quit_requested and interventions and hil_frames > 0
            and not expert_failed and quality["strict_success"] and joints_legal
        )
        accepted = bool(valid_hil and not cli.headless_smoke)
        task_env.episode_metadata["diagnostic_only"] = bool(cli.headless_smoke)
        task_env.episode_metadata["accepted_for_training"] = accepted
        record = {
            "task": task_name, "seed": seed, "rollout_index": rollout_index,
            "policy_server_metadata": policy_metadata,
            "policy_steps": policy_steps, "intervention_requests": requests,
            "interventions": interventions, "segments": segments,
            "control_mask": list(task_env.control_mask), "hil_frames": hil_frames,
            "official_task_success": official_success, **quality,
            "final_snapshot": final_snapshot,
            "joints_legal": bool(joints_legal), "joint_absmax": float(joint_absmax),
            "aborted": aborted, "quit_requested": quit_requested,
            "expert_failed": expert_failed, "valid_hil": valid_hil,
            "diagnostic_only": bool(cli.headless_smoke),
            "accepted_for_training": accepted,
            "rollout_seconds": round(time.time() - started, 2),
        }
        raw_dir = finalize_episode(
            task_env, cli.output_dir, saved_index, prompt,
            save=valid_hil, supervisor_label="success" if official_success else "failure",
            segments=segments, extra=record,
        )
        saved = raw_dir is not None
        record["raw_dir"] = str(raw_dir) if raw_dir else None
        if raw_dir is not None and cli.headless_smoke:
            (raw_dir / "DIAGNOSTIC_ONLY.json").write_text(json.dumps({
                "diagnostic_only": True, "accepted_for_training": False,
                "reason": "frozen-seed HIL collector smoke; never training data",
            }, indent=2) + "\n")
        append_jsonl(cli.output_dir / "rollouts.jsonl", record)
        return record, quit_requested
    finally:
        safe_close_env(task_env, clear_cache=not saved)


def main_for_task(task_name: str, argv: list[str] | None = None) -> int:
    cli = parse_args(task_name, argv)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    if (cli.output_dir / "rollouts.jsonl").exists() or (cli.output_dir / "session.json").exists():
        raise FileExistsError("refusing to overwrite or resume a prior collection directory")
    adapter = _load_adapter(task_name)
    user, args = _build_runtime(task_name, cli, adapter)
    client, policy_metadata = build_openpi_policy_client(cli.host, cli.port, task_name)
    completed = saved = 0
    try:
        for index in range(cli.episodes):
            if saved >= cli.target_saved:
                break
            seed = cli.seed_start + index
            record, quit_requested = run_episode(
                task_name, cli, adapter, client, policy_metadata, args,
                seed, index, saved,
            )
            completed += 1
            saved += int(record["valid_hil"])
            print(f"[HG-DAgger] {task_name} seed={seed} valid={record['valid_hil']} "
                  f"saved={saved}/{cli.target_saved}", flush=True)
            if quit_requested:
                break
    finally:
        close_openpi_policy_client(client)
    summary = {
        "task": task_name, "rollouts": completed, "saved_valid_hil": saved,
        "target_saved": cli.target_saved, "complete": saved >= cli.target_saved,
        "headless_smoke": cli.headless_smoke,
        "policy": "published task-specific pi0.5", "seed_start": cli.seed_start,
        "policy_server_metadata": policy_metadata,
        "expert_prefilter": False,
    }
    (cli.output_dir / "session.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0
