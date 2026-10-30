#!/usr/bin/env python3
"""Evaluate one published RoboTwin pi0.5 task checkpoint on one raw seed.

No scripted expert runs before the policy.  A simulator or server exception is
an infrastructure error, not a failed policy rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", str(SCRIPT_DIR.parent))).resolve()
sys.path.insert(0, str(ROBOTWIN_ROOT / "scripts"))

from eval_policy_xpolicylab import (  # noqa: E402
    build_policy_client,
    class_decorator,
    close_policy_client,
    is_episode_end,
    load_task_args,
    normalize_action_chunk,
    prepare_policy_case,
    reset_policy,
    robotwin_obs_to_xpolicylab,
    safe_close_env,
    xpolicylab_action_to_robotwin,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--frequency", type=int, default=50)
    parser.add_argument("--execute-horizon", type=int, choices=(10, 50), default=50)
    parser.add_argument("--step-limit", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-failure-video", action="store_true",
                        help="Diagnostic rerun only: save camera video for a failed policy rollout")
    parser.add_argument("--record-all-video", action="store_true",
                        help="Diagnostic rerun only: save camera video regardless of rollout outcome")
    parser.add_argument("--trace-handover-state", action="store_true",
                        help="Write privileged box/target/arm state at each action-chunk boundary")
    parser.add_argument("--trace-task-state", action="store_true",
                        help="Diagnostic-only chunk-boundary state trace for basket or can-pot")
    parser.add_argument("--recover-at-step", type=int, default=None,
                        help="Diagnostic only: stop policy at a chunk boundary and run expert suffix")
    parser.add_argument("--handover-handback-stage-actions", type=int, default=None,
                        help="Diagnostic only: execute this many expert stage actions, then resume the policy")
    parser.add_argument("--task-state-handback-stage-actions", type=int, default=None,
                        help="Diagnostic-only move_can_pot expert stages before policy handback")
    parser.add_argument("--trace-burger-final", action="store_true",
                        help="Diagnostic only: capture physical placement at termination and after settling")
    parser.add_argument("--burger-settle-steps", type=int, default=250,
                        help="Additional physics-only scene steps after burger diagnostic termination")
    return parser.parse_args()


def burger_snapshot(task_env) -> dict:
    """Privileged physical readout, never used by the policy or success-rate loop."""
    tray, burger, fries = task_env.tray, task_env.hamburg, task_env.frenchfries
    tray_left = [float(v) for v in tray.get_functional_point(0, "pose").p]
    tray_right = [float(v) for v in tray.get_functional_point(1, "pose").p]
    burger_point = [float(v) for v in burger.get_functional_point(0, "pose").p]
    fries_point = [float(v) for v in fries.get_functional_point(0, "pose").p]
    contact_pairs = set()
    for contact in task_env.scene.get_contacts():
        try:
            contact_pairs.add(frozenset((
                contact.bodies[0].entity.name, contact.bodies[1].entity.name
            )))
        except (AttributeError, IndexError):
            continue
    tray_name = tray.get_name()
    return {
        "tray_name": tray_name,
        "burger_name": burger.get_name(),
        "fries_name": fries.get_name(),
        "tray_left_point": tray_left,
        "tray_right_point": tray_right,
        "burger_point": burger_point,
        "fries_point": fries_point,
        "burger_minus_tray_z_m": burger_point[2] - tray_left[2],
        "fries_minus_tray_z_m": fries_point[2] - tray_right[2],
        "burger_tray_contact": frozenset((tray_name, burger.get_name())) in contact_pairs,
        "fries_tray_contact": frozenset((tray_name, fries.get_name())) in contact_pairs,
        "left_gripper_open": bool(task_env.is_left_gripper_open()),
        "right_gripper_open": bool(task_env.is_right_gripper_open()),
        "original_check_success": bool(task_env.check_success()),
    }


def main() -> int:
    cli = parse_args()
    if cli.trace_handover_state and cli.task_name != "handover_block":
        raise ValueError("--trace-handover-state is only valid for handover_block")
    if cli.trace_task_state and cli.task_name not in ("place_can_basket", "move_can_pot"):
        raise ValueError("--trace-task-state supports only basket and can-pot")
    if cli.trace_task_state and cli.trace_handover_state:
        raise ValueError("select only one task state trace")
    if cli.recover_at_step is not None and (
        not (cli.trace_handover_state or cli.trace_task_state) or cli.recover_at_step <= 0
        or cli.recover_at_step % cli.execute_horizon != 0
    ):
        raise ValueError("--recover-at-step requires a state trace and a positive chunk boundary")
    if cli.handover_handback_stage_actions is not None and (
        not cli.trace_handover_state or cli.recover_at_step is None
        or cli.handover_handback_stage_actions <= 0
    ):
        raise ValueError("--handover-handback-stage-actions requires recovery and a positive count")
    if cli.task_state_handback_stage_actions is not None and (
        cli.task_name != "move_can_pot" or not cli.trace_task_state
        or cli.recover_at_step is None or cli.task_state_handback_stage_actions <= 0
    ):
        raise ValueError("--task-state-handback-stage-actions is diagnostic move_can_pot recovery only")
    if cli.trace_burger_final and cli.task_name != "place_burger_fries":
        raise ValueError("--trace-burger-final is only valid for place_burger_fries")
    if cli.burger_settle_steps < 0:
        raise ValueError("--burger-settle-steps must be nonnegative")
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    record_path = cli.output_dir / "eval_records.jsonl"
    if record_path.exists():
        raise RuntimeError(f"refusing to overwrite existing result: {record_path}")

    task_config = ROBOTWIN_ROOT / "env_cfg/task_config/demo_clean.yml"
    if not task_config.is_file():
        raise RuntimeError(f"missing demo_clean task config: {task_config}")

    user_args = {
        "task_name": cli.task_name,
        "task_config": "demo_clean",
        "policy_name": "Pi_05_RobotTwin",
        "host": cli.host,
        "port": cli.port,
        "protocol": "ws",
        "xpolicylab_root": str(ROBOTWIN_ROOT / "XPolicyLab"),
        "ckpt_name": f"public_pi05_{cli.task_name}",
        "ckpt_setting": f"public_pi05_{cli.task_name}",
        "action_type": "joint",
        "evaluation_id": f"candidate-{cli.task_name}-{cli.seed}",
        "trial_id": f"candidate-{cli.seed}",
        "action_case_id": f"candidate-{cli.task_name}-{cli.seed}",
    }
    args, _ = load_task_args(user_args)
    args.update({
        "eval_instruction": "seen",
        "eval_mode": True,
        "render_freq": 0,
        "need_plan": True,
        "save_data": False,
        "save_video": False,
        "save_path": str(cli.output_dir),
    })
    user_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    user_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    task_env = class_decorator(cli.task_name)
    model_client = build_policy_client(user_args)
    try:
        task_env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=False, **args)
        task_env.step_lim = cli.step_limit
        task_env.set_instruction(cli.prompt)
        if hasattr(task_env, "_update_render"):
            task_env._update_render()
        viewer = getattr(task_env, "viewer", None)
        if viewer is not None:
            viewer.render()
        prepare_policy_case(model_client, cli.task_name, cli.seed, cli.prompt, "joint")
        reset_policy(model_client)

        if cli.record_failure_video or cli.record_all_video:
            task_env.save_data = True
            task_env.save_video = True
            task_env.save_dir = str(cli.output_dir)
            task_env.FRAME_IDX = 0
            task_env.folder_path = {}
            task_env.ep_num = 0
            task_env.control_mask = []
            task_env.current_control_source = "policy"
            task_env._take_picture()

        policy_steps = 0
        trace = None
        previous_snapshot = None
        rescue_result = None
        handback_executed = False
        if cli.trace_handover_state:
            from handover_block_recovery import handover_snapshot

            trace = (cli.output_dir / "handover_state_trace.jsonl").open("w")
            previous_snapshot = handover_snapshot(task_env, policy_steps)
            trace.write(json.dumps(previous_snapshot) + "\n")
            trace.flush()
        if cli.trace_task_state:
            if cli.task_name == "place_can_basket":
                from place_can_basket_recovery import basket_snapshot, classify_basket_state
                task_snapshot, task_classify = basket_snapshot, classify_basket_state
            else:
                from move_can_pot_recovery import move_can_snapshot, classify_move_can_state
                task_snapshot, task_classify = move_can_snapshot, classify_move_can_state
            trace = (cli.output_dir / "task_state_trace.jsonl").open("w")
            initial = task_snapshot(task_env, policy_steps)
            trace.write(json.dumps({"snapshot": initial,
                                    "decision": task_classify(initial)}) + "\n")
            trace.flush()
        while not is_episode_end(task_env):
            observation = task_env.get_obs()
            xpl_obs = robotwin_obs_to_xpolicylab(
                observation,
                instruction=cli.prompt,
                env_idx=0,
                frequency=cli.frequency,
                task_env=task_env,
            )
            model_client.call(func_name="update_obs", obs=xpl_obs)
            action_chunk = normalize_action_chunk(model_client.call(func_name="get_action"))
            if len(action_chunk) != 50:
                raise RuntimeError(f"expected 50 predicted actions, got {len(action_chunk)}")
            for action in action_chunk[: cli.execute_horizon]:
                flat_action, action_type = xpolicylab_action_to_robotwin(
                    action,
                    action_type="joint",
                    current_observation=observation,
                )
                task_env.current_control_source = "policy"
                task_env.take_action(flat_action, action_type=action_type)
                policy_steps += 1
                if (cli.record_failure_video or cli.record_all_video) and policy_steps % 5 == 0:
                    task_env._take_picture()
                if is_episode_end(task_env):
                    break
                observation = task_env.get_obs()
                xpl_obs = robotwin_obs_to_xpolicylab(
                    observation,
                    instruction=cli.prompt,
                    env_idx=0,
                    frequency=cli.frequency,
                    task_env=task_env,
                )
                model_client.call(func_name="update_obs", obs=xpl_obs)
            if cli.trace_task_state:
                snapshot = task_snapshot(task_env, policy_steps)
                decision = task_classify(snapshot)
                trace.write(json.dumps({"snapshot": snapshot, "decision": decision}) + "\n")
                trace.flush()
                if (cli.recover_at_step is not None and policy_steps >= cli.recover_at_step
                        and rescue_result is None):
                    if cli.task_name == "place_can_basket":
                        from place_can_basket_recovery import iter_basket_expert_suffix
                        suffix = iter_basket_expert_suffix
                    else:
                        from move_can_pot_recovery import iter_move_can_expert_suffix
                        suffix = iter_move_can_expert_suffix
                    if cli.task_state_handback_stage_actions is not None:
                        events = []
                        stage_actions = 0
                        suffix_iterator = suffix(task_env, decision)
                        for event in suffix_iterator:
                            events.append(event)
                            if event["phase"] == "done":
                                break
                            stage_actions += 1
                            if stage_actions >= cli.task_state_handback_stage_actions:
                                break
                        after_expert = task_snapshot(task_env, policy_steps)
                        post_decision = task_classify(after_expert)
                        expert_finished = bool(events and events[-1]["phase"] == "done")
                        handback_ready = bool(
                            not expert_finished and stage_actions > 0
                            and getattr(task_env, "plan_success", False)
                            and not task_env.check_success() and post_decision.get("ready")
                        )
                        rescue_result = {
                            "mode": "staged_handback",
                            "policy_steps_at_takeover": policy_steps,
                            "initial_snapshot": snapshot,
                            "initial_decision": decision,
                            "expert_events": events,
                            "expert_stage_actions": stage_actions,
                            "expert_executed": stage_actions > 0,
                            "expert_finished": expert_finished,
                            "post_expert_decision": post_decision,
                            "post_expert_snapshot": after_expert,
                            "handback_ready": handback_ready,
                        }
                        if handback_ready:
                            reset_policy(model_client)
                            handback_executed = True
                    else:
                        events = list(suffix(task_env, decision))
                        suffix_result = events[-1]["result"]
                        rescue_result = {
                            "mode": "diagnostic_expert_suffix",
                            "policy_steps_at_takeover": policy_steps,
                            "initial_snapshot": snapshot,
                            "initial_decision": decision,
                            "expert_events": events,
                            "expert_executed": bool(suffix_result["executed_stage_ids"]),
                            "suffix_result": suffix_result,
                            "post_expert_snapshot": task_snapshot(task_env, policy_steps),
                            "final_task_success": bool(task_env.check_success()),
                        }
                    (cli.output_dir / "rescue_result.json").write_text(
                        json.dumps(rescue_result, ensure_ascii=False, indent=2) + "\n"
                    )
                    if not handback_executed:
                        break
            elif trace is not None:
                snapshot = handover_snapshot(task_env, policy_steps)
                trace.write(json.dumps(snapshot) + "\n")
                trace.flush()
                if (cli.recover_at_step is not None and policy_steps >= cli.recover_at_step
                        and rescue_result is None):
                    from handover_block_recovery import (
                        classify_handover_state, handover_snapshot,
                        iter_expert_suffix, run_expert_suffix,
                    )

                    decision = classify_handover_state(snapshot, previous_snapshot)
                    if cli.handover_handback_stage_actions is None:
                        rescue_result = run_expert_suffix(task_env, decision)
                    else:
                        events = []
                        stage_actions = 0
                        for event in iter_expert_suffix(task_env, decision):
                            events.append(event)
                            if event["phase"] == "done":
                                break
                            stage_actions += 1
                            if stage_actions >= cli.handover_handback_stage_actions:
                                break
                        after_expert = handover_snapshot(task_env, policy_steps)
                        post_decision = classify_handover_state(after_expert, snapshot)
                        expert_finished = events[-1]["phase"] == "done"
                        handback_ready = bool(
                            not expert_finished and task_env.plan_success
                            and not after_expert["task_success"]
                            and post_decision.get("ready")
                        )
                        rescue_result = {
                            "mode": "staged_handback", "initial_decision": decision,
                            "expert_events": events, "expert_stage_actions": stage_actions,
                            "expert_finished": expert_finished,
                            "post_expert_decision": post_decision,
                            "post_expert_snapshot": after_expert,
                            "handback_ready": handback_ready,
                            "policy_steps_at_takeover": policy_steps,
                        }
                        if handback_ready:
                            reset_policy(model_client)
                            handback_executed = True
                    (cli.output_dir / "rescue_result.json").write_text(
                        json.dumps(rescue_result, ensure_ascii=False, indent=2) + "\n"
                    )
                    if not handback_executed:
                        break
                previous_snapshot = snapshot

        original_eval_success = bool(task_env.eval_success)
        original_final_check_success = bool(task_env.check_success())
        if cli.record_failure_video or cli.record_all_video:
            if (cli.record_all_video or not original_eval_success) and int(task_env.FRAME_IDX) >= 2:
                task_env.merge_pkl_to_hdf5_video(instructions=[cli.prompt])
            else:
                cache = getattr(task_env, "folder_path", {}).get("cache")
                if cache and Path(cache).exists():
                    task_env.remove_data_cache()
        if rescue_result is not None and rescue_result.get("mode") == "staged_handback":
            rescue_result["policy_steps_after_handback"] = (
                policy_steps - rescue_result["policy_steps_at_takeover"]
                if handback_executed else 0
            )
            rescue_result["final_task_success"] = original_final_check_success
            (cli.output_dir / "rescue_result.json").write_text(
                json.dumps(rescue_result, ensure_ascii=False, indent=2) + "\n"
            )
        if cli.trace_burger_final:
            burger_before = burger_snapshot(task_env)
            for _ in range(cli.burger_settle_steps):
                task_env.scene.step()
            burger_after = burger_snapshot(task_env)
            (cli.output_dir / "burger_physical_state.json").write_text(
                json.dumps({
                    "seed": cli.seed, "extra_physics_steps": cli.burger_settle_steps,
                    "at_policy_termination": burger_before,
                    "after_settling": burger_after,
                }, ensure_ascii=False, indent=2) + "\n"
            )

        record = {
            "task": cli.task_name,
            "seed": cli.seed,
            "success": original_eval_success,
            "final_check_success": original_final_check_success,
            "policy_steps": policy_steps,
            "generated_horizon": 50,
            "execute_horizon": cli.execute_horizon,
            "frequency_field": cli.frequency,
            "step_limit": cli.step_limit,
            "expert_prefilter": False,
            "video_path": "video/episode_0000000.mp4" if (cli.record_all_video or (cli.record_failure_video and not original_eval_success)) else None,
            "expert_intervention": bool(
                rescue_result is not None and rescue_result.get("expert_executed", True)
            ),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if rescue_result is not None:
            record["rescue_success"] = bool(
                rescue_result.get("success", rescue_result.get("final_task_success", False))
            )
            record["rescue_branch"] = rescue_result["initial_decision"].get("branch")
            record["handback_executed"] = handback_executed
        record_path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 3
    finally:
        if "trace" in locals() and trace is not None:
            trace.close()
        safe_close_env(task_env, clear_cache=True)
        close_policy_client(model_client)


if __name__ == "__main__":
    raise SystemExit(main())
