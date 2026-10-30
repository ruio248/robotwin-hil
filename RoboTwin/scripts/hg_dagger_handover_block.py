#!/usr/bin/env python3
"""Native handover_block HIL collection with conservative staged recovery.

The released pi0.5 policy acts until ``i``. The remaining chunk is discarded,
and the scripted expert plans from the *live* scene. ``r`` hands back only at
an action boundary after a fresh state check; ``x`` discards the rollout.
Only successful, joint-legal episodes with recorded HIL frames are saved.
This collector never uses the scripted expert to preselect rollout seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(os.environ.get(
    "ROBOTWIN_ROOT", str(Path(__file__).resolve().parents[1])
)).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_policy_xpolicylab import (  # noqa: E402
    build_policy_client, class_decorator, close_policy_client, is_episode_end,
    load_task_args, normalize_action_chunk, prepare_policy_case, reset_policy,
    robotwin_obs_to_xpolicylab, safe_close_env, xpolicylab_action_to_robotwin,
)
from handover_block_recovery import (  # noqa: E402
    classify_handover_state, handover_snapshot, iter_expert_suffix,
)
from hg_dagger_handover import (  # noqa: E402
    HumanInterventionInput, append_jsonl, append_segment, finalize_episode,
)


PROMPT = ("use the left arm to grasp the red block on the table, "
          "handover it to the right arm and place it on the blue pad")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed-start", type=int, default=40000)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--target-saved", type=int, default=30)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--step-limit", type=int, default=800)
    p.add_argument("--frequency", type=int, default=50)
    p.add_argument("--save-freq", type=int, default=15)
    p.add_argument("--render-freq", type=int, default=5)
    p.add_argument("--headless-smoke", action="store_true",
                   help="Test only: allow headless auto-i/r; never a collection run")
    p.add_argument("--auto-intervene-step", type=int, default=-1)
    p.add_argument("--auto-handback-stage-actions", type=int, default=-1)
    p.add_argument("--auto-abort-step", type=int, default=-1)
    cli = p.parse_args()
    if cli.episodes < 1 or cli.target_saved < 1 or cli.save_freq < 1:
        p.error("episodes, target-saved and save-freq must be positive")
    if cli.step_limit < 1 or cli.frequency < 1:
        p.error("step-limit and frequency must be positive")
    if cli.seed_start < 40000 and not cli.headless_smoke:
        p.error("production collection must not overlap development/test seeds")
    auto_values = (cli.auto_intervene_step, cli.auto_handback_stage_actions,
                   cli.auto_abort_step)
    if any(value >= 0 for value in auto_values) and not cli.headless_smoke:
        p.error("auto-i/r/x hooks are restricted to --headless-smoke")
    if any(value == 0 or value < -1 for value in auto_values):
        p.error("auto-i/r/x trigger steps must be positive or disabled (-1)")
    if cli.auto_handback_stage_actions >= 1 and cli.auto_intervene_step < 1:
        p.error("automatic r requires a positive automatic i trigger")
    if cli.headless_smoke and (cli.episodes != 1 or cli.target_saved != 1):
        p.error("headless smoke is exactly one diagnostic rollout")
    return cli


def runtime_args(cli: argparse.Namespace) -> tuple[dict, dict]:
    user = {
        "task_name": "handover_block", "task_config": "demo_clean",
        "policy_name": "Pi_05_RobotTwin", "host": cli.host, "port": cli.port,
        "protocol": "ws", "xpolicylab_root": str(ROOT / "XPolicyLab"),
        "ckpt_name": "public_pi05_handover_block",
        "ckpt_setting": "public_pi05_handover_block", "action_type": "joint",
        "evaluation_id": "hg-dagger-handover-block", "trial_id": "hg-dagger-handover-block",
        "action_case_id": "hg-dagger-handover-block",
    }
    args, _ = load_task_args(user)
    args.update({
        "eval_instruction": "seen", "eval_mode": True,
        "render_freq": cli.render_freq, "need_plan": True,
        "save_data": False, "save_video": False,
        "save_path": str(cli.output_dir), "save_freq": cli.save_freq,
    })
    user["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    user["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    return user, args


def run_episode(cli: argparse.Namespace, client, args: dict, seed: int,
                rollout_index: int, saved_index: int) -> tuple[dict, bool]:
    task_env = class_decorator("handover_block")
    saved = False
    started = time.time()
    try:
        task_env.setup_demo(now_ep_num=saved_index, seed=seed, is_test=False, **args)
        task_env.step_lim = cli.step_limit
        task_env.set_instruction(PROMPT)
        if not cli.headless_smoke:
            viewer = getattr(task_env, "viewer", None)
            if viewer is None and not sys.stdin.isatty():
                raise RuntimeError("interactive collection needs a focused viewer or terminal")
            if viewer is not None:
                task_env._update_render()
                viewer.render()
        prepare_policy_case(client, "handover_block", seed, PROMPT, "joint")
        reset_policy(client)

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
            "task": "handover_block", "seed": seed, "instruction": PROMPT,
            "policy": "published task-specific pi0.5", "expert_prefilter": False,
            "frequency_field": cli.frequency, "generated_horizon": 50,
            "execute_horizon": 50, "save_freq": cli.save_freq,
        }
        task_env._take_picture()

        source, source_start = "policy", int(task_env.FRAME_IDX)
        segments: list[dict] = []
        interventions: list[dict] = []
        requests: list[dict] = []
        policy_steps = 0
        expert_actions = 0
        mode = "policy"
        recovery = None
        aborted = False
        quit_requested = False
        expert_failed = False
        previous_snapshot = handover_snapshot(task_env, policy_steps)

        def switch_source(new_source: str) -> None:
            nonlocal source, source_start
            append_segment(segments, source, source_start, int(task_env.FRAME_IDX))
            source, source_start = new_source, int(task_env.FRAME_IDX)

        def request_takeover() -> None:
            nonlocal mode, recovery, expert_actions
            current = handover_snapshot(task_env, policy_steps)
            decision = classify_handover_state(current, previous_snapshot)
            requests.append({"policy_steps": policy_steps, "decision": decision})
            if not decision.get("ready"):
                print(f"[HIL] takeover rejected: {decision.get('reason')}", flush=True)
                return
            recovery = iter_expert_suffix(task_env, decision)
            expert_actions = 0
            interventions.append({
                "policy_steps_at_takeover": policy_steps,
                "decision": decision, "stage_events": [], "handback": False,
            })
            switch_source("hil")
            mode = "expert"
            print(f"[HIL] i accepted: {decision['branch']}; chunk discarded", flush=True)

        with HumanInterventionInput(task_env, {
            "i": "intervene", "r": "handback", "x": "abort", "q": "quit"
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
                    if event == "intervene":
                        request_takeover()
                        if mode == "expert":
                            continue
                    observation = task_env.get_obs()
                    client.call(func_name="update_obs", obs=robotwin_obs_to_xpolicylab(
                        observation, instruction=PROMPT, env_idx=0,
                        frequency=cli.frequency, task_env=task_env,
                    ))
                    action_chunk = normalize_action_chunk(client.call(func_name="get_action"))
                    if len(action_chunk) != 50:
                        raise RuntimeError(f"expected 50 policy actions, got {len(action_chunk)}")
                    interrupted = False
                    for action in action_chunk:
                        event = keyboard.poll()
                        if (cli.auto_intervene_step > 0 and not interventions
                                and policy_steps >= cli.auto_intervene_step):
                            event = "intervene"
                        if cli.auto_abort_step > 0 and policy_steps >= cli.auto_abort_step:
                            event = "abort"
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
                        client.call(func_name="update_obs", obs=robotwin_obs_to_xpolicylab(
                            observation, instruction=PROMPT, env_idx=0,
                            frequency=cli.frequency, task_env=task_env,
                        ))
                    if quit_requested or aborted:
                        break
                    if not interrupted:
                        previous_snapshot = handover_snapshot(task_env, policy_steps)
                else:
                    if (cli.auto_handback_stage_actions > 0
                            and expert_actions >= cli.auto_handback_stage_actions):
                        event = "handback"
                    if event == "handback":
                        snapshot = handover_snapshot(task_env, policy_steps)
                        decision = classify_handover_state(snapshot, previous_snapshot)
                        if decision.get("ready") and task_env.plan_success:
                            interventions[-1]["handback"] = True
                            interventions[-1]["post_expert_decision"] = decision
                            recovery.close()
                            switch_source("policy")
                            task_env.current_control_source = "policy"
                            task_env.current_stage_id = 0
                            reset_policy(client)
                            mode = "policy"
                            previous_snapshot = snapshot
                            continue
                        print(f"[HIL] handback denied: {decision.get('reason')}", flush=True)
                    event = next(recovery, None)
                    if event is None:
                        expert_failed = True
                        break
                    interventions[-1]["stage_events"].append(event)
                    if event["phase"] == "done":
                        interventions[-1]["expert_result"] = event["result"]
                        expert_failed = not bool(event["result"].get("success"))
                        break
                    expert_actions += 1
                    if not event.get("plan_success"):
                        expert_failed = True
                        break

        append_segment(segments, source, source_start, int(task_env.FRAME_IDX))
        final_success = bool(task_env.check_success())
        joints_legal, joint_absmax = task_env.planned_joints_legal()
        hil_frames = task_env.control_mask.count("hil")
        valid_hil = bool(
            not aborted and not quit_requested and interventions and hil_frames > 0
            and not expert_failed and final_success and joints_legal
        )
        accepted_for_training = bool(valid_hil and not cli.headless_smoke)
        task_env.episode_metadata["diagnostic_only"] = bool(cli.headless_smoke)
        task_env.episode_metadata["accepted_for_training"] = accepted_for_training
        record = {
            "task": "handover_block", "seed": seed, "rollout_index": rollout_index,
            "policy_steps": policy_steps, "intervention_requests": requests,
            "interventions": interventions, "segments": segments,
            "control_mask": list(task_env.control_mask),
            "hil_frames": hil_frames, "final_check_success": final_success,
            "joints_legal": bool(joints_legal), "joint_absmax": float(joint_absmax),
            "aborted": aborted, "quit_requested": quit_requested,
            "expert_failed": expert_failed, "valid_hil": valid_hil,
            "diagnostic_only": bool(cli.headless_smoke),
            "accepted_for_training": accepted_for_training,
            "rollout_seconds": round(time.time() - started, 2),
        }
        raw_dir = finalize_episode(
            task_env, cli.output_dir, saved_index, PROMPT,
            save=valid_hil, supervisor_label="success" if final_success else "failure",
            segments=segments, extra=record,
        )
        saved = raw_dir is not None
        record["raw_dir"] = str(raw_dir) if raw_dir else None
        append_jsonl(cli.output_dir / "rollouts.jsonl", record)
        return record, quit_requested
    finally:
        safe_close_env(task_env, clear_cache=not saved)


def main() -> int:
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    if (cli.output_dir / "rollouts.jsonl").exists():
        raise FileExistsError("refusing to resume/overwrite prior collection report")
    user, args = runtime_args(cli)
    client = build_policy_client(user)
    completed = 0
    saved = 0
    try:
        for index in range(cli.episodes):
            if saved >= cli.target_saved:
                break
            seed = cli.seed_start + index
            record, quit_requested = run_episode(cli, client, args, seed, index, saved)
            completed += 1
            saved += bool(record["valid_hil"])
            print(f"[HIL] seed={seed} valid={record['valid_hil']} saved={saved}/{cli.target_saved}",
                  flush=True)
            if quit_requested:
                break
    finally:
        close_policy_client(client)
    summary = {
        "task": "handover_block", "rollouts": completed, "saved_valid_hil": saved,
        "target_saved": cli.target_saved, "complete": saved >= cli.target_saved,
        "headless_smoke": cli.headless_smoke,
        "policy": "published task-specific pi0.5", "seeds_start": cli.seed_start,
        "expert_prefilter": False,
    }
    (cli.output_dir / "session.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
