"""Initial-scene expert screening; never used for live-state recovery.

The expert runs with recording/viewer disabled. Only an accepted scene is
rebuilt for the policy, then given an instruction by the SFT evaluator's
build_instruction function. No simulator or policy imports are needed here.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

from .recording import append_jsonl


SCREEN_LOG = "scene_screening.jsonl"


def screening_history(output_dir: Path) -> dict[int, dict]:
    """Latest event per candidate, including started/interrupted candidates."""
    path = output_dir / SCREEN_LOG
    if not path.exists():
        return {}
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)  # Corrupt journals require inspection, not reuse.
            latest[int(row["seed"])] = row
    return latest


def screen_expert_scene(env, args, *, seed, episode_index, close_env, output_dir):
    """Run the official expert's plan, success, and joint-limit gates.

    A technical error is not an ordinary rejected scene. Preserve its traceback
    and stop before querying the policy, rather than hiding it in a yield rate.
    """
    event = {"seed": int(seed), "status": "expert_started", "policy_steps": 0,
             "counted_as_policy_rollout": False}
    append_jsonl(output_dir / SCREEN_LOG, event)
    started = time.monotonic()
    screen_args = dict(args, render_freq=0, save_data=False, save_video=False,
                       eval_video_save_dir=None, eval_mode=True, need_plan=True)
    try:
        env.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **screen_args)
        # HIL's safe-grasp error handler writes here; no adapter recovery is run.
        env.episode_metadata = {"seed": int(seed), "control_source": "expert_screen"}
        env.current_control_source = "expert_screen"
        episode_info = deepcopy(env.play_once())
        planned = bool(env.plan_success)
        # Match the official short-circuit gate: do not inspect an unfinished
        # expert terminal state when planning already failed.
        succeeded = bool(env.check_success()) if planned else None
        # The SFT collector checks planned_joints_legal() only after both
        # planning and task success. Keep that order for the DAgger seed gate.
        joints_legal, joint_absmax = (
            env.planned_joints_legal() if planned and succeeded else (None, None)
        )
        accepted = bool(planned and succeeded and joints_legal)
        event.update(status="expert_passed" if accepted else "expert_rejected",
                     plan_success=planned, check_success=succeeded,
                     joints_legal=bool(joints_legal) if joints_legal is not None else None,
                     joint_absmax=float(joint_absmax) if joint_absmax is not None else None,
                     expert_episode_info=episode_info,
                     expert_stage_error=deepcopy(env.episode_metadata.get("expert_stage_error")),
                     seconds=round(time.monotonic() - started, 3))
        append_jsonl(output_dir / SCREEN_LOG, event)
        return event
    except Exception as exc:
        event.update(status="expert_error", exception_type=type(exc).__name__,
                     error=str(exc), traceback=traceback.format_exc(),
                     seconds=round(time.monotonic() - started, 3))
        append_jsonl(output_dir / SCREEN_LOG, event)
        raise
    finally:
        close_env(env)


def prepare_screened_scene(env, args, *, seed, episode_index, adapter,
                           build_instruction, instruction_pool_size, close_env,
                           output_dir):
    """Return per-episode provenance, or None for a normal expert rejection."""
    event = screen_expert_scene(
        env, args, seed=seed, episode_index=episode_index, close_env=close_env,
        output_dir=output_dir,
    )
    if event["status"] != "expert_passed":
        return None
    stage = "policy_setup"
    try:
        # Never start from the expert's terminal state or replay its actions.
        env.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **args)
        stage = "instruction"
        # Same function and 50-description pool as the documented SFT test_num=50.
        # This call is AFTER reset, BEFORE rendering/adapter hooks/policy reset.
        instruction = str(build_instruction(
            args, event["expert_episode_info"], "seen", instruction_pool_size,
        ))
        if not instruction.strip() or instruction == args["task_name"]:
            raise ValueError("Official seen instruction generation fell back to task name")
        adapter.initialize_episode(env)
        event = {**event, "status": "policy_ready", "instruction_type": "seen",
                 "instruction_pool_size": int(instruction_pool_size),
                 "instruction": instruction}
        env.episode_metadata["initial_scene_screening"] = deepcopy(event)
        append_jsonl(output_dir / SCREEN_LOG, event)
        return event
    except Exception as exc:
        append_jsonl(output_dir / SCREEN_LOG, {
            **event, "status": f"{stage}_error", "exception_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        })
        close_env(env)
        raise


def screening_qualifies_for_training(screening, *, seed: int, instruction: str) -> bool:
    """Require a completed, same-seed official expert check before formal HIL use."""
    return bool(
        isinstance(screening, dict)
        and screening.get("status") == "policy_ready"
        and screening.get("seed") == int(seed)
        and screening.get("instruction") == instruction
        and screening.get("plan_success") is True
        and screening.get("check_success") is True
        and screening.get("joints_legal") is True
    )
