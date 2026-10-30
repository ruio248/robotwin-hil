#!/usr/bin/env python3
"""Evaluate a learned policy on fixed RoboTwin seeds for a DART scan.

This measures autonomous learned-policy task success. It does not run the
scripted expert as a pre-filter and does not inject rollout-time noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--policy-name", default="Pi_05_RobotTwin")
    parser.add_argument("--task-name", default="handover_to_tray")
    parser.add_argument("--task-config", default="handover_to_tray_v2_promptfix")
    parser.add_argument("--seed-start", type=int, default=31000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--step-limit", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def summarize(records: list[dict], cli: argparse.Namespace) -> dict:
    terminal = [row for row in records if row.get("status") == "completed"]
    successes = sum(bool(row["success"]) for row in terminal)
    infra = [row for row in records if row.get("status") == "infra_error"]
    expected = list(range(cli.seed_start, cli.seed_start + cli.episodes))
    seen = [int(row["seed"]) for row in records]
    return {
        "status": "complete" if len(terminal) == cli.episodes and not infra and seen == expected else "incomplete",
        "metric": "autonomous_learned_policy_full_task_success_rate",
        "episodes_expected": cli.episodes,
        "episodes_completed": len(terminal),
        "infrastructure_errors": len(infra),
        "successes": successes,
        "success_rate": successes / len(terminal) if terminal else None,
        "seed_start": cli.seed_start,
        "seed_end": cli.seed_start + cli.episodes - 1,
        "seeds_missing_or_order_mismatch": seen != expected,
        "frequency_hz": cli.frequency,
        "step_limit": cli.step_limit,
        "action_type": "absolute_joint",
        "action_horizon": 10,
        "expert_prefilter": False,
        "human_intervention": False,
        "prompt": PROMPT,
        "task_config": cli.task_config,
        "records": records,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    cli = parse_args()
    if cli.episodes < 1 or cli.frequency != 30 or cli.step_limit < 1:
        raise ValueError("episodes/step-limit must be positive and frequency must be 30 Hz")
    cli.output_dir = cli.output_dir.expanduser().resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = cli.output_dir / "eval_config.json"
    config = {
        "host": cli.host,
        "port": cli.port,
        "policy_name": cli.policy_name,
        "task_name": cli.task_name,
        "task_config": cli.task_config,
        "seed_start": cli.seed_start,
        "episodes": cli.episodes,
        "frequency_hz": cli.frequency,
        "step_limit": cli.step_limit,
        "prompt": PROMPT,
        "action_type": "absolute_joint",
        "action_horizon": 10,
        "expert_prefilter": False,
        "human_intervention": False,
    }
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise FileExistsError(f"existing evaluation config differs: {config_path}")
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    records_path = cli.output_dir / "eval_records.jsonl"
    existing = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()] if records_path.exists() else []
    if existing and not cli.resume:
        raise FileExistsError(f"existing records require --resume: {records_path}")
    done = {int(row["seed"]) for row in existing if row.get("status") == "completed"}
    if len(done) != sum(row.get("status") == "completed" for row in existing):
        raise ValueError("duplicate completed seed in existing records")

    usr_args = {
        "bench_name": "RoboTwin",
        "task_name": cli.task_name,
        "task_config": cli.task_config,
        "ckpt_setting": "dart_policy_success_scan",
        "policy_name": cli.policy_name,
        "host": cli.host,
        "port": cli.port,
        "protocol": "ws",
        "root_dir": str(ROOT),
        "xpolicylab_root": str(ROOT / "XPolicyLab"),
        "action_type": "joint",
    }
    args, _ = load_task_args(usr_args)
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_log"] = False
    args["save_data"] = False
    args["collect_data"] = False
    args["save_video"] = False
    task_env = class_decorator(cli.task_name)
    model_client = None
    records = list(existing)
    try:
        model_client = build_policy_client(usr_args)
        for rollout_index in range(cli.episodes):
            seed = cli.seed_start + rollout_index
            if seed in done:
                continue
            row = {
                "seed": seed,
                "status": "started",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            try:
                task_env.setup_demo(now_ep_num=rollout_index, seed=seed, is_test=True, **args)
                task_env.step_lim = cli.step_limit
                task_env.set_instruction(PROMPT)
                prepare_policy_case(model_client, cli.task_name, seed, PROMPT, "joint")
                reset_policy(model_client)
                policy_steps = 0
                policy_decisions = 0
                while not is_episode_end(task_env):
                    observation = task_env.get_obs()
                    xpl_obs = robotwin_obs_to_xpolicylab(
                        observation, instruction=PROMPT, env_idx=0,
                        frequency=cli.frequency, task_env=task_env,
                    )
                    model_client.call(func_name="update_obs", obs=xpl_obs)
                    chunk = normalize_action_chunk(model_client.call(func_name="get_action"))
                    if len(chunk) != 10:
                        raise RuntimeError(f"expected 10-action chunk; received {len(chunk)}")
                    policy_decisions += 1
                    for action in chunk:
                        flat_action, robotwin_action_type = xpolicylab_action_to_robotwin(
                            action, action_type="joint", current_observation=observation,
                        )
                        task_env.take_action(flat_action, action_type=robotwin_action_type)
                        policy_steps += 1
                        if is_episode_end(task_env):
                            break
                        observation = task_env.get_obs()
                        xpl_obs = robotwin_obs_to_xpolicylab(
                            observation, instruction=PROMPT, env_idx=0,
                            frequency=cli.frequency, task_env=task_env,
                        )
                        model_client.call(func_name="update_obs", obs=xpl_obs)
                success = bool(task_env.eval_success)
                notify_trial_end(model_client, cli.task_name, seed, success)
                row.update(
                    status="completed",
                    success=success,
                    policy_steps=policy_steps,
                    policy_decisions=policy_decisions,
                    total_env_steps=int(task_env.take_action_cnt),
                    final_check_success=bool(task_env.check_success()),
                )
                done.add(seed)
            except Exception as exc:
                row.update(
                    status="infra_error",
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=10),
                )
                records.append(row)
                with records_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                print(json.dumps({"seed": seed, "status": row["status"], "error": row["error"]}), flush=True)
                break
            finally:
                safe_close_env(task_env)
            records.append(row)
            with records_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            print(json.dumps({"seed": seed, "status": row["status"], "success": row.get("success")}), flush=True)
    finally:
        safe_close_env(task_env)
        if model_client is not None:
            close_policy_client(model_client)

    report = summarize(records, cli)
    (cli.output_dir / "aggregate_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "status", "episodes_expected", "episodes_completed", "infrastructure_errors", "successes", "success_rate"
    )}, ensure_ascii=False), flush=True)
    if report["status"] != "complete":
        raise RuntimeError(f"evaluation incomplete; inspect {records_path}")


if __name__ == "__main__":
    main()
