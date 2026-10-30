#!/usr/bin/env python3
"""Headless, no-intervention stage-six policy evaluation on fixed DAgger seeds.

Stages 1-5 use the same scripted expert prefix as one_stage_dagger.py.  Only
ready stage-six handoffs enter the denominator.  The enhanced arm replays the
same seeds and excludes handoffs outside the predeclared pairing tolerance.
No expert recovery or human input occurs after the handoff.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from coverage_sampling.core import DecisionSampler, config_record
from coverage_sampling.options import config_from_args
from eval_policy_xpolicylab import (
    build_policy_client, class_decorator, close_policy_client, is_episode_end,
    normalize_action_chunk, notify_trial_end, prepare_policy_case, reset_policy,
    robotwin_obs_to_xpolicylab, safe_close_env, sample_absolute_joint_candidates,
    xpolicylab_action_to_robotwin,
)
from one_stage_dagger import handoff_pair_differences, handoff_pair_ready, handoff_record
from policy_eval_record import PROMPT, build_runtime_args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "enhanced", "gaussian"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18305)
    parser.add_argument("--policy-name", default="Pi_05_RobotTwin")
    parser.add_argument("--ckpt-name", default="v2_promptfix_9999")
    parser.add_argument("--task-config", default="handover_to_tray_v2_promptfix")
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--render-freq", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument("--render-max-num-materials", type=int, default=128)
    parser.add_argument("--render-max-num-textures", type=int, default=512)
    parser.add_argument("--step-limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paired-baseline-records", type=Path)
    parser.add_argument("--critic", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--encoder-weights", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--policy-microbatch", type=int, default=32)
    parser.add_argument("--gaussian-sigma", type=float)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_record(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records, cli, sampling_config):
    ready = [r for r in records if r["status"] == "completed"]
    successes = sum(bool(r["success"]) for r in ready)
    realized_noise = [
        float(r["mean_gaussian_noise_rms"])
        for r in ready if r.get("mean_gaussian_noise_rms") is not None
    ]
    return {
        "mode": cli.mode,
        "seed_start": cli.seed_start,
        "episodes_requested": cli.episodes,
        "episodes_recorded": len(records),
        "prefix_failed": sum(r["status"] == "prefix_failed" for r in records),
        "paired_prefix_excluded": sum(r["status"] == "paired_prefix_excluded" for r in records),
        "ready_stage6": len(ready),
        "autonomous_successes": successes,
        "stage6_success_rate": successes / len(ready) if ready else None,
        "planned_joints_legal": sum(bool(r.get("joints_legal")) for r in ready),
        "mean_realized_gaussian_noise_rms": (
            sum(realized_noise) / len(realized_noise) if realized_noise else None
        ),
        "checkpoint_name": cli.ckpt_name,
        "task_config": cli.task_config,
        "frequency": cli.frequency,
        "port": cli.port,
        "sampling_config": config_record(sampling_config) if sampling_config else None,
        "records": records,
    }


def main():
    cli = parse_args()
    if cli.episodes < 1 or cli.render_freq != 0 or cli.frequency != 30:
        raise ValueError("This comparison requires positive episodes, headless rendering and 30 Hz")
    if cli.mode in {"enhanced", "gaussian"} and not cli.paired_baseline_records:
        raise ValueError("Enhanced and Gaussian arms require --paired-baseline-records")
    if cli.mode == "baseline" and cli.paired_baseline_records:
        raise ValueError("Baseline arm does not use paired records")
    cli.output_dir = cli.output_dir.expanduser().resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = cli.output_dir / "records.jsonl"
    existing = load_records(records_path)
    if existing and not cli.resume:
        raise FileExistsError(f"Existing records require --resume: {records_path}")
    done = {int(r["seed"]) for r in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate seed in existing records")
    baseline_by_seed = {}
    if cli.paired_baseline_records:
        baseline_by_seed = {int(r["seed"]): r for r in load_records(cli.paired_baseline_records)}

    sampling_config = scorer = None
    if cli.mode in {"enhanced", "gaussian"}:
        if cli.mode == "gaussian" and cli.gaussian_sigma is None:
            raise ValueError("Gaussian arm requires --gaussian-sigma")
        sampling_args = {
            "es_mode": cli.mode, "es_window_start": 0, "es_window_end": 899,
            "es_num_candidates": cli.num_candidates,
            "es_policy_microbatch": cli.policy_microbatch,
            "es_horizon": 10, "es_seed": 42,
            "task_name": "handover_to_tray", "action_type": "joint", "eval_batch": False,
        }
        if cli.mode == "gaussian":
            sampling_args["es_gaussian_sigma"] = cli.gaussian_sigma
        else:
            sampling_args.update({
                "es_scorer": "discriminator_chunk",
                "es_critic": str(cli.critic) if cli.critic is not None else None,
                "es_calibration": str(cli.calibration) if cli.calibration is not None else None,
                "es_calibration_key": "disc_s42",
                "es_encoder_weights": str(cli.encoder_weights) if cli.encoder_weights is not None else None,
                "es_device": cli.device,
            })
        for name in ("critic", "calibration", "encoder_weights"):
            if cli.mode == "enhanced" and getattr(cli, name) is None:
                raise ValueError(f"Enhanced arm requires --{name.replace('_', '-')}")
        sampling_config = config_from_args(sampling_args)
        if sampling_config.requires_scores:
            from coverage_sampling.critic import create_online_scorer
            scorer = create_online_scorer(sampling_args, sampling_config)

    user_args, args = build_runtime_args(cli)
    task_env = class_decorator("handover_to_tray")
    model_client = build_policy_client(user_args)
    records = list(existing)
    try:
        if sampling_config is not None:
            info = model_client.call(func_name="candidate_sampling_info")
            if not isinstance(info, dict) or info.get("max_batch_size", 0) < cli.policy_microbatch:
                raise RuntimeError(f"Policy server cannot produce batch={cli.policy_microbatch}: {info!r}")
        for rollout_index in range(cli.episodes):
            seed = cli.seed_start + rollout_index
            if seed in done:
                continue
            if cli.mode in {"enhanced", "gaussian"} and seed not in baseline_by_seed:
                raise ValueError(f"Missing baseline seed {seed}")
            task_env.setup_demo(now_ep_num=rollout_index, seed=seed, is_test=False, **args)
            if cli.step_limit is not None:
                task_env.step_lim = cli.step_limit
            task_env.set_instruction(PROMPT)
            task_env.current_control_source = "expert_prefix"
            prefix_ok = bool(task_env.play_prefix_until_placement())
            handoff = handoff_record(task_env)
            handoff["prefix_plan_success"] = prefix_ok
            record = {"seed": seed, "status": None, "handoff": handoff,
                      "prefix_steps": int(task_env.take_action_cnt),
                      "timestamp": datetime.now().isoformat(timespec="seconds")}
            if not handoff["ready"]:
                record["status"] = "prefix_failed"
            elif cli.mode in {"enhanced", "gaussian"}:
                baseline = baseline_by_seed[seed]
                if baseline["status"] != "completed":
                    record["status"] = "paired_prefix_excluded"
                    handoff["paired_prefix_error"] = "baseline_prefix_not_ready"
                else:
                    diffs = handoff_pair_differences(baseline["handoff"], handoff)
                    handoff["paired_differences"] = diffs
                    if not handoff_pair_ready(diffs):
                        record["status"] = "paired_prefix_excluded"
                        handoff["paired_prefix_error"] = "handoff_state_mismatch"
            if record["status"] is None:
                prepare_policy_case(model_client, "handover_to_tray", seed, PROMPT, "joint")
                reset_policy(model_client)
                task_env.current_control_source = "policy"
                policy_steps = 0
                decision_step = 0
                sampler = DecisionSampler(sampling_config, None, seed) if sampling_config else None
                ess_values = []
                shift_values = []
                gaussian_rms_values = []
                gaussian_max_values = []
                while not is_episode_end(task_env):
                    observation = task_env.get_obs()
                    xpl_obs = robotwin_obs_to_xpolicylab(
                        observation, instruction=PROMPT, env_idx=0,
                        frequency=cli.frequency, task_env=task_env)
                    model_client.call(func_name="update_obs", obs=xpl_obs)
                    if sampler:
                        def sample_candidates(count):
                            return sample_absolute_joint_candidates(
                                model_client, observation, "joint", count,
                                sampling_config.policy_microbatch)

                        chunk, selection = sampler.select(
                            decision_step, lambda: None,
                            sample_candidates=sample_candidates,
                            score_candidates=(scorer.bind_observation(observation)
                                              if scorer is not None else None))
                        ess_values.append(float(selection["ess_fraction"]))
                        shift_values.append(float(selection["joint_rms_shift_from_uniform"]))
                        if selection.get("gaussian_noise_arm_rms") is not None:
                            gaussian_rms_values.append(float(selection["gaussian_noise_arm_rms"]))
                            gaussian_max_values.append(float(selection["gaussian_noise_arm_max_abs"]))
                    else:
                        chunk = normalize_action_chunk(model_client.call(func_name="get_action"))
                    if len(chunk) != 10:
                        raise RuntimeError(f"Expected 10-action chunk, received {len(chunk)}")
                    decision_step += 1
                    for action in chunk:
                        flat_action, action_type = xpolicylab_action_to_robotwin(
                            action, action_type="joint", current_observation=observation)
                        task_env.take_action(flat_action, action_type=action_type)
                        policy_steps += 1
                        if is_episode_end(task_env):
                            break
                        # Match HG-DAgger's HIL_CHUNK_OBSERVATIONS=1 path:
                        # enhanced sampling reads/encodes a fresh observation only
                        # at the next decision boundary; baseline keeps the
                        # original per-action update_obs behaviour.
                        if not sampler:
                            observation = task_env.get_obs()
                            xpl_obs = robotwin_obs_to_xpolicylab(
                                observation, instruction=PROMPT, env_idx=0,
                                frequency=cli.frequency, task_env=task_env)
                            model_client.call(func_name="update_obs", obs=xpl_obs)
                joints_legal, joint_absmax = task_env.planned_joints_legal()
                record.update(status="completed", success=bool(task_env.eval_success),
                              policy_steps=policy_steps, policy_decisions=decision_step,
                              total_env_steps=int(task_env.take_action_cnt),
                              final_check_success=bool(task_env.check_success()),
                              final_success_metrics=task_env.success_metrics(),
                              mean_ess_fraction=sum(ess_values) / len(ess_values) if ess_values else None,
                              mean_joint_rms_shift=sum(shift_values) / len(shift_values) if shift_values else None,
                              mean_gaussian_noise_rms=(
                                  sum(gaussian_rms_values) / len(gaussian_rms_values)
                                  if gaussian_rms_values else None
                              ),
                              max_gaussian_noise_abs=(
                                  max(gaussian_max_values) if gaussian_max_values else None
                              ),
                              joints_legal=bool(joints_legal),
                              joint_absmax=float(joint_absmax))
            records.append(record)
            append_record(records_path, record)
            notify_trial_end(model_client, "handover_to_tray", seed, bool(record.get("success", False)))
            safe_close_env(task_env)
            print(f"[STAGE6] mode={cli.mode} seed={seed} status={record['status']} "
                  f"success={record.get('success')} steps={record.get('policy_steps')}", flush=True)
            summary = summarize(records, cli, sampling_config)
            (cli.output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        traceback.print_exc()
        safe_close_env(task_env, clear_cache=True)
        return 3
    finally:
        close_policy_client(model_client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
