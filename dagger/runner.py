#!/usr/bin/env python3
"""Canonical human-gated DAgger runner with task recovery adapters.

The policy controls the live scene until the supervisor presses ``i``.  The
remaining policy action chunk is discarded immediately, and the privileged
scripted expert replans from the current simulator state. Raw trajectories
preserve per-frame policy/HIL ownership and operator save
decisions. Offline export selects expert segments for training.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


from dagger.runtime import activate_runtime, configure_imports, runtime_root
from dagger.adapters import get_adapter, registered_tasks
from dagger.episode_protocol import SCREEN_LOG, prepare_screened_scene, screening_history
from dagger.recording import (
    read_json_object, write_json, append_jsonl, append_segment,
    finalize_episode, discard_recovery_cache,
)
from dagger.takeover import (
    DEFAULT_KEYS, OperatorInterrupt, HumanInterventionInput,
    choose_recovery_stage, prompt_choice,
)
from dagger.viewer import (
    render_initial_frame, setup_viewer_diagnostics, install_viewer_frame_limit,
    configure_spectator_camera,
)
from dagger.adapters.handover_to_tray import PROMPT, STAGE_LABELS, RESERVED_SEED_RANGES

ROBOTWIN_ROOT = runtime_root()


def _load_collection_helpers(root: Path) -> None:
    import importlib
    configure_imports(root, source_fallback=True)
    exports = {
        "coverage_sampling.options": ("add_arguments", "config_from_args"),
        "hg_dagger_resume": ("collection_lock", "inspect_collection", "preserve_incomplete_cache"),
        "one_stage_dagger": (
            "baseline_handoffs", "baseline_seeds", "handoff_pair_differences",
            "handoff_pair_ready", "is_valid_hil",
        ),
        "policy_rollout_archive": ("archive_policy_rollout",),
    }
    for module, names in exports.items():
        helpers = importlib.import_module(module)
        for name in names:
            target = "add_sampling_arguments" if name == "add_arguments" else name
            globals()[target] = getattr(helpers, name)


def _load_runtime_helpers():
    # Simulator imports run only after selecting the actual runtime and cwd.
    import importlib
    helpers = importlib.import_module("eval_policy_xpolicylab")
    for name in (
        "build_instruction", "build_policy_client", "class_decorator", "close_policy_client",
        "is_episode_end", "load_task_args", "normalize_action_chunk",
        "notify_trial_end", "prepare_policy_case", "reset_policy",
        "robotwin_obs_to_xpolicylab", "safe_close_env",
        "sample_absolute_joint_chunk", "sample_absolute_joint_candidates",
        "xpolicylab_action_to_robotwin",
    ):
        globals()[name] = getattr(helpers, name)


def reserved_seed_label(seed: int, task: str = "handover_to_tray") -> str | None:
    ranges = RESERVED_SEED_RANGES if task == "handover_to_tray" else get_adapter(task).reserved_seed_ranges
    for low, high, label in ranges:
        if low <= seed <= high:
            return label
    return None


def format_collection_progress(records: list[dict[str, Any]], saved_hil: int,
                               target: int, max_rollouts: int) -> str:
    """Display saved-data progress and episode-level takeover rates."""
    width = 20
    filled = min(width, width * min(saved_hil, target) // target)
    bar = "#" * filled + "-" * (width - filled)
    valid = [row for row in records if int(row.get("policy_steps", 0)) > 0
             or int(row.get("manual_takeover_requests", 0)) > 0]
    requests = sum(int(row.get("manual_takeover_requests", 0)) > 0 for row in valid)
    triggered = sum(int(row.get("intervention_count", 0)) > 0 for row in valid)
    denominator = len(valid)
    prefix = (f"[PROGRESS] [{bar}] {saved_hil}/{target} valid saved HIL "
              f"({100 * min(saved_hil, target) / target:.0f}%); "
              f"rollouts={len(records)}/{max_rollouts}; ")
    if not denominator:
        return prefix + "takeover rate=n/a (no valid rollout)"
    return (prefix + f"takeover requested={requests}/{denominator} "
            f"({100 * requests / denominator:.1f}%) "
            f"triggered={triggered}/{denominator} "
            f"({100 * triggered / denominator:.1f}%)")


def build_seed_stream(cli: argparse.Namespace):
    """Yield collection seeds: sequential from --seed-start, or random in range."""
    task = getattr(cli, "task", "handover_to_tray")
    if cli.seed_mode == "sequential":
        count = cli.max_scene_attempts if getattr(cli, "episode_protocol", None) == "official-seen" else cli.episodes
        for seed in range(int(cli.seed_start), int(cli.seed_start) + int(count)):
            label = reserved_seed_label(seed, task)
            if label:
                raise ValueError(
                    f"Refusing to collect DAgger data on reserved seed {seed} ({label}). "
                    "Use fresh collection seeds such as 40000+."
                )
            yield seed
        return

    rng = random.Random(int(cli.rng_seed))
    used: set[int] = set()
    low, high = int(cli.seed_min), int(cli.seed_max)
    if low > high:
        raise ValueError("--seed-min must be <= --seed-max")
    # A finite pool must terminate instead of spinning when it is exhausted.
    available = sum(reserved_seed_label(seed, task) is None for seed in range(low, high + 1))
    while len(used) < available:
        seed = rng.randint(low, high)
        if seed in used or reserved_seed_label(seed, task):
            continue
        used.add(seed)
        yield seed


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def joint_bias_dims(bias_dims: str, left_dim: int, right_dim: int) -> list[int]:
    if bias_dims.strip().lower() == "joints":
        dims = list(range(left_dim))
        dims += [left_dim + 1 + index for index in range(right_dim)]
        return dims
    return [int(value) for value in bias_dims.split(",") if value.strip() != ""]


def apply_action_bias(action: np.ndarray, dims: list[int], magnitude: float) -> np.ndarray:
    biased = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    for dim in dims:
        if 0 <= dim < biased.shape[0]:
            biased[dim] += magnitude
    return biased


def validate_policy_action_chunk(adapter, action_chunk, observation) -> None:
    """Check decoded actions, not the wire format's list of action dictionaries.

    Use the same converter as execution without mutating the response or making
    another policy query. Keep per-step conversion in the execution loop so its
    observation-dependent fallbacks and takeover boundaries remain unchanged.
    """
    decoded = []
    for action in action_chunk:
        flat_action, kind = xpolicylab_action_to_robotwin(
            action, action_type="joint", current_observation=observation,
        )
        if kind != "qpos":
            raise ValueError(f"{adapter.name} requires absolute joint actions, got {kind}")
        decoded.append(flat_action)
    adapter.validate_action_chunk(decoded)


def build_runtime_args(cli: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    user_args: dict[str, Any] = {
        "task_name": cli.task,
        "task_config": cli.task_config,
        "policy_name": cli.policy_name,
        "host": cli.host,
        "port": int(cli.port),
        "protocol": "ws",
        "xpolicylab_root": str(ROBOTWIN_ROOT / "XPolicyLab"),
        "ckpt_name": cli.ckpt_name,
        "ckpt_setting": cli.ckpt_name,
        "action_type": "joint",
        "evaluation_id": f"hg-dagger-{cli.task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "trial_id": f"hg-dagger-{cli.task}",
        "action_case_id": f"hg-dagger-{cli.task}",
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
    configure_spectator_camera(args, os.environ.get("HIL_VIEWER_CAMERA_PRESET", "official"))
    user_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    user_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    return user_args, args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Choose the helper snapshot before importing any of its modules.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--runtime-root", type=Path, default=ROBOTWIN_ROOT)
    selected, _ = bootstrap.parse_known_args(argv)
    selected_root = selected.runtime_root.expanduser().resolve()
    _load_collection_helpers(selected_root)
    parser = argparse.ArgumentParser(
        description="Human-gated DAgger with task recovery adapters (press i to intervene)."
    )
    parser.add_argument("--task", choices=registered_tasks(), default="handover_to_tray")
    parser.add_argument("--runtime-root", type=Path, default=ROBOTWIN_ROOT,
                        help="Generated RoboTwin HIL runtime (or ROBOTWIN_ROOT)")
    parser.add_argument("--instruction", help="Exact instruction for this session")
    parser.add_argument("--episode-protocol", choices=("official-seen", "legacy-fixed"),
                        help="official-seen: initial expert check + per-episode official seen prompt")
    parser.add_argument("--instruction-pool-size", type=int, default=50,
                        help="Official description pool size (SFT test_num=50); not the rollout count")
    parser.add_argument("--max-scene-attempts", type=int,
                        help="Candidate seed budget for official-seen; default 10 times --episodes")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18300)
    parser.add_argument("--policy-name", default="Pi_05")
    parser.add_argument("--ckpt-name", default=None)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument(
        "--one-stage", action="store_true",
        help="Scripted stages 1-5, then policy-only stage 6; auto-select recovery branch.",
    )
    parser.add_argument(
        "--paired-baseline-report", type=Path,
        help="Enhanced arm: replay every seed in a completed one-stage baseline report.",
    )
    parser.add_argument(
        "--seed-mode",
        choices=["sequential", "random"],
        default="random",
        help=(
            "random: sample uniformly from [seed-min, seed-max] and skip reserved "
            "demo/dev/test ranges; sequential: use seed-start + rollout index."
        ),
    )
    parser.add_argument("--seed-min", type=int, default=40000)
    parser.add_argument("--seed-max", type=int, default=99999)
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help=(
            "Seed for the --seed-mode random offset stream. Leave unset to draw "
            "fresh OS entropy each run, so repeated runs explore different "
            "collection seeds. Pass a fixed value only for a reproducible order."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=["hil", "expert"],
        default="hil",
        help=(
            "hil: a saved episode with HIL frames counts toward --target-saved; "
            "expert: additionally require expert recovery success."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Maximum policy rollouts; rejected expert-screen candidates do not count.",
    )
    parser.add_argument(
        "--target-saved",
        type=int,
        default=20,
        help="Stop once this many valid (saved) HIL episodes have been collected.",
    )
    parser.add_argument("--frequency", type=int, default=None)
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
        default=None,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume counters and saved raw episodes in --output-dir after a completed session.",
    )
    parser.add_argument("--save-data", default="true")
    parser.add_argument(
        "--save-policy-rollouts", action="store_true",
        help="Also archive completed policy-controlled segments separately under policy_rollouts/; never use them as HIL labels.",
    )
    parser.add_argument(
        "--save-video",
        default="false",
        help=(
            "Deprecated. Collection is raw-first: raw frames are kept and MP4 "
            "export happens offline via export_hg_dagger_dataset.py."
        ),
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=None,
        help="Override the environment step limit for short no-takeover recordings.",
    )
    parser.add_argument(
        "--bias-start-step",
        type=int,
        default=-1,
        help="Start applying a permanent action bias from this policy step (-1 disables).",
    )
    parser.add_argument("--bias-magnitude", type=float, default=0.0)
    parser.add_argument("--bias-dims", default="joints")
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
    parser.add_argument(
        "--auto-label",
        choices=["success", "failure", "none"],
        default="none",
        help="Test-only hook: skip the manual success/failure prompt.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["auto", "manual"],
        default="auto",
        help=(
            "auto: label success/failure from check_success(); "
            "manual: ask the supervisor to press s/f."
        ),
    )
    parser.add_argument(
        "--auto-save",
        choices=["true", "false", "none", "valid_hil"],
        default="none",
        help="Test-only hook: skip the manual save/discard prompt.",
    )
    parser.add_argument(
        "--auto-handback-step",
        type=int,
        default=-1,
        help="Test-only hook: hand back to the policy after this many expert stages.",
    )
    parser.add_argument(
        "--auto-abort-step",
        type=int,
        default=-1,
        help="Test-only hook: simulate pressing x after this many policy steps.",
    )
    parser.add_argument(
        "--takeover-eval", action="store_true",
        help="Run matched-seed rollouts and report the manual i-key request rate.",
    )
    parser.add_argument(
        "--es-activation", choices=("fixed", "manual"), default="fixed",
        help="fixed: use the configured decision window; manual: press e to start a fresh window.",
    )
    parser.add_argument(
        "--es-manual-duration", type=int, default=10,
        help="Number of policy decisions enhanced after pressing e in manual mode.",
    )
    add_sampling_arguments(parser)
    cli = parser.parse_args(argv)
    cli.runtime_root = selected_root
    # Preserve invocation-relative paths before activate_runtime changes cwd
    # for RoboTwin's asset/config lookups.
    for name in ("output_dir", "paired_baseline_report"):
        value = getattr(cli, name)
        if value is not None:
            setattr(cli, name, value.expanduser().resolve())
    for name in ("es_calibration", "es_critic", "es_encoder_weights", "es_log_dir"):
        value = getattr(cli, name)
        if value is not None:
            setattr(cli, name, str(Path(value).expanduser().resolve()))
    adapter = get_adapter(cli.task)
    cli.task_config = cli.task_config or adapter.default_task_config
    cli.ckpt_name = cli.ckpt_name or adapter.default_checkpoint_name
    cli.frequency = cli.frequency if cli.frequency is not None else adapter.default_frequency
    cli.episode_protocol = cli.episode_protocol or adapter.default_episode_protocol
    if cli.episode_protocol == "official-seen":
        if cli.instruction is not None:
            parser.error("official-seen generates each episode's prompt; --instruction is incompatible")
        if cli.one_stage or cli.takeover_eval:
            parser.error("official-seen is not the legacy one-stage/matched takeover protocol")
    else:
        cli.instruction = cli.instruction if cli.instruction is not None else adapter.prompt
        if not cli.instruction or not cli.instruction.strip():
            parser.error("--instruction must not be empty")
    if cli.max_scene_attempts is None:
        cli.max_scene_attempts = 10 * cli.episodes
    if cli.max_scene_attempts < 1 or cli.instruction_pool_size < 1:
        parser.error("max-scene-attempts and instruction-pool-size must be positive")
    cli.output_dir = cli.output_dir or (
        cli.runtime_root / "data" / f"{cli.task}_hg_dagger_r1" / cli.task / "aloha_agilex"
    )
    if cli.one_stage and not adapter.supports_one_stage:
        parser.error(f"{cli.task} does not support --one-stage")
    if cli.es_mode != "off" and not adapter.supports_sampling:
        parser.error(f"{cli.task}: task-matched full-horizon sampling is not validated; use --es-mode off")
    if min(cli.episodes, cli.target_saved, cli.frequency, cli.save_freq) < 1:
        parser.error("episodes, target-saved, frequency and save-freq must be positive")
    return cli


def main(argv: list[str] | None = None) -> int:
    global ROBOTWIN_ROOT
    cli = parse_args(argv)
    adapter = get_adapter(cli.task)
    instruction = cli.instruction
    ROBOTWIN_ROOT = activate_runtime(cli.runtime_root)
    _load_runtime_helpers()
    if cli.one_stage:
        if cli.target_mode != "expert" or cli.seed_mode != "sequential":
            raise ValueError("One-stage collection requires --target-mode expert and sequential seeds")
        if cli.es_activation != "fixed" or cli.takeover_eval or cli.auto_handback_step >= 0:
            raise ValueError("One-stage collection requires fixed sampling and full expert recovery")
        if cli.bias_magnitude or cli.label_mode != "auto":
            raise ValueError("One-stage comparison disallows action bias and manual outcome labels")
        if cli.paired_baseline_report and cli.es_mode not in {"enhanced", "gaussian"}:
            raise ValueError("Paired baseline report is only for enhanced or gaussian arms")
    elif cli.paired_baseline_report or cli.auto_save == "valid_hil":
        raise ValueError("Paired baseline report and valid_hil autosave require --one-stage")
    paired_report_sha = None
    paired_seeds = None
    paired_handoffs = None
    if cli.paired_baseline_report:
        source = cli.paired_baseline_report.expanduser().resolve()
        paired_seeds = baseline_seeds(
            source, checkpoint=cli.ckpt_name, task_config=cli.task_config,
            frequency=int(cli.frequency),
        )
        paired_report_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        paired_handoffs = baseline_handoffs(source)
        cli.episodes = len(paired_seeds)
    if cli.render_freq <= 0 and os.environ.get("HIL_DIAG_ALLOW_NO_VIEWER") != "1":
        raise ValueError("--render-freq must be positive for human supervision")
    if cli.es_activation == "manual":
        if cli.es_mode == "off" or cli.es_manual_duration < 1:
            raise ValueError("Manual sampling requires vanilla/enhanced/gaussian mode and a positive duration")
        cli.es_window_start = 0
        cli.es_window_end = cli.es_manual_duration - 1
    if cli.takeover_eval:
        if cli.acceptance or cli.seed_mode != "sequential":
            raise ValueError("Takeover comparison requires sequential matched seeds and no acceptance mode")
        if cli.auto_intervene_step >= 0 or cli.auto_abort_step >= 0:
            raise ValueError("Takeover comparison cannot use automatic intervention or abort hooks")
        if cli.es_window_start is None or cli.es_window_end is None:
            raise ValueError("Takeover comparison requires the same declared window in all three arms")
    if cli.resume and cli.acceptance:
        raise ValueError("Acceptance runs cannot resume a collection")
    if (cli.es_mode != "off" or cli.takeover_eval) and cli.bias_magnitude:
        raise ValueError("Action bias is incompatible with simulated candidate scoring")
    from coverage_sampling.core import config_record
    sampling_config = config_from_args({
        **vars(cli), "task_name": cli.task, "action_type": "joint", "eval_batch": False,
    })
    scorer = None
    if sampling_config is not None and sampling_config.requires_scores:
        from coverage_sampling.critic import create_online_scorer

        scorer = create_online_scorer(vars(cli), sampling_config)
        print(f"[SAMPLING] {sampling_config.score_basis}; beta={sampling_config.beta:g}; "
              f"scale={sampling_config.score_scale:g}; K={sampling_config.num_candidates}; "
              f"policy_batch={sampling_config.policy_microbatch}; full-chunk weighted mean", flush=True)
    elif sampling_config is not None:
        print(
            f"[SAMPLING] output-space isotropic Gaussian; sigma={sampling_config.gaussian_sigma:g} rad; "
            f"K={sampling_config.num_candidates}; policy_batch={sampling_config.policy_microbatch}; "
            "base=uniform candidate mean; arm joints only; grippers unchanged",
            flush=True,
        )
    if cli.acceptance:
        cli.episodes = 1
        cli.save_data = "false"
    if cli.seed_mode == "random" and cli.rng_seed is None:
        # Fresh entropy per run: repeated launches must not replay the same
        # collection-seed sequence, otherwise "random" adds no diversity.
        cli.rng_seed = random.SystemRandom().randrange(1, 2**31)
        print(
            f"[SEEDS] random mode, run rng-seed={cli.rng_seed} "
            "(pass --rng-seed to reproduce this order)"
        )
    seed_iter = iter(paired_seeds) if paired_seeds is not None else build_seed_stream(cli)
    cli.output_dir = cli.output_dir.expanduser().resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = collection_lock(cli.output_dir)
    expected_resume = {
        "sampling_mode": cli.es_mode,
        "sampling_activation": cli.es_activation,
        "sampling_config": config_record(sampling_config) if sampling_config is not None else None,
        "target_mode": cli.target_mode,
        "checkpoint_name": cli.ckpt_name,
        "task_config": cli.task_config,
        "instruction": instruction,
        "frequency": int(cli.frequency),
        "seed_mode": cli.seed_mode,
        "critic_sha256": scorer.metadata["critic_sha256"] if scorer is not None else None,
    }
    if cli.episode_protocol == "official-seen":
        expected_resume.update(episode_protocol=cli.episode_protocol,
                               instruction_pool_size=cli.instruction_pool_size)
    if adapter.requires_training_gate:
        expected_resume.update(task=cli.task, adapter=adapter.name)
    if cli.save_policy_rollouts:
        expected_resume["save_policy_rollouts"] = True
    if cli.one_stage:
        expected_resume.update(one_stage=True, paired_baseline_sha256=paired_report_sha)
    progress = inspect_collection(cli.output_dir, target_mode=cli.target_mode,
                                  expected=expected_resume, resume=cli.resume)
    screen_history = screening_history(cli.output_dir)
    if screen_history and cli.episode_protocol != "official-seen":
        raise ValueError("Cannot resume a screened collection with legacy-fixed protocol")
    if cli.resume:
        previous_range = read_json_object(Path(progress.source_report)).get("seed_range")
        expected_start = [int(cli.seed_min), int(cli.seed_max)] if cli.seed_mode == "random" else None
        if expected_start is not None and previous_range != expected_start:
            raise ValueError(f"Resume seed range mismatch: {previous_range} != {expected_start}")
        if cli.seed_mode == "sequential" and previous_range[0] != int(cli.seed_start):
            raise ValueError("Resume seed-start differs from the previous session")
        preserved = preserve_incomplete_cache(cli.output_dir)
        print(f"[RESUME] next rollout={progress.next_rollout_index + 1}/{cli.episodes}, "
              f"next raw episode={progress.next_episode_index}, "
              f"saved valid HIL={progress.saved_valid_hil}/{cli.target_saved}")
        for path in preserved:
            print(f"[RESUME] preserved unfinished frames: {path}")

    user_args, args = build_runtime_args(cli)
    task_env = class_decorator(cli.task)
    model_client = build_policy_client(user_args)
    save_data = parse_bool(cli.save_data)
    if cli.save_policy_rollouts and not save_data:
        raise ValueError("--save-policy-rollouts requires --save-data true")
    if parse_bool(cli.save_video):
        print(
            "[HG-DAGGER] --save-video is deprecated in collection; "
            "raw frames are saved and MP4 export is done offline.",
            flush=True,
        )
    auto_label = None if cli.auto_label == "none" else cli.auto_label
    auto_save = (False if cli.takeover_eval else None) if cli.auto_save == "none" else (
        "valid_hil" if cli.auto_save == "valid_hil" else parse_bool(cli.auto_save)
    )
    bias_dims = joint_bias_dims(
        cli.bias_dims,
        int(user_args["left_arm_dim"]),
        int(user_args["right_arm_dim"]),
    )
    data_episode_index = progress.next_episode_index
    records: list[dict[str, Any]] = list(progress.records)
    used_seeds = set(progress.used_seeds)
    used_seeds.update(screen_history)
    scene_attempts = len(screen_history)
    scene_budget_exhausted = False
    aborted = False
    sampling_log = None

    print("\n" + "=" * 72)
    print(f"RoboTwin {cli.task} human-gated DAgger")
    if cli.one_stage:
        print("One-stage: scripted expert stages 1-5; policy controls stage 6 only.")
        print("Keys: i=takeover (automatic expert recovery), x=abort episode, q=quit.")
    else:
        print("Keys: i=takeover, r=hand back to policy, x=abort episode, q=quit.")
    if cli.es_activation == "manual":
        print(f"Press e during policy control to toggle {cli.es_manual_duration} sampled decisions.")
    print("After each episode: s/f=success/failure, y/n=save/discard.")
    print(f"Prompt: {instruction or 'official seen, generated and frozen separately per accepted scene'}")
    print(f"Episode protocol: {cli.episode_protocol}")
    print(f"Sampling: {cli.es_mode}; takeover evaluation: {cli.takeover_eval}")
    print(f"Output: {cli.output_dir}")
    print("=" * 72 + "\n")

    try:
        if sampling_config is not None and min(
            sampling_config.num_candidates, sampling_config.policy_microbatch
        ) > 8:
            requested_batch = min(sampling_config.num_candidates, sampling_config.policy_microbatch)
            try:
                capability = model_client.call(func_name="candidate_sampling_info")
                if not isinstance(capability, dict) or capability.get("max_batch_size", 0) < requested_batch:
                    raise ValueError(f"Server capability: {capability!r}")
            except Exception as exc:
                raise RuntimeError(
                    f"Policy batch={requested_batch} requires the updated Pi0.5 service. "
                    "Restart the policy server with the updated adapter before collecting; "
                    "an already running process does not reload source changes."
                ) from exc
        rollout_index = progress.next_rollout_index
        saved_hil_count = progress.saved_valid_hil
        aborted_rollouts = progress.aborted_rollouts
        session_started = time.time()
        print(format_collection_progress(records, saved_hil_count,
                                         int(cli.target_saved), int(cli.episodes)), flush=True)
        while (
            rollout_index < int(cli.episodes)
            and (cli.takeover_eval or paired_seeds is not None or saved_hil_count < int(cli.target_saved))
        ):
            if cli.episode_protocol == "official-seen" and scene_attempts >= cli.max_scene_attempts:
                scene_budget_exhausted = True
                print("[EXPERT-SCREEN] Candidate budget exhausted; no unchecked scenes will run.")
                break
            seed = next(seed_iter, None)
            while seed is not None and seed in used_seeds:
                seed = next(seed_iter, None)
            if seed is None:
                scene_budget_exhausted = True
                print("[SEEDS] Candidate pool exhausted; ending collection without reusing seeds.")
                break
            used_seeds.add(seed)
            rollout_started = time.time()
            args["save_data"] = False
            args["need_plan"] = True
            args["eval_mode"] = True
            initial_scene_screening = None
            if cli.episode_protocol == "official-seen":
                scene_attempts += 1
                print(f"[EXPERT-SCREEN] candidate={scene_attempts}/{cli.max_scene_attempts} seed={seed}; no training frames", flush=True)
                initial_scene_screening = prepare_screened_scene(
                    task_env, args, seed=seed, episode_index=data_episode_index,
                    adapter=adapter, build_instruction=build_instruction,
                    instruction_pool_size=cli.instruction_pool_size,
                    close_env=safe_close_env, output_dir=cli.output_dir,
                )
                if initial_scene_screening is None:
                    print(f"[EXPERT-SCREEN] rejected seed={seed}; not a policy failure", flush=True)
                    continue
                instruction = initial_scene_screening["instruction"]
                print(f"[EXPERT-SCREEN] accepted seed={seed}; prompt={instruction}", flush=True)
            else:
                task_env.setup_demo(
                    now_ep_num=data_episode_index,
                    seed=seed,
                    is_test=False,
                    **args,
                )
                adapter.initialize_episode(task_env)
            if cli.step_limit is not None:
                task_env.step_lim = int(cli.step_limit)
            task_env.set_instruction(instruction)
            render_initial_frame(task_env)
            if rollout_index == 0:
                setup_viewer_diagnostics(task_env)
            install_viewer_frame_limit(task_env)
            stage6_handoff = None
            if cli.one_stage:
                task_env.current_control_source = "expert_prefix"
                prefix_ok = adapter.play_prefix_until_placement(task_env)
                stage6_handoff = adapter.handoff_record(task_env)
                stage6_handoff["prefix_plan_success"] = bool(prefix_ok)
                paired_prefix_error = None
                if paired_handoffs is not None:
                    baseline_handoff = paired_handoffs[int(seed)]
                    if not baseline_handoff.get("ready"):
                        paired_prefix_error = "baseline_prefix_not_ready"
                    elif stage6_handoff["ready"]:
                        differences = handoff_pair_differences(
                            baseline_handoff, stage6_handoff,
                        )
                        stage6_handoff["paired_differences"] = differences
                        if not handoff_pair_ready(differences):
                            paired_prefix_error = "handoff_state_mismatch"
                if not stage6_handoff["ready"] or paired_prefix_error:
                    status = "paired_prefix_excluded" if paired_prefix_error else "prefix_failed"
                    stage6_handoff["paired_prefix_error"] = paired_prefix_error
                    print(f"\n[ONE-STAGE] seed={seed} stage-6 handoff excluded: "
                          f"{stage6_handoff['state']}; pairing={paired_prefix_error}", flush=True)
                    records.append({
                        "rollout_index": rollout_index, "seed": int(seed),
                        "one_stage": True, "rollout_status": status,
                        "stage6_handoff": stage6_handoff, "policy_steps": 0,
                        "manual_takeover_requests": 0, "intervention_count": 0,
                        "autonomous_success": False, "save_decision": False,
                        "rollout_seconds": round(time.time() - rollout_started, 2),
                    })
                    notify_trial_end(model_client, cli.task, seed, False)
                    safe_close_env(task_env)
                    rollout_index += 1
                    continue
                print(f"\n[ONE-STAGE] seed={seed} expert stages 1-5 complete; "
                      "policy starts at stage 6.", flush=True)
            prepare_policy_case(model_client, cli.task, seed, instruction, "joint")
            reset_policy(model_client)

            # Record the full episode from this point. ``_take_picture`` tags
            # each saved frame with the current control source, so the HDF5
            # keeps a per-frame policy/HIL mask.
            task_env.save_data = save_data
            task_env.save_dir = str(cli.output_dir)
            # Raw-first collection: never encode MP4/HDF5 inside the control loop.
            task_env.save_video = False
            task_env.FRAME_IDX = 0
            task_env.folder_path = {}
            task_env.ep_num = int(data_episode_index)
            task_env.control_mask = []
            task_env.current_control_source = "policy"
            if save_data:
                task_env._take_picture()

            mode = "policy"
            policy_steps = 0
            decision_step = 0
            intervention_count = 0
            manual_takeover_requests = 0
            takeover_request_events: list[dict[str, Any]] = []
            sampling_activation_events: list[dict[str, Any]] = []
            segments: list[dict[str, Any]] = []
            interventions: list[dict[str, Any]] = []
            current_source = "policy"
            source_start_step = 0
            recovery_iter = None
            expert_stage_count = 0
            last_expert_progress: dict[str, Any] | None = None
            quit_requested = False
            episode_aborted = False
            last_expert_result: dict[str, Any] | None = None

            def switch_source(new_source: str) -> None:
                nonlocal current_source, source_start_step
                if current_source == new_source:
                    return
                append_segment(
                    segments,
                    current_source,
                    source_start_step,
                    int(task_env.FRAME_IDX),
                )
                current_source = new_source
                source_start_step = int(task_env.FRAME_IDX)

            def finish_sampling_log(status: str, error: str | None = None) -> None:
                nonlocal sampling_log
                if sampling_log is None:
                    return
                sampling_log.write({
                    "event": "hil_outcome", "status": status,
                    "manual_takeover_requests": manual_takeover_requests,
                    "interventions": intervention_count,
                    "policy_steps": policy_steps,
                })
                sampling_log.finish(bool(task_env.eval_success), error, policy_steps)
                sampling_log = None

            print(
                f"\n\033[96m[ROLLOUT {rollout_index + 1}/max {cli.episodes} | "
                f"saved valid HIL ({cli.target_mode}) {saved_hil_count}/{cli.target_saved}] "
                f"seed={seed}; "
                f"{'e=sampling, ' if cli.es_activation == 'manual' else ''}"
                +
                ("i=takeover, x=abort episode, q=quit.\033[0m"
                 if cli.one_stage else
                 "i=takeover, r=handback, x=abort episode, q=quit.\033[0m")
            )

            key_map = {**DEFAULT_KEYS, **({"e": "sampling_toggle"} if cli.es_activation == "manual" else {})}
            if cli.one_stage:
                key_map.pop("r", None)
            with HumanInterventionInput(task_env, key_map=key_map) as keyboard:
                sampler = None
                manual_window = None

                def poll_live_control() -> None:
                    # Draw and poll on the main thread; never step physics here.
                    task_env._render_viewer_if_available()
                    pending = keyboard.poll()
                    if pending in {"intervene", "abort", "quit", "sampling_toggle"}:
                        print(f"\n[CONTROL] {pending} requested; waiting for policy RPC to drain.", flush=True)
                        raise OperatorInterrupt(pending)

                if sampling_config is not None:
                    from coverage_sampling.core import DecisionSampler, EpisodeLog, ManualSamplingWindow

                    if cli.es_activation == "manual":
                        manual_window = ManualSamplingWindow(cli.es_manual_duration)

                    def toggle_sampling() -> None:
                        enabled = manual_window.toggle(decision_step)
                        record = {
                            "event": "sampling_activation", "enabled": enabled,
                            "decision": int(decision_step), "policy_steps": int(policy_steps),
                            "frame_idx": int(task_env.FRAME_IDX),
                            "window_end_exclusive": int(decision_step + cli.es_manual_duration) if enabled else None,
                        }
                        sampling_activation_events.append(record)
                        sampling_log.write(record)
                        status = (f"ON for decisions {decision_step}–{decision_step + cli.es_manual_duration - 1}"
                                  if enabled else "OFF")
                        print(f"\n[SAMPLING] {status}; policy remains in control.", flush=True)

                    sampler = DecisionSampler(sampling_config, None, seed)
                    log_dir = Path(cli.es_log_dir) if cli.es_log_dir else cli.output_dir / "sampling"
                    sampling_log = EpisodeLog(
                        log_dir / f"episode_{rollout_index:04d}_seed_{seed}.jsonl",
                        sampling_config,
                        {**(scorer.metadata if scorer is not None else {
                            "scorer_kind": None,
                            "score_semantics": None,
                            "gaussian_noise_space": "absolute_joint_chunk_output",
                            "gaussian_noise_grippers": False,
                        }), "episode_seed": seed, "instruction": instruction,
                         "policy_checkpoint": cli.ckpt_name, "task_config": cli.task_config,
                         "frequency": int(cli.frequency), "evaluation_type": "human_gated",
                         "selection_score_basis": sampling_config.score_basis,
                         "sampling_activation": cli.es_activation,
                         "manual_duration": cli.es_manual_duration if manual_window is not None else None,
                         "executor": "Base_Task.take_action(qpos)",
                         "window_units": ("zero-based policy decisions relative to e activation; inclusive"
                                          if manual_window is not None else
                                          "zero-based policy decision calls; inclusive")},
                    )
                while True:
                    event = keyboard.poll()
                    if event == "quit":
                        quit_requested = True
                        break
                    if event == "abort":
                        episode_aborted = True
                        break
                    if is_episode_end(task_env):
                        break

                    if mode == "policy":
                        if event == "sampling_toggle":
                            toggle_sampling()
                            continue
                        if event == "intervene":
                            manual_takeover_requests += 1
                            request_event = {
                                "policy_steps": int(policy_steps), "decision": int(decision_step),
                                "frame_idx": int(task_env.FRAME_IDX),
                                "elapsed_seconds": round(time.time() - rollout_started, 2),
                                "accepted": False,
                            }
                            takeover_request_events.append(request_event)
                            if sampling_log is not None:
                                sampling_log.write({"event": "takeover_request", "policy_steps": policy_steps,
                                                    "decision": decision_step, "source": "human"})
                            if cli.one_stage:
                                recovery_iter = adapter.recovery_stage_iterator(task_env)
                                chosen_stage = None
                            else:
                                recovery_iter, chosen_stage = choose_recovery_stage(
                                    task_env, keyboard, adapter
                                )
                            if recovery_iter is None:
                                print(
                                    "\n\033[93m[HG-DAGGER] 已取消本次接管，继续策略执行。\033[0m"
                                )
                                continue
                            request_event["accepted"] = True
                            switch_source("hil")
                            intervention_count += 1
                            interventions.append(
                                {
                                    "intervention_index": intervention_count,
                                    "policy_steps_before_intervention": int(policy_steps),
                                    "start_step": int(task_env.FRAME_IDX),
                                    "handback_step": None,
                                    "expert_done": False,
                                    "chosen_stage_id": chosen_stage,
                                }
                            )
                            mode = "expert"
                            print(
                                "\n\033[93m[HG-DAGGER] i detected: policy chunk discarded.\033[0m"
                            )
                            print("\033[93m[HG-DAGGER] Expert takeover; "
                                  + ("automatic recovery to completion." if cli.one_stage
                                     else "press r to hand back.") + "\033[0m")
                            continue

                        observation = task_env.get_obs()
                        xpl_obs = robotwin_obs_to_xpolicylab(
                            observation,
                            instruction=instruction,
                            env_idx=0,
                            frequency=int(cli.frequency),
                            task_env=task_env,
                        )
                        responsive_wait = getattr(model_client, "responsive_wait", None)
                        # Keep painting while uploading an observation. Control
                        # keys remain queued until the action RPC below drains.
                        with (responsive_wait(task_env._render_viewer_if_available)
                              if responsive_wait else nullcontext()):
                            model_client.call(func_name="update_obs", obs=xpl_obs)
                        chunk_interrupted = False
                        manual_trigger = True
                        selection = None
                        if sampler is not None:
                            def sample_chunk():
                                poll_live_control()
                                responsive_wait = getattr(model_client, "responsive_wait", None)
                                with (responsive_wait(poll_live_control) if responsive_wait else nullcontext()):
                                    chunk = sample_absolute_joint_chunk(model_client, observation, "joint")
                                poll_live_control()
                                return chunk

                            def sample_candidates(count):
                                poll_live_control()
                                responsive_wait = getattr(model_client, "responsive_wait", None)
                                with (responsive_wait(poll_live_control) if responsive_wait else nullcontext()):
                                    chunks = sample_absolute_joint_candidates(
                                        model_client, observation, "joint", count,
                                        sampling_config.policy_microbatch,
                                    )
                                poll_live_control()
                                return chunks

                            try:
                                action_chunk, selection = sampler.select(
                                    decision_step, sample_chunk,
                                    sample_candidates=sample_candidates,
                                    score_candidates=(scorer.bind_observation(observation)
                                                      if scorer is not None else None),
                                    active_override=manual_window.active(decision_step) if manual_window is not None else None,
                                )
                            except OperatorInterrupt as interruption:
                                event = interruption.event
                                action_chunk = []
                                chunk_interrupted = True
                                if event == "quit":
                                    quit_requested = True
                            else:
                                sampling_log.decision(selection)
                        else:
                            try:
                                with (responsive_wait(poll_live_control) if responsive_wait else nullcontext()):
                                    action_chunk = normalize_action_chunk(
                                        model_client.call(func_name="get_action")
                                    )
                                poll_live_control()
                            except OperatorInterrupt as interruption:
                                event = interruption.event
                                action_chunk = []
                                chunk_interrupted = True
                                if event == "quit":
                                    quit_requested = True
                        if len(action_chunk) == 0 and not chunk_interrupted:
                            raise RuntimeError("Policy returned an empty action chunk.")
                        if not chunk_interrupted:
                            validate_policy_action_chunk(adapter, action_chunk, observation)
                        if not chunk_interrupted:
                            decision_step += 1

                        for action_idx, action in enumerate(action_chunk):
                            event = keyboard.poll()
                            if event in {"quit", "abort", "intervene", "sampling_toggle"}:
                                if event == "quit":
                                    quit_requested = True
                                chunk_interrupted = True
                                break

                            flat_action, robotwin_action_type = xpolicylab_action_to_robotwin(
                                action,
                                action_type="joint",
                                current_observation=observation,
                            )
                            if (
                                cli.bias_start_step >= 0
                                and policy_steps >= cli.bias_start_step
                                and cli.bias_magnitude
                            ):
                                flat_action = apply_action_bias(
                                    flat_action,
                                    bias_dims,
                                    cli.bias_magnitude,
                                )
                            should_score = scorer is not None and selection["active"] and action_idx == 0
                            actual_coverage = None
                            if should_score:
                                actual_coverage = (selection["blended_chunk_score"]
                                                   if sampling_config.score_representation == "full_chunk"
                                                   else scorer(observation, flat_action))
                            pending = keyboard.poll() if scorer is not None else None
                            if pending in {"quit", "abort", "intervene", "sampling_toggle"}:
                                event = pending
                                if event == "quit":
                                    quit_requested = True
                                chunk_interrupted = True
                                break
                            task_env.current_control_source = "policy"
                            task_env.take_action(
                                flat_action,
                                action_type=robotwin_action_type,
                            )
                            policy_steps += 1
                            if sampling_log is not None:
                                predicted = None
                                if selection["active"] and selection["selected"] is not None:
                                    curve = selection["branches"][selection["selected"]]["coverage"]
                                    if action_idx < len(curve):
                                        predicted = curve[action_idx]
                                sampling_log.executed(
                                    decision_step - 1, action_idx, actual_coverage,
                                    selection["active"], flat_action,
                                    coverage_basis=sampling_config.score_basis,
                                    coverage_time="pre_execution_decision_observation",
                                    predicted_coverage=predicted,
                                    prediction_error=actual_coverage - predicted if predicted is not None else None,
                                    success=bool(task_env.eval_success), rollout_step=policy_steps,
                                )
                            if save_data and policy_steps % int(cli.save_freq) == 0:
                                task_env._take_picture()

                            if is_episode_end(task_env):
                                chunk_interrupted = True
                                break
                            if (
                                cli.auto_intervene_step >= 0
                                and policy_steps >= cli.auto_intervene_step
                                and intervention_count == 0
                            ):
                                event = "intervene"
                                manual_trigger = False
                                chunk_interrupted = True
                                break
                            if (
                                cli.auto_abort_step >= 0
                                and policy_steps >= cli.auto_abort_step
                            ):
                                event = "abort"
                                chunk_interrupted = True
                                break

                            # Pi0.5 consumes only the latest observation at the next
                            # chunk boundary. The critic scores the first action at
                            # that observation; later actions need no extra image.
                            chunk_observations = (
                                cli.policy_name == "Pi_05_RobotTwin"
                                and sampler is not None
                                and os.environ.get("HIL_CHUNK_OBSERVATIONS", "0") == "1"
                            )
                            if not chunk_observations:
                                observation = task_env.get_obs()
                            if not chunk_observations:
                                xpl_obs = robotwin_obs_to_xpolicylab(
                                    observation,
                                    instruction=instruction,
                                    env_idx=0,
                                    frequency=int(cli.frequency),
                                    task_env=task_env,
                                )
                                model_client.call(func_name="update_obs", obs=xpl_obs)

                        if quit_requested:
                            break
                        if is_episode_end(task_env):
                            break
                        if chunk_interrupted and event == "abort":
                            episode_aborted = True
                            break
                        if chunk_interrupted and event == "sampling_toggle":
                            toggle_sampling()
                            continue
                        if chunk_interrupted and event == "intervene":
                            if manual_trigger:
                                manual_takeover_requests += 1
                                request_event = {
                                    "policy_steps": int(policy_steps), "decision": int(decision_step),
                                    "frame_idx": int(task_env.FRAME_IDX),
                                    "elapsed_seconds": round(time.time() - rollout_started, 2),
                                    "accepted": False,
                                }
                                takeover_request_events.append(request_event)
                                if sampling_log is not None:
                                    sampling_log.write({"event": "takeover_request", "policy_steps": policy_steps,
                                                        "decision": decision_step, "source": "human"})
                            if cli.one_stage:
                                recovery_iter = adapter.recovery_stage_iterator(task_env)
                                chosen_stage = None
                            else:
                                recovery_iter, chosen_stage = choose_recovery_stage(
                                    task_env, keyboard, adapter
                                )
                            if recovery_iter is None:
                                print(
                                    "\n\033[93m[HG-DAGGER] 已取消本次接管，继续策略执行。\033[0m"
                                )
                                continue
                            if manual_trigger:
                                request_event["accepted"] = True
                            switch_source("hil")
                            intervention_count += 1
                            interventions.append(
                                {
                                    "intervention_index": intervention_count,
                                    "policy_steps_before_intervention": int(policy_steps),
                                    "start_step": int(task_env.FRAME_IDX),
                                    "handback_step": None,
                                    "expert_done": False,
                                    "chosen_stage_id": chosen_stage,
                                }
                            )
                            mode = "expert"
                            print(
                                "\n\033[93m[HG-DAGGER] i detected: policy chunk discarded.\033[0m"
                            )
                            print("\033[93m[HG-DAGGER] Expert takeover; "
                                  + ("automatic recovery to completion." if cli.one_stage
                                     else "press r to hand back.") + "\033[0m")
                            continue
                    else:
                        # Expert mode. The supervisor can hand back at any
                        # stage boundary. Each ``next()`` executes one stage.
                        if not cli.one_stage and (event == "handback" or (
                            cli.auto_handback_step >= 0
                            and expert_stage_count >= cli.auto_handback_step
                        )):
                            if interventions:
                                interventions[-1]["handback_step"] = int(task_env.FRAME_IDX)
                                interventions[-1]["expert_progress"] = last_expert_progress
                            switch_source("policy")
                            mode = "policy"
                            reset_policy(model_client)
                            if sampler is not None:
                                sampler.checked_replay = False
                            print("\033[93m[HG-DAGGER] Hand back to policy.\033[0m")
                            continue

                        task_env.current_control_source = "hil"
                        step = next(recovery_iter, None)
                        expert_stage_count += 1
                        if step is None:
                            step = {"phase": "done", "result": {
                                "success": False, "plan_success": False,
                                "reason": "expert_iterator_ended_without_result",
                            }}
                        if step.get("phase") == "done":
                            last_expert_result = step.get("result") or {}
                            if interventions:
                                interventions[-1]["expert_done"] = bool(last_expert_result.get("success"))
                                interventions[-1]["end_step"] = int(task_env.FRAME_IDX)
                                interventions[-1]["expert_result"] = last_expert_result
                            if last_expert_result.get("success"):
                                break
                            reason = last_expert_result.get("reason") or "expert_recovery_failed"
                            print(f"[HG-DAGGER] 专家恢复未完成：{reason}", flush=True)
                            if cli.one_stage or cli.acceptance or cli.auto_intervene_step >= 0:
                                # Noninteractive diagnostics retain their finite fail result.
                                break
                            # A failed iterator is not a terminal environment outcome.
                            # Hold the scene, keep pumping the viewer, and let the
                            # operator retry, hand back, abort, or quit explicitly.
                            if interventions:
                                interventions[-1].setdefault("recovery_failures", []).append(last_expert_result)
                            while True:
                                response = prompt_choice(
                                    task_env, {"i": "retry", "r": "handback", "x": "abort"},
                                    "[HG-DAGGER] 本局暂停（未结束）：i=重新选择阶段，r=交还策略，x=作废本局，q=退出并保留缓存",
                                )
                                if response != "retry":
                                    break
                                retry_iter, chosen_stage = choose_recovery_stage(task_env, keyboard, adapter)
                                if retry_iter is None:
                                    print("[HG-DAGGER] 已取消重选，仍保持暂停；请明确交还或退出。", flush=True)
                                    continue
                                recovery_iter = retry_iter
                                expert_stage_count = 0
                                if interventions:
                                    interventions[-1].setdefault("recovery_retries", []).append(
                                        {"chosen_stage_id": chosen_stage, "frame_idx": int(task_env.FRAME_IDX)})
                                break
                            if response == "abort":
                                episode_aborted = True
                                break
                            if response == "handback":
                                if interventions:
                                    interventions[-1]["handback_step"] = int(task_env.FRAME_IDX)
                                switch_source("policy")
                                mode = "policy"
                                reset_policy(model_client)
                                if sampler is not None:
                                    sampler.checked_replay = False
                                print("[HG-DAGGER] Hand back to policy; requesting a fresh chunk.", flush=True)
                            continue
                        last_expert_progress = step

            # Close the final open segment.
            append_segment(
                segments,
                current_source,
                source_start_step,
                int(task_env.FRAME_IDX),
            )

            if quit_requested:
                aborted = True
                finish_sampling_log("quit")
                notify_trial_end(model_client, cli.task, seed, False)
                safe_close_env(task_env)
                break

            if episode_aborted:
                # The supervisor judged this rollout unrecoverable: drop it
                # immediately instead of waiting for the episode to finish or
                # sitting through the save prompt.
                aborted_rollouts += 1
                finish_sampling_log("aborted")
                discard_recovery_cache(task_env)
                notify_trial_end(model_client, cli.task, seed, False)
                print(
                    f"\n\033[93m[EPISODE] aborted by supervisor after "
                    f"{policy_steps} policy steps "
                    f"(frame_idx={int(task_env.FRAME_IDX)}, "
                    f"interventions={intervention_count}); discarded, next rollout.\033[0m"
                )
                aborted_record = {
                    "rollout_index": rollout_index, "seed": int(seed),
                    "one_stage": bool(cli.one_stage),
                    "stage6_handoff": stage6_handoff,
                    "sampling_mode": cli.es_mode, "rollout_status": "aborted",
                    "policy_steps": int(policy_steps),
                    "manual_takeover_requests": manual_takeover_requests,
                    "takeover_request_events": takeover_request_events,
                    "sampling_activation_events": sampling_activation_events,
                    "intervention_count": intervention_count,
                    "interventions": interventions,
                    "autonomous_success": False, "save_decision": False,
                    "instruction": instruction,
                    "initial_scene_screening": initial_scene_screening,
                    "rollout_seconds": round(time.time() - rollout_started, 2),
                }
                records.append(aborted_record)
                if cli.takeover_eval:
                    append_jsonl(cli.output_dir / "takeover_rollouts.jsonl", aborted_record)
                safe_close_env(task_env)
                rollout_index += 1
                print(format_collection_progress(records, saved_hil_count,
                                                 int(cli.target_saved), int(cli.episodes)), flush=True)
                continue

            joints_legal, joint_absmax = task_env.planned_joints_legal()
            final_success_metrics = adapter.success_metrics(task_env)
            final_check_success = task_env.check_success()
            hil_frames = sum(
                1 for item in task_env.control_mask if str(item).lower() == "hil"
            )
            valid_hil = is_valid_hil(
                interventions=intervention_count, hil_frames=hil_frames,
                expert_result=last_expert_result, final_success=bool(final_check_success),
                joints_legal=bool(joints_legal),
            ) if cli.one_stage else False
            if adapter.requires_training_gate:
                diagnostic_run = bool(cli.auto_intervene_step >= 0 or cli.auto_handback_step >= 0
                                      or cli.step_limit is not None or cli.render_freq <= 0 or cli.bias_magnitude)
                if adapter.default_episode_protocol == "official-seen" and cli.episode_protocol != "official-seen":
                    diagnostic_run = True
                valid_hil = bool(not diagnostic_run and intervention_count > 0 and hil_frames > 0 and final_check_success
                                 and joints_legal and final_success_metrics.get("physical_quality_gate")
                                 and final_success_metrics.get("recovery_ok")
                                 and (cli.target_mode != "expert" or (last_expert_result or {}).get("success")))

            if auto_label is not None:
                supervisor_label = auto_label
            elif cli.label_mode == "auto":
                # The program already decides task success; the supervisor only
                # decides whether this trajectory is worth keeping.
                supervisor_label = "success" if final_check_success else "failure"
                print(
                    f"[TRAJECTORY] auto label={supervisor_label} "
                    f"(check_success={bool(final_check_success)})"
                )
            else:
                supervisor_label = prompt_choice(
                    task_env,
                    {"s": "success", "f": "failure"},
                    "[TRAJECTORY] Mark trajectory: [s]uccess / [f]ailure",
                )

            if adapter.requires_training_gate:
                valid_hil = bool(valid_hil and supervisor_label == "success")

            if auto_save == "valid_hil":
                do_save = valid_hil
                print(f"[ONE-STAGE] valid HIL={valid_hil}; auto-save={do_save}", flush=True)
            elif auto_save is None:
                do_save = (
                    prompt_choice(
                        task_env,
                        {"y": "save", "n": "discard"},
                        "[TRAJECTORY] Save episode? [y]es / [n]o",
                    )
                    == "save"
                )
            else:
                do_save = bool(auto_save)

            episode_record = {
                "rollout_index": rollout_index,
                "rollout_status": "completed",
                "one_stage": bool(cli.one_stage),
                "stage6_handoff": stage6_handoff,
                "valid_hil": bool(valid_hil),
                "sampling_mode": cli.es_mode,
                "episode_index": int(data_episode_index),
                "seed": int(seed),
                "instruction": instruction,
                "initial_scene_screening": initial_scene_screening,
                "policy_steps": int(policy_steps),
                "policy_decisions": int(decision_step),
                "manual_takeover_requests": manual_takeover_requests,
                "takeover_request_events": takeover_request_events,
                "sampling_activation_events": sampling_activation_events,
                "intervention_count": intervention_count,
                "interventions": interventions,
                "segments": segments,
                "control_mask": list(task_env.control_mask),
                "autonomous_success": bool(
                    (final_check_success and intervention_count == 0)
                    if cli.one_stage or adapter.requires_training_gate else task_env.eval_success
                ),
                "final_check_success": bool(final_check_success),
                "final_success_metrics": final_success_metrics,
                "expert_result": last_expert_result,
                "expert_progress": last_expert_progress,
                "hil_frames": int(hil_frames),
                "supervisor_label": supervisor_label,
                "joints_legal": bool(joints_legal),
                "joint_absmax": float(joint_absmax),
                "save_decision": bool(do_save),
                "rollout_seconds": round(time.time() - rollout_started, 2),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            if adapter.requires_training_gate:
                episode_record.update(task=cli.task, adapter=adapter.name,
                                      accepted_for_training=bool(do_save and valid_hil),
                                      diagnostic_only=not valid_hil)

            if cli.save_policy_rollouts:
                policy_cache = getattr(task_env, "folder_path", {}).get("cache")
                policy_archive = archive_policy_rollout(
                    Path(policy_cache) if policy_cache else None,
                    cli.output_dir,
                    rollout_index=rollout_index,
                    seed=int(seed),
                    control_mask=list(task_env.control_mask),
                    save_freq=int(cli.save_freq),
                    sampling_mode=cli.es_mode,
                    autonomous_success=bool(episode_record["autonomous_success"]),
                )
                episode_record["policy_rollout_dir"] = (
                    str(policy_archive) if policy_archive else None
                )

            raw_dir = finalize_episode(
                task_env,
                cli.output_dir,
                data_episode_index,
                instruction,
                save=do_save,
                supervisor_label=supervisor_label,
                segments=segments,
                extra=episode_record,
            )
            episode_record["raw_dir"] = str(raw_dir) if raw_dir else None
            episode_record["hdf5_path"] = None
            if do_save:
                data_episode_index += 1
                expert_success = bool((last_expert_result or {}).get("success"))
                if hil_frames > 0 and (
                    (valid_hil if cli.one_stage or adapter.requires_training_gate else
                     (cli.target_mode != "expert" or expert_success))
                ):
                    saved_hil_count += 1
            records.append(episode_record)
            if cli.takeover_eval:
                append_jsonl(cli.output_dir / "takeover_rollouts.jsonl", episode_record)
            finish_sampling_log("completed")
            notify_trial_end(
                model_client,
                cli.task,
                seed,
                bool(task_env.eval_success),
            )
            print(
                f"[EPISODE] label={supervisor_label} save={do_save} "
                f"interventions={intervention_count}"
            )
            print(format_collection_progress(records, saved_hil_count,
                                             int(cli.target_saved), int(cli.episodes)), flush=True)
            safe_close_env(task_env)

            if cli.acceptance:
                checks = {
                    "intervention_detected": intervention_count > 0,
                    "policy_chunk_discarded": intervention_count > 0,
                    "expert_invoked": intervention_count > 0,
                    "plan_success": bool((last_expert_result or {}).get("plan_success")),
                    "task_success": bool(task_env.eval_success),
                    "joints_legal": bool(joints_legal),
                }
                passed = all(checks.values())
                report = {
                    "acceptance_passed": passed,
                    "criterion": "all takeover and expert-recovery checks are true",
                    "checks": checks,
                    "record": records[-1],
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
            rollout_index += 1
    except OperatorInterrupt as exc:
        if exc.event != "quit":
            raise
        aborted = True
        finish_sampling_log("quit")
        safe_close_env(task_env)
        print("[EPISODE] Quit at operator prompt; unsaved raw cache retained.", flush=True)
    except Exception:
        print("\n\033[91mHG-DAgger session error:\033[0m")
        print(traceback.format_exc())
        if sampling_log is not None:
            try:
                sampling_log.finish(False, traceback.format_exc(), 0)
            except Exception:
                pass
        safe_close_env(task_env, clear_cache=True)
        return 3
    finally:
        close_policy_client(model_client)
        lock_handle.close()

    screen_history = screening_history(cli.output_dir)
    session_report = {
        "task": cli.task,
        "adapter": adapter.name,
        "runtime_root": str(ROBOTWIN_ROOT),
        "aborted": aborted,
        "one_stage": bool(cli.one_stage),
        "paired_baseline_report": str(cli.paired_baseline_report.expanduser().resolve()) if cli.paired_baseline_report else None,
        "paired_baseline_sha256": paired_report_sha,
        "sampling_mode": cli.es_mode,
        "save_policy_rollouts": bool(cli.save_policy_rollouts),
        "sampling_activation": cli.es_activation,
        "sampling_manual_duration": cli.es_manual_duration if cli.es_activation == "manual" else None,
        "sampling_config": config_record(sampling_config) if sampling_config is not None else None,
        "sampling_window": [cli.es_window_start, cli.es_window_end],
        "critic_sha256": scorer.metadata["critic_sha256"] if scorer is not None else None,
        "takeover_eval": bool(cli.takeover_eval),
        "policy_name": cli.policy_name,
        "policy_host": cli.host,
        "policy_port": int(cli.port),
        "checkpoint_name": cli.ckpt_name,
        "task_config": cli.task_config,
        "instruction": cli.instruction,
        "episode_protocol": cli.episode_protocol,
        "instruction_pool_size": cli.instruction_pool_size if cli.episode_protocol == "official-seen" else None,
        "scene_attempts": scene_attempts,
        "max_scene_attempts": cli.max_scene_attempts,
        "scene_budget_exhausted": scene_budget_exhausted,
        "scene_screening_log": SCREEN_LOG if cli.episode_protocol == "official-seen" else None,
        "expert_screen_rejected": sum(row["status"] == "expert_rejected" for row in screen_history.values()),
        "expert_screen_passed": sum(row.get("plan_success") is True and row.get("check_success") is True
                                    for row in screen_history.values()),
        "frequency": int(cli.frequency),
        "step_limit": cli.step_limit,
        "seed_mode": cli.seed_mode,
        "rng_seed": cli.rng_seed,
        "seed_range": (
            [int(cli.seed_min), int(cli.seed_max)]
            if cli.seed_mode == "random"
            else [int(cli.seed_start), int(cli.seed_start) + int(
                cli.max_scene_attempts if cli.episode_protocol == "official-seen" else cli.episodes) - 1]
        ),
        "target_mode": cli.target_mode,
        "target_saved": int(cli.target_saved),
        "max_rollouts": int(cli.episodes),
        "resumed_from": progress.source_report,
        "next_rollout_index": int(rollout_index),
        "next_episode_index": int(data_episode_index),
        "saved_valid_hil_episodes": int(saved_hil_count),
        "eligible_stage6_rollouts": sum(
            item.get("one_stage") and item.get("rollout_status") == "completed"
            for item in records
        ) if cli.one_stage else None,
        "prefix_failures": sum(
            item.get("rollout_status") == "prefix_failed" for item in records
        ) if cli.one_stage else None,
        "paired_prefix_exclusions": sum(
            item.get("rollout_status") == "paired_prefix_excluded" for item in records
        ) if cli.one_stage else None,
        "valid_hil_events": sum(bool(item.get("valid_hil")) for item in records) if cli.one_stage or adapter.requires_training_gate else None,
        "recovery_failures": sum(
            item.get("one_stage") and item.get("intervention_count", 0) > 0
            and not item.get("valid_hil") and item.get("rollout_status") == "completed"
            for item in records
        ) if cli.one_stage else None,
        "rollouts": len(records),
        "total_rollouts": len(records),
        "aborted_rollouts": int(aborted_rollouts),
        "interventions": sum(int(item.get("intervention_count", 0)) for item in records),
        "saved_episodes": sum(1 for item in records if item.get("save_decision")),
        "saved_hil_episodes": sum(
            1
            for item in records
            if item.get("save_decision") and int(item.get("hil_frames", 0)) > 0
        ),
        "success_labels": sum(
            1 for item in records if item.get("supervisor_label") == "success"
        ),
        "autonomous_successes": sum(
            1 for item in records if item.get("autonomous_success")
        ),
        "total_seconds": round(progress.prior_seconds + time.time() - session_started, 2),
        "this_session_seconds": round(time.time() - session_started, 2),
        "seconds_per_rollout": round(
            (progress.prior_seconds + time.time() - session_started) / len(records), 2
        ) if records else None,
        "seconds_per_saved_episode": round(
            (progress.prior_seconds + time.time() - session_started)
            / max(1, sum(1 for item in records if item.get("save_decision"))),
            2,
        ),
        "records": records,
    }
    valid_records = [
        item for item in records
        if int(item.get("policy_steps", 0)) > 0 or int(item.get("manual_takeover_requests", 0)) > 0
    ]
    requested = sum(int(item.get("manual_takeover_requests", 0)) > 0 for item in valid_records)
    triggered = sum(int(item.get("intervention_count", 0)) > 0 for item in valid_records)
    session_report["takeover_measurement"] = {
        "valid_rollouts": len(valid_records),
        "manual_request_episodes": requested,
        "manual_request_probability": requested / len(valid_records) if valid_records else None,
        "recovery_trigger_episodes": triggered,
        "recovery_trigger_probability": triggered / len(valid_records) if valid_records else None,
        "excluded_zero_step_rollouts": len(records) - len(valid_records),
        "aborted_valid_rollouts": sum(item.get("rollout_status") == "aborted" for item in valid_records),
    }
    session_report["seeds"] = [item.get("seed") for item in records]
    report_path = cli.output_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite session report: {report_path}")
    write_json(report_path, session_report)
    print(f"session_report={report_path}")
    if cli.takeover_eval:
        print(f"[TAKEOVER] {requested}/{len(valid_records)} valid rollouts requested takeover "
              f"({session_report['takeover_measurement']['manual_request_probability']})")
    return 130 if aborted else 4 if scene_budget_exhausted else 0


if __name__ == "__main__":
    raise SystemExit(main())
