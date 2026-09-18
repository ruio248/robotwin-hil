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
import random
import select
import shutil
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


RESERVED_SEED_RANGES = (
    (1, 579, "demonstration (1-579)"),
    (30000, 30022, "development (30000-30022)"),
    (31000, 31103, "frozen test (31000-31103)"),
)


def reserved_seed_label(seed: int) -> str | None:
    for low, high, label in RESERVED_SEED_RANGES:
        if low <= seed <= high:
            return label
    return None


def build_seed_stream(cli: argparse.Namespace):
    """Yield collection seeds: sequential from --seed-start, or random in range."""
    if cli.seed_mode == "sequential":
        for seed in range(int(cli.seed_start), int(cli.seed_start) + int(cli.episodes)):
            label = reserved_seed_label(seed)
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
    while True:
        seed = rng.randint(low, high)
        if seed in used or reserved_seed_label(seed):
            continue
        used.add(seed)
        yield seed


DEFAULT_KEYS = {
    "i": "intervene",
    "r": "handback",
    "x": "abort",
    "q": "quit",
}

# Human-readable stage names shown at takeover time. Stage 2 (source_lift) is
# an intermediate motion folded into the resume_handover entry point (3);
# stage 4 (receiver_grasp) also serves as the direct "grasp-and-place from
# air" entry point when the bar is already hovering near the tray.
STAGE_LABELS = {
    1: "source_grasp               左臂抓取红杆",
    2: "source_lift               左臂抬起",
    3: "handover_pose             移向交接位姿",
    4: "receiver_grasp            右臂抓取（含悬空直接放）",
    5: "source_release_and_retreat 左臂松开并后撤",
    6: "guided_tray_placement     放入蓝托盘",
}


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


class HumanInterventionInput:
    """Poll SAPIEN's focused window and, optionally, the launching terminal."""

    def __init__(self, task_env, key_map: dict[str, str] | None = None):
        self.task_env = task_env
        self.key_map = {
            key.lower(): value
            for key, value in (key_map or DEFAULT_KEYS).items()
        }
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
        for key, event in self.key_map.items():
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
        for key, event in sorted(
            self.key_map.items(), key=lambda item: -len(item[0])
        ):
            if key in data:
                return event
        return None

    def poll(self) -> str | None:
        return self._poll_viewer() or self._poll_terminal()

    def read_line(self, prompt: str) -> str:
        """Temporarily leave cbreak mode and read one line from the terminal.

        Returns an empty string when there is no interactive terminal, so the
        caller can safely fall back to automatic stage selection in headless
        runs.
        """
        if self._fd is None or self._term_attrs is None:
            return ""
        sys.stdout.write(prompt)
        sys.stdout.flush()
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._term_attrs)
            line = sys.stdin.readline()
        except (OSError, termios.error, ValueError):
            line = ""
        finally:
            try:
                tty.setcbreak(self._fd)
            except (OSError, termios.error):
                pass
        return (line or "").strip()


def choose_recovery_stage(task_env, keyboard) -> tuple[Any, int | None]:
    """Show the inferred recovery stage and let the supervisor confirm it or
    override the entry stage before the expert starts moving.

    Returns ``(recovery_iter, chosen_stage_id)``. ``(None, None)`` means the
    supervisor cancelled this takeover and the policy should keep control.
    """
    state = task_env.infer_recovery_state()
    print("\n\033[96m[HG-DAGGER] 任务共 6 个阶段:\033[0m")
    for stage_id in range(1, 7):
        print(f"  {stage_id}. {STAGE_LABELS[stage_id]}")
    print("\n\033[96m[HG-DAGGER] 专家自动判定:\033[0m")
    print(f"  branch={state['branch']}  entry_stage={state['stage_id']}")
    print(
        f"  source_holds={state['source_holds']} "
        f"receiver_holds={state['receiver_holds']} "
        f"placed={state['placed']} near_tray={state.get('near_tray', False)}"
    )
    print(f"  bar_pos={state.get('bar_position')}")

    while True:
        answer = keyboard.read_line(
            "\n阶段判断正确吗? [Y/Enter]=确认自动, 输入 1-6 重选, q=取消本次接管: "
        ).strip().lower()
        if answer in ("", "y", "yes"):
            return task_env.recovery_stage_iterator(), None
        if answer == "q":
            return None, None
        if answer.isdigit() and 1 <= int(answer) <= 6:
            chosen = int(answer)
            return task_env.recovery_stage_iterator(start_stage_id=chosen), chosen
        print("\033[91m  无效输入，请输入 Y / 1-6 / q\033[0m")


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


def append_segment(
    segments: list[dict[str, Any]],
    source: str,
    start_step: int,
    end_step: int,
) -> None:
    """Record a contiguous control segment when it has at least one step."""
    if int(end_step) > int(start_step):
        segments.append(
            {
                "source": source,
                "start_step": int(start_step),
                "end_step": int(end_step),
            }
        )


def prompt_choice(
    task_env,
    choices: dict[str, str],
    prompt: str,
) -> str:
    """Block until the supervisor emits one of the configured key events."""
    print(prompt, flush=True)
    with HumanInterventionInput(task_env, choices) as poller:
        while True:
            event = poller.poll()
            if event in choices.values():
                return event
            time.sleep(0.02)


def finalize_episode(
    task_env,
    output_dir: Path,
    episode_index: int,
    instruction: str,
    *,
    save: bool,
    supervisor_label: str,
    segments: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Persist or discard the raw recorded trajectory after supervisor review.

    Collection deliberately stops at the raw stage: frames stay as the native
    per-frame pkl cache and metadata stays in ``episode.json``. HDF5, MP4 and
    LeRobot conversion happen later in ``export_hg_dagger_dataset.py`` so the
    control loop never pays for encoding.
    """
    task_env.episode_metadata["control_mask"] = list(task_env.control_mask)
    task_env.episode_metadata["segments"] = segments
    task_env.episode_metadata["supervisor_label"] = supervisor_label
    task_env.episode_metadata["saved"] = bool(save)
    task_env.info["task_metadata"] = task_env.episode_metadata
    task_env.info["control_mask"] = list(task_env.control_mask)
    task_env.info["segments"] = segments
    task_env.info["supervisor_label"] = supervisor_label
    task_env.info["saved"] = bool(save)

    if not save:
        discard_recovery_cache(task_env)
        return None

    if int(task_env.FRAME_IDX) < 2:
        raise RuntimeError("Episode produced fewer than two recorded frames; cannot save.")

    raw_dir = output_dir / "raw" / f"episode_{episode_index:07d}"
    frames_dir = raw_dir / "frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = getattr(task_env, "folder_path", {}).get("cache")
    if cache and Path(cache).exists():
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        shutil.move(str(cache), str(frames_dir))

    write_json(
        raw_dir / "episode.json",
        {
            "episode_index": int(episode_index),
            "instruction": instruction,
            "seed": int(getattr(task_env, "episode_seed", -1)),
            "supervisor_label": supervisor_label,
            "save_freq": int(task_env.save_freq or 15),
            "frequency": int(task_env.save_freq or 15),
            "control_mask": list(task_env.control_mask),
            "segments": segments,
            "episode_metadata": task_env.episode_metadata,
            "info": task_env.info,
            "record": extra or {},
        },
    )

    append_jsonl(
        output_dir / "episodes.jsonl",
        {
            "episode_index": episode_index,
            "saved": True,
            "supervisor_label": supervisor_label,
            "control_mask": list(task_env.control_mask),
            "segments": segments,
            "raw_dir": str(raw_dir),
            "hil_frames": (extra or {}).get("hil_frames"),
            "rollout_seconds": (extra or {}).get("rollout_seconds"),
            "expert_success": ((extra or {}).get("expert_result") or {}).get("success"),
            "expert_branch": ((extra or {}).get("expert_result") or {}).get("branch"),
            "intervention_count": (extra or {}).get("intervention_count"),
        },
    )
    return raw_dir


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
        help="Maximum number of policy rollouts to run (seed_start + index).",
    )
    parser.add_argument(
        "--target-saved",
        type=int,
        default=10,
        help="Stop once this many valid (saved) HIL episodes have been collected.",
    )
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
        choices=["true", "false", "none"],
        default="none",
        help="Test-only hook: skip the manual save/discard prompt.",
    )
    parser.add_argument(
        "--auto-handback-step",
        type=int,
        default=-1,
        help="Test-only hook: hand back to the policy after this many expert stages.",
    )
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    if cli.render_freq <= 0:
        raise ValueError("--render-freq must be positive for human supervision")
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
    seed_iter = build_seed_stream(cli)
    cli.output_dir = cli.output_dir.expanduser().resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    user_args, args = build_runtime_args(cli)
    task_env = class_decorator("handover_to_tray")
    model_client = build_policy_client(user_args)
    save_data = parse_bool(cli.save_data)
    if parse_bool(cli.save_video):
        print(
            "[HG-DAGGER] --save-video is deprecated in collection; "
            "raw frames are saved and MP4 export is done offline.",
            flush=True,
        )
    auto_label = None if cli.auto_label == "none" else cli.auto_label
    auto_save = None if cli.auto_save == "none" else parse_bool(cli.auto_save)
    bias_dims = joint_bias_dims(
        cli.bias_dims,
        int(user_args["left_arm_dim"]),
        int(user_args["right_arm_dim"]),
    )
    data_episode_index = next_episode_index(cli.output_dir)
    records: list[dict[str, Any]] = []
    aborted = False

    print("\n" + "=" * 72)
    print("RoboTwin handover_to_tray human-gated DAgger")
    print("Keys: i=takeover, r=hand back to policy, x=abort episode, q=quit.")
    print("After each episode: s/f=success/failure, y/n=save/discard.")
    print(f"Prompt: {PROMPT}")
    print(f"Output: {cli.output_dir}")
    print("=" * 72 + "\n")

    try:
        rollout_index = 0
        saved_hil_count = 0
        aborted_rollouts = 0
        session_started = time.time()
        while (
            rollout_index < int(cli.episodes)
            and saved_hil_count < int(cli.target_saved)
        ):
            seed = next(seed_iter)
            rollout_started = time.time()
            args["save_data"] = False
            args["need_plan"] = True
            args["eval_mode"] = True
            task_env.setup_demo(
                now_ep_num=data_episode_index,
                seed=seed,
                is_test=False,
                **args,
            )
            if cli.step_limit is not None:
                task_env.step_lim = int(cli.step_limit)
            task_env.set_instruction(PROMPT)
            render_initial_frame(task_env)
            prepare_policy_case(model_client, "handover_to_tray", seed, PROMPT, "joint")
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
            intervention_count = 0
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

            print(
                f"\n\033[96m[ROLLOUT {rollout_index + 1}/max {cli.episodes} | "
                f"saved valid HIL ({cli.target_mode}) {saved_hil_count}/{cli.target_saved}] "
                f"seed={seed}; "
                "i=takeover, r=handback, x=abort episode, q=quit.\033[0m"
            )

            with HumanInterventionInput(task_env) as keyboard:
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
                        if event == "intervene":
                            recovery_iter, chosen_stage = choose_recovery_stage(
                                task_env, keyboard
                            )
                            if recovery_iter is None:
                                print(
                                    "\n\033[93m[HG-DAGGER] 已取消本次接管，继续策略执行。\033[0m"
                                )
                                continue
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
                            print(
                                "\033[93m[HG-DAGGER] Expert takeover; press r to hand back.\033[0m"
                            )
                            continue

                        observation = task_env.get_obs()
                        xpl_obs = robotwin_obs_to_xpolicylab(
                            observation,
                            instruction=PROMPT,
                            env_idx=0,
                            frequency=int(cli.frequency),
                            task_env=task_env,
                        )
                        model_client.call(func_name="update_obs", obs=xpl_obs)
                        action_chunk = normalize_action_chunk(
                            model_client.call(func_name="get_action")
                        )
                        if not action_chunk:
                            raise RuntimeError("Policy returned an empty action chunk.")

                        chunk_interrupted = False
                        for action in action_chunk:
                            event = keyboard.poll()
                            if event == "quit":
                                quit_requested = True
                                chunk_interrupted = True
                                break
                            if event == "intervene":
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
                            task_env.current_control_source = "policy"
                            task_env.take_action(
                                flat_action,
                                action_type=robotwin_action_type,
                            )
                            policy_steps += 1
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
                                chunk_interrupted = True
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

                        if quit_requested:
                            break
                        if is_episode_end(task_env):
                            break
                        if chunk_interrupted and event == "intervene":
                            recovery_iter, chosen_stage = choose_recovery_stage(
                                task_env, keyboard
                            )
                            if recovery_iter is None:
                                print(
                                    "\n\033[93m[HG-DAGGER] 已取消本次接管，继续策略执行。\033[0m"
                                )
                                continue
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
                            print(
                                "\033[93m[HG-DAGGER] Expert takeover; press r to hand back.\033[0m"
                            )
                            continue
                    else:
                        # Expert mode. The supervisor can hand back at any
                        # stage boundary. Each ``next()`` executes one stage.
                        if event == "handback" or (
                            cli.auto_handback_step >= 0
                            and expert_stage_count >= cli.auto_handback_step
                        ):
                            if interventions:
                                interventions[-1]["handback_step"] = int(task_env.FRAME_IDX)
                                interventions[-1]["expert_progress"] = last_expert_progress
                            switch_source("policy")
                            mode = "policy"
                            reset_policy(model_client)
                            print("\033[93m[HG-DAGGER] Hand back to policy.\033[0m")
                            continue

                        task_env.current_control_source = "hil"
                        step = next(recovery_iter, None)
                        expert_stage_count += 1
                        if step is None:
                            break
                        if step.get("phase") == "done":
                            if interventions:
                                interventions[-1]["expert_done"] = True
                                interventions[-1]["end_step"] = int(task_env.FRAME_IDX)
                            last_expert_result = step.get("result") or {}
                            break
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
                notify_trial_end(model_client, "handover_to_tray", seed, False)
                safe_close_env(task_env)
                break

            if episode_aborted:
                # The supervisor judged this rollout unrecoverable: drop it
                # immediately instead of waiting for the episode to finish or
                # sitting through the save prompt.
                aborted_rollouts += 1
                discard_recovery_cache(task_env)
                notify_trial_end(model_client, "handover_to_tray", seed, False)
                print(
                    f"\n\033[93m[EPISODE] aborted by supervisor at step "
                    f"{int(task_env.FRAME_IDX)} "
                    f"(interventions={intervention_count}); discarded, next rollout.\033[0m"
                )
                safe_close_env(task_env)
                rollout_index += 1
                continue

            joints_legal, joint_absmax = task_env.planned_joints_legal()
            final_success_metrics = task_env.success_metrics()
            final_check_success = task_env.check_success()
            hil_frames = sum(
                1 for item in task_env.control_mask if str(item).lower() == "hil"
            )

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

            if auto_save is None:
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
                "episode_index": int(data_episode_index),
                "seed": int(seed),
                "instruction": PROMPT,
                "policy_steps": int(policy_steps),
                "intervention_count": intervention_count,
                "interventions": interventions,
                "segments": segments,
                "control_mask": list(task_env.control_mask),
                "autonomous_success": bool(task_env.eval_success),
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

            raw_dir = finalize_episode(
                task_env,
                cli.output_dir,
                data_episode_index,
                PROMPT,
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
                    cli.target_mode != "expert" or expert_success
                ):
                    saved_hil_count += 1
            records.append(episode_record)
            notify_trial_end(
                model_client,
                "handover_to_tray",
                seed,
                bool(task_env.eval_success),
            )
            print(
                f"[EPISODE] label={supervisor_label} save={do_save} "
                f"interventions={intervention_count}"
            )
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
    except Exception:
        print("\n\033[91mHG-DAgger session error:\033[0m")
        print(traceback.format_exc())
        safe_close_env(task_env, clear_cache=True)
        return 3
    finally:
        close_policy_client(model_client)

    session_report = {
        "aborted": aborted,
        "seed_mode": cli.seed_mode,
        "rng_seed": cli.rng_seed,
        "seed_range": (
            [int(cli.seed_min), int(cli.seed_max)]
            if cli.seed_mode == "random"
            else [int(cli.seed_start), int(cli.seed_start) + int(cli.episodes) - 1]
        ),
        "target_mode": cli.target_mode,
        "target_saved": int(cli.target_saved),
        "max_rollouts": int(cli.episodes),
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
        "total_seconds": round(time.time() - session_started, 2),
        "seconds_per_rollout": round(
            (time.time() - session_started) / len(records), 2
        ) if records else None,
        "seconds_per_saved_episode": round(
            (time.time() - session_started)
            / max(1, sum(1 for item in records if item.get("save_decision"))),
            2,
        ),
        "records": records,
    }
    session_report["seeds"] = [item.get("seed") for item in records]
    report_path = cli.output_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(report_path, session_report)
    print(f"session_report={report_path}")
    return 130 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
