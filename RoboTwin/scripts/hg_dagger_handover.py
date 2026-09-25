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
from dataclasses import asdict
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
    sample_absolute_joint_chunk,
    xpolicylab_action_to_robotwin,
)
from coverage_sampling.options import add_arguments as add_sampling_arguments, config_from_args  # noqa: E402


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


class OperatorInterrupt(Exception):
    """A real operator key arrived while policy sampling was in progress."""

    def __init__(self, event: str):
        super().__init__(event)
        self.event = event

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
        self._viewer_held: set[str] = set()

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
                pressed = bool(window.key_press(key))
                down = bool(window.key_down(key))
                if event == "sampling_toggle":
                    if not down:
                        self._viewer_held.discard(key)
                    elif key not in self._viewer_held:
                        self._viewer_held.add(key)
                        return event
                    if pressed and not down:
                        return event
                elif pressed or down:
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
    print(
        f"  末端到杆距离: source={state.get('source_ee_to_bar')} "
        f"receiver={state.get('receiver_ee_to_bar')} (米)"
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
        if os.environ.get("HIL_DIAG_ALLOW_NO_VIEWER") == "1":
            # Diagnostic-only escape hatch: run this entry point fully headless.
            return
        raise RuntimeError(
            "Human-gated mode needs a SAPIEN viewer. Set --render-freq to a positive value "
            "and launch inside the 5090 desktop session."
        )
    task_env._update_render()
    viewer.render()


def setup_viewer_diagnostics(task_env) -> None:
    """Environment-gated diagnostics for the SAPIEN viewer cost.

    HIL_DIAG_VIEWER=1      time every viewer.render()/scene.update_render() call
    HIL_DIAG_SKIP_VIEWER=1 keep the window but turn viewer.render() into a no-op
    """
    if os.environ.get("HIL_DIAG_VIEWER") != "1" and os.environ.get("HIL_DIAG_SKIP_VIEWER") != "1":
        return

    def install_timer(target, attribute, label, every):
        original = getattr(target, attribute)
        stats = {"count": 0, "total": 0.0, "max": 0.0}

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                stats["count"] += 1
                stats["total"] += elapsed
                stats["max"] = max(stats["max"], elapsed)
                if stats["count"] % every == 0:
                    print(
                        f"[DIAG] {label} n={stats['count']} "
                        f"total={stats['total']:.2f}s "
                        f"avg={stats['total'] / stats['count'] * 1000:.2f}ms "
                        f"max={stats['max'] * 1000:.2f}ms",
                        flush=True,
                    )

        setattr(target, attribute, timed)

    # These exist with or without a viewer, so time them first.
    install_timer(task_env, "take_action", "take_action", 25)
    install_timer(task_env, "get_obs", "get_obs", 25)
    install_timer(task_env, "_update_render", "scene.update_render", 200)

    viewer = getattr(task_env, "viewer", None)
    if viewer is None:
        print("[DIAG] no viewer instance; still timing env paths", flush=True)
        return

    window = getattr(viewer, "window", None)
    print(
        f"[DIAG] viewer resolution={getattr(viewer, 'resolution', None)} "
        f"shader_dir={getattr(viewer, 'shader_dir', None)} "
        f"paused={getattr(viewer, 'paused', None)} "
        f"window_size={getattr(window, 'size', None)}",
        flush=True,
    )
    for plugin in getattr(viewer, "plugins", []):
        camera_index = getattr(plugin, "camera_index", None)
        focused_camera = getattr(plugin, "focused_camera", None)
        if camera_index is None and focused_camera is None:
            continue
        print(
            f"[DIAG] plugin {type(plugin).__name__}: camera_index={camera_index} "
            f"focused_camera={focused_camera}",
            flush=True,
        )

    if os.environ.get("HIL_DIAG_SKIP_VIEWER") == "1":
        viewer.render = lambda *args, **kwargs: None
        print("[DIAG] viewer.render disabled (window kept)", flush=True)
        return

    install_timer(viewer, "render", "viewer.render", 25)
    print("[DIAG] timing enabled for take_action/get_obs/update_render/viewer.render", flush=True)


def install_viewer_frame_limit(task_env) -> None:
    """Redraw the viewer at most ``HIL_VIEWER_MAX_FPS`` times per second.

    ``Base_Task.take_action`` redraws the viewer on entry and exit regardless
    of ``--render-freq``, so a supervised rollout pays for two viewer renders
    per policy step (~2 x 15 ms at 960x540, more at higher resolutions).  The
    supervisor only needs the window to look smooth, so throttle the redraw by
    wall-clock time instead. ``HIL_VIEWER_MAX_FPS=0`` restores the old
    behaviour of rendering on every call.
    """
    if getattr(task_env, "_hil_viewer_fps_installed", False):
        return
    task_env._hil_viewer_fps_installed = True

    raw = os.environ.get("HIL_VIEWER_MAX_FPS", "10").strip()
    try:
        max_fps = float(raw)
    except ValueError:
        print(f"[VIEWER] invalid HIL_VIEWER_MAX_FPS={raw!r}; keeping 10 fps", flush=True)
        max_fps = 10.0

    viewer = getattr(task_env, "viewer", None)
    if viewer is None or max_fps <= 0:
        print(f"[VIEWER] frame limit disabled (HIL_VIEWER_MAX_FPS={raw})", flush=True)
        return

    min_interval = 1.0 / max_fps
    original_render = viewer.render
    state = {"last": 0.0, "rendered": 0, "skipped": 0}

    def throttled_render(*args, **kwargs):
        now = time.perf_counter()
        if now - state["last"] < min_interval:
            state["skipped"] += 1
            return None
        state["last"] = now
        state["rendered"] += 1
        return original_render(*args, **kwargs)

    viewer.render = throttled_render
    print(
        f"[VIEWER] render limited to {max_fps:g} fps "
        f"(min interval {min_interval * 1000:.0f} ms); "
        "set HIL_VIEWER_MAX_FPS=0 to disable",
        flush=True,
    )


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
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    if cli.render_freq <= 0 and os.environ.get("HIL_DIAG_ALLOW_NO_VIEWER") != "1":
        raise ValueError("--render-freq must be positive for human supervision")
    if cli.es_activation == "manual":
        if cli.es_mode == "off" or cli.es_manual_duration < 1:
            raise ValueError("Manual sampling requires vanilla/enhanced mode and a positive duration")
        cli.es_window_start = 0
        cli.es_window_end = cli.es_manual_duration - 1
    if cli.takeover_eval:
        if cli.acceptance or cli.seed_mode != "sequential":
            raise ValueError("Takeover comparison requires sequential matched seeds and no acceptance mode")
        if cli.auto_intervene_step >= 0 or cli.auto_abort_step >= 0:
            raise ValueError("Takeover comparison cannot use automatic intervention or abort hooks")
        if cli.es_window_start is None or cli.es_window_end is None:
            raise ValueError("Takeover comparison requires the same declared window in all three arms")
    if (cli.es_mode != "off" or cli.takeover_eval) and cli.bias_magnitude:
        raise ValueError("Action bias is incompatible with simulated candidate scoring")
    sampling_config = config_from_args({
        **vars(cli), "task_name": "handover_to_tray", "action_type": "joint", "eval_batch": False,
    })
    scorer = None
    if sampling_config is not None:
        from coverage_sampling.critic import OnlineCoverage

        scorer = OnlineCoverage(cli.es_critic, cli.es_encoder_weights, device=cli.es_device)
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
    auto_save = (False if cli.takeover_eval else None) if cli.auto_save == "none" else parse_bool(cli.auto_save)
    bias_dims = joint_bias_dims(
        cli.bias_dims,
        int(user_args["left_arm_dim"]),
        int(user_args["right_arm_dim"]),
    )
    data_episode_index = next_episode_index(cli.output_dir)
    records: list[dict[str, Any]] = []
    aborted = False
    sampling_log = None

    print("\n" + "=" * 72)
    print("RoboTwin handover_to_tray human-gated DAgger")
    print("Keys: i=takeover, r=hand back to policy, x=abort episode, q=quit.")
    if cli.es_activation == "manual":
        print(f"Press e during policy control to toggle {cli.es_manual_duration} enhanced decisions.")
    print("After each episode: s/f=success/failure, y/n=save/discard.")
    print(f"Prompt: {PROMPT}")
    print(f"Sampling: {cli.es_mode}; takeover evaluation: {cli.takeover_eval}")
    print(f"Output: {cli.output_dir}")
    print("=" * 72 + "\n")

    try:
        rollout_index = 0
        saved_hil_count = 0
        aborted_rollouts = 0
        session_started = time.time()
        while (
            rollout_index < int(cli.episodes)
            and (cli.takeover_eval or saved_hil_count < int(cli.target_saved))
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
            if rollout_index == 0:
                setup_viewer_diagnostics(task_env)
                install_viewer_frame_limit(task_env)
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
                "i=takeover, r=handback, x=abort episode, q=quit.\033[0m"
            )

            key_map = {**DEFAULT_KEYS, **({"e": "sampling_toggle"} if cli.es_activation == "manual" else {})}
            with HumanInterventionInput(task_env, key_map=key_map) as keyboard:
                sampler = None
                manual_window = None
                if sampling_config is not None:
                    from coverage_sampling.core import DecisionSampler, EpisodeLog, ManualSamplingWindow
                    from coverage_sampling.robotwin import RobotwinRollout

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
                        print(f"\n[ENHANCED] {status}; policy remains in control.", flush=True)

                    def poll_live_control() -> None:
                        # Lookahead never displays hypothetical states. Process
                        # pending keys only after restoring the real scene.
                        task_env._render_viewer_if_available()
                        pending = keyboard.poll()
                        if pending in {"intervene", "abort", "quit", "sampling_toggle"}:
                            raise OperatorInterrupt(pending)

                    sampler = DecisionSampler(
                        sampling_config,
                        RobotwinRollout(task_env, scorer, on_restored=poll_live_control),
                        seed,
                    )
                    log_dir = Path(cli.es_log_dir) if cli.es_log_dir else cli.output_dir / "sampling"
                    sampling_log = EpisodeLog(
                        log_dir / f"episode_{rollout_index:04d}_seed_{seed}.jsonl",
                        sampling_config,
                        {**scorer.metadata, "episode_seed": seed, "instruction": PROMPT,
                         "policy_checkpoint": cli.ckpt_name, "task_config": cli.task_config,
                         "frequency": int(cli.frequency), "evaluation_type": "human_gated",
                         "sampling_activation": cli.es_activation,
                         "manual_duration": cli.es_manual_duration if manual_window is not None else None,
                         "executor": "Base_Task.take_action(qpos)",
                         "window_units": "zero-based policy decision calls; inclusive"},
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
                            recovery_iter, chosen_stage = choose_recovery_stage(
                                task_env, keyboard
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
                        chunk_interrupted = False
                        manual_trigger = True
                        selection = None
                        if sampler is not None:
                            def sample_chunk():
                                poll_live_control()
                                chunk = sample_absolute_joint_chunk(model_client, observation, "joint")
                                poll_live_control()
                                return chunk

                            try:
                                action_chunk, selection = sampler.select(
                                    decision_step, sample_chunk,
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
                            action_chunk = normalize_action_chunk(
                                model_client.call(func_name="get_action")
                            )
                        if len(action_chunk) == 0 and not chunk_interrupted:
                            raise RuntimeError("Policy returned an empty action chunk.")
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
                            actual_coverage = scorer(observation, flat_action) if scorer is not None else None
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
                                if selection["active"]:
                                    curve = selection["branches"][selection["selected"]]["coverage"]
                                    if action_idx < len(curve):
                                        predicted = curve[action_idx]
                                sampling_log.executed(
                                    decision_step - 1, action_idx, actual_coverage,
                                    selection["active"], flat_action,
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
                            recovery_iter, chosen_stage = choose_recovery_stage(
                                task_env, keyboard
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
                            if sampler is not None:
                                sampler.checked_replay = False
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
                finish_sampling_log("quit")
                notify_trial_end(model_client, "handover_to_tray", seed, False)
                safe_close_env(task_env)
                break

            if episode_aborted:
                # The supervisor judged this rollout unrecoverable: drop it
                # immediately instead of waiting for the episode to finish or
                # sitting through the save prompt.
                aborted_rollouts += 1
                finish_sampling_log("aborted")
                discard_recovery_cache(task_env)
                notify_trial_end(model_client, "handover_to_tray", seed, False)
                print(
                    f"\n\033[93m[EPISODE] aborted by supervisor after "
                    f"{policy_steps} policy steps "
                    f"(frame_idx={int(task_env.FRAME_IDX)}, "
                    f"interventions={intervention_count}); discarded, next rollout.\033[0m"
                )
                if cli.takeover_eval:
                    aborted_record = {
                        "rollout_index": rollout_index, "seed": int(seed),
                        "sampling_mode": cli.es_mode, "rollout_status": "aborted",
                        "policy_steps": int(policy_steps),
                        "manual_takeover_requests": manual_takeover_requests,
                        "takeover_request_events": takeover_request_events,
                        "sampling_activation_events": sampling_activation_events,
                        "intervention_count": intervention_count,
                        "interventions": interventions,
                        "autonomous_success": False, "save_decision": False,
                        "rollout_seconds": round(time.time() - rollout_started, 2),
                    }
                    records.append(aborted_record)
                    append_jsonl(cli.output_dir / "takeover_rollouts.jsonl", aborted_record)
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
                "rollout_status": "completed",
                "sampling_mode": cli.es_mode,
                "episode_index": int(data_episode_index),
                "seed": int(seed),
                "instruction": PROMPT,
                "policy_steps": int(policy_steps),
                "policy_decisions": int(decision_step),
                "manual_takeover_requests": manual_takeover_requests,
                "takeover_request_events": takeover_request_events,
                "sampling_activation_events": sampling_activation_events,
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
            if cli.takeover_eval:
                append_jsonl(cli.output_dir / "takeover_rollouts.jsonl", episode_record)
            finish_sampling_log("completed")
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
        if sampling_log is not None:
            try:
                sampling_log.finish(False, traceback.format_exc(), 0)
            except Exception:
                pass
        safe_close_env(task_env, clear_cache=True)
        return 3
    finally:
        close_policy_client(model_client)

    session_report = {
        "aborted": aborted,
        "sampling_mode": cli.es_mode,
        "sampling_activation": cli.es_activation,
        "sampling_manual_duration": cli.es_manual_duration if cli.es_activation == "manual" else None,
        "sampling_config": asdict(sampling_config) if sampling_config is not None else None,
        "sampling_window": [cli.es_window_start, cli.es_window_end],
        "critic_sha256": scorer.metadata["critic_sha256"] if scorer is not None else None,
        "takeover_eval": bool(cli.takeover_eval),
        "policy_name": cli.policy_name,
        "policy_host": cli.host,
        "policy_port": int(cli.port),
        "checkpoint_name": cli.ckpt_name,
        "task_config": cli.task_config,
        "instruction": PROMPT,
        "frequency": int(cli.frequency),
        "step_limit": cli.step_limit,
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
    write_json(report_path, session_report)
    print(f"session_report={report_path}")
    if cli.takeover_eval:
        print(f"[TAKEOVER] {requested}/{len(valid_records)} valid rollouts requested takeover "
              f"({session_report['takeover_measurement']['manual_request_probability']})")
    return 130 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
