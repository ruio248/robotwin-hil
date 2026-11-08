"""Operator input and task-independent recovery-stage confirmation."""
from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty

DEFAULT_KEYS = {"i": "intervene", "r": "handback", "x": "abort", "q": "quit"}


def recovery_reason_category(reason: str) -> str:
    """Group a rejection for analysis without changing whether motion is allowed."""
    if reason == "object_moving_wait_for_settle":
        return "temporary_wait"
    if "requires_specialized_recovery" in reason or "requires_manual_unstack" in reason:
        return "unsupported_primitive"
    if reason.startswith("stage_precondition_mismatch"):
        return "incompatible_stage"
    if "ambiguous" in reason or "partial_grasp" in reason:
        return "ambiguous_state"
    if reason.startswith("unobservable_state"):
        return "technical_unobservable"
    if reason in {"outside_recovery_workspace", "unheld_object_not_supported"}:
        return "safety_precondition"
    return "unclassified"


class OperatorInterrupt(Exception):
    """A real operator key arrived while policy sampling was in progress."""

    def __init__(self, event: str):
        super().__init__(event)
        self.event = event

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

    def _pump_viewer(self) -> None:
        """Service desktop events on the main thread without advancing physics."""
        viewer = getattr(self.task_env, "viewer", None)
        if viewer is None or getattr(viewer, "closed", False):
            return
        pump = getattr(viewer, "pump_events", None)
        if callable(pump):
            pump()
        else:
            # The native SAPIEN viewer processes input as part of rendering.
            render = getattr(viewer, "render", None)
            if callable(render):
                render()

    def poll(self) -> str | None:
        self._pump_viewer()
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
            while True:
                self._pump_viewer()
                if getattr(getattr(self.task_env, "viewer", None), "closed", False):
                    raise OperatorInterrupt("quit")
                readable, _, _ = select.select([self._fd], [], [], 0.02)
                if readable:
                    # Canonical TTY mode makes the fd readable after Enter.
                    line = sys.stdin.readline()
                    if not line:
                        raise OperatorInterrupt("quit")
                    break
        except (OSError, termios.error, ValueError):
            line = ""
        finally:
            try:
                tty.setcbreak(self._fd)
            except (OSError, termios.error):
                pass
        return (line or "").strip()

def prompt_choice(
    task_env,
    choices: dict[str, str],
    prompt: str,
) -> str:
    """Block until the supervisor emits one of the configured key events."""
    print(prompt, flush=True)
    with HumanInterventionInput(task_env, {**choices, "q": "quit"}) as poller:
        while True:
            event = poller.poll()
            if event == "quit":
                raise OperatorInterrupt("quit")
            if event in choices.values():
                return event
            time.sleep(0.02)


def choose_recovery_stage(task_env, keyboard, adapter=None):
    """Confirm the adapter's inferred branch, override a stage, or cancel.

    Constructing an iterator must not execute any expert motion. The runner
    changes control ownership before it advances the returned iterator.
    """
    if adapter is None:
        from dagger.adapters import get_adapter
        adapter = get_adapter("handover_to_tray")
    state = adapter.infer_recovery_state(task_env)
    labels = adapter.stage_labels
    print(f"\n[HG-DAGGER] 任务共 {len(labels)} 个阶段:")
    for stage_id, label in sorted(labels.items()):
        print(f"  {stage_id}. {label}")
    print("\n[HG-DAGGER] 专家自动判定:")
    for line in adapter.describe_recovery_state(state):
        print(f"  {line}")
    choices = "/".join(str(stage) for stage in sorted(labels))
    while True:
        answer = keyboard.read_line(
            f"\n阶段判断正确吗? [Y/Enter]=确认自动, 输入 {choices} 重选, q/r=取消接管并交还策略: "
        ).strip().lower()
        if answer == "" and isinstance(keyboard, HumanInterventionInput) and keyboard._fd is None:
            print("[HG-DAGGER] 无交互终端，取消未确认的接管。", flush=True)
            return None, None
        if answer in ("q", "r"):
            return None, None
        if answer in ("", "y", "yes"):
            chosen = None
        elif answer.isdigit() and int(answer) in labels:
            chosen = int(answer)
        else:
            print(f"无效输入，请输入 Y / {choices} / q / r")
            continue
        state = adapter.infer_recovery_state(task_env)
        reason = adapter.recovery_stage_rejection(state, chosen)
        if reason:
            print(f"[HG-DAGGER] 阶段选择未执行：{reason}。本局未结束，现场保持不动。", flush=True)
            print("请重选可执行阶段；q/r 取消本次接管。阶段编号不能覆盖抓持/支撑前置条件。", flush=True)
            metadata = getattr(task_env, "episode_metadata", {})
            metadata.setdefault("recovery_selection_rejections", []).append(
                {"chosen_stage_id": chosen, "reason": reason,
                 "reason_category": recovery_reason_category(reason), "state": state,
                 "frame_idx": getattr(task_env, "FRAME_IDX", None)}
            )
            if isinstance(keyboard, HumanInterventionInput) and keyboard._fd is None:
                # read_line returns empty without a TTY; do not spin forever or
                # silently turn a rejected choice into an accepted intervention.
                print("[HG-DAGGER] 无交互终端，取消被拒绝的接管。", flush=True)
                return None, None
            continue
        kwargs = {"start_stage_id": chosen}
        if state.get("partial_place_candidate") is not None:
            kwargs["confirmed_partial_place"] = True
        return adapter.recovery_stage_iterator(task_env, **kwargs), chosen
