"""Operator input and task-independent recovery-stage confirmation."""
from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty

DEFAULT_KEYS = {"i": "intervene", "r": "handback", "x": "abort", "q": "quit"}


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
            f"\n阶段判断正确吗? [Y/Enter]=确认自动, 输入 {choices} 重选, q=取消本次接管: "
        ).strip().lower()
        if answer in ("", "y", "yes"):
            return adapter.recovery_stage_iterator(task_env), None
        if answer == "q":
            return None, None
        if answer.isdigit() and int(answer) in labels:
            chosen = int(answer)
            return adapter.recovery_stage_iterator(task_env, start_stage_id=chosen), chosen
        print(f"无效输入，请输入 Y / {choices} / q")
