"""Window input must remain live while the operator is deciding what to save."""
import os
import sys
import termios
import threading
from io import StringIO
from types import SimpleNamespace

import pytest

from dagger.takeover import HumanInterventionInput, OperatorInterrupt, prompt_choice


class WaitingViewer:
    closed = False

    def __init__(self, key=None):
        self.window = self
        self.key = key
        self.pumps = 0
        self.waiting = threading.Event()

    def pump_events(self):
        self.pumps += 1
        if self.pumps >= 3:
            self.waiting.set()

    def key_press(self, key):
        return self.pumps >= 3 and key == self.key

    def key_down(self, key):
        return False

    def render(self):
        raise AssertionError("A prompt should not redraw the simulated scene")


def test_save_prompt_receives_viewer_key_without_physics(monkeypatch):
    monkeypatch.setattr(sys, "stdin", StringIO())
    viewer = WaitingViewer("y")
    answer = prompt_choice(SimpleNamespace(viewer=viewer), {"y": "save", "n": "discard"}, "Save?")
    assert answer == "save"
    assert viewer.waiting.is_set()


def test_quit_from_save_prompt_is_not_a_save_or_discard(monkeypatch):
    monkeypatch.setattr(sys, "stdin", StringIO())
    with pytest.raises(OperatorInterrupt) as caught:
        prompt_choice(SimpleNamespace(viewer=WaitingViewer("q")), {"y": "save", "n": "discard"}, "Save?")
    assert caught.value.event == "quit"


def test_stage_prompt_keeps_pumping_until_terminal_line_and_restores_tty(monkeypatch):
    master, slave = os.openpty()
    viewer = WaitingViewer()
    try:
        original = termios.tcgetattr(slave)
        # No terminal emulator consumes echoed output in this PTY test. On
        # macOS an undrained echo makes TCSADRAIN wait indefinitely.
        original[3] &= ~(termios.ECHO | termios.ECHONL)
        termios.tcsetattr(slave, termios.TCSANOW, original)
        original = termios.tcgetattr(slave)
        with os.fdopen(os.dup(slave), "r") as terminal:
            monkeypatch.setattr(sys, "stdin", terminal)

            def answer_after_window_responds():
                if viewer.waiting.wait(timeout=2):
                    os.write(master, b"6\n")
                else:
                    # Unblock a regressed implementation, then fail the assertion.
                    os.write(master, b"timeout\n")

            writer = threading.Thread(target=answer_after_window_responds)
            writer.start()
            with HumanInterventionInput(SimpleNamespace(viewer=viewer)) as keyboard:
                answer = keyboard.read_line("Stage? ")
            writer.join(timeout=3)
        assert answer == "6"
        assert viewer.waiting.is_set()
        restored = termios.tcgetattr(slave)
        # PENDIN is a kernel bookkeeping flag on Darwin, not a mode we own.
        mode_mask = termios.ICANON | termios.ECHO | termios.ISIG | termios.IEXTEN
        assert restored[3] & mode_mask == original[3] & mode_mask
        assert restored[0] == original[0]
        assert restored[6][termios.VMIN] == original[6][termios.VMIN]
        assert restored[6][termios.VTIME] == original[6][termios.VTIME]
    finally:
        os.close(master)
        os.close(slave)
