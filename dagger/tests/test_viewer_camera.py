"""Spectator pose is independent of task and policy observations."""
from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from dagger.viewer import configure_spectator_camera, render_initial_frame


def test_official_spectator_pose_is_unchanged():
    args = {"render_freq": 10, "camera_xyz_x": .4}
    configure_spectator_camera(args, "official")
    assert args == {"render_freq": 10, "camera_xyz_x": .4}


def test_robot_rear_spectator_is_centered_behind_aloha():
    args = {"render_freq": 10, "collect_head_camera": True,
            "camera_xyz_x": .4, "camera_rpy_y": 2.45}
    configure_spectator_camera(args, "robot-rear")
    assert args["camera_xyz_x"] == 0.0
    assert args["camera_xyz_y"] < -.65
    assert args["camera_rpy_y"] == pytest.approx(-math.pi / 2)
    assert args["camera_rpy_p"] < 0
    assert args["collect_head_camera"] is True
    assert args["render_freq"] == 10


def test_robot_front_spectator_is_centered_across_the_table():
    args = {"render_freq": 10, "collect_head_camera": True}
    configure_spectator_camera(args, "robot-front")
    assert args["camera_xyz_x"] == 0.0
    assert args["camera_xyz_y"] > 0.0
    assert args["camera_rpy_y"] == pytest.approx(math.pi / 2)
    assert args["camera_rpy_p"] < 0.0
    assert args["collect_head_camera"] is True
    assert args["render_freq"] == 10


def test_unknown_spectator_preset_fails_before_simulation():
    with pytest.raises(ValueError, match="HIL_VIEWER_CAMERA_PRESET"):
        configure_spectator_camera({}, "diagonal")


def test_front_preset_reaches_live_viewer_before_first_frame(monkeypatch):
    """Task kwargs may be dropped by setup_scene; the live viewer must change."""
    monkeypatch.setitem(sys.modules, "x11_fullscreen", SimpleNamespace(
        request_fullscreen_for_current_process=lambda: None))
    calls = []
    viewer = SimpleNamespace(manages_fullscreen=True,
                             set_camera_xyz=lambda *xyz: calls.append(("xyz", xyz)),
                             set_camera_rpy=lambda *rpy: calls.append(("rpy", rpy)),
                             render=lambda: calls.append("render"))
    env = SimpleNamespace(viewer=viewer, _update_render=lambda: calls.append("update"))
    monkeypatch.setenv("HIL_VIEWER_IMAGE_SOURCE", "spectator")
    monkeypatch.setenv("HIL_VIEWER_CAMERA_PRESET", "robot-front")
    render_initial_frame(env)
    assert calls == [("xyz", (0.0, 0.35, 1.25)), ("rpy", (0.0, -0.50, math.pi / 2)),
                     "update", "render"]


def test_head_camera_display_binds_existing_policy_camera(monkeypatch):
    monkeypatch.setitem(sys.modules, "x11_fullscreen", SimpleNamespace(
        request_fullscreen_for_current_process=lambda: None))
    head = object()
    calls = []
    viewer = SimpleNamespace(manages_fullscreen=True,
                             set_display_camera=lambda camera, name: calls.append((camera, name)),
                             render=lambda: calls.append("render"))
    env = SimpleNamespace(viewer=viewer,
                          cameras=SimpleNamespace(static_camera_name=["head_camera"],
                                                  static_camera_list=[head]),
                          _update_render=lambda: calls.append("update"))
    monkeypatch.setenv("HIL_VIEWER_IMAGE_SOURCE", "head_camera")
    render_initial_frame(env)
    assert calls == [(head, "head_camera"), "update", "render"]


def test_head_camera_display_fails_if_camera_not_loaded(monkeypatch):
    monkeypatch.setitem(sys.modules, "x11_fullscreen", SimpleNamespace(
        request_fullscreen_for_current_process=lambda: None))
    env = SimpleNamespace(viewer=SimpleNamespace(manages_fullscreen=True),
                          cameras=SimpleNamespace(static_camera_name=[], static_camera_list=[]))
    monkeypatch.setenv("HIL_VIEWER_IMAGE_SOURCE", "head_camera")
    with pytest.raises(RuntimeError, match="loaded head_camera"):
        render_initial_frame(env)


def test_head_wrist_display_binds_three_existing_cameras(monkeypatch):
    monkeypatch.setitem(sys.modules, "x11_fullscreen", SimpleNamespace(
        request_fullscreen_for_current_process=lambda: None))
    head, left, right = object(), object(), object()
    calls = []
    viewer = SimpleNamespace(manages_fullscreen=True,
                             set_display_cameras=lambda cameras: calls.append(cameras),
                             render=lambda: calls.append("render"))
    env = SimpleNamespace(viewer=viewer,
                          cameras=SimpleNamespace(static_camera_name=["head_camera"],
                                                  static_camera_list=[head], left_camera=left,
                                                  right_camera=right),
                          _update_render=lambda: calls.append("update"))
    monkeypatch.setenv("HIL_VIEWER_IMAGE_SOURCE", "head_wrist")
    render_initial_frame(env)
    assert calls == [{"head_camera": head, "left_camera": left, "right_camera": right},
                     "update", "render"]


def test_head_wrist_display_requires_both_wrist_cameras(monkeypatch):
    monkeypatch.setitem(sys.modules, "x11_fullscreen", SimpleNamespace(
        request_fullscreen_for_current_process=lambda: None))
    env = SimpleNamespace(viewer=SimpleNamespace(manages_fullscreen=True,
                                                 set_display_cameras=lambda cameras: None),
                          cameras=SimpleNamespace(static_camera_name=["head_camera"],
                                                  static_camera_list=[object()], left_camera=object(),
                                                  right_camera=None))
    monkeypatch.setenv("HIL_VIEWER_IMAGE_SOURCE", "head_wrist")
    with pytest.raises(RuntimeError, match="both wrist cameras"):
        render_initial_frame(env)
