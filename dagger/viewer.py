"""Main-thread viewer rendering and existing performance diagnostics."""
from __future__ import annotations

import os
import time
import math


def configure_spectator_camera(args: dict, preset: str) -> None:
    """Set only the interactive spectator pose, never policy observations."""
    if preset == "official":
        return
    if preset == "robot-front":
        # Center the single spectator on the table's +y side, looking toward
        # the two Aloha arms at y=-0.65. FPSCameraController applies -yaw.
        args.update(camera_xyz_x=0.0, camera_xyz_y=0.35, camera_xyz_z=1.25,
                    camera_rpy_r=0.0, camera_rpy_p=-0.50,
                    camera_rpy_y=math.pi / 2)
        return
    if preset != "robot-rear":
        raise ValueError(f"Unknown HIL_VIEWER_CAMERA_PRESET={preset!r}; use official, robot-front or robot-rear")
    # Aloha bases are on y=-0.65. Put the viewer on their shared rear axis,
    # facing +y toward the table center (SAPIEN FPS yaw rotates about -z).
    args.update(camera_xyz_x=0.0, camera_xyz_y=-1.05, camera_xyz_z=1.55,
                camera_rpy_r=0.0, camera_rpy_p=-0.65,
                camera_rpy_y=-1.5707963267948966)


def render_initial_frame(task_env) -> None:
    from x11_fullscreen import request_fullscreen_for_current_process

    viewer = getattr(task_env, "viewer", None)
    if viewer is None:
        if os.environ.get("HIL_DIAG_ALLOW_NO_VIEWER") == "1":
            # Diagnostic-only escape hatch: run this entry point fully headless.
            return
        raise RuntimeError(
            "Human-gated mode needs a SAPIEN viewer. Set --render-freq to a positive value "
            "and launch inside the 5090 desktop session."
        )
    image_source = os.environ.get("HIL_VIEWER_IMAGE_SOURCE", "spectator").strip()
    if image_source in ("head_camera", "head_wrist"):
        cameras = getattr(task_env, "cameras", None)
        names = getattr(cameras, "static_camera_name", ())
        if "head_camera" not in names:
            raise RuntimeError("head_camera viewer requires the preview backend and a loaded head_camera")
        head = cameras.static_camera_list[names.index("head_camera")]
        if image_source == "head_wrist":
            if not hasattr(viewer, "set_display_cameras") or not all(
                getattr(cameras, name, None) is not None for name in ("left_camera", "right_camera")
            ):
                raise RuntimeError("head_wrist viewer requires the preview backend and both wrist cameras")
            viewer.set_display_cameras({
                "head_camera": head,
                "left_camera": cameras.left_camera,
                "right_camera": cameras.right_camera,
            })
        else:
            if not hasattr(viewer, "set_display_camera"):
                raise RuntimeError("head_camera viewer requires the preview backend")
            viewer.set_display_camera(head, "head_camera")
    elif image_source == "spectator":
        # _base_task._init calls setup_scene() without forwarding task kwargs.
        # Apply the preset to the actual viewer after setup, before first draw.
        preset = os.environ.get("HIL_VIEWER_CAMERA_PRESET", "official").strip()
        camera_args = {}
        configure_spectator_camera(camera_args, preset)
        if camera_args:
            xyz = tuple(camera_args[f"camera_xyz_{axis}"] for axis in "xyz")
            rpy = tuple(camera_args[f"camera_rpy_{axis}"] for axis in "rpy")
            viewer.set_camera_xyz(*xyz)
            viewer.set_camera_rpy(*rpy)
            print(f"[HIL-VIEWER] applied spectator preset={preset} xyz={xyz} rpy={rpy}", flush=True)
    else:
        raise ValueError(f"Unknown HIL_VIEWER_IMAGE_SOURCE={image_source!r}")
    task_env._update_render()
    viewer.render()
    # SAPIEN exposes resize but not a portable fullscreen API. Ask the active
    # X11 window manager to fullscreen only the X window owned by this process.
    # Keep this best-effort so headless diagnostics and non-X11 sessions work.
    if not getattr(viewer, "manages_fullscreen", False) and getattr(task_env, "_hil_fullscreen_viewer", None) is not viewer:
        request_fullscreen_for_current_process()
        task_env._hil_fullscreen_viewer = viewer

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
    """Refresh the live viewer during physics substeps, with a wall-clock cap.

    Entry/exit-only drawing makes a long action appear as a jump. Use the
    existing render-update hook inside the physics loop as an opportunity to
    draw. Lookahead sets render_freq=0, so hypothetical states remain hidden.
    This does not change physics steps, commands, recording or model images.
    """
    viewer = getattr(task_env, "viewer", None)
    if viewer is None or getattr(task_env, "_hil_viewer_bound_instance", None) is viewer:
        return
    task_env._hil_viewer_bound_instance = viewer

    raw = os.environ.get("HIL_VIEWER_MAX_FPS", "30").strip()
    try:
        max_fps = float(raw)
    except ValueError:
        print(f"[VIEWER] invalid HIL_VIEWER_MAX_FPS={raw!r}; keeping 30 fps", flush=True)
        max_fps = 30.0

    if max_fps <= 0:
        print(f"[VIEWER] frame limit disabled (HIL_VIEWER_MAX_FPS={raw})", flush=True)
        return

    min_interval = 1.0 / max_fps
    original_render = viewer.render
    state = {"last": 0.0, "rendered": 0, "skipped": 0,
             "report_time": time.perf_counter(), "report_frames": 0, "max_gap": 0.0}
    task_env._hil_viewer_stats = state

    def throttled_render(*args, **kwargs):
        now = time.perf_counter()
        if now - state["last"] < min_interval:
            state["skipped"] += 1
            return None
        if state["last"]:
            state["max_gap"] = max(state["max_gap"], now - state["last"])
        state["last"] = now
        state["rendered"] += 1
        result = original_render(*args, **kwargs)
        if os.environ.get("HIL_DIAG_VIEWER") == "1" and now - state["report_time"] >= 5:
            elapsed = now - state["report_time"]
            fps = (state["rendered"] - state["report_frames"]) / elapsed
            print(f"\n[VIEWER] measured={fps:.1f} fps max_gap={state['max_gap']:.3f}s", flush=True)
            state.update(report_time=now, report_frames=state["rendered"], max_gap=0.0)
        return result

    viewer.render = throttled_render
    original_update = getattr(task_env, "_hil_viewer_original_update", task_env._update_render)
    task_env._hil_viewer_original_update = original_update

    def update_live_viewer(*args, **kwargs):
        result = original_update(*args, **kwargs)
        if task_env.render_freq and getattr(task_env, "viewer", None) is not None:
            task_env._render_viewer_if_available()
        return result

    task_env._update_render = update_live_viewer
    print(
        f"[VIEWER] render limited to {max_fps:g} fps "
        f"(min interval {min_interval * 1000:.0f} ms); "
        "drawing live physics substeps; set HIL_VIEWER_MAX_FPS=0 for legacy drawing",
        flush=True,
    )
