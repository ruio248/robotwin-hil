"""Main-thread viewer rendering and existing performance diagnostics."""
from __future__ import annotations

import os
import time


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
