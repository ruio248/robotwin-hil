"""Best-effort EWMH fullscreen request for the current process's X11 window."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time


class _ClientMessageData(ctypes.Union):
    _fields_ = [
        ("b", ctypes.c_char * 20),
        ("s", ctypes.c_short * 10),
        ("l", ctypes.c_long * 5),
    ]


class _ClientMessageEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("message_type", ctypes.c_ulong),
        ("format", ctypes.c_int),
        ("data", _ClientMessageData),
    ]


class _XEvent(ctypes.Union):
    _fields_ = [
        ("type", ctypes.c_int),
        ("xclient", _ClientMessageEvent),
        ("pad", ctypes.c_long * 24),
    ]


def _window_pid(x11, display, window: int, pid_atom: int, cardinal_atom: int) -> int | None:
    actual_type = ctypes.c_ulong()
    actual_format = ctypes.c_int()
    item_count = ctypes.c_ulong()
    bytes_after = ctypes.c_ulong()
    property_data = ctypes.POINTER(ctypes.c_ubyte)()
    status = x11.XGetWindowProperty(
        display,
        window,
        pid_atom,
        0,
        1,
        0,
        cardinal_atom,
        ctypes.byref(actual_type),
        ctypes.byref(actual_format),
        ctypes.byref(item_count),
        ctypes.byref(bytes_after),
        ctypes.byref(property_data),
    )
    try:
        if status != 0 or actual_format.value != 32 or item_count.value < 1 or not property_data:
            return None
        # Xlib represents format-32 properties as native unsigned longs.
        return int(ctypes.cast(property_data, ctypes.POINTER(ctypes.c_ulong))[0])
    finally:
        if property_data:
            x11.XFree(property_data)


def _current_process_windows(x11, display, root: int, pid_atom: int, cardinal_atom: int) -> list[int]:
    root_return = ctypes.c_ulong()
    parent_return = ctypes.c_ulong()
    children_return = ctypes.POINTER(ctypes.c_ulong)()
    child_count = ctypes.c_uint()
    status = x11.XQueryTree(
        display,
        root,
        ctypes.byref(root_return),
        ctypes.byref(parent_return),
        ctypes.byref(children_return),
        ctypes.byref(child_count),
    )
    if not status:
        return []
    try:
        return [
            int(children_return[index])
            for index in range(child_count.value)
            if _window_pid(x11, display, int(children_return[index]), pid_atom, cardinal_atom)
            == os.getpid()
        ]
    finally:
        if children_return:
            x11.XFree(children_return)


def request_fullscreen_for_current_process(timeout_seconds: float = 2.0) -> bool:
    """Request fullscreen for this process's top-level X11 window.

    The request is sent through EWMH to the window manager. It never changes
    the desktop resolution or targets another application's window. Returns
    False when fullscreen was disabled, X11 is unavailable, or no owned window
    appeared before the timeout.
    """
    if os.environ.get("HIL_VIEWER_FULLSCREEN", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False

    library = ctypes.util.find_library("X11")
    if not library or not os.environ.get("DISPLAY"):
        return False

    x11 = ctypes.CDLL(library)
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XGetWindowProperty.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
    ]
    x11.XFree.argtypes = [ctypes.c_void_p]
    x11.XFree.restype = ctypes.c_int
    x11.XQueryTree.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
        ctypes.POINTER(ctypes.c_uint),
    ]
    x11.XQueryTree.restype = ctypes.c_int
    x11.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long, ctypes.POINTER(_XEvent)]
    x11.XSendEvent.restype = ctypes.c_int
    x11.XFlush.argtypes = [ctypes.c_void_p]

    display = x11.XOpenDisplay(os.environ["DISPLAY"].encode())
    if not display:
        return False

    try:
        root = x11.XDefaultRootWindow(display)
        pid_atom = x11.XInternAtom(display, b"_NET_WM_PID", 0)
        cardinal_atom = x11.XInternAtom(display, b"CARDINAL", 0)
        state_atom = x11.XInternAtom(display, b"_NET_WM_STATE", 0)
        fullscreen_atom = x11.XInternAtom(display, b"_NET_WM_STATE_FULLSCREEN", 0)
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        windows: list[int] = []
        while time.monotonic() <= deadline:
            windows = _current_process_windows(x11, display, root, pid_atom, cardinal_atom)
            if windows:
                break
            time.sleep(0.1)
        if not windows:
            print("[HIL-VIEWER] fullscreen skipped: no current-process X11 window found", flush=True)
            return False

        # A single HG-DAgger process creates one SAPIEN viewer. If another
        # same-process window exists, fullscreen the first owned top-level.
        event = _XEvent()
        event.xclient = _ClientMessageEvent(
            type=33,  # ClientMessage
            serial=0,
            send_event=1,
            display=display,
            window=windows[0],
            message_type=state_atom,
            format=32,
        )
        event.xclient.data.l[0] = 1  # _NET_WM_STATE_ADD
        event.xclient.data.l[1] = fullscreen_atom
        event.xclient.data.l[2] = 0
        event.xclient.data.l[3] = 1  # application request
        # SubstructureNotifyMask | SubstructureRedirectMask on the root.
        mask = (1 << 19) | (1 << 20)
        sent = x11.XSendEvent(display, root, 0, mask, ctypes.byref(event))
        x11.XFlush(display)
        if sent:
            print(f"[HIL-VIEWER] fullscreen requested for this process's window 0x{windows[0]:x}", flush=True)
            return True
        print("[HIL-VIEWER] fullscreen request could not be sent", flush=True)
        return False
    finally:
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay(display)
