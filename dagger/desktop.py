"""Resolve the operator's X11 desktop without confusing xrdp with the monitor.

Only reads display-server arguments and probes X11 authorization. It does not
change a desktop session or the user's global environment.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


@dataclass(frozen=True)
class Desktop:
    display: str
    authority: str
    kind: str


def display_servers(proc_root: Path = Path("/proc")) -> list[Desktop]:
    found = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = process.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
            if not args or Path(args[0]).name not in {"Xorg", "Xwayland"}:
                continue
            display = next(arg for arg in args[1:] if re.fullmatch(r":\d+(?:\.\d+)?", arg))
            authority = Path(args[args.index("-auth") + 1])
            if not authority.is_absolute():
                authority = process.joinpath("cwd").resolve() / authority
            # Authenticate only with this user's readable authorization file.
            if authority.stat().st_uid != os.getuid() or not os.access(authority, os.R_OK):
                continue
            config = args[args.index("-config") + 1] if "-config" in args else ""
            remote = "xrdp" in Path(config).parts
            found.append(Desktop(display, str(authority), "remote" if remote else "local"))
        except (OSError, ValueError, IndexError, StopIteration, UnicodeError):
            continue
    return list(dict.fromkeys(found))


def accessible(desktop: Desktop) -> bool:
    env = {**os.environ, "DISPLAY": desktop.display, "XAUTHORITY": desktop.authority}
    try:
        result = subprocess.run(
            ["xwininfo", "-root"], env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=3, check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def select_desktop(target="auto", *, environ=None, servers=None, probe=accessible) -> Desktop:
    env = os.environ if environ is None else environ
    if target not in {"auto", "current", "local", "remote"}:
        raise ValueError("HIL_DESKTOP must be auto, current, local or remote")
    if target in {"auto", "current"} and env.get("DISPLAY"):
        inherited = Desktop(env["DISPLAY"], env.get("XAUTHORITY", str(Path.home() / ".Xauthority")), "current")
        if probe(inherited):
            return inherited
    if target == "current":
        raise RuntimeError("The inherited DISPLAY/XAUTHORITY cannot open an X11 window")
    desired = "local" if target == "auto" else target
    candidates = display_servers() if servers is None else servers
    valid = [item for item in candidates if item.kind == desired and probe(item)]
    if len(valid) != 1:
        raise RuntimeError(
            f"Expected one accessible {desired} desktop, found {len(valid)}. "
            "Open its terminal, then use HIL_DESKTOP=current with that terminal's DISPLAY/XAUTHORITY."
        )
    return valid[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("auto", "current", "local", "remote"), default="auto")
    args = parser.parse_args()
    desktop = select_desktop(args.target)
    print(f"[HIL-VIEWER] desktop={desktop.kind} DISPLAY={desktop.display} XAUTHORITY={desktop.authority}", file=sys.stderr)
    # The launcher evaluates only these explicitly shell-quoted assignments.
    for name, value in (("DISPLAY", desktop.display), ("XAUTHORITY", desktop.authority), ("SDL_VIDEODRIVER", "x11")):
        print(f"export {name}={shlex.quote(value)}")
    print("unset WAYLAND_DISPLAY")


if __name__ == "__main__":
    main()
