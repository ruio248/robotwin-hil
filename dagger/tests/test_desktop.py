from pathlib import Path
import os
import shutil
import subprocess

import pytest

from dagger.desktop import Desktop, select_desktop, display_servers


LOCAL = Desktop(":1", "/run/user/1000/local-auth", "local")
REMOTE = Desktop(":10", "/home/operator/.Xauthority", "remote")


@pytest.mark.parametrize("override,expected", [(None, "remote"), ("local", "local")])
def test_stack_launcher_defaults_to_windows_app_but_allows_override(tmp_path, override, expected):
    repo = Path(__file__).resolve().parents[2]
    wrapper = tmp_path / "dagger/run_stack_blocks_three.sh"
    wrapper.parent.mkdir()
    shutil.copy2(repo / "dagger/run_stack_blocks_three.sh", wrapper)
    launcher = tmp_path / "robotwin_hil/local_serving/run_hg_dagger_manual.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text('printf "%s %s\\n" "$MANUAL_TASK_NAME" "$HIL_DESKTOP"\n')
    environ = {key: value for key, value in os.environ.items() if key != "HIL_DESKTOP"}
    if override is not None:
        environ["HIL_DESKTOP"] = override
    result = subprocess.run(["bash", str(wrapper)], env=environ,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"stack_blocks_three {expected}"


def test_local_override_ignores_inherited_remote_display():
    selected = select_desktop("local", environ={"DISPLAY": ":10"}, servers=[REMOTE, LOCAL], probe=lambda _: True)
    assert selected == LOCAL


def test_auto_preserves_terminal_desktop_and_ssh_without_display_uses_local():
    assert select_desktop("auto", environ={"DISPLAY": ":10", "XAUTHORITY": REMOTE.authority}, servers=[LOCAL, REMOTE], probe=lambda _: True).display == ":10"
    assert select_desktop("auto", environ={}, servers=[REMOTE, LOCAL], probe=lambda _: True) == LOCAL


def test_missing_local_does_not_silently_open_remote():
    with pytest.raises(RuntimeError, match="local desktop, found 0"):
        select_desktop("local", environ={}, servers=[REMOTE], probe=lambda _: True)


def test_remote_is_explicit_and_ambiguous_desktops_are_rejected():
    assert select_desktop("remote", environ={}, servers=[LOCAL, REMOTE], probe=lambda _: True) == REMOTE
    with pytest.raises(RuntimeError, match="found 2"):
        select_desktop("local", environ={}, servers=[LOCAL, Desktop(":2", "auth2", "local")], probe=lambda _: True)


def test_reads_xwayland_and_xrdp_auth_without_reading_auth_contents(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    local_auth = home / "local-auth"
    local_auth.touch()
    (home / ".Xauthority").touch()
    for pid, args in (("10", ["/usr/bin/Xwayland", ":1", "-auth", str(local_auth)]),
                      ("20", ["/usr/lib/xorg/Xorg", ":10", "-auth", ".Xauthority", "-config", "xrdp/xorg.conf"])):
        process = tmp_path / pid
        process.mkdir()
        (process / "cmdline").write_bytes("\0".join(args).encode() + b"\0")
        (process / "cwd").symlink_to(home)
    assert set(display_servers(tmp_path)) == {
        Desktop(":1", str(local_auth), "local"),
        Desktop(":10", str(home / ".Xauthority"), "remote"),
    }


def test_launcher_resolves_desktop_from_source_not_stale_runtime(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    launcher = tmp_path / "robotwin_hil/local_serving/run_hg_dagger_manual.sh"
    launcher.parent.mkdir(parents=True)
    shutil.copy2(repo / "robotwin_hil/local_serving/run_hg_dagger_manual.sh", launcher)
    source_dagger = tmp_path / "dagger"
    source_dagger.mkdir()
    (source_dagger / "__init__.py").touch()
    shutil.copy2(repo / "dagger/desktop.py", source_dagger / "desktop.py")
    stale_dagger = tmp_path / "outputs/robotwin_hil_runtime/RoboTwin/dagger"
    stale_dagger.mkdir(parents=True)
    (stale_dagger / "__init__.py").touch()
    (stale_dagger / "runner.py").touch()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "xwininfo").write_text("#!/bin/sh\nexit 0\n")
    (binaries / "xwininfo").chmod(0o755)
    result = subprocess.run(
        ["bash", str(launcher)], cwd=stale_dagger.parent,
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
             "ROBOTWIN_ROOT": str(stale_dagger.parent), "HIL_DESKTOP": "current",
             "HIL_DISPLAY_CHECK_ONLY": "1", "HIL_DRY_RUN": "0", "DISPLAY": ":92"},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "desktop connection ready: DISPLAY=:92" in result.stdout
