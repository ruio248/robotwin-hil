"""Build an isolated runtime from pinned RoboTwin plus the HIL extensions.

This is a compatibility bridge for the existing scripts, whose imports assume
they live inside a RoboTwin checkout. It never writes into the upstream
submodule; the assembled runtime is generated under ignored ``outputs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone


UPSTREAM_COMMIT = "6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "RoboTwin"
OVERLAY = REPO_ROOT / "robotwin_hil" / "overlay"
POLICY_COMPAT = REPO_ROOT / "robotwin_hil" / "compat" / "XPolicyLab"
DEFAULT_TARGET = REPO_ROOT / "outputs" / "robotwin_hil_runtime" / "RoboTwin"
METADATA_NAME = ".robotwin_hil_runtime.json"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(SOURCE), *args], text=True).strip()


def _check_upstream() -> None:
    if _git("rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise RuntimeError(f"RoboTwin must be at {UPSTREAM_COMMIT}")
    if _git("status", "--porcelain", "--ignore-submodules=all"):
        raise RuntimeError("RoboTwin has local changes; keep the upstream checkout pristine")
    if not (SOURCE / "envs" / "_base_task.py").is_file():
        raise RuntimeError("RoboTwin submodule is not initialized")
    if not (OVERLAY / "envs" / "handover_to_tray.py").is_file():
        raise RuntimeError("HIL overlay is incomplete")
    if not (POLICY_COMPAT / "policy" / "Pi_05_RobotTwin").is_dir():
        raise RuntimeError("HIL XPolicyLab compatibility source is incomplete")


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for label, root in (("overlay", OVERLAY), ("policy_compat", POLICY_COMPAT)):
        for path in sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink()):
            relative = path.relative_to(root).as_posix()
            digest.update(f"{label}/{relative}\0".encode())
            if path.is_symlink():
                digest.update(f"link:{os.readlink(path)}".encode())
            else:
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
    return digest.hexdigest()


def _copy_overlay(destination: Path) -> int:
    count = 0
    for source in sorted(p for p in OVERLAY.rglob("*") if p.is_file() or p.is_symlink()):
        target = destination / source.relative_to(OVERLAY)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.is_file():
            target.unlink()
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, target)
        count += 1
    return count


def build(target: Path = DEFAULT_TARGET, *, rebuild: bool = False) -> Path:
    _check_upstream()
    target = target.resolve()
    fingerprint = _source_fingerprint()
    metadata = target / METADATA_NAME
    if target.exists():
        if metadata.is_file():
            old = json.loads(metadata.read_text())
            if old.get("upstream_commit") == UPSTREAM_COMMIT and old.get("extension_sha256") == fingerprint:
                print(f"Runtime already current: {target}")
                return target
        if not rebuild:
            raise RuntimeError(f"Existing runtime at {target}; pass --rebuild to preserve it as a backup")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".robotwin_hil_build_", dir=target.parent))
    try:
        # Upstream XPolicyLab is pinned in the official submodule for provenance;
        # this historical HIL runtime instead uses the compatible adapter snapshot.
        shutil.copytree(
            SOURCE,
            staging,
            dirs_exist_ok=True,
            symlinks=True,
            ignore=lambda directory, names: {name for name in names if name == ".git" or (Path(directory) == SOURCE and name == "XPolicyLab")},
        )
        shutil.copytree(POLICY_COMPAT, staging / "XPolicyLab", symlinks=True)
        overlay_count = _copy_overlay(staging)
        (staging / METADATA_NAME).write_text(
            json.dumps(
                {
                    "upstream_commit": UPSTREAM_COMMIT,
                    "extension_sha256": fingerprint,
                    "overlay_files": overlay_count,
                    "built_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source": "robotwin_hil/overlay + robotwin_hil/compat/XPolicyLab",
                },
                indent=2,
            )
            + "\n"
        )
        if target.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = target.with_name(f"{target.name}.backup-{stamp}")
            if backup.exists():
                raise RuntimeError(f"Backup already exists: {backup}")
            target.rename(backup)
            try:
                staging.rename(target)
            except Exception:
                backup.rename(target)
                raise
            print(f"Previous runtime preserved: {backup}")
        else:
            staging.rename(target)
    finally:
        if staging.exists():
            # Only a fresh, tool-created staging directory is removed on error.
            shutil.rmtree(staging)
    print(f"Built runtime: {target} ({overlay_count} overlay files)")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--rebuild", action="store_true", help="Back up an existing runtime before replacing it")
    args = parser.parse_args()
    build(args.output, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
