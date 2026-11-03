"""Resolve the generated HIL runtime before simulator imports and asset I/O."""
from __future__ import annotations

import os
from pathlib import Path
import sys


PACKAGE_PARENT = Path(__file__).resolve().parents[1]


def runtime_root() -> Path:
    explicit = os.environ.get("ROBOTWIN_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if (PACKAGE_PARENT / ".robotwin_hil_runtime.json").is_file():
        return PACKAGE_PARENT
    return PACKAGE_PARENT / "outputs" / "robotwin_hil_runtime" / "RoboTwin"


def configure_imports(root: Path, *, source_fallback: bool = False) -> None:
    """Select one helper snapshot; never silently reuse another runtime's modules."""
    root = root.expanduser().resolve()
    source_helpers = PACKAGE_PARENT / "robotwin_hil" / "overlay" / "scripts"
    helpers = root / "scripts"
    # Parsing --help in a source checkout does not require an assembled runtime.
    # Actual execution calls this again without the fallback in activate_runtime.
    if source_fallback and not (helpers / "coverage_sampling/options.py").is_file():
        helpers = source_helpers
    origins = {
        "coverage_sampling": helpers, "tail_value": helpers, "hg_dagger_resume": helpers,
        "one_stage_dagger": helpers, "policy_rollout_archive": helpers,
        "x11_fullscreen": helpers, "eval_policy_xpolicylab": root / "scripts",
        "envs": root / "envs", "XPolicyLab": root / "XPolicyLab",
    }
    for name, module in tuple(sys.modules.items()):
        expected = origins.get(name.split(".")[0])
        filename = getattr(module, "__file__", None)
        if expected is not None and filename is not None:
            if not Path(filename).resolve().is_relative_to(expected):
                raise RuntimeError(
                    f"{name} is already loaded from {filename}, outside {expected}. "
                    "Start a fresh process with the intended --runtime-root; "
                    "mixing runtime versions is not supported."
                )
    for directory in (root, helpers):
        if directory.is_dir():
            path = str(directory)
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)


def activate_runtime(root: Path) -> Path:
    root = root.expanduser().resolve()
    required = (
        ".robotwin_hil_runtime.json", "envs/_base_task.py",
        "scripts/eval_policy_xpolicylab.py", "XPolicyLab",
        "scripts/coverage_sampling/options.py", "scripts/hg_dagger_resume.py",
        "scripts/one_stage_dagger.py", "scripts/policy_rollout_archive.py",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(
            f"Incomplete HIL runtime {root}: {missing}. "
            "Build with python -m robotwin_hil.prepare_runtime; "
            "pass --runtime-root or set ROBOTWIN_ROOT."
        )
    configure_imports(root)
    os.environ["ROBOTWIN_ROOT"] = str(root)
    # RoboTwin assets and task configs use cwd-relative paths.
    os.chdir(root)
    return root
