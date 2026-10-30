#!/usr/bin/env python3
"""Diagnostic-only runner that injects the local basket recovery helper.

The shared public evaluation script remains untouched. This wrapper preloads
the task-specific recovery module before delegating to that evaluator, allowing
an isolated one-seed smoke of a local recovery change.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER = SCRIPT_DIR / "place_can_basket_recovery.py"
EVALUATOR = Path(os.environ.get(
    "PUBLIC_PI05_EVALUATOR", str(SCRIPT_DIR / "eval_public_pi05_candidate.py")
)).resolve()

spec = importlib.util.spec_from_file_location("place_can_basket_recovery", HELPER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load diagnostic recovery helper: {HELPER}")
helper = importlib.util.module_from_spec(spec)
sys.modules["place_can_basket_recovery"] = helper
spec.loader.exec_module(helper)
runpy.run_path(str(EVALUATOR), run_name="__main__")
