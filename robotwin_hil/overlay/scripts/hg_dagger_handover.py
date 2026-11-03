#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in top-level dagger/."""
from pathlib import Path
import sys

# Works inside the assembled runtime and when imported from the source tree.
for ancestor in Path(__file__).resolve().parents:
    if (ancestor / "dagger" / "__init__.py").is_file():
        if str(ancestor) not in sys.path:
            sys.path.insert(0, str(ancestor))
        break
else:
    raise ImportError("dagger package missing; rebuild with python -m robotwin_hil.prepare_runtime")

from dagger.runner import (
    PROMPT, STAGE_LABELS, RESERVED_SEED_RANGES, ROBOTWIN_ROOT,
    reserved_seed_label, build_seed_stream, parse_bool, joint_bias_dims,
    apply_action_bias, build_runtime_args, parse_args, main,
)
from dagger.takeover import (
    DEFAULT_KEYS, OperatorInterrupt, HumanInterventionInput,
    choose_recovery_stage, prompt_choice,
)
from dagger.recording import (
    read_json_object, write_json, append_jsonl, append_segment,
    finalize_episode, discard_recovery_cache,
)
from dagger.viewer import (
    render_initial_frame, setup_viewer_diagnostics, install_viewer_frame_limit,
)

if __name__ == "__main__":
    raise SystemExit(main())
