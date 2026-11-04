#!/usr/bin/env bash
# Task/display defaults only: lifecycle and recording stay shared.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MANUAL_TASK_NAME=stack_blocks_three
# Windows App uses xrdp, not the local physical monitor. Explicit overrides win.
export HIL_DESKTOP="${HIL_DESKTOP:-remote}"
exec bash "$repo_root/robotwin_hil/local_serving/run_hg_dagger_manual.sh" "$@"
