#!/usr/bin/env bash
# PR#510 left-to-right block handover; Windows App/xrdp by default.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MANUAL_TASK_NAME=handover_block
export HIL_DESKTOP="${HIL_DESKTOP:-remote}"
exec bash "$repo_root/robotwin_hil/local_serving/run_hg_dagger_manual.sh" "$@"
