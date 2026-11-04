#!/usr/bin/env bash
# Task/display defaults only; server lifecycle is deliberately separate.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MANUAL_TASK_NAME=rotate_qrcode
export HIL_DESKTOP="${HIL_DESKTOP:-remote}"
exec bash "$repo_root/robotwin_hil/local_serving/run_hg_dagger_manual.sh" "$@"
