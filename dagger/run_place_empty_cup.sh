#!/usr/bin/env bash
# Task/display defaults only; server lifecycle is deliberately separate.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MANUAL_TASK_NAME=place_empty_cup
export HIL_DESKTOP="${HIL_DESKTOP:-remote}"
export HIL_VIEWER_IMAGE_SOURCE="${HIL_VIEWER_IMAGE_SOURCE:-spectator}"
export HIL_VIEWER_CAMERA_PRESET="${HIL_VIEWER_CAMERA_PRESET:-robot-front}"
export HIL_VIEWER_RENDER_RESOLUTION="${HIL_VIEWER_RENDER_RESOLUTION:-320x240}"
export HIL_VIEWER_WINDOW_RESOLUTION="${HIL_VIEWER_WINDOW_RESOLUTION:-1200x900}"
exec bash "$repo_root/robotwin_hil/local_serving/run_hg_dagger_manual.sh" "$@"
