#!/usr/bin/env bash
set -euo pipefail

REPO=/hdd/robotwin-hil
ROOT="$REPO/RoboTwin"
POLICY="$ROOT/XPolicyLab/policy/Pi_05_RobotTwin"
CONFIG="$ROOT/XPolicyLab/pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml"

export CUDA_VISIBLE_DEVICES="${POLICY_GPU_ID:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export OPENPI_LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-$REPO/.cache/openpi}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO/.cache/xdg}"
export TMPDIR="${TMPDIR:-$REPO/.cache/tmp}"
mkdir -p "$OPENPI_LOCAL_CACHE_ROOT" "$XDG_CACHE_HOME" "$TMPDIR"

cd "$ROOT"
# shellcheck disable=SC1091
source "$POLICY/openpi/.venv/bin/activate"
export PYTHONPATH="$ROOT:$POLICY/openpi/src${PYTHONPATH:+:$PYTHONPATH}"

exec python -u XPolicyLab/setup_policy_server.py \
  --config_path "$CONFIG" \
  --host 127.0.0.1 \
  --port 18300
