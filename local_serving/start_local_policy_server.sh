#!/usr/bin/env bash
set -euo pipefail

REPO="${ROBOTWIN_HIL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="$REPO/RoboTwin"
POLICY="$ROOT/XPolicyLab/policy/Pi_05_RobotTwin"
CONFIG="${POLICY_CONFIG_PATH:-$ROOT/XPolicyLab/pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml}"

export CUDA_VISIBLE_DEVICES="${POLICY_GPU_ID:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export OPENPI_LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-$REPO/.cache/openpi}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO/.cache/xdg}"
export TMPDIR="${TMPDIR:-$REPO/.cache/tmp}"
mkdir -p "$OPENPI_LOCAL_CACHE_ROOT" "$XDG_CACHE_HOME" "$TMPDIR"

cd "$ROOT"
policy_python="${OPENPI_PYTHON:-$POLICY/openpi/.venv/bin/python}"
# Worktrees share the installed runtime; PYTHONPATH below selects this checkout's
# adapter and OpenPI implementation (including get_action_candidates).
if [[ ! -x "$policy_python" && -z "${OPENPI_PYTHON:-}" ]]; then
  policy_python=/hdd/robotwin-hil/RoboTwin/XPolicyLab/policy/Pi_05_RobotTwin/openpi/.venv/bin/python
fi
[[ -x "$policy_python" ]] || { echo "OpenPI Python not found: $policy_python" >&2; exit 1; }
export PYTHONPATH="$ROOT:$POLICY/openpi/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Policy root=$REPO python=$policy_python JAX memory fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION"
exec "$policy_python" -u XPolicyLab/setup_policy_server.py \
  --config_path "$CONFIG" \
  --host 127.0.0.1 \
  --port "${MANUAL_POLICY_PORT:-18300}"
