#!/usr/bin/env bash
# Local (127.0.0.1) HG-DAgger takeover validation on the Ubuntu/4090 host.
#
# Usage:
#   bash run_local_takeover_validation.sh [seed] [auto_intervene_step]
#
# With no auto_intervene_step the run waits for a real ``i`` key press in the
# SAPIEN window; pass a step to fire the takeover automatically at that policy
# step (test hook inside hg_dagger_handover.py).
set -euo pipefail

REPO=/hdd/robotwin-hil
ROOT="$REPO/RoboTwin"
PORT="${HG_DAGGER_PORT:-18300}"
SEED="${1:-40002}"
STEP="${2:-}"
OUTPUT="${HG_DAGGER_ACCEPTANCE_OUTPUT:-/media/ruio/hdd/robotwin-hil/outputs/hg_dagger_acceptance}"

# The Ubuntu host exports a global proxy; the policy client must not use it for
# the localhost websocket connection.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/home/ruio/.Xauthority}"

ARGS=(
  --acceptance
  --host 127.0.0.1
  --port "$PORT"
  --policy-name Pi_05_RobotTwin
  --ckpt-name v2_promptfix_9999
  --task-config handover_to_tray_v2_promptfix
  --seed-start "$SEED"
  --render-freq 5
  --frequency 30
  --output-dir "$OUTPUT"
)
if [[ -n "$STEP" ]]; then
  ARGS+=(--auto-intervene-step "$STEP" --auto-label success --auto-save false)
fi

cd "$REPO"
exec bash "$REPO/enter_robotwin_hil.sh" python -u scripts/hg_dagger_handover.py "${ARGS[@]}"
