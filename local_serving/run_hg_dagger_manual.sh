#!/usr/bin/env bash
# Interactive HG-DAgger: policy rollout with manual i/r takeover in the SAPIEN window.
set -u

cd /hdd/robotwin-hil
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/home/ruio/.Xauthority}"
# Fixed intrinsic render resolution (framebuffer), not OS window shrink.
export HIL_VIEWER_RESOLUTION="${HIL_VIEWER_RESOLUTION:-1280x720}"

LOG=/tmp/hg_dagger_manual_$(date +%Y%m%d_%H%M%S).log

bash ./enter_robotwin_hil.sh python -u scripts/hg_dagger_handover.py \
  --host 127.0.0.1 \
  --port 18300 \
  --policy-name Pi_05_RobotTwin \
  --ckpt-name v2_promptfix_9999 \
  --task-config handover_to_tray_v2_promptfix \
  --seed-start "${MANUAL_SEED_START:-40003}" \
  --episodes "${MANUAL_EPISODES:-3}" \
  --render-freq 10 \
  --frequency 30 \
  --save-data true \
  --output-dir "${MANUAL_OUTPUT_DIR:-/media/ruio/hdd/robotwin-hil/outputs/hg_dagger_manual_test}" \
  2>&1 | tee "$LOG"

echo
echo "log=$LOG"
echo "DONE. Press Enter to close."
read -r _
