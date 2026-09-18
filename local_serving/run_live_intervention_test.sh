#!/usr/bin/env bash
# Launch the test-seed live-intervention (DAgger takeover) evaluation inside a
# desktop terminal so the SAPIEN window is visible.
set -u

cd /hdd/robotwin-hil
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/home/ruio/.Xauthority}"

LOG=/tmp/live_intervention_test_$(date +%Y%m%d_%H%M%S).log

bash ./enter_robotwin_hil.sh python -u scripts/live_intervention_eval.py \
  --host 127.0.0.1 \
  --port 18300 \
  --policy-name Pi_05_RobotTwin \
  --ckpt-name v2_promptfix_9999 \
  --task-config handover_to_tray_v2_promptfix \
  --seed-start "${LIVE_SEED_START:-31000}" \
  --episodes "${LIVE_EPISODES:-100}" \
  --render-freq "${LIVE_RENDER_FREQ:-10}" \
  --frequency 30 \
  --save-videos "${LIVE_SAVE_VIDEOS:-none}" \
  --intervene-step "${LIVE_INTERVENE_STEP:-600}" \
  --output-dir "${LIVE_OUTPUT_DIR:-/media/ruio/hdd/robotwin-hil/outputs/live_intervention_test100}" \
  2>&1 | tee "$LOG"

echo
echo "log=$LOG"
echo "DONE. Press Enter to close."
read -r _
