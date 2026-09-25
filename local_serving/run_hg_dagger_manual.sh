#!/usr/bin/env bash
# Interactive HIL rollout. Set HIL_TAKEOVER_EVAL=1 for matched-seed off/vanilla/enhanced trials.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export ROBOTWIN_HIL_ROOT="$repo_root"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/home/ruio/.Xauthority}"
export HIL_VIEWER_RESOLUTION="${HIL_VIEWER_RESOLUTION:-960x540}"
export HIL_VIEWER_MAX_FPS="${HIL_VIEWER_MAX_FPS:-10}"

sampling_mode="${HIL_ES_MODE:-off}"
case "$sampling_mode" in off|vanilla|enhanced) ;; *) echo "Invalid HIL_ES_MODE=$sampling_mode" >&2; exit 2 ;; esac
takeover_eval="${HIL_TAKEOVER_EVAL:-0}"
case "$takeover_eval" in 0|1) ;; *) echo "HIL_TAKEOVER_EVAL must be 0 or 1" >&2; exit 2 ;; esac

if [[ "$takeover_eval" == 1 ]]; then
  seed_mode="${MANUAL_SEED_MODE:-sequential}"
  save_data="${MANUAL_SAVE_DATA:-false}"
  auto_save="${MANUAL_AUTO_SAVE:-false}"
  episodes="${MANUAL_MAX_ROLLOUTS:-20}"
  default_output="$repo_root/outputs/enhanced_sampling/human_takeover/$sampling_mode"
else
  seed_mode="${MANUAL_SEED_MODE:-random}"
  save_data="${MANUAL_SAVE_DATA:-true}"
  auto_save="${MANUAL_AUTO_SAVE:-none}"
  episodes="${MANUAL_MAX_ROLLOUTS:-200}"
  default_output="$repo_root/outputs/hg_dagger_collection_r1"
fi
output_dir="${MANUAL_OUTPUT_DIR:-$default_output}"

sampling_args=(--es-mode "$sampling_mode")
if [[ "$takeover_eval" == 1 ]]; then
  : "${HIL_ES_WINDOW_START:?Set the same zero-based decision window for all three arms}"
  : "${HIL_ES_WINDOW_END:?Set the same zero-based decision window for all three arms}"
  sampling_args+=(--es-window-start "$HIL_ES_WINDOW_START" --es-window-end "$HIL_ES_WINDOW_END")
fi
if [[ "$sampling_mode" != off ]]; then
  : "${HIL_ES_CRITIC:?Set the frozen coverage critic checkpoint}"
  : "${HIL_ES_ENCODER_WEIGHTS:?Set the local ResNet18 weights}"
  if [[ "$takeover_eval" != 1 ]]; then
    : "${HIL_ES_WINDOW_START:?Set the active decision window}"
    : "${HIL_ES_WINDOW_END:?Set the active decision window}"
    sampling_args+=(--es-window-start "$HIL_ES_WINDOW_START" --es-window-end "$HIL_ES_WINDOW_END")
  fi
  sampling_args+=(
    --es-critic "$HIL_ES_CRITIC"
    --es-encoder-weights "$HIL_ES_ENCODER_WEIGHTS"
    --es-device "${HIL_ES_DEVICE:-cpu}"
    --es-num-candidates "${HIL_ES_NUM_CANDIDATES:-4}"
    --es-horizon "${HIL_ES_HORIZON:-10}"
    --es-beta "${HIL_ES_BETA:-10}"
    --es-seed "${HIL_ES_SEED:-42}"
    --es-log-dir "${HIL_ES_LOG_DIR:-$output_dir/sampling}"
  )
fi

run_args=(
  python -u scripts/hg_dagger_handover.py
  --host 127.0.0.1
  --port "${MANUAL_POLICY_PORT:-18300}"
  --policy-name "${MANUAL_POLICY_NAME:-Pi_05_RobotTwin}"
  --ckpt-name "${MANUAL_CKPT_NAME:-v2_promptfix_9999}"
  --task-config "${MANUAL_TASK_CONFIG:-handover_to_tray_v2_promptfix}"
  --seed-start "${MANUAL_SEED_START:-40000}"
  --seed-mode "$seed_mode"
  --seed-min "${MANUAL_SEED_MIN:-40000}"
  --seed-max "${MANUAL_SEED_MAX:-99999}"
  --episodes "$episodes"
  --target-saved "${MANUAL_TARGET_SAVED:-50}"
  --target-mode "${MANUAL_TARGET_MODE:-hil}"
  --render-freq "${MANUAL_RENDER_FREQ:-10}"
  --frequency "${MANUAL_FREQUENCY:-30}"
  --save-data "$save_data"
  --auto-save "$auto_save"
  --output-dir "$output_dir"
  "${sampling_args[@]}"
)
if [[ "$takeover_eval" == 1 ]]; then
  run_args+=(--takeover-eval)
fi

if [[ "${HIL_DRY_RUN:-0}" == 1 ]]; then
  printf 'bash ./enter_robotwin_hil.sh'
  printf ' %q' "${run_args[@]}"
  printf '\n'
  exit 0
fi

log="/tmp/hg_dagger_${sampling_mode}_$(date +%Y%m%d_%H%M%S).log"
set +e
bash ./enter_robotwin_hil.sh "${run_args[@]}" 2>&1 | tee "$log"
run_status=${PIPESTATUS[0]}
set -e
echo "log=$log"
if [[ -t 0 ]]; then
  echo "DONE. Press Enter to close."
  read -r _
fi
exit "$run_status"
