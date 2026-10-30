#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data-training/robotwin-hil-lyt
OPENPI="$ROOT/RoboTwin/XPolicyLab/policy/Pi_05_RobotTwin/openpi"
PYTHON="$ROOT/piperx/openpi/.venv/bin/python"
BASE_CHECKPOINT="$ROOT/checkpoints/pi05_robotwin_handover_to_tray_v2_promptfix/robotwin_handover_to_tray_v2_promptfix_bf16_inference/9999/params"
SOURCE_RUN="$ROOT/outputs/iwr_hg_dagger_episode_safe_300"
RUN="$ROOT/outputs/iwr_hg_dagger_episode_safe_300_bs64_retry1"
ASSETS="$ROOT/outputs/iwr_hg_dagger_episode_safe_300_assets"
CONFIG=pi05_robotwin_handover_to_tray_v2_promptfix
NORM_ID=iwr_hg_dagger_episode_safe_v1
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
EXP_NAME=iwr_hg_dagger_episode_safe_300_bs64_retry1

mkdir -p "$RUN" "$RUN/tmp" "$ROOT/cache/openpi" "$ROOT/cache/huggingface"
exec 9>"$RUN/orchestrator.lock"
if ! flock -n 9; then
  echo "Another episode-safe DAgger launcher already holds the retry run lock." >&2
  exit 2
fi

test -x "$PYTHON"
test -d "$BASE_CHECKPOINT"
test -f "$SOURCE_RUN/iwr_balanced_mix.json"
test -f "$ASSETS/$CONFIG/$NORM_ID/norm_stats.json"

echo "waiting_for_gpu $(date -u +%FT%TZ) minimum_free_mib=$MIN_FREE_MIB" | tee -a "$RUN/orchestration_state.txt"
while true; do
  selected=""
  selected_free=0
  for gpu in 0 1 2 3 4 5 6 7; do
    read -r free_mib util < <(
      nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits \
        | awk -F, '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); print $1, $2}'
    )
    if [[ "$free_mib" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] \
      && (( free_mib >= MIN_FREE_MIB && util < 35 )); then
      selected="$gpu"
      selected_free="$free_mib"
      break
    fi
  done

  if [[ -n "$selected" ]]; then
    echo "training_started $(date -u +%FT%TZ) gpu=$selected free_mib=$selected_free batch_size=64" | tee -a "$RUN/orchestration_state.txt"
    cd "$OPENPI"
    export HF_LEROBOT_HOME="$ROOT/lerobot_datasets"
    export HF_HOME="$ROOT/cache/huggingface"
    export HF_DATASETS_CACHE="$ROOT/cache/huggingface/datasets"
    export OPENPI_DATA_HOME="$ROOT/cache/openpi"
    export OPENPI_CACHE_DIR="$ROOT/cache/openpi"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export WANDB_MODE=offline
    export TMPDIR="$RUN/tmp"
    set +e
    CUDA_VISIBLE_DEVICES="$selected" "$PYTHON" scripts/train_iwr_balanced_5050.py "$CONFIG" \
      --baseline-repo-id ruio248/robotwin_handover_to_tray_v2_promptfix \
      --dagger-repo-id iwr_hg_dagger_corrected_compat \
      --norm-stats-asset-id "$NORM_ID" \
      --mix-manifest "$SOURCE_RUN/iwr_balanced_mix.json" \
      --assets-base-dir "$ASSETS" \
      --checkpoint-base-dir "$RUN/checkpoints" \
      --weight-loader-params "$BASE_CHECKPOINT" \
      --exp-name "$EXP_NAME" \
      --batch-size 64 \
      --num-train-steps 300 \
      --num-workers 8 \
      --fsdp-devices 1 \
      --lora --lora-only --no-ema \
      >"$RUN/train.log" 2>&1
    status=$?
    set -e
    if (( status == 0 )); then
      echo "training_complete $(date -u +%FT%TZ) exit_code=0" | tee -a "$RUN/orchestration_state.txt"
    else
      echo "training_failed $(date -u +%FT%TZ) exit_code=$status" | tee -a "$RUN/orchestration_state.txt"
    fi
    printf '%s\n' "$status" > "$RUN/train.exit"
    exit "$status"
  fi

  printf 'waiting_for_gpu %s free_mib_by_gpu=' "$(date -u +%FT%TZ)" >> "$RUN/gpu_wait.log"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | paste -sd, - >> "$RUN/gpu_wait.log"
  sleep "$POLL_SECONDS"
done
