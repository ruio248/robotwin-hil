#!/usr/bin/env bash
set -euo pipefail

# Ubuntu entry point.  By default it extracts the selected HIL round, rebuilds
# equal-source norm_stats, and runs only the loader smoke.  Set RUN_TRAIN=1 to
# start policy training after the smoke passes.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname -- "$ROOT")"
OPENPI_ROOT="${OPENPI_ROOT:-$ROOT/XPolicyLab/policy/Pi_05_RobotTwin/openpi}"
PYTHON="${PYTHON:-$OPENPI_ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-pi05_robotwin_handover_to_tray_v2_promptfix}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/iwr_hg_dagger_balanced_5050}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$OUTPUT_DIR/assets}"
NORM_STATS_ASSET_ID="${NORM_STATS_ASSET_ID:-iwr_hg_dagger_balanced_5050}"
BASELINE_REPO_ID="${BASELINE_REPO_ID:-baseline}"
DAGGER_REPO_ID="${DAGGER_REPO_ID:-dagger}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-2500}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
LORA="${LORA:-1}"
LORA_ONLY="${LORA_ONLY:-1}"
NO_EMA="${NO_EMA:-1}"

if [[ -n "${HIL_RAW_ROOTS:-}" ]]; then
  read -r -a RAW_ROOTS <<< "$HIL_RAW_ROOTS"
else
  RAW_ROOTS=("$REPO_ROOT/outputs/hg_dagger_collection_r15/raw")
fi

PREPARE_ARGS=(
  --output-dir "$OUTPUT_DIR"
  --config-name "$CONFIG"
  --assets-base-dir "$ASSETS_BASE_DIR"
  --norm-stats-asset-id "$NORM_STATS_ASSET_ID"
  --overwrite
)
for raw_root in "${RAW_ROOTS[@]}"; do
  PREPARE_ARGS+=(--raw-root "$raw_root")
done
if [[ -n "${BASELINE_ROOT:-}" ]]; then
  PREPARE_ARGS+=(--baseline-root "$BASELINE_ROOT")
else
  export HF_LEROBOT_HOME="$OUTPUT_DIR"
fi

"$PYTHON" "$ROOT/scripts/prepare_hg_dagger_dataset.py" "${PREPARE_ARGS[@]}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$OUTPUT_DIR}"
MODEL_TRAIN_ARGS=()
if [[ "$LORA" == "1" ]]; then
  MODEL_TRAIN_ARGS+=(--lora)
fi
if [[ "$LORA_ONLY" == "1" ]]; then
  MODEL_TRAIN_ARGS+=(--lora-only)
fi
if [[ "$NO_EMA" == "1" ]]; then
  MODEL_TRAIN_ARGS+=(--no-ema)
fi
"$PYTHON" "$OPENPI_ROOT/scripts/train_iwr_balanced_5050.py" "$CONFIG" \
  --baseline-repo-id "$BASELINE_REPO_ID" \
  --dagger-repo-id "$DAGGER_REPO_ID" \
  --norm-stats-asset-id "$NORM_STATS_ASSET_ID" \
  --mix-manifest "$OUTPUT_DIR/iwr_balanced_mix.json" \
  --assets-base-dir "$ASSETS_BASE_DIR" \
  --batch-size "$BATCH_SIZE" \
  "${MODEL_TRAIN_ARGS[@]}" \
  --num-workers "$NUM_WORKERS" \
  --loader-smoke-batches 2

if [[ "$RUN_TRAIN" == "1" ]]; then
  TRAIN_ARGS=(
    "$CONFIG"
    --baseline-repo-id "$BASELINE_REPO_ID"
    --dagger-repo-id "$DAGGER_REPO_ID"
    --norm-stats-asset-id "$NORM_STATS_ASSET_ID"
    --mix-manifest "$OUTPUT_DIR/iwr_balanced_mix.json"
    --assets-base-dir "$ASSETS_BASE_DIR"
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR:-$OUTPUT_DIR/checkpoints}"
    --exp-name "${EXP_NAME:-iwr_hg_dagger_balanced_5050}"
    --batch-size "$BATCH_SIZE"
    --num-train-steps "$NUM_TRAIN_STEPS"
    --num-workers "$NUM_WORKERS"
    --fsdp-devices "$FSDP_DEVICES"
  )
  TRAIN_ARGS+=("${MODEL_TRAIN_ARGS[@]}")
  if [[ -n "${WEIGHT_LOADER_PARAMS:-}" ]]; then
    TRAIN_ARGS+=(--weight-loader-params "$WEIGHT_LOADER_PARAMS")
  fi
  "$PYTHON" "$OPENPI_ROOT/scripts/train_iwr_balanced_5050.py" "${TRAIN_ARGS[@]}"
fi
