#!/usr/bin/env bash
# Ubuntu desktop preset for the existing e/i/r HG-DAgger workflow.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HIL_ES_MODE=enhanced
export HIL_ES_ACTIVATION=manual
export HIL_ES_DEVICE="${HIL_ES_DEVICE:-cuda}"
export HIL_VIEWER_RESOLUTION="${HIL_VIEWER_RESOLUTION:-1600x900}"
export HIL_VIEWER_FULLSCREEN="${HIL_VIEWER_FULLSCREEN:-1}"
export HIL_VIEWER_MAX_FPS="${HIL_VIEWER_MAX_FPS:-45}"
export HIL_CHUNK_OBSERVATIONS="${HIL_CHUNK_OBSERVATIONS:-1}"
export HIL_ES_CRITIC="${HIL_ES_CRITIC:-/hdd/robotwin-hil/outputs/sft_policy_eval_100/coverage_v2_alpha_ablation_step10000_20260924/inference_checkpoints/alpha_1p0.pt}"
export HIL_ES_ENCODER_WEIGHTS="${HIL_ES_ENCODER_WEIGHTS:-/home/ruio/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth}"
export MANUAL_OUTPUT_DIR="${MANUAL_OUTPUT_DIR:-$repo_root/outputs/hg_dagger_collection_smooth_$(date +%Y%m%d_%H%M%S)}"

exec bash "$repo_root/local_serving/run_hg_dagger_manual.sh"
