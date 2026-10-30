#!/usr/bin/env bash
# Opt-in full-chunk discriminator; does not start/restart the policy service.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fit_root="$repo_root/outputs/coverage_benchmark_v1_20261009/fit"

export HIL_ES_SCORER=discriminator_chunk
export HIL_ES_MODE="${HIL_ES_MODE:-enhanced}"
export HIL_ES_ACTIVATION="${HIL_ES_ACTIVATION:-manual}"
export HIL_ES_CRITIC="${HIL_ES_CRITIC:-$fit_root/disc_s42.pt}"
export HIL_ES_CALIBRATION="${HIL_ES_CALIBRATION:-$fit_root/calibration.json}"
export HIL_ES_CALIBRATION_KEY="${HIL_ES_CALIBRATION_KEY:-disc_s42}"
export HIL_ES_ENCODER_WEIGHTS="${HIL_ES_ENCODER_WEIGHTS:-/home/ruio/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth}"
export HIL_ES_NUM_CANDIDATES="${HIL_ES_NUM_CANDIDATES:-32}"
export HIL_ES_POLICY_MICROBATCH="${HIL_ES_POLICY_MICROBATCH:-8}"
export HIL_ES_HORIZON=10
export HIL_VIEWER_RENDER_RESOLUTION="${HIL_VIEWER_RENDER_RESOLUTION:-320x180}"
export HIL_VIEWER_WINDOW_RESOLUTION="${HIL_VIEWER_WINDOW_RESOLUTION:-1280x720}"
export HIL_VIEWER_FULLSCREEN="${HIL_VIEWER_FULLSCREEN:-0}"
export MANUAL_OUTPUT_DIR="${MANUAL_OUTPUT_DIR:-$repo_root/outputs/hg_dagger_discriminator_$(date +%Y%m%d_%H%M%S)}"

exec bash "$repo_root/local_serving/run_hg_dagger_manual.sh"
