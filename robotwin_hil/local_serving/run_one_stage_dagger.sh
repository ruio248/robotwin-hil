#!/usr/bin/env bash
# Run the same expert-prefix/stage-6 protocol for baseline, enhanced or Gaussian.
# The policy service is external; this script never starts or stops it.
# Optional: MANUAL_SAVE_POLICY_ROLLOUTS=1 archives completed policy segments
# separately from HIL labels under <output-dir>/policy_rollouts/.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
arm="${ONE_STAGE_ARM:-baseline}"
case "$arm" in
  baseline) export HIL_ES_MODE=off ;;
  enhanced) export HIL_ES_MODE=enhanced ;;
  gaussian) export HIL_ES_MODE=gaussian ;;
  *) echo "ONE_STAGE_ARM must be baseline, enhanced or gaussian" >&2; exit 2 ;;
esac

export HIL_ONE_STAGE=1
export HIL_ES_ACTIVATION=fixed
export HIL_ES_WINDOW_START=0
export HIL_ES_WINDOW_END=899
export MANUAL_SEED_MODE=sequential
export MANUAL_SEED_START="${MANUAL_SEED_START:-100000}"
export MANUAL_TARGET_MODE=expert
export MANUAL_TARGET_SAVED="${MANUAL_TARGET_SAVED:-30}"
export MANUAL_MAX_ROLLOUTS="${MANUAL_MAX_ROLLOUTS:-200}"
export MANUAL_AUTO_SAVE="${MANUAL_AUTO_SAVE:-valid_hil}"
export MANUAL_POLICY_PORT="${MANUAL_POLICY_PORT:-18304}"
export HIL_VIEWER_RENDER_RESOLUTION="${HIL_VIEWER_RENDER_RESOLUTION:-320x180}"
export HIL_VIEWER_WINDOW_RESOLUTION="${HIL_VIEWER_WINDOW_RESOLUTION:-1280x720}"
export HIL_VIEWER_FULLSCREEN="${HIL_VIEWER_FULLSCREEN:-0}"
export MANUAL_OUTPUT_DIR="${MANUAL_OUTPUT_DIR:-$repo_root/outputs/one_stage_dagger_${arm}_$(date +%Y%m%d_%H%M%S)}"

if [[ "$arm" != baseline ]]; then
  : "${HIL_ONE_STAGE_PAIRED_REPORT:?Point this to the completed baseline session_*.json}"
  export HIL_ES_NUM_CANDIDATES="${HIL_ES_NUM_CANDIDATES:-32}"
  export HIL_ES_POLICY_MICROBATCH="${HIL_ES_POLICY_MICROBATCH:-32}"
  export HIL_ES_HORIZON=10
fi

if [[ "$arm" == enhanced ]]; then
  fit_root="$repo_root/outputs/coverage_benchmark_v1_20261009/fit"
  export HIL_ES_SCORER=discriminator_chunk
  export HIL_ES_CRITIC="${HIL_ES_CRITIC:-$fit_root/disc_s42.pt}"
  export HIL_ES_CALIBRATION="${HIL_ES_CALIBRATION:-$fit_root/calibration.json}"
  export HIL_ES_CALIBRATION_KEY="${HIL_ES_CALIBRATION_KEY:-disc_s42}"
  export HIL_ES_ENCODER_WEIGHTS="${HIL_ES_ENCODER_WEIGHTS:-/home/ruio/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth}"
elif [[ "$arm" == gaussian ]]; then
  : "${HIL_ES_GAUSSIAN_SIGMA:?Set the calibrated arm-joint chunk noise sigma in radians}"
fi

exec bash "$repo_root/robotwin_hil/local_serving/run_hg_dagger_manual.sh"
