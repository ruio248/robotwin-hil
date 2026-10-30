#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OUT="$ROOT/outputs/d_outcome_auc_seed2_20261026"
RESULT_DIR="${RESULT_DIR:-$OUT}"
PAIRS="$ROOT/outputs/d_action_pairs_screen_20261026_seed1_n30"
TRACES="$ROOT/outputs/d_outcome_traces_seed2_20261026/formal"
CODE="$ROOT/control/goal_d_pair_pipeline_20261026"
PY=/media/ruio/hdd/Openpi-PACE-eval-20261022/.venv/bin/python
R3M="$ROOT/control/d_pair_seed1_regeneration_20261026/assets/r3m18/pytorch_model.bin"

test "$(findmnt -n -o SOURCE /media/ruio/hdd)" = /dev/sda1
test -s "$R3M"
mkdir -p "$RESULT_DIR"
export PYTHONPATH="$CODE:$ROOT/control/d_pair_seed1_regeneration_20261026:$ROOT/robotwin-hil-enhanced-sampling/RoboTwin/scripts"

for task in "$@"; do
  output="$RESULT_DIR/$task.json"
  test -f "$OUT/d_checkpoints/$task/manifest.json"
  test -f "$OUT/d_checkpoints/$task/metrics.json"
  test -f "$PAIRS/${task}_seed1_n30/r3m_feature_manifest.json"
  if test -e "$output"; then
    "$PY" - "$output" "$task" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert data.get("complete") is True and data.get("task")==sys.argv[2]
PY
    echo "verified_preexisting $task"
    continue
  fi
  echo "scoring $task"
  "$PY" -u "$CODE/score_outcome_traces.py" \
    --task "$task" \
    --trace-run "$TRACES/$task" \
    --d-root "$OUT/d_checkpoints/$task" \
    --r3m-asset "$R3M" \
    --feature-manifest "$PAIRS/${task}_seed1_n30/r3m_feature_manifest.json" \
    --output "$output" --device cuda:0 >"$RESULT_DIR/$task.log" 2>&1
  "$PY" - "$output" "$task" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert data.get("complete") is True and data.get("task")==sys.argv[2]
print(sys.argv[2], data["outcome_auc"])
PY
done
