#!/usr/bin/env bash
# Capture three fresh, confirmed policy failures for each available released-cache task.
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OUT="$ROOT/failure_case_review_20261027"
PYTHON=/hdd/miniconda3/envs/robotwin_hil/bin/python
SCRIPT="$ROOT/control/record_cached_pi05_failure_cases_20261027.py"
PORT=18485

[[ "$(findmnt -n -o SOURCE /media/ruio/hdd)" == /dev/sda1 ]]
[[ -x "$PYTHON" && -f "$SCRIPT" ]]
mkdir -p "$OUT"
exec 9>"$OUT/runner.lock"
flock -n 9 || { echo 'failure-video runner already active' >&2; exit 3; }
cd "$ROOT"
export PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH"

run_one() {
  local task="$1" run="$2"
  local records="$ROOT/pace/runs/$run/$task/fixed_h50/seed0/evaluation/episodes.jsonl"
  [[ -f "$records" ]]
  printf '%s starting %s\n' "$(date -Is)" "$task" >> "$OUT/state.log"
  "$PYTHON" -u "$SCRIPT" --root "$ROOT" --task "$task" \
    --records "$records" --output "$OUT/$task" --max-failures 3 \
    --gpu 0 --port "$PORT" > "$OUT/${task}.log" 2>&1
  "$PYTHON" - "$OUT/$task/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert s['complete'] and s['confirmed_failure_videos'] >= 3, s
PY
  printf '%s complete %s\n' "$(date -Is)" "$task" >> "$OUT/state.log"
}

run_one place_empty_cup 20261025-234848-752317
run_one beat_block_hammer 20261025-211324-657950
run_one lift_pot 20261026-021121-869993
run_one stack_blocks_two 20261025-165743-156563
printf '%s complete all four cached-scene tasks\n' "$(date -Is)" >> "$OUT/state.log"
