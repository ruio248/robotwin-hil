#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OUT="$ROOT/requested_three_task_case_review_20261027"
PYTHON=/hdd/miniconda3/envs/robotwin_hil/bin/python
RECORDER="$ROOT/control/record_cached_pi05_outcome_cases_20261027.py"
PORT=18485

[[ "$(findmnt -n -o SOURCE /media/ruio/hdd)" == /dev/sda1 ]]
[[ -x "$PYTHON" && -f "$RECORDER" ]]
mkdir -p "$OUT"
exec 9>"$OUT/runner.lock"
flock -n 9 || { echo 'case-review runner already active' >&2; exit 3; }
cd "$ROOT"
export PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH"

run_one() {
  local task="$1" run="$2" outcome="$3"
  local records="$ROOT/pace/runs/$run/$task/fixed_h50/seed0/evaluation/episodes.jsonl"
  local output="$OUT/$task/$outcome"
  [[ -f "$records" ]]
  printf '%s starting %s %s\n' "$(date -Is)" "$task" "$outcome" >> "$OUT/state.log"
  "$PYTHON" -u "$RECORDER" --root "$ROOT" --task "$task" \
    --records "$records" --output "$output" --case-outcome "$outcome" \
    --max-cases 3 --gpu 0 --port "$PORT" \
    > "$OUT/${task}_${outcome}.log" 2>&1
  "$PYTHON" - "$output/summary.json" "$outcome" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary['complete'] and summary[f'confirmed_{sys.argv[2]}_videos'] >= 3, summary
PY
  printf '%s complete %s %s\n' "$(date -Is)" "$task" "$outcome" >> "$OUT/state.log"
}

run_one pick_dual_bottles 20261025-163018-443001 failure
run_one pick_dual_bottles 20261025-163018-443001 success
run_one open_microwave 20261026-012534-739684 failure
run_one open_microwave 20261026-012534-739684 success
printf '%s complete all Ubuntu cases\n' "$(date -Is)" >> "$OUT/state.log"
