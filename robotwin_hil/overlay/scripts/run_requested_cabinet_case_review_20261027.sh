#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data-training/robotwin-hil-lyt/outputs/requested_three_task_case_review_20261027
PYTHON=/data-training/robotwin-hil-lyt/piperx/openpi/.venv/bin/python
mkdir -p "$ROOT/control" "$ROOT/failure/control" "$ROOT/success/control"
exec 9>"$ROOT/runner.lock"
flock -n 9 || { echo 'cabinet case-review runner already active' >&2; exit 3; }

for outcome in failure success; do
  printf '%s starting put_object_cabinet %s\n' "$(date -Is)" "$outcome" >> "$ROOT/state.log"
  REVIEW_OUTPUT_DIR="$ROOT/$outcome" REVIEW_ONLY_TASK=put_object_cabinet \
    REVIEW_OUTCOME="$outcome" "$PYTHON" -u \
    "$ROOT/control/record_three_requested_cases_20261027.py" \
    > "$ROOT/${outcome}.log" 2>&1
  "$PYTHON" - "$ROOT/$outcome/put_object_cabinet/summary.json" "$outcome" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary['complete'] and summary[f'confirmed_{sys.argv[2]}_videos'] >= 3, summary
PY
  printf '%s complete put_object_cabinet %s\n' "$(date -Is)" "$outcome" >> "$ROOT/state.log"
done
printf '%s complete cabinet cases\n' "$(date -Is)" >> "$ROOT/state.log"
