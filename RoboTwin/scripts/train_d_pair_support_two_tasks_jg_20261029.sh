#!/usr/bin/env bash
set -Eeuo pipefail

# Five-fold, episode-held-out training using the established task-screening D.
BASE=/home/ruihao/d_pair_support_turn_switch_dump_bin_bigbin_20261029
DATA_ROOT="$BASE/data"
RESULT_ROOT="$BASE/results"
CONTROL="$BASE/control"
CODE=/home/ruihao/d_pair_screening_seed1_20261026/code
PY=/home/ruihao/train_critic/.venv/bin/python
GPU=0
TASKS=(turn_switch dump_bin_bigbin)

[[ -x "$PY" && -f "$CODE/coverage_benchmark/train_screened_task_discriminators.py" ]]
[[ -f "$DATA_ROOT/postprocess_manifest.json" ]]
mkdir -p "$CONTROL"
STATE="$CONTROL/state.json"
LOG="$CONTROL/orchestrator.log"
[[ ! -e "$STATE" ]]
exec 9>"$CONTROL/runner.lock"
flock -n 9 || { echo "another D training runner holds $CONTROL/runner.lock" >&2; exit 3; }

write_state() {
  "$PY" - "$STATE" "$1" "${2:-}" "${3:-}" "${4:-0}" "${5:-}" <<'PY'
import json, os, sys
from pathlib import Path
p, status, task, phase, completed, detail = sys.argv[1:]
value = {"status": status, "task": task or None, "phase": phase or None,
         "completed_tasks": int(completed), "tasks": ["turn_switch", "dump_bin_bigbin"],
         "scene_seed_split": 1, "episodes_per_task": 30, "folds": 5,
         "steps_per_fold": 10000, "batch_size": 256, "seed": 42,
         "device": "cuda:0", "detail": detail or None}
tmp = p + ".tmp"
Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
os.replace(tmp, p)
PY
}

record() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
completed=0
current_task=""
current_stage=preflight
on_error() {
  rc=$?
  write_state failed "$current_task" "$current_stage" "$completed" "exit=$rc command=$BASH_COMMAND" || true
  record "failed task=${current_task:-none} stage=$current_stage rc=$rc completed_tasks=$completed"
  exit "$rc"
}
trap on_error ERR

"$PY" - "$DATA_ROOT/postprocess_manifest.json" "$DATA_ROOT" "${TASKS[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest.get("complete") is True
root, tasks = Path(sys.argv[2]), sys.argv[3:]
rows = {row["task"]: row for row in manifest.get("tasks", [])}
assert set(rows) == set(tasks)
for task in tasks:
    task_root = root / f"{task}_seed1_n30"
    state = json.loads((task_root / "state.json").read_text())
    summary = json.loads((task_root / "summary.json").read_text())
    feature = json.loads((task_root / "r3m_feature_manifest.json").read_text())
    assert state.get("status") == "complete" and state.get("completed_episodes") == 30
    assert summary.get("complete") is True and summary.get("episodes") == 30
    assert summary.get("expert_successes") == 30 and len(set(summary.get("scene_seeds", []))) == 30
    assert feature.get("complete") is True and feature.get("episodes") == 30
    for relative, expected in rows[task]["files_sha256"].items():
        actual = hashlib.sha256((task_root / relative).read_bytes()).hexdigest()
        assert actual == expected, (task, relative)
PY

[[ ! -e "$RESULT_ROOT" ]]
mkdir -p "$RESULT_ROOT"
write_state running "" preflight 0
wait_for_slot() {
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if (( used <= 4096 )); then return 0; fi
    record "waiting_for_slot gpu=$GPU memory_used_mib=$used"
    sleep 30
  done
}

for task in "${TASKS[@]}"; do
  current_task="$task"
  current_stage=verify_no_existing_result
  task_root="$DATA_ROOT/${task}_seed1_n30"
  task_log="$CONTROL/${task}.log"
  [[ ! -e "$RESULT_ROOT/$task" ]]
  wait_for_slot
  current_stage=train_five_fold_episode_heldout_D
  write_state running "$task" "$current_stage" "$completed"
  record "starting task=$task split=episode_held_out folds=5 steps_per_fold=10000 batch=256 gpu=$GPU"
  if env PYTHONPATH="$CODE" "$PY" -u "$CODE/coverage_benchmark/train_screened_task_discriminators.py" \
      --data-root "$DATA_ROOT" --task-data-dir "$task_root" --task "$task" \
      --output-root "$RESULT_ROOT" --steps 10000 --folds 5 --batch-size 256 \
      --save-every 1000 --seed 42 --device "cuda:$GPU" >"$task_log" 2>&1; then
    "$PY" - "$RESULT_ROOT/$task" "$task" <<'PY'
import json, sys
from pathlib import Path
out, task = Path(sys.argv[1]), sys.argv[2]
metrics = json.loads((out / "metrics.json").read_text())
manifest = json.loads((out / "manifest.json").read_text())
assert metrics.get("complete") is True and metrics.get("task") == task
assert metrics.get("data", {}).get("scene_seed_split") == 1
assert len(metrics.get("per_episode_out_of_fold", [])) == 30
assert manifest.get("complete") is True and manifest.get("episodes") == 30
PY
  else
    rc=$?
    record "train_failed task=$task rc=$rc log=$task_log"
    exit "$rc"
  fi
  completed=$((completed + 1))
  current_stage=task_complete
  write_state running "$task" "$current_stage" "$completed"
  record "complete task=$task completed_tasks=$completed"
done

current_task=""
current_stage=aggregate_metrics
env PYTHONPATH="$CODE" "$PY" -u "$CODE/coverage_benchmark/aggregate_d_task_screening.py" \
  --results-root "$RESULT_ROOT" --tasks "${TASKS[@]}"
write_state complete "" done "$completed"
record "complete aggregate=$RESULT_ROOT/aggregate_d_task_screening.json"
