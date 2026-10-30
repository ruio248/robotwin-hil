#!/usr/bin/env bash
set -Eeuo pipefail

BASE=/home/ruihao/d_pair_screening_seed1_20261026
DATA_ROOT="$BASE/data"
RESULT_ROOT="$BASE/results"
CONTROL="$BASE/control"
CODE="$BASE/code"
PY=/home/ruihao/train_critic/.venv/bin/python
GPU=0
TASKS=(
  beat_block_hammer
  click_alarmclock
  click_bell
  move_playingcard_away
  press_stapler
  place_empty_cup
  place_object_stand
  place_phone_stand
  open_microwave
  lift_pot
)

[[ -x "$PY" && -f "$CODE/coverage_benchmark/train_screened_task_discriminators.py" ]]
[[ -f "$DATA_ROOT/postprocess_manifest.json" ]]
[[ ! -e "$CONTROL" ]]
mkdir -p "$CONTROL"
STATE="$CONTROL/state.json"
LOG="$CONTROL/run.log"

write_state() {
  "$PY" - "$STATE" "$1" "${2:-}" "${3:-}" "${4:-0}" <<'PY'
import json, os, sys
from pathlib import Path
p, status, task, phase, completed = sys.argv[1:]
payload={"status":status,"task":task or None,"phase":phase or None,
         "completed_tasks":int(completed),"scene_seed_split":1,
         "data_root":"/home/ruihao/d_pair_screening_seed1_20261026/data",
         "results_root":"/home/ruihao/d_pair_screening_seed1_20261026/results",
         "device":"cuda:0","folds":5,"steps_per_fold":10000,"seed":42}
tmp=p+".tmp"; Path(tmp).write_text(json.dumps(payload,indent=2)+"\n"); os.replace(tmp,p)
PY
}
record() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
completed=0
current_task=""
on_error() {
  rc=$?
  write_state failed "$current_task" "exit_${rc}" "$completed" || true
  record "failed task=${current_task:-none} rc=$rc completed_tasks=$completed"
  exit "$rc"
}
trap on_error ERR

"$PY" - "$DATA_ROOT/postprocess_manifest.json" "$DATA_ROOT" "${TASKS[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text())
assert manifest.get("complete") is True
root=Path(sys.argv[2]); tasks=sys.argv[3:]
rows={x["task"]:x for x in manifest.get("tasks",[])}
assert set(rows)==set(tasks)
for task in tasks:
    task_root=root/f"{task}_seed1_n30"
    state=json.loads((task_root/"state.json").read_text())
    summary=json.loads((task_root/"summary.json").read_text())
    feature=json.loads((task_root/"r3m_feature_manifest.json").read_text())
    assert state.get("status")=="complete" and state.get("scene_seed_split")==1
    assert summary.get("complete") is True and summary.get("episodes")==30
    assert len(set(summary.get("scene_seeds",[])))==30
    assert feature.get("complete") is True and feature.get("episodes")==30
    for relative, expected in rows[task]["files_sha256"].items():
        path=task_root/relative
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual==expected, (task,relative,expected,actual)
PY

wait_for_slot() {
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if (( used <= 4096 )); then return 0; fi
    record "waiting_for_slot gpu${GPU}_memory_used_mib=$used"
    sleep 30
  done
}

for task in "${TASKS[@]}"; do
  current_task="$task"
  task_root="$DATA_ROOT/${task}_seed1_n30"
  task_log="$CONTROL/${task}.log"
  wait_for_slot
  if [[ -f "$RESULT_ROOT/$task/manifest.json" && -f "$RESULT_ROOT/$task/metrics.json" ]]; then
    "$PY" - "$RESULT_ROOT/$task" "$task" <<'PY'
import json,sys
from pathlib import Path
out,task=Path(sys.argv[1]),sys.argv[2]
m=json.loads((out/"metrics.json").read_text()); f=json.loads((out/"manifest.json").read_text())
assert m.get("complete") is True and m.get("task")==task
assert m.get("data",{}).get("scene_seed_split")==1
assert len(m.get("per_episode_out_of_fold",[]))==30
assert f.get("complete") is True and f.get("episodes")==30
assert (out/"metrics.json").is_file()
PY
    completed=$((completed + 1))
    write_state running "$task" "verified_preexisting_task_complete" "$completed"
    record "verified_preexisting task=$task completed_tasks=$completed"
    continue
  fi
  write_state running "$task" "train_5fold_episode_heldout_D" "$completed"
  record "starting task=$task seed=42 folds=5 steps_per_fold=10000 device=cuda:$GPU"
  if env PYTHONPATH="$CODE" \
    "$PY" -u "$CODE/coverage_benchmark/train_screened_task_discriminators.py" \
      --data-root "$DATA_ROOT" --task-data-dir "$task_root" --task "$task" \
      --output-root "$RESULT_ROOT" --steps 10000 --folds 5 --batch-size 256 \
      --save-every 1000 --seed 42 --device "cuda:$GPU" >"$task_log" 2>&1; then
    "$PY" - "$RESULT_ROOT/$task" "$task" <<'PY'
import json,sys
from pathlib import Path
out,task=Path(sys.argv[1]),sys.argv[2]
m=json.loads((out/"metrics.json").read_text()); f=json.loads((out/"manifest.json").read_text())
assert m.get("complete") is True and m.get("task")==task
assert m.get("data",{}).get("scene_seed_split")==1
assert len(m.get("per_episode_out_of_fold",[]))==30
assert f.get("complete") is True and f.get("episodes")==30
PY
  else
    rc=$?
    record "train_failed task=$task rc=$rc log=$task_log"
    exit "$rc"
  fi
  completed=$((completed + 1))
  write_state running "$task" "task_complete" "$completed"
  record "complete task=$task completed_tasks=$completed"
done

env PYTHONPATH="$CODE" "$PY" -u "$CODE/coverage_benchmark/aggregate_d_task_screening.py" \
  --results-root "$RESULT_ROOT" --tasks "${TASKS[@]}"
write_state complete "" "done" "$completed"
record "complete aggregate=$RESULT_ROOT/aggregate_d_task_screening.json"
