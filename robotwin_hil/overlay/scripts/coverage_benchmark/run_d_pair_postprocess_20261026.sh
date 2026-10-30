#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
ROBOTWIN=/media/ruio/hdd/robotwin-hil-enhanced-sampling/RoboTwin
CONTROL="$ROOT/control/goal_d_pair_pipeline_20261026"
QUEUE="${D_PAIR_QUEUE:-$CONTROL/screened_tasks_v2}"
PAIR_ROOT="${D_PAIR_ROOT:-$ROOT/outputs/d_action_pairs_screen_20261026_v2}"
RESULT_ROOT="${D_PAIR_RESULT_ROOT:-$PAIR_ROOT/d_screening_results}"
R3M="${D_PAIR_R3M_PATH:-$CONTROL/assets/r3m18/pytorch_model.bin}"
CLIENT_PYTHON=/hdd/miniconda3/envs/robotwin_hil/bin/python
SERVER_PYTHON="$OPENPI/.venv/bin/python"
TASKS=(beat_block_hammer click_alarmclock click_bell move_playingcard_away press_stapler place_empty_cup place_object_stand place_phone_stand open_microwave lift_pot)
PORT=18489

# RobotWin imports read assets/objects relative to the process working directory.
cd "$ROOT"

POSTPROCESS_DIR="${D_PAIR_POSTPROCESS_DIR:-$CONTROL/postprocess}"
mkdir -p "$POSTPROCESS_DIR"
STATE="$POSTPROCESS_DIR/state.json"
LOG="$POSTPROCESS_DIR/run.log"
printf '{"status":"waiting_for_pairs","completed_tasks":0,"started_at":"%s"}\n' "$(date -Is)" > "$STATE"

record() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
write_state() {
  "$CLIENT_PYTHON" - "$STATE" "$1" "${2:-}" "${3:-}" "${4:-0}" <<'PY'
import json,sys,tempfile,os
p,status,task,phase,count=sys.argv[1:]
d={"status":status,"task":task or None,"phase":phase or None,"completed_tasks":int(count)}
tmp=p+".tmp"
with open(tmp,"w") as f: f.write(json.dumps(d,indent=2)+"\n")
os.replace(tmp,p)
PY
}
trap 'rc=$?; write_state failed "${task:-}" "exit_${rc}" "${completed_tasks:-0}" || true' ERR

while true; do
  if "$CLIENT_PYTHON" - "$PAIR_ROOT" "${TASKS[@]}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); tasks=sys.argv[2:]; missing=[]
for task in tasks:
 p=root/f"{task}_seed0_n30"/"summary.json"
 try: d=json.loads(p.read_text())
 except Exception: missing.append(task); continue
 seeds=d.get("scene_seeds",[])
 if not (d.get("complete") is True and d.get("task")==task and d.get("episodes")==30
         and d.get("expert_successes")==30 and len(seeds)==30 and len(set(seeds))==30
         and d.get("pairs",0)>0):
  missing.append(task)
print("pair_summaries_ready" if not missing else "waiting_for_pair_summaries missing="+",".join(missing))
raise SystemExit(0 if not missing else 1)
PY
  then
    break
  fi
  status=missing
  if [[ -f "$QUEUE/orchestrator_state.json" ]]; then
    status=$("$CLIENT_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status","missing"))' "$QUEUE/orchestrator_state.json")
  fi
  record "collection_state=$status; awaiting complete per-task summaries (including any audited expert-scene recovery)"
  sleep 30
done
completed_tasks=0
write_state running "" "validating_collection" "$completed_tasks"

"$CLIENT_PYTHON" - "$PAIR_ROOT" "${TASKS[@]}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); tasks=sys.argv[2:]
for task in tasks:
 d=json.loads((root/f"{task}_seed0_n30"/"summary.json").read_text())
 assert d.get("complete") is True and d.get("task")==task
 assert d.get("episodes")==30 and d.get("expert_successes")==30
 seeds=d.get("scene_seeds",[])
 assert len(seeds)==30 and len(set(seeds))==30, (task,seeds)
 assert d.get("pairs",0)>0
 assert d.get("pair_semantics")=="expert_next_50Hz_joint_samples_vs_policy_first_H_absolute_joint_actions"
PY
test -s "$R3M"
test "$(sha256sum "$R3M" | cut -d' ' -f1)" = 58d7bfcda80d81a4c54582e0dd3b0a73aaa2cafff111325eff729419ea989c1e
export OPENPI_DATA_HOME="$ROOT/openpi_data" HF_HOME="$ROOT/.hf-home" XLA_PYTHON_CLIENT_MEM_FRACTION=0.55

wait_slot() {
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    if (( used <= 4096 )) && ! ss -ltnH | grep -q ":$PORT[[:space:]]"; then return 0; fi
    record "waiting_for_postprocess_slot gpu0_mib=$used port=$PORT"
    sleep 30
  done
}

for task in "${TASKS[@]}"; do
  pair_dir="$PAIR_ROOT/${task}_seed0_n30"
  result_dir="$RESULT_ROOT/$task"
  if [[ -e "$result_dir/manifest.json" ]]; then
    record "refusing_existing_result task=$task path=$result_dir"
    exit 3
  fi
  wait_slot
  write_state running "$task" "annotate_expert_subplan_stages" "$completed_tasks"
  record "annotate_expert_subplan_stages task=$task"
  env PYTHONPATH="$ROOT/pace/src:$ROOT:$ROOT/description/utils" \
      "$CLIENT_PYTHON" -u "$CONTROL/coverage_benchmark/annotate_pair_stages.py" \
      --task "$task" --task-root "$pair_dir" --pace-root "$ROOT/pace" \
      --robotwin-root "$ROOT" --collector "$CONTROL/collect_matched_pairs_v2.py"
  wait_slot
  write_state running "$task" "augment_candidates_k4" "$completed_tasks"
  record "augment_candidates_k4 task=$task"
  env PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH" \
      PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src:$ROOT/pace/src:$ROBOTWIN/scripts" \
      "$CLIENT_PYTHON" -u "$CONTROL/coverage_benchmark/augment_matched_policy_candidates.py" \
      --task "$task" --input-root "$pair_dir" --checkpoint-root "$ROOT/checkpoints" \
      --openpi-root "$OPENPI" --server-python "$SERVER_PYTHON" --gpu 0 --port "$PORT" --candidates 4
  wait_slot
  write_state running "$task" "encode_frozen_r3m" "$completed_tasks"
  record "encode_r3m task=$task"
  env PYTHONPATH="$CONTROL:$ROBOTWIN/scripts:$ROOT/pace/src" \
      "$SERVER_PYTHON" -u "$CONTROL/coverage_benchmark/encode_matched_pairs_r3m.py" \
      --task-root "$pair_dir" --asset "$R3M" --device cuda:0 --batch-size 64
  wait_slot
  write_state running "$task" "train_5fold_D" "$completed_tasks"
  record "train_oof_D task=$task steps_per_fold=10000 folds=5"
  env PYTHONPATH="$CONTROL:$ROBOTWIN/scripts:$ROOT/pace/src" \
      "$SERVER_PYTHON" -u "$CONTROL/coverage_benchmark/train_screened_task_discriminators.py" \
      --data-root "$PAIR_ROOT" --task "$task" --output-root "$RESULT_ROOT" \
      --steps 10000 --folds 5 --batch-size 256 --save-every 1000 --seed 42 --device cuda:0
  completed_tasks=$((completed_tasks + 1))
  write_state running "$task" "task_complete" "$completed_tasks"
done

env "$CLIENT_PYTHON" -u "$CONTROL/coverage_benchmark/aggregate_d_task_screening.py" \
  --results-root "$RESULT_ROOT" --tasks "${TASKS[@]}"
write_state complete "" "done" "$completed_tasks"
record "complete result_root=$RESULT_ROOT"
