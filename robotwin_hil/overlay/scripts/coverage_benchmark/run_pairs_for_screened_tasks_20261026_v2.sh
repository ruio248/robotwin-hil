#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
SCRIPT="$ROOT/control/goal_d_pair_pipeline_20261026/collect_matched_pairs_v2.py"
CONTROL="$ROOT/control/goal_d_pair_pipeline_20261026/screened_tasks_v2"
OUTPUT="$ROOT/outputs/d_action_pairs_screen_20261026_v2"
CLIENT_PYTHON=/hdd/miniconda3/envs/robotwin_hil/bin/python
SERVER_PYTHON="$OPENPI/.venv/bin/python"
GPU=0
PORT=18489
CANDIDATE_SCENES=40
TARGET_EPISODES=30
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

mkdir -p "$CONTROL" "$OUTPUT"
STATE="$CONTROL/orchestrator_state.json"
LOG="$CONTROL/orchestrator.log"
printf 'started %s\n' "$(date -Is)" > "$CONTROL/events.log"
cat > "$STATE" <<EOF
{"status":"running","tasks":[],"candidate_scenes_per_task":$CANDIDATE_SCENES,"target_successful_expert_episodes":$TARGET_EPISODES,"gpu":$GPU,"port":$PORT,"started_at":"$(date -Is)"}
EOF

record_event() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$CONTROL/events.log" "$LOG"
}

wait_for_resources() {
  local n used
  for n in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if (( used <= 4096 )) && ! ss -ltnH 2>/dev/null | grep -q ":$PORT[[:space:]]"; then
      return 0
    fi
    record_event "waiting_for_slot gpu=$GPU memory_used_mib=$used port=$PORT"
    sleep 30
  done
  return 1
}

task_records='[]'
failed=0
for task in "${TASKS[@]}"; do
  offset=0
  # Exclude the already-used pilot scene from the D fitting/validation cohort.
  [[ "$task" != click_alarmclock ]] || offset=1
  task_out="$OUTPUT/${task}_seed0_n${TARGET_EPISODES}"
  task_log="$CONTROL/${task}.log"
  if [[ -e "$task_out" ]]; then
    record_event "failed_existing_output task=$task path=$task_out"
    failed=1
    continue
  fi
  if ! wait_for_resources; then
    record_event "failed_resource_wait task=$task"
    failed=1
    break
  fi
  record_event "starting task=$task candidates=$CANDIDATE_SCENES target_expert_successes=$TARGET_EPISODES scene_offset=$offset gpu=$GPU port=$PORT"
  if env \
    PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH" \
    OPENPI_DATA_HOME="$ROOT/openpi_data" \
    HF_HOME="$ROOT/.hf-home" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
    "$CLIENT_PYTHON" -u "$SCRIPT" \
      --task "$task" \
      --checkpoint-root "$ROOT/checkpoints" \
      --checkpoint-id "public_pi05_${task}" \
      --output "$task_out" \
      --pace-root "$ROOT/pace" \
      --robotwin-root "$ROOT" \
      --openpi-root "$OPENPI" \
      --server-python "$SERVER_PYTHON" \
      --scene-seed 0 \
      --scene-offset "$offset" \
      --episodes "$CANDIDATE_SCENES" \
      --target-episodes "$TARGET_EPISODES" \
      --gpu "$GPU" \
      --port "$PORT" \
      --horizon 10 > "$task_log" 2>&1; then
    if "$CLIENT_PYTHON" - "$task_out/state.json" <<'PY'
import json, sys
state=json.load(open(sys.argv[1]))
raise SystemExit(0 if state.get("status")=="complete" and state.get("completed_episodes")==30 and state.get("total_pairs",0)>0 else 1)
PY
    then
      record_event "complete task=$task pairs_dir=$task_out"
      task_status=complete
    else
      record_event "failed_validation task=$task state=$task_out/state.json"
      failed=1
      task_status=failed_validation
    fi
  else
    rc=$?
    record_event "failed_runner task=$task exit=$rc log=$task_log"
    failed=1
    task_status=failed_runner
  fi
  task_records=$("$CLIENT_PYTHON" - "$task_records" "$task" "$task_status" "$task_out" <<'PY'
import json,sys
rows=json.loads(sys.argv[1]); rows.append({"task":sys.argv[2],"status":sys.argv[3],"output":sys.argv[4]})
print(json.dumps(rows))
PY
)
  "$CLIENT_PYTHON" - "$STATE" "$task_records" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); d["tasks"]=json.loads(sys.argv[2]); p.write_text(json.dumps(d,indent=2)+"\n")
PY
done

"$CLIENT_PYTHON" - "$STATE" "$task_records" "$failed" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); d["tasks"]=json.loads(sys.argv[2]); d["status"]="complete" if sys.argv[3]=="0" and len(d["tasks"])==10 and all(x["status"]=="complete" for x in d["tasks"]) else "partial_failed"; d["finished_at"]=__import__("datetime").datetime.now().astimezone().isoformat(); p.write_text(json.dumps(d,indent=2)+"\n")
PY
record_event "finished status=$("$CLIENT_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$STATE")"
[[ "$failed" == 0 ]]
