#!/usr/bin/env bash
# Independent policy-visited trajectory cohort for D-vs-outcome validation.
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
OUT="$ROOT/outputs/d_outcome_traces_seed2_20261026"
CONTROL="$OUT/control"
POST="$ROOT/control/d_pair_seed1_regeneration_20261026/postprocess/state.json"
PY=/hdd/miniconda3/envs/robotwin_hil/bin/python
SERVER_PY="$OPENPI/.venv/bin/python"
SHIM="$ROOT/pace/scripts/trace_python_shim.py"
GPU=0
PORT=18490
TASKS=(beat_block_hammer click_alarmclock click_bell move_playingcard_away
       press_stapler place_empty_cup place_object_stand place_phone_stand
       open_microwave lift_pot)

mkdir -p "$CONTROL"
exec 9>"$CONTROL/runner.lock"
flock -n 9 || { echo 'outcome trace queue already running' >&2; exit 3; }
STATE="$CONTROL/state.json"
LOG="$CONTROL/runner.log"
completed=0
current_task=""
write_state() {
  "$PY" - "$STATE" "$1" "${2:-}" "${3:-}" "$completed" <<'PY'
import json,os,sys
from pathlib import Path
path,status,task,phase,completed=sys.argv[1:]
tmp=path+'.tmp'
Path(tmp).write_text(json.dumps({'status':status,'task':task or None,'phase':phase or None,
 'completed_tasks':int(completed),'scene_seed_split':2,'episodes_per_task':50,
 'diagnostic_only':True,'accepted_for_training':False},indent=2)+'\n')
os.replace(tmp,path)
PY
}
record() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
on_error() {
  local rc=$?
  write_state failed "$current_task" "exit_${rc}" || true
  record "failed task=${current_task:-none} exit=$rc completed=$completed"
  exit "$rc"
}
trap on_error ERR

[[ "$(findmnt -n -o SOURCE /media/ruio/hdd)" == /dev/sda1 ]]
[[ -x "$PY" && -x "$SERVER_PY" && -x "$SHIM" ]]
[[ -s "$ROOT/pace/configs/expert_scenes.json" ]]
[[ "$(df -BG --output=avail /media/ruio/hdd | tail -1 | tr -d ' G')" -ge 100 ]]
write_state waiting_postprocess
while true; do
  status=$(
    "$PY" - "$POST" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); print(json.loads(p.read_text()).get('status','unknown') if p.exists() else 'missing')
PY
  )
  if [[ "$status" == complete ]]; then break; fi
  if [[ "$status" == failed || "$status" == blocked_collection ]]; then
    write_state blocked "" "upstream_postprocess_${status}"
    record "upstream postprocess status=$status; no rollout traces started"
    exit 2
  fi
  sleep 60
done

export PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH"
export OPENPI_DATA_HOME="$ROOT/openpi_data" HF_HOME="$ROOT/.hf-home"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
cd "$ROOT"

wait_for_slot() {
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if (( used <= 4096 )) && ! ss -ltnH 2>/dev/null | grep -q ":$PORT[[:space:]]"; then return 0; fi
    sleep 30
  done
}

verify() {
  "$PY" - "$1" "$2" "$3" <<'PY'
import json,sys
from pathlib import Path
root,task,n=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3])
launch=json.loads((root/'launch.json').read_text())
meta=json.loads((root/'evaluation/metadata.json').read_text())
rows=[json.loads(s) for s in (root/'evaluation/episodes.jsonl').read_text().splitlines() if s]
assert launch['status']=='complete' and meta['status']=='complete'
assert meta['task']==task and meta['method']=='fixed_h50' and meta['seed']==2
assert meta['expert_check_mode']=='cached_scenes'
assert len(rows)==n and len({r['scene_seed'] for r in rows})==n
assert not any(r.get('infra_error') or r.get('error') for r in rows)
for row in rows:
  path=root/'evaluation/trace'/f"episode_{row['episode']:03d}"/'trace.json'
  t=json.loads(path.read_text())
  assert t['complete'] and t['diagnostic_only'] and not t['accepted_for_training']
  assert t['scene_seed']==row['scene_seed'] and t['success']==row['success']
  assert len(t['queries'])==row['queries']
print(task, sum(r['success'] for r in rows), n)
PY
}

for task in "${TASKS[@]}"; do
  current_task="$task"
  [[ -d "$ROOT/checkpoints/$task/params" ]]
  [[ -s "$ROOT/checkpoints/$task/assets/robotwin/norm_stats.json" ]]
  wait_for_slot
  write_state running "$task" smoke
  record "smoke task=$task seed2 episodes=1"
  "$PY" -u "$ROOT/pace/scripts/_eval_worker.py" \
    --task "$task" --server-python "$SERVER_PY" --client-python "$SHIM" \
    --checkpoint-root "$ROOT/checkpoints" --openpi-root "$OPENPI" \
    --robotwin-root "$ROOT" --output "$OUT/smoke/$task" \
    --gpu "$GPU" --port "$PORT" --seed 2 --episodes 1 \
    --method fixed --horizon 50 --startup-timeout 600 --ultrafast \
    >"$CONTROL/${task}_smoke.log" 2>&1
  verify "$OUT/smoke/$task" "$task" 1
  wait_for_slot
  write_state running "$task" formal
  record "formal task=$task seed2 episodes=50"
  "$PY" -u "$ROOT/pace/scripts/_eval_worker.py" \
    --task "$task" --server-python "$SERVER_PY" --client-python "$SHIM" \
    --checkpoint-root "$ROOT/checkpoints" --openpi-root "$OPENPI" \
    --robotwin-root "$ROOT" --output "$OUT/formal/$task" \
    --gpu "$GPU" --port "$PORT" --seed 2 --episodes 50 \
    --method fixed --horizon 50 --startup-timeout 600 --ultrafast \
    >"$CONTROL/${task}_formal.log" 2>&1
  verify "$OUT/formal/$task" "$task" 50
  completed=$((completed+1))
  write_state running "$task" task_complete
  record "complete task=$task completed=$completed"
done
write_state complete "" done
record "complete ten tasks; formal trace root=$OUT/formal"
