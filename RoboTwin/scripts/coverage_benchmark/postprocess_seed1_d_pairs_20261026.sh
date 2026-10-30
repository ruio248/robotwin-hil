#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
ROBOTWIN=/media/ruio/hdd/robotwin-hil-enhanced-sampling/RoboTwin
CONTROL="$ROOT/control/d_pair_seed1_regeneration_20261026"
PAIR_ROOT="$ROOT/outputs/d_action_pairs_screen_20261026_seed1_n30"
POST="$CONTROL/postprocess"
R3M="$CONTROL/assets/r3m18/pytorch_model.bin"
CLIENT_PYTHON=/hdd/miniconda3/envs/robotwin_hil/bin/python
SERVER_PYTHON="$OPENPI/.venv/bin/python"
# Collection used 18489; keep the postprocess server on a fresh port so the
# collector's just-closed TCP socket cannot fail the bind preflight.
PORT=18491
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

mkdir -p "$POST"
STATE="$POST/state.json"
LOG="$POST/run.log"
write_state() {
  "$CLIENT_PYTHON" - "$STATE" "$1" "${2:-}" "${3:-}" "${4:-0}" <<'PY'
import json, os, sys
from pathlib import Path
p, status, task, phase, completed = sys.argv[1:]
tmp = p + ".tmp"
Path(tmp).write_text(json.dumps({"status":status,"task":task or None,
    "phase":phase or None,"completed_tasks":int(completed),
    "scene_seed_split":1,"pair_root":"/media/ruio/hdd/pace_pi05_eval_20261022/outputs/d_action_pairs_screen_20261026_seed1_n30"},indent=2)+"\n")
os.replace(tmp,p)
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

while true; do
  status=$("$CLIENT_PYTHON" - "$CONTROL/state.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists(): print("missing")
else: print(json.loads(p.read_text()).get("status","unknown"))
PY
)
  if [[ "$status" == complete ]]; then break; fi
  if [[ "$status" == failed || "$status" == partial_failed ]]; then
    write_state blocked_collection "" "collection_failed" 0
    record "collection_failed; postprocess will not run automatically"
    exit 2
  fi
  record "waiting_for_collection status=$status"
  sleep 60
done

# RoboTwin task assets are relative to the experiment root during stage replay.
cd "$ROOT"

"$CLIENT_PYTHON" - "$PAIR_ROOT" "${TASKS[@]}" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
for task in sys.argv[2:]:
 p=root/f"{task}_seed1_n30"
 d=json.loads((p/"summary.json").read_text())
 s=json.loads((p/"state.json").read_text())
 assert d.get("complete") is True and d.get("episodes")==30 and d.get("expert_successes")==30
 assert s.get("scene_seed_split")==1 and s.get("strictly_same_observation") is True
 assert len(set(d.get("scene_seeds",[])))==30
PY

test -s "$R3M"
export OPENPI_DATA_HOME="$ROOT/openpi_data" HF_HOME="$ROOT/.hf-home" XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
wait_for_slot() {
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if (( used <= 4096 )) && ! ss -ltnH 2>/dev/null | grep -q ":$PORT[[:space:]]"; then return 0; fi
    record "waiting_for_slot gpu=${GPU}_mib=$used port=$PORT"
    sleep 30
  done
}

for task in "${TASKS[@]}"; do
  current_task="$task"
  task_root="$PAIR_ROOT/${task}_seed1_n30"
  wait_for_slot
  write_state running "$task" "annotate_expert_subplan_stages" "$completed"
  record "annotate task=$task"
  env PYTHONPATH="$ROOT/pace/src:$ROOT:$ROOT/description/utils" \
    "$CLIENT_PYTHON" -u "$CONTROL/coverage_benchmark/annotate_pair_stages.py" \
      --task "$task" --task-root "$task_root" --pace-root "$ROOT/pace" \
      --robotwin-root "$ROOT" --collector "$CONTROL/collect_matched_pairs_v2.py"

  wait_for_slot
  write_state running "$task" "augment_same_observation_policy_candidates_k4" "$completed"
  record "augment_k4 task=$task"
  env PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH" \
      PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src:$ROOT/pace/src:$ROBOTWIN/scripts" \
    "$CLIENT_PYTHON" -u "$CONTROL/coverage_benchmark/augment_matched_policy_candidates.py" \
      --task "$task" --input-root "$task_root" --checkpoint-root "$ROOT/checkpoints" \
      --openpi-root "$OPENPI" --server-python "$SERVER_PYTHON" \
      --gpu "$GPU" --port "$PORT" --candidates 4

  wait_for_slot
  write_state running "$task" "encode_frozen_r3m_features" "$completed"
  record "encode_r3m task=$task"
  env PYTHONPATH="$CONTROL:$ROBOTWIN/scripts:$ROOT/pace/src" \
    "$SERVER_PYTHON" -u "$CONTROL/coverage_benchmark/encode_matched_pairs_r3m.py" \
      --task-root "$task_root" --asset "$R3M" --device cuda:0 --batch-size 64

  completed=$((completed + 1))
  write_state running "$task" "task_postprocess_complete" "$completed"
  record "task_postprocess_complete task=$task completed_tasks=$completed"
done

"$CLIENT_PYTHON" - "$PAIR_ROOT" "$POST/postprocess_manifest.json" "${TASKS[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
tasks = sys.argv[3:]
rows=[]
for task in tasks:
 task_root=root/f"{task}_seed1_n30"
 files={}
 for p in sorted(task_root.rglob("*")):
  if p.is_file() and p.name in {"pairs.npz","metadata.json","stage_labels.npz","stage_replay.json","policy_candidates_k4.npz","policy_candidates_k4.json","features_r3m18.npz","r3m_feature_manifest.json"}:
   files[str(p.relative_to(task_root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 rows.append({"task":task,"episode_count":30,"scene_seed_split":1,"files_sha256":files})
output.write_text(json.dumps({"complete":True,"tasks":rows},indent=2)+"\n")
PY
cp "$POST/postprocess_manifest.json" "$PAIR_ROOT/postprocess_manifest.json"
write_state complete "" "done" "$completed"
record "complete postprocess_root=$PAIR_ROOT"
