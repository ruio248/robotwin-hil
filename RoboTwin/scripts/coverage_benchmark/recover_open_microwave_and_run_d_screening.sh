#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
CONTROL="$ROOT/control/goal_d_pair_pipeline_20261026"
PAIR_ROOT="$ROOT/outputs/d_action_pairs_screen_20261026_v2"
COLLECTOR="$CONTROL/collect_matched_pairs_recoverable.py"
POSTPROCESS="$CONTROL/run_d_pair_postprocess_recovery_20261026.sh"
CLIENT_PYTHON=/hdd/miniconda3/envs/robotwin_hil/bin/python
SERVER_PYTHON="$OPENPI/.venv/bin/python"
PORT=18489
LOG="$CONTROL/recover_open_microwave_and_run_d_screening.log"

record() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

wait_original_pid_gone() {
  local pid="$1" needle="$2" cmd
  while [[ -r "/proc/$pid/cmdline" ]]; do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    [[ "$cmd" == *"$needle"* ]] || return 0
    record "waiting_for_original_process pid=$pid cmd=$cmd"
    sleep 20
  done
}

record "recovery_supervisor_started host=$(hostname)"
wait_original_pid_gone 635944 run_pairs_for_screened_tasks_20261026_v2.sh
wait_original_pid_gone 637845 run_d_pair_postprocess_20261026.sh

python3 - "$PAIR_ROOT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
op=root/'open_microwave_seed0_n30/state.json'
lp=root/'lift_pot_seed0_n30/summary.json'
o=json.loads(op.read_text()); l=json.loads(lp.read_text())
assert o.get('task')=='open_microwave' and o.get('status')=='failed'
assert o.get('error')=='AssertionError: target_pose cannot be None for move action.'
assert o.get('attempted_scenes')==1 and [x['scene_seed'] for x in o.get('episode_summaries',[])]==[100000]
assert l.get('complete') is True and l.get('episodes')==30 and len(set(l.get('scene_seeds',[])))==30
for task in ['beat_block_hammer','click_alarmclock','click_bell','move_playingcard_away','press_stapler','place_empty_cup','place_object_stand','place_phone_stand']:
 d=json.loads((root/f'{task}_seed0_n30/summary.json').read_text())
 assert d.get('complete') is True and d.get('episodes')==30 and len(set(d.get('scene_seeds',[])))==30
print('recovery_preconditions_ok')
PY

while ss -ltnH 2>/dev/null | grep -q ":$PORT[[:space:]]"; do
  record "waiting_for_policy_port port=$PORT"
  sleep 20
done
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  (( used <= 4096 )) && break
  record "waiting_for_gpu_slot gpu0_mib=$used"
  sleep 30
done

record "resume_open_microwave from_saved_episode_and_first_unattempted_scene"
if ! env \
  PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH" \
  OPENPI_DATA_HOME="$ROOT/openpi_data" \
  HF_HOME="$ROOT/.hf-home" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
  "$CLIENT_PYTHON" -u "$COLLECTOR" \
    --task open_microwave \
    --checkpoint-root "$ROOT/checkpoints" \
    --checkpoint-id public_pi05_open_microwave \
    --output "$PAIR_ROOT/open_microwave_seed0_n30" \
    --pace-root "$ROOT/pace" \
    --robotwin-root "$ROOT" \
    --openpi-root "$OPENPI" \
    --server-python "$SERVER_PYTHON" \
    --scene-seed 0 --scene-offset 0 --episodes 40 --target-episodes 30 \
    --gpu 0 --port "$PORT" --horizon 10 --resume \
    >> "$CONTROL/screened_tasks_v2/open_microwave_recovery.log" 2>&1; then
  record "open_microwave_recovery_failed; stopping_without_starting_D_training"
  exit 1
fi

"$CLIENT_PYTHON" - "$PAIR_ROOT/open_microwave_seed0_n30/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d.get('complete') is True and d.get('task')=='open_microwave'
assert d.get('episodes')==30 and d.get('expert_successes')==30
assert len(d.get('scene_seeds',[]))==30 and len(set(d['scene_seeds']))==30
assert any(x.get('scene_seed')==100001 and x.get('reason')=='expert_grasp_pose_missing'
           and x.get('counted_as_policy_failure') is False for x in d.get('rejected_scenes',[]))
print('open_microwave_recovery_summary_valid')
PY

for task in beat_block_hammer click_alarmclock click_bell move_playingcard_away press_stapler place_empty_cup place_object_stand place_phone_stand open_microwave lift_pot; do
  result_dir="$PAIR_ROOT/d_screening_results/$task"
  [[ ! -e "$result_dir/manifest.json" ]] || { record "refusing_existing_result task=$task"; exit 3; }
done

record "starting_recovery_aware_D_postprocess"
bash "$POSTPROCESS" >> "$CONTROL/postprocess/recovery_runner.log" 2>&1
record "recovery_and_D_postprocess_process_exited"
