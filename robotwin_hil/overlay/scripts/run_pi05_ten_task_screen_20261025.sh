#!/usr/bin/env bash
# Screen ten new published, task-specific pi0.5 checkpoints on Ubuntu.
# A one-scene smoke is excluded from each 50-scene formal evaluation.
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
CONTROL="$ROOT/control/ten_task_screen_20261025"
TASKS=(
  beat_block_hammer click_alarmclock click_bell move_playingcard_away
  press_stapler place_empty_cup place_object_stand place_phone_stand
  open_microwave lift_pot
)
GPU=0
PORT=18484
mkdir -p "$CONTROL"
exec 9>"$CONTROL/runner.lock"
flock -n 9 || { echo 'ten-task screen already running' >&2; exit 3; }

state() { printf '%s %s\n' "$(date '+%F %T %z')" "$*" | tee -a "$CONTROL/state.log"; }
fail_trap() {
  local rc=$?
  if (( rc != 0 )); then state "failed exit=$rc line=${BASH_LINENO[0]:-unknown}"; fi
}
trap fail_trap EXIT

[[ "$(findmnt -n -o SOURCE /media/ruio/hdd)" == /dev/sda1 ]]
[[ -f "$ROOT/checkpoints/manifest.json" ]]
[[ -f "$ROOT/pace/configs/expert_scenes.json" ]]
[[ -x /hdd/miniconda3/envs/robotwin_hil/bin/python ]]
export PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH"
export OPENPI_DATA_HOME="$ROOT/openpi_data"
export HF_HOME="$ROOT/.hf-home"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
cd "$ROOT"

python - "$ROOT/checkpoints/manifest.json" "$ROOT/pace/configs/expert_scenes.json" "${TASKS[@]}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
scenes = json.load(open(sys.argv[2]))['tasks']
models = {item['task'] for item in manifest['models']}
for task in sys.argv[3:]:
    assert task in models, f'missing published checkpoint: {task}'
    assert task in scenes and len(scenes[task]['0']) >= 50 and len(scenes[task]['2']) >= 1, f'missing cached scenes: {task}'
print('validated ten manifest entries and cached-scene cohorts')
PY

wait_for_slot() {
  local i used free
  for ((i=0; i<120; i++)); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if (( used <= 4096 && free >= 18000 )) && ! ss -ltnH 2>/dev/null | grep -q ":$PORT[[:space:]]"; then
      return 0
    fi
    sleep 30
  done
  state "resource_wait_timeout gpu=$GPU port=$PORT"
  return 1
}

verify_run() {
  local run_root="$1" task="$2" expected="$3" seed="$4"
  python - "$run_root/$task/fixed_h50/seed$seed/evaluation" "$task" "$expected" "$seed" <<'PY'
import json, pathlib, sys
root, task, expected, seed = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
meta = json.loads((root / 'metadata.json').read_text())
assert meta['status'] == 'complete' and meta['task'] == task, meta
assert meta['method'] == 'fixed_h50' and meta['seed'] == seed, meta
assert meta['expert_check_mode'] == 'cached_scenes', meta
rows = [json.loads(line) for line in (root / 'episodes.jsonl').read_text().splitlines() if line]
assert len(rows) == expected, (task, len(rows), expected)
seeds = [row['scene_seed'] for row in rows]
assert len(set(seeds)) == expected, (task, seeds)
assert not any(row.get('infra_error') or row.get('error') for row in rows), task
print(f'{task}: {sum(bool(row["success"]) for row in rows)}/{expected}; {len(seeds)} unique scenes')
PY
}

run_one() {
  local stage="$1" task="$2" episodes="$3" seed="$4"
  local logfile="$CONTROL/${stage}_${task}.log" run_root
  wait_for_slot
  state "starting stage=$stage task=$task episodes=$episodes gpu=$GPU port=$PORT"
  bash "$ROOT/pace/scripts/eval_all.sh" \
    --method fixed --horizon 50 --seed "$seed" --episodes "$episodes" \
    --gpus "$GPU" --jobs-per-gpu 1 --tasks "$task" \
    --start-port "$PORT" --ultrafast > "$logfile" 2>&1
  run_root=$(awk -F': ' '/^Results: / {print $2; exit}' "$logfile")
  [[ -n "$run_root" && -d "$run_root/$task/fixed_h50/seed$seed/evaluation" ]]
  verify_run "$run_root" "$task" "$episodes" "$seed"
  printf '%s %s %s\n' "$stage" "$task" "$run_root" >> "$CONTROL/run_paths.txt"
  state "complete stage=$stage task=$task run_root=$run_root"
}

for task in "${TASKS[@]}"; do
  state "downloading_and_verifying task=$task"
  python "$ROOT/download_from_manifest.py" \
    --manifest "$ROOT/checkpoints/manifest.json" --output "$ROOT/checkpoints" \
    "$task" > "$CONTROL/download_${task}.log" 2>&1
  grep -q 'all selected task files verified' "$CONTROL/download_${task}.log"
  run_one smoke "$task" 1 2
  run_one formal "$task" 50 0
done

python - "$CONTROL/run_paths.txt" "$CONTROL/aggregate_ten_tasks.json" "${TASKS[@]}" <<'PY'
import json, pathlib, sys
paths_file, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
expected = set(sys.argv[3:])
rows = {}
for line in paths_file.read_text().splitlines():
    stage, task, run_root = line.split(' ', 2)
    if stage != 'formal':
        continue
    root = pathlib.Path(run_root) / task / 'fixed_h50' / 'seed0' / 'evaluation'
    meta = json.loads((root / 'metadata.json').read_text())
    episodes = [json.loads(x) for x in (root / 'episodes.jsonl').read_text().splitlines() if x]
    seeds = [x['scene_seed'] for x in episodes]
    rows[task] = {
        'successes': sum(bool(x['success']) for x in episodes),
        'episodes': len(episodes),
        'scene_seeds': seeds,
        'unique_scene_seeds': len(set(seeds)),
        'checkpoint_config': meta['server_metadata']['config'],
        'task_config': meta['task_config'],
        'expert_check_mode': meta['expert_check_mode'],
        'evaluation_method': meta['method'],
        'run_root': str(root),
    }
assert set(rows) == expected and all(x['episodes'] == 50 and x['unique_scene_seeds'] == 50 for x in rows.values())
out.write_text(json.dumps({'complete': True, 'tasks': rows}, indent=2) + '\n')
PY
state 'complete ten_task_screen'
