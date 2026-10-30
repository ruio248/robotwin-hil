#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
EXP=/media/ruio/hdd/h50_support_20261027
PY=/hdd/miniconda3/envs/robotwin_hil/bin/python
MODEL_PY="$OPENPI/.venv/bin/python"
R3M="$ROOT/control/d_pair_seed1_regeneration_20261026/assets/r3m18/pytorch_model.bin"
SHIM="$EXP/code/trace_h50_shim.py"
PORT=18497
TASKS=(open_microwave beat_block_hammer place_empty_cup stack_blocks_two)

[[ "$(findmnt -n -o SOURCE --target "$ROOT")" == /dev/sda1 ]]
[[ -x "$SHIM" && -x "$PY" ]]
exec 9>"$EXP/control/evaluator.lock"
flock -n 9 || { echo 'H50 evaluator already running' >&2; exit 3; }
export PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH"
export OPENPI_DATA_HOME="$ROOT/openpi_data" HF_HOME="$ROOT/.hf-home"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export PYTHONPATH="/hdd/robotwin-hil-enhanced-sampling/RoboTwin/scripts"
cd "$ROOT"

while ! flock -n "$EXP/control/collector.lock" -c true; do
  sleep 60
done
for task in "${TASKS[@]}"; do
  for split in 0 1; do
    "$PY" - "$EXP/data/${task}_seed${split}_n25" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
s=json.loads((p/'summary.json').read_text())
m=json.loads((p/'r3m_feature_manifest.json').read_text())
assert s['complete'] and s['episodes']==25 and m['complete'] and m['episodes']==25
assert m['action_horizon']==50
PY
  done
done

for task in "${TASKS[@]}"; do
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    if (( used <= 4096 )); then break; fi
    sleep 30
  done
  model="$EXP/models/$task"
  [[ ! -e "$model" ]] || { echo "refusing existing D model $model" >&2; exit 4; }
  printf '%s train D %s\n' "$(date -Is)" "$task"
  "$MODEL_PY" -u "$EXP/code/train_support_h50.py" \
    --task "$task" \
    --task-root "$EXP/data/${task}_seed0_n25" \
    --task-root "$EXP/data/${task}_seed1_n25" \
    --output "$model" --steps 10000 --batch-size 256 --seed 42 --device cuda:0 \
    >"$EXP/control/${task}_train.log" 2>&1
  "$PY" - "$model" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
m=json.loads((p/'protocol.json').read_text())
assert m['status']=='complete' and m['expert_episodes']==50 and m['horizon']==50
assert m['training_steps']==10000 and len(m['scene_seeds'])==50
assert (p/'model.pt').is_file()
PY
done

for task in "${TASKS[@]}"; do
  for phase in smoke formal; do
    while true; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
      if (( used <= 4096 )) && ! ss -ltnH | grep -q ":$PORT[[:space:]]"; then break; fi
      sleep 30
    done
    output="$EXP/outcome/$phase/$task"
    [[ ! -e "$output" ]] || { echo "refusing existing evaluation $output" >&2; exit 4; }
    episodes=50
    [[ "$phase" == smoke ]] && episodes=1
    printf '%s %s %s episodes=%s\n' "$(date -Is)" "$task" "$phase" "$episodes"
    "$PY" -u "$ROOT/pace/scripts/_eval_worker.py" \
      --task "$task" --server-python "$OPENPI/.venv/bin/python" --client-python "$SHIM" \
      --checkpoint-root "$ROOT/checkpoints" --openpi-root "$OPENPI" \
      --robotwin-root "$ROOT" --output "$output" \
      --gpu 0 --port "$PORT" --seed 2 --episodes "$episodes" \
      --method fixed --horizon 50 --startup-timeout 600 --ultrafast \
      >"$EXP/control/${task}_${phase}_outcome.log" 2>&1
    "$PY" - "$output" "$task" "$episodes" <<'PY'
import json,sys
from pathlib import Path
p,task,n=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3])
launch=json.loads((p/'launch.json').read_text())
meta=json.loads((p/'evaluation/metadata.json').read_text())
rows=[json.loads(x) for x in (p/'evaluation/episodes.jsonl').read_text().splitlines()]
assert launch['status']=='complete' and meta['status']=='complete'
assert meta['task']==task and meta['method']=='fixed_h50' and meta['seed']==2
assert len(rows)==n and len({x['scene_seed'] for x in rows})==n
assert not any(x.get('infra_error') or x.get('error') for x in rows)
for row in rows:
  q=p/'evaluation/trace'/f"episode_{row['episode']:03d}"/'trace.json'
  trace=json.loads(q.read_text())
  assert trace['complete'] and trace['scene_seed']==row['scene_seed']
  assert trace['success']==row['success'] and len(trace['queries'])==row['queries']
print(task, n, sum(row['success'] for row in rows))
PY
  done
  "$MODEL_PY" -u "$EXP/code/score_outcome_h50.py" \
    --task "$task" --eval-root "$EXP/outcome/formal/$task" \
    --checkpoint "$EXP/models/$task/model.pt" --r3m-asset "$R3M" \
    --output "$EXP/metrics/${task}.json" --device cuda:0 \
    >"$EXP/control/${task}_score.log" 2>&1
done
printf '%s four H50 D models, fixed-H50 outcome cohorts, and D-vs-outcome metrics complete\n' "$(date -Is)"
