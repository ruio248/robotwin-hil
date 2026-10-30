#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
EXP=/media/ruio/hdd/h50_support_20261027
PY=/hdd/miniconda3/envs/robotwin_hil/bin/python
MODEL_PY="$OPENPI/.venv/bin/python"
R3M="$ROOT/control/d_pair_seed1_regeneration_20261026/assets/r3m18/pytorch_model.bin"
TASK=open_microwave
PORT=18496
export PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH"
export OPENPI_DATA_HOME="$ROOT/openpi_data" HF_HOME="$ROOT/.hf-home"
export PYTHONPATH="/hdd/robotwin-hil-enhanced-sampling/RoboTwin/scripts:$EXP/code"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
[[ "$(findmnt -n -o SOURCE --target "$ROOT")" == /dev/sda1 ]]
exec 9>"$EXP/control/${TASK}_continuation.lock"
flock -n 9 || { echo 'open_microwave continuation already running' >&2; exit 3; }
cd "$ROOT"

wait_for_slot() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    if (( used <= 4096 )) && ! ss -ltnH | grep -q ":$PORT[[:space:]]"; then return; fi
    sleep 30
  done
}

# Parent queue was intentionally replaced while this self-owned first collector
# was writing.  Let that collector exit normally; never discard its data.
while ps -p 981422 -o args= 2>/dev/null | grep -q 'collect_matched_pairs_h50.py --task open_microwave'; do
  sleep 60
done
first="$EXP/data/${TASK}_seed0_n25"
"$PY" - "$first" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); s=json.loads((p/'summary.json').read_text())
assert s['complete'] and s['episodes']==25 and len(set(s['scene_seeds']))==25
PY
wait_for_slot
"$MODEL_PY" -u "$EXP/code/encode_pairs_h50.py" \
  --task-root "$first" --asset "$R3M" --device cuda:0 --batch-size 64 \
  >"$EXP/control/${TASK}_seed0_encode.log" 2>&1

second="$EXP/data/${TASK}_seed1_n25"
[[ ! -e "$second" ]] || { echo "refusing existing cohort $second" >&2; exit 4; }
wait_for_slot
echo "$(date -Is) collect task=$TASK split=1"
"$PY" -u "$EXP/code/collect_matched_pairs_h50.py" \
  --task "$TASK" --checkpoint-root "$ROOT/checkpoints" \
  --checkpoint-id "public_pi05_$TASK" --output "$second" \
  --pace-root "$ROOT/pace" --robotwin-root "$ROOT" --openpi-root "$OPENPI" \
  --server-python "$MODEL_PY" --scene-seed 1 \
  --episodes 100 --target-episodes 25 --gpu 0 --port "$PORT" --horizon 50 \
  >"$EXP/control/${TASK}_seed1_collect.log" 2>&1
"$PY" - "$second" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); s=json.loads((p/'summary.json').read_text())
assert s['complete'] and s['episodes']==25 and len(set(s['scene_seeds']))==25
PY
"$MODEL_PY" -u "$EXP/code/encode_pairs_h50.py" \
  --task-root "$second" --asset "$R3M" --device cuda:0 --batch-size 64 \
  >"$EXP/control/${TASK}_seed1_encode.log" 2>&1

wait_for_slot
model="$EXP/models/$TASK"
[[ ! -e "$model" ]] || { echo "refusing existing model $model" >&2; exit 4; }
echo "$(date -Is) train task=$TASK"
"$MODEL_PY" -u "$EXP/code/train_support_h50.py" \
  --task "$TASK" --task-root "$first" --task-root "$second" \
  --output "$model" --steps 10000 --batch-size 256 --seed 42 --device cuda:0 \
  >"$EXP/control/${TASK}_train.log" 2>&1
"$PY" - "$model" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); m=json.loads((p/'protocol.json').read_text())
assert m['status']=='complete' and m['expert_episodes']==50
assert m['horizon']==50 and len(m['scene_seeds'])==50
PY

for phase in smoke formal; do
  wait_for_slot
  output="$EXP/outcome/$phase/$TASK"
  [[ ! -e "$output" ]] || { echo "refusing existing evaluation $output" >&2; exit 4; }
  episodes=50
  [[ "$phase" == smoke ]] && episodes=1
  echo "$(date -Is) evaluate task=$TASK phase=$phase"
  "$PY" -u "$ROOT/pace/scripts/_eval_worker.py" \
    --task "$TASK" --server-python "$MODEL_PY" \
    --client-python "$EXP/code/trace_h50_shim.py" \
    --checkpoint-root "$ROOT/checkpoints" --openpi-root "$OPENPI" \
    --robotwin-root "$ROOT" --output "$output" \
    --gpu 0 --port "$PORT" --seed 2 --episodes "$episodes" \
    --method fixed --horizon 50 --startup-timeout 600 --ultrafast \
    >"$EXP/control/${TASK}_${phase}_eval.log" 2>&1
  "$PY" - "$output" "$episodes" <<'PY'
import json,sys
from pathlib import Path
p,n=Path(sys.argv[1]),int(sys.argv[2])
launch=json.loads((p/'launch.json').read_text())
meta=json.loads((p/'evaluation/metadata.json').read_text())
rows=[json.loads(x) for x in (p/'evaluation/episodes.jsonl').read_text().splitlines()]
assert launch['status']=='complete' and meta['status']=='complete'
assert meta['method']=='fixed_h50' and meta['seed']==2
assert len(rows)==n and len({x['scene_seed'] for x in rows})==n
assert not any(x.get('infra_error') or x.get('error') for x in rows)
PY
done
"$MODEL_PY" -u "$EXP/code/score_outcome_h50.py" \
  --task "$TASK" --eval-root "$EXP/outcome/formal/$TASK" \
  --checkpoint "$model/model.pt" --r3m-asset "$R3M" \
  --output "$EXP/metrics/${TASK}.json" --device cuda:0 \
  >"$EXP/control/${TASK}_score.log" 2>&1
echo "$(date -Is) complete task=$TASK"
