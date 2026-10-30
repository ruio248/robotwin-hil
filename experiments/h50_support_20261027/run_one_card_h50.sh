#!/usr/bin/env bash
# One isolated task/card: 50 expert episodes (25+25), D training, and 50 held-out outcomes.
set -Eeuo pipefail

: "${EXP:?isolated experiment root required}"
: "${TASK:?RoboTwin task required}"
: "${SIM_PY:?Python with SAPIEN required}"
: "${MODEL_PY:?Python with Torch required}"
SERVER_PY="${SERVER_PY:-$MODEL_PY}"
: "${GPU:?physical GPU index required}"
: "${PORT:?dedicated policy service port required}"
ROOT="$EXP/root"
OPENPI="$EXP/openpi"
R3M="$EXP/code/r3m18_pytorch_model.bin"
TASK_OUT="$EXP/data/$TASK"
mkdir -p "$EXP/control" "$EXP/tmp" "$EXP/cache" "$TASK_OUT"
exec 9>"$EXP/control/${TASK}.lock"
flock -n 9 || { echo "task already running: $TASK" >&2; exit 3; }
export TMPDIR="$EXP/tmp" XDG_CACHE_HOME="$EXP/cache" HF_HOME="$EXP/cache/hf"
export CUDA_CACHE_PATH="$EXP/cache/cuda" MPLCONFIGDIR="$EXP/cache/matplotlib"
export OPENPI_DATA_HOME="$ROOT/openpi_data"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export CUDA_VISIBLE_DEVICES="$GPU"
export PATH="$(dirname "$SIM_PY"):$PATH"
export PYTHONPATH="$EXP/deps:$EXP/code:$OPENPI/src:$OPENPI/pace/src:$OPENPI/packages/openpi-client/src"
cd "$ROOT"

[[ -x "$SIM_PY" && -x "$MODEL_PY" && -x "$SERVER_PY" && -x "$EXP/code/trace_h50_shim.py" ]]
[[ -s "$R3M" && -s "$ROOT/pace/configs/expert_scenes.json" ]]
[[ -s "$EXP/checkpoints/$TASK/assets/robotwin/norm_stats.json" ]]
[[ -d "$EXP/checkpoints/$TASK/params" ]]

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if (( used <= 4096 )) && ! ss -ltnH | grep -q ":$PORT[[:space:]]"; then return; fi
    sleep 30
  done
}

for split in 0 1; do
  out="$TASK_OUT/seed${split}_n25"
  if [[ -e "$out" ]]; then
    [[ "$split" == 0 && "${RESUME_FIRST_COMPLETE:-0}" == 1 ]] || { echo "refusing existing cohort $out" >&2; exit 4; }
    echo "$(date -Is) reuse completed task=$TASK split=$split"
  else
    wait_for_gpu
    echo "$(date -Is) collect task=$TASK split=$split GPU=$GPU"
    "$SIM_PY" -u "$EXP/code/collect_matched_pairs_h50.py" \
    --task "$TASK" --checkpoint-root "$EXP/checkpoints" \
    --checkpoint-id "public_pi05_$TASK" --output "$out" \
    --pace-root "$ROOT/pace" --robotwin-root "$ROOT" --openpi-root "$OPENPI" \
    --server-python "$SERVER_PY" --scene-seed "$split" \
    --episodes 100 --target-episodes 25 --gpu "$GPU" --port "$PORT" --horizon 50 \
    >"$EXP/control/${TASK}_seed${split}_collect.log" 2>&1
  fi
  "$SIM_PY" - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); s=json.loads((p/'summary.json').read_text())
assert s['complete'] and s['episodes']==25 and len(set(s['scene_seeds']))==25
assert s['server_metadata']['action_horizon']==50
PY
  "$MODEL_PY" -u "$EXP/code/encode_pairs_h50.py" \
    --task-root "$out" --asset "$R3M" --device cuda:0 --batch-size 64 \
    >"$EXP/control/${TASK}_seed${split}_encode.log" 2>&1
done

wait_for_gpu
model="$EXP/models/$TASK"
[[ ! -e "$model" ]] || { echo "refusing existing model $model" >&2; exit 4; }
echo "$(date -Is) train task=$TASK GPU=$GPU"
"$MODEL_PY" -u "$EXP/code/train_support_h50.py" \
  --task "$TASK" --task-root "$TASK_OUT/seed0_n25" \
  --task-root "$TASK_OUT/seed1_n25" --output "$model" \
  --steps 10000 --batch-size 256 --seed 42 --device cuda:0 \
  >"$EXP/control/${TASK}_train.log" 2>&1
"$SIM_PY" - "$model" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); m=json.loads((p/'protocol.json').read_text())
assert m['status']=='complete' and m['horizon']==50 and m['expert_episodes']==50
assert len(m['scene_seeds'])==50 and (p/'model.pt').is_file()
PY

for phase in smoke formal; do
  wait_for_gpu
  output="$EXP/outcome/$phase/$TASK"
  [[ ! -e "$output" ]] || { echo "refusing existing rollout $output" >&2; exit 4; }
  episodes=50
  [[ "$phase" == smoke ]] && episodes=1
  echo "$(date -Is) evaluate task=$TASK phase=$phase episodes=$episodes GPU=$GPU"
  "$SIM_PY" -u "$ROOT/pace/scripts/_eval_worker.py" \
    --task "$TASK" --server-python "$SERVER_PY" \
    --client-python "$EXP/code/trace_h50_shim.py" \
    --checkpoint-root "$EXP/checkpoints" --openpi-root "$OPENPI" \
    --robotwin-root "$ROOT" --output "$output" \
    --gpu "$GPU" --port "$PORT" --seed 2 --episodes "$episodes" \
    --method fixed --horizon 50 --startup-timeout 600 --ultrafast \
    >"$EXP/control/${TASK}_${phase}_eval.log" 2>&1
  "$SIM_PY" - "$output" "$TASK" "$episodes" <<'PY'
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
  t=json.loads((p/'evaluation/trace'/f"episode_{row['episode']:03d}"/'trace.json').read_text())
  assert t['complete'] and t['scene_seed']==row['scene_seed']
  assert t['success']==row['success'] and len(t['queries'])==row['queries']
print(task,n,sum(row['success'] for row in rows))
PY
done

"$MODEL_PY" -u "$EXP/code/score_outcome_h50.py" \
  --task "$TASK" --eval-root "$EXP/outcome/formal/$TASK" \
  --checkpoint "$model/model.pt" --r3m-asset "$R3M" \
  --output "$EXP/metrics/${TASK}.json" --device cuda:0 \
  >"$EXP/control/${TASK}_score.log" 2>&1
echo "$(date -Is) complete task=$TASK"
