#!/usr/bin/env bash
# Resume only the held-out evaluation after the isolated source-provenance fix.
set -Eeuo pipefail

EXP=/home/ruihao/h50_support_parallel_20261027
ROOT="$EXP/root"
OPENPI="$EXP/openpi"
TASK=stack_blocks_two
SIM_PY=/home/ruihao/robotwin_hil/conda/bin/python
SERVER_PY=/home/ruihao/openpi-rtc-sync/.venv/bin/python
MODEL_PY=/home/ruihao/train_critic/.venv/bin/python
PORT=18497
export TMPDIR="$EXP/tmp" XDG_CACHE_HOME="$EXP/cache" HF_HOME="$EXP/cache/hf"
export CUDA_CACHE_PATH="$EXP/cache/cuda" MPLCONFIGDIR="$EXP/cache/matplotlib"
export OPENPI_DATA_HOME="$ROOT/openpi_data" XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export CUDA_VISIBLE_DEVICES=0
export PATH="$(dirname "$SIM_PY"):$PATH"
export PYTHONPATH="$EXP/deps:$EXP/code:$OPENPI/src:$OPENPI/pace/src:$OPENPI/packages/openpi-client/src"
cd "$ROOT"
exec 9>"$EXP/control/${TASK}_eval.lock"
flock -n 9 || { echo "evaluation already running: $TASK" >&2; exit 3; }

"$SIM_PY" - "$EXP" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); m=json.loads((p/'models/stack_blocks_two/protocol.json').read_text())
assert m['status']=='complete' and m['horizon']==50 and m['expert_episodes']==50
assert len(set(m['scene_seeds']))==50 and (p/'models/stack_blocks_two/model.pt').is_file()
for split in (0,1):
    s=json.loads((p/f'data/stack_blocks_two/seed{split}_n25/summary.json').read_text())
    assert s['complete'] and s['episodes']==25
PY

for phase in smoke formal; do
  out="$EXP/outcome/$phase/$TASK"
  [[ ! -e "$out" ]] || { echo "refusing existing output $out" >&2; exit 4; }
  count=50
  [[ "$phase" == smoke ]] && count=1
  echo "$(date -Is) evaluate $TASK $phase count=$count"
  "$SIM_PY" -u "$ROOT/pace/scripts/_eval_worker.py" \
    --task "$TASK" --server-python "$SERVER_PY" \
    --client-python "$EXP/code/trace_h50_shim.py" \
    --checkpoint-root "$EXP/checkpoints" --openpi-root "$OPENPI" \
    --robotwin-root "$ROOT" --output "$out" \
    --gpu 0 --port "$PORT" --seed 2 --episodes "$count" \
    --method fixed --horizon 50 --startup-timeout 600 --ultrafast \
    >"$EXP/control/${TASK}_${phase}_eval_retry.log" 2>&1
  "$SIM_PY" - "$out" "$TASK" "$count" <<'PY'
import json,sys
from pathlib import Path
p,task,n=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3])
launch=json.loads((p/'launch.json').read_text())
meta=json.loads((p/'evaluation/metadata.json').read_text())
rows=[json.loads(x) for x in (p/'evaluation/episodes.jsonl').read_text().splitlines()]
assert launch['status']=='complete' and meta['status']=='complete'
assert meta['task']==task and meta['method']=='fixed_h50' and meta['seed']==2
assert meta['code_provenance']=='isolated_copy_without_git'
assert len(rows)==n and len({x['scene_seed'] for x in rows})==n
assert not any(x.get('infra_error') or x.get('error') for x in rows)
for row in rows:
    trace=json.loads((p/'evaluation/trace'/f"episode_{row['episode']:03d}"/'trace.json').read_text())
    assert trace['complete'] and trace['scene_seed']==row['scene_seed']
    assert trace['success']==row['success'] and len(trace['queries'])==row['queries']
print(task,n,sum(row['success'] for row in rows))
PY
done

"$MODEL_PY" -u "$EXP/code/score_outcome_h50.py" \
  --task "$TASK" --eval-root "$EXP/outcome/formal/$TASK" \
  --checkpoint "$EXP/models/$TASK/model.pt" \
  --r3m-asset "$EXP/code/r3m18_pytorch_model.bin" \
  --output "$EXP/metrics/$TASK.json" --device cuda:0 \
  >"$EXP/control/${TASK}_score_retry.log" 2>&1
echo "$(date -Is) complete $TASK held-out evaluation and scoring"
