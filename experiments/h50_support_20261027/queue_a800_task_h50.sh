#!/usr/bin/env bash
# Wait for this task's public checkpoint to become complete, smoke once, then run.
set -Eeuo pipefail

EXP=/data-training/robotwin-hil-lyt/outputs/h50_support_parallel_20261027
SIM_PY=/data-training/qplanning_repro/.venv-robotwin-copy/bin/python
MODEL_PY=/data-training/robotwin-hil-lyt/piperx/openpi/.venv/bin/python
: "${TASK:?}"
: "${GPU:?}"
: "${PORT:?}"
[[ "$TASK" == beat_block_hammer || "$TASK" == place_empty_cup ]]
[[ "$GPU" == 4 || "$GPU" == 7 ]]
export EXP TASK GPU PORT SIM_PY MODEL_PY
export TMPDIR="$EXP/tmp" XDG_CACHE_HOME="$EXP/cache" HF_HOME="$EXP/cache/hf"
export CUDA_CACHE_PATH="$EXP/cache/cuda" MPLCONFIGDIR="$EXP/cache/matplotlib"
export OPENPI_DATA_HOME="$EXP/root/openpi_data"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$EXP/deps:$EXP/code:$EXP/openpi/src:$EXP/openpi/pace/src:$EXP/openpi/packages/openpi-client/src"
cd "$EXP/root"
exec 9>"$EXP/control/${TASK}_queue.lock"
flock -n 9 || { echo "task queue already running: $TASK" >&2; exit 3; }

manifest="$EXP/control/manifest_mirror.json"
for attempt in $(seq 1 240); do
  if "$SIM_PY" - "$manifest" "$EXP/checkpoints" "$TASK" 2>/dev/null <<'PY'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
root=Path(sys.argv[2]); task=sys.argv[3]
model=next(x for x in m['models'] if x['task']==task)
assert all((root/f['path']).is_file() and (root/f['path']).stat().st_size==f['bytes'] for f in model['files'])
PY
  then break; fi
  if (( attempt == 240 )); then echo "checkpoint wait timed out: $TASK" >&2; exit 5; fi
  sleep 30
done
"$SIM_PY" - "$manifest" "$EXP/checkpoints" "$TASK" <<'PY'
import hashlib,json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
root=Path(sys.argv[2]); task=sys.argv[3]
model=next(x for x in m['models'] if x['task']==task)
for entry in model['files']:
    p=root/entry['path']; h=hashlib.sha256()
    with p.open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024),b''): h.update(block)
    if h.hexdigest()!=entry['sha256']: raise ValueError(f'checkpoint SHA mismatch: {p}')
print('checkpoint verified',task,len(model['files']),flush=True)
PY

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  if (( used <= 4096 )) && ! ss -ltnH | grep -q ":$PORT[[:space:]]"; then break; fi
  sleep 30
done
smoke="$EXP/data/smoke_${TASK}_seed0"
[[ ! -e "$smoke" ]] || { echo "refusing existing smoke $smoke" >&2; exit 4; }
echo "$(date -Is) smoke task=$TASK GPU=$GPU"
"$SIM_PY" -u "$EXP/code/collect_matched_pairs_h50.py" \
  --task "$TASK" --checkpoint-root "$EXP/checkpoints" \
  --checkpoint-id "public_pi05_$TASK" --output "$smoke" \
  --pace-root "$EXP/root/pace" --robotwin-root "$EXP/root" --openpi-root "$EXP/openpi" \
  --server-python "$MODEL_PY" --scene-seed 0 --episodes 20 --target-episodes 1 \
  --gpu "$GPU" --port "$PORT" --horizon 50 \
  >"$EXP/control/${TASK}_smoke.log" 2>&1
"$SIM_PY" - "$smoke" <<'PY'
import json,sys,numpy as np
from pathlib import Path
p=Path(sys.argv[1]); s=json.loads((p/'summary.json').read_text())
assert s['complete'] and s['episodes']==1 and s['server_metadata']['action_horizon']==50
q=next(p.glob('episode_*_scene_*/pairs.npz'))
with np.load(q,allow_pickle=False) as z:
    n=len(z['state'])
    assert z['expert_chunks'].shape==(n,50,14)
    assert z['policy_chunks'].shape==(n,50,14)
    assert all(z[x].shape[0]==n for x in ('images_head','images_left','images_right'))
print('same-observation H50 smoke verified',s['task'],flush=True)
PY
echo "$(date -Is) launch formal task=$TASK GPU=$GPU"
exec bash "$EXP/code/run_one_card_h50.sh"
