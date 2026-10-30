#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
EXP=/media/ruio/hdd/h50_support_20261027
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
PY=/hdd/miniconda3/envs/robotwin_hil/bin/python
ENCODER_PY="$OPENPI/.venv/bin/python"
R3M="$ROOT/control/d_pair_seed1_regeneration_20261026/assets/r3m18/pytorch_model.bin"
PORT=18496
TASKS=(open_microwave beat_block_hammer place_empty_cup stack_blocks_two)

[[ "$(findmnt -n -o SOURCE --target "$ROOT")" == /dev/sda1 ]]
[[ -s "$R3M" && -x "$PY" && -x "$ENCODER_PY" ]]
exec 9>"$EXP/control/collector.lock"
flock -n 9 || { echo "H50 collector already running" >&2; exit 3; }
export PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH"
export OPENPI_DATA_HOME="$ROOT/openpi_data" HF_HOME="$ROOT/.hf-home"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55

for task in "${TASKS[@]}"; do
  for split in 0 1; do
    out="$EXP/data/${task}_seed${split}_n25"
    log="$EXP/control/${task}_seed${split}_collect.log"
    [[ ! -e "$out" ]] || { echo "refusing existing collection $out" >&2; exit 4; }
    while true; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
      if (( used <= 4096 )) && ! ss -ltnH | grep -q ":$PORT[[:space:]]"; then break; fi
      sleep 30
    done
    printf '%s collect %s split=%s\n' "$(date -Is)" "$task" "$split"
    "$PY" -u "$EXP/code/collect_matched_pairs_h50.py" \
      --task "$task" --checkpoint-root "$ROOT/checkpoints" \
      --checkpoint-id "public_pi05_${task}" --output "$out" \
      --pace-root "$ROOT/pace" --robotwin-root "$ROOT" --openpi-root "$OPENPI" \
      --server-python "$OPENPI/.venv/bin/python" --scene-seed "$split" \
      --episodes 100 --target-episodes 25 --gpu 0 --port "$PORT" \
      --horizon 50 >"$log" 2>&1
    "$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); s=json.loads((p/"summary.json").read_text()); assert s["complete"] and s["episodes"]==25 and len(set(s["scene_seeds"]))==25' "$out"
    "$ENCODER_PY" -u "$EXP/code/encode_pairs_h50.py" \
      --task-root "$out" --asset "$R3M" --device cuda:0 --batch-size 64 \
      >"$EXP/control/${task}_seed${split}_encode.log" 2>&1
    "$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); m=json.loads((p/"r3m_feature_manifest.json").read_text()); assert m["complete"] and m["episodes"]==25 and m["action_horizon"]==50' "$out"
    printf '%s complete %s split=%s\n' "$(date -Is)" "$task" "$split"
  done
done
printf '%s all eight 25-episode H50 collections and R3M encodings complete\n' "$(date -Is)"
