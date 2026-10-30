#!/usr/bin/env bash
# Dedicated offline chunk-critic experiment; never touches interactive HIL port 18300.
set -euo pipefail

repo_root="${TAIL_CHUNK_REPO_ROOT:?set TAIL_CHUNK_REPO_ROOT}"
run_root="${TAIL_CHUNK_RUN_ROOT:?set TAIL_CHUNK_RUN_ROOT}"
base_cache="${TAIL_CHUNK_BASE_CACHE:?set TAIL_CHUNK_BASE_CACHE}"
dataset_root="${TAIL_CHUNK_DATASET_ROOT:?set TAIL_CHUNK_DATASET_ROOT}"
python_bin="${TAIL_CHUNK_PYTHON:?set TAIL_CHUNK_PYTHON}"
policy_checkpoint="$run_root/baseline_9999_with_assets"
policy_pid_file="$run_root/policy.pid"
status_file="$run_root/orchestration_state.txt"
cache_dir="$run_root/cache_full_450"
train_dir="$run_root/run_full_h10_alpha1_10000"

mkdir -p "$run_root"
write_status() {
  printf 'phase=%s\ntime=%s\n' "$1" "$(date -Is)" > "$status_file"
}
stop_owned_policy() {
  if [[ -f "$policy_pid_file" ]]; then
    policy_pid="$(head -1 "$policy_pid_file")"
    if [[ "$policy_pid" =~ ^[0-9]+$ ]] &&
       ps -p "$policy_pid" -o args= 2>/dev/null | grep -Fq 'serve_tail_chunk_policy.py'; then
      kill -TERM "$policy_pid"
      printf 'Stopped owned diagnostic policy server pid=%s\n' "$policy_pid"
    fi
  fi
}
on_exit() {
  code=$?
  if [[ "$code" -ne 0 ]]; then
    write_status "failed_exit_${code}"
    stop_owned_policy
  fi
}
trap on_exit EXIT

cd "$repo_root"
policy_pid="$(head -1 "$policy_pid_file")"
if [[ ! "$policy_pid" =~ ^[0-9]+$ ]] ||
   ! ps -p "$policy_pid" -o args= | grep -Fq 'serve_tail_chunk_policy.py'; then
  echo 'Dedicated policy service PID is absent or not ours' >&2
  exit 1
fi
if ! ss -ltn | grep -Fq '127.0.0.1:18303'; then
  echo 'Dedicated policy port 18303 is not listening' >&2
  exit 1
fi
write_status cache
resume=()
if [[ -f "$cache_dir/manifest.json" ]]; then
  resume=(--resume)
fi
"$python_bin" -u scripts/tail_chunk_data.py \
  --base-cache "$base_cache" \
  --dataset-root "$dataset_root" \
  --policy-url ws://127.0.0.1:18303 \
  --policy-checkpoint "$policy_checkpoint" \
  --output-dir "$cache_dir" \
  --num-candidates 4 --horizon 10 "${resume[@]}"

"$python_bin" - "$cache_dir/manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding='utf-8'))
assert manifest['complete'] and len(manifest['episodes']) == 450
assert sum(e['split'] == 'train' for e in manifest['episodes']) == 405
assert sum(e['split'] == 'val' for e in manifest['episodes']) == 45
print('Complete chunk cache:', sum(e['anchors'] for e in manifest['episodes']), 'anchors', flush=True)
PY
stop_owned_policy
write_status training
CUDA_VISIBLE_DEVICES="${TAIL_CHUNK_TRAIN_GPU:-3}" "$python_bin" -u scripts/train_tail_chunk.py \
  --cache-dir "$cache_dir" \
  --output-dir "$train_dir" \
  --steps 10000 --batch-size 256 --width 256 --lr 1e-4 \
  --weight-decay 1e-4 --gamma .99 --alpha 1.0 \
  --grad-clip 1.0 --target-tau .005 --score-limit 1e4 \
  --eval-every 500 --log-every 50 --seed 42
write_status complete
