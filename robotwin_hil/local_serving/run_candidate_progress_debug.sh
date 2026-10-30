#!/usr/bin/env bash
# Own a dedicated policy server; never reuse/kill the interactive HIL service.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${ROBOTWIN_ROOT:-$REPO/outputs/robotwin_hil_runtime/RoboTwin}"
OUT="${CANDIDATE_DEBUG_OUTPUT:-$REPO/outputs/candidate_progress_k4_8_16_32_20261003}"
DATA="${CANDIDATE_DEBUG_DATA:-$REPO/outputs/hg_dagger_collection_smooth_20260930_163549}"
PY="${CANDIDATE_DEBUG_PYTHON:-/hdd/miniconda3/envs/robotwin_hil/bin/python}"
PORT="${CANDIDATE_DEBUG_PORT:-18302}"
CONFIG="$ROOT/XPolicyLab/pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml"
CRITIC=/hdd/robotwin-hil/outputs/sft_policy_eval_100/coverage_v2_alpha_ablation_step10000_20260924/inference_checkpoints/alpha_1p0.pt
ENCODER=/home/ruio/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
mkdir -p "$OUT"
exec 9>"$OUT/orchestrator.lock"
flock -n 9 || { echo "This diagnostic is already running"; exit 1; }
printf '%s\n' "$$" > "$OUT/orchestrator.pid"
policy_pid=""
cleanup() {
    status=$?
    if [[ -n "$policy_pid" ]] && kill -0 "$policy_pid" 2>/dev/null; then
        kill -TERM "$policy_pid"
        wait "$policy_pid" || true
    fi
    if [[ $status -ne 0 ]]; then
        printf 'failed exit=%s\n' "$status" > "$OUT/orchestration_state.txt"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$REPO"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
printf 'unit_tests\n' > "$OUT/orchestration_state.txt"
"$PY" -m unittest discover -s "$ROOT/scripts/tests" -p test_candidate_progress.py -v
"$PY" - "$PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PY
printf 'starting_dedicated_policy\n' > "$OUT/orchestration_state.txt"
MANUAL_POLICY_PORT="$PORT" XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    bash robotwin_hil/local_serving/start_local_policy_server.sh > "$OUT/policy_server.log" 2>&1 &
policy_pid=$!
printf '%s\n' "$policy_pid" > "$OUT/policy_server.pid"
"$PY" - "$PORT" "$policy_pid" <<'PY'
import os, socket, sys, time
port, pid = map(int, sys.argv[1:])
for _ in range(180):
    os.kill(pid, 0)
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            break
    except OSError:
        time.sleep(2)
else:
    raise TimeoutError('Dedicated policy server did not listen within 360 seconds')
PY
COMMON=(--collection-root "$DATA" --policy-config "$CONFIG" --policy-url "ws://127.0.0.1:$PORT"
        --batch-size 4 --seed 42 --critic-checkpoint "$CRITIC" --encoder-weights "$ENCODER" --resume)
printf 'smoke\n' > "$OUT/orchestration_state.txt"
"$PY" -u "$ROOT/scripts/debug_candidate_progress.py" "${COMMON[@]}" \
    --output-dir "$OUT/smoke" --episode-limit 1 --frames-per-episode 3
printf 'smoke_resume\n' > "$OUT/orchestration_state.txt"
"$PY" -u "$ROOT/scripts/debug_candidate_progress.py" "${COMMON[@]}" \
    --output-dir "$OUT/smoke" --episode-limit 1 --frames-per-episode 3 --report-only
printf 'sampling_full\n' > "$OUT/orchestration_state.txt"
"$PY" -u "$ROOT/scripts/debug_candidate_progress.py" "${COMMON[@]}" --output-dir "$OUT/full" &
worker_pid=$!
printf '%s\n' "$worker_pid" > "$OUT/worker.pid"
wait "$worker_pid"
printf 'complete\n' > "$OUT/orchestration_state.txt"
