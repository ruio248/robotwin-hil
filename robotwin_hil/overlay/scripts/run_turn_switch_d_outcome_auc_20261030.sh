#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/media/ruio/hdd/dump_bin_bigbin_a800_eval_20261023
ROBOTWIN="$ROOT/robotwin"
TRAIN="$ROOT"
OUT=/media/ruio/hdd/turn_switch_d_auc_own5k_20261030
SIM_PY=/hdd/miniconda3/envs/robotwin_hil/bin/python
SERVER_PY=/hdd/Openpi-PACE-eval-20261022/.venv/bin/python
SERVER_GPU=0
SIM_GPU=0
PORT=18492
EVALUATOR="$ROBOTWIN/scripts/record_turn_switch_policy_outcome_trace_20261030.py"
SCORER="$ROBOTWIN/scripts/score_turn_switch_success_failure_d_auc_20261030.py"
DROOT="$OUT/frozen_d"
R3M="$DROOT/assets/r3m18/pytorch_model.bin"

test -x "$SIM_PY" && test -x "$SERVER_PY"
test -f "$EVALUATOR" && test -f "$SCORER"
test "$(findmnt -n -o SOURCE /media/ruio/hdd)" = /dev/sda1
test -s "$OUT/checkpoint/5000/params/manifest.ocdbt"
test -s "$OUT/checkpoint/5000/assets/ruio248/robotwin_turn_switch_demo_clean50_fixedprompt_proprio_20261018/norm_stats.json"
read -r checkpoint_count checkpoint_bytes < <(find "$OUT/checkpoint/5000" -type f -printf '%s\n' | awk '{total+=$1; count++} END {printf "%d %.0f\n", count, total}')
test "$checkpoint_count" = 37 && test "$checkpoint_bytes" = 12438807635
test "$(sha256sum "$OUT/checkpoint/5000/params/manifest.ocdbt" | cut -d' ' -f1)" = 503dcd0e832ca7090725599f1d5f9bf5a1057b180fcbfc92abeeb26cc20796c0
test -s "$DROOT/metrics.json" && test -s "$DROOT/manifest.json" && test -s "$R3M"
test -s "$DROOT/r3m_feature_manifest.json"
test -s "$OUT/control/source_server.yml"
test ! -e "$OUT/control/state.json"
mkdir -p "$OUT/control" "$OUT/cache/hf" "$OUT/cache/openpi" "$OUT/cache/jax" "$OUT/tmp/xdg" "$OUT/tmp/warp" "$OUT/smoke" "$OUT/formal"
exec 9>"$OUT/control/runner.lock"
flock -n 9 || { echo "runner lock is already held" >&2; exit 3; }

write_state() {
  python3 - "$OUT/control/state.json" "$1" "${2:-}" <<'PY'
import json,os,sys,time
from pathlib import Path
p,phase,detail=sys.argv[1:]
v={"task":"turn_switch","phase":phase,"status":"complete" if phase=="complete" else "running",
   "detail":detail or None,"scene_seeds":list(range(31000,31050)),
   "protocol":"own state=True 5k; demo_clean; H10; 30Hz; step_limit=400",
   "diagnostic_only":True,"accepted_for_training":False,"updated_unix":time.time()}
t=p+".tmp";Path(t).write_text(json.dumps(v,indent=2)+"\n");os.replace(t,p)
PY
}
port_open() {
  python3 - "$PORT" <<'PY'
import socket,sys
s=socket.socket();s.settimeout(.5);r=s.connect_ex(("127.0.0.1",int(sys.argv[1])));s.close();raise SystemExit(0 if r==0 else 1)
PY
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
server_pid=""
cleanup() {
  rc=$?
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$server_pid" 2>/dev/null || true
  fi
  if (( rc != 0 )); then write_state failed "runner exited rc=$rc; inspect control logs" || true; fi
}
trap cleanup EXIT

if (( $(gpu_used "$SERVER_GPU") > 2048 || $(gpu_used "$SIM_GPU") > 2048 )); then
  echo "Refusing busy GPUs server=$SERVER_GPU:$(gpu_used "$SERVER_GPU")MiB sim=$SIM_GPU:$(gpu_used "$SIM_GPU")MiB" >&2; exit 4
fi
if port_open; then echo "Refusing occupied port $PORT" >&2; exit 5; fi
SERVER_CONFIG="$OUT/control/server.yml"
if [[ ! -e "$SERVER_CONFIG" ]]; then
  python3 - "$OUT/control/source_server.yml" "$SERVER_CONFIG" "$PORT" "$OUT/checkpoint/5000" <<'PY'
import sys
from pathlib import Path
s,d,port,checkpoint=sys.argv[1:]; lines=Path(s).read_text().splitlines()
for key,value in (("port",port),("model_path",checkpoint)):
    ids=[i for i,x in enumerate(lines) if x.startswith(f"{key}:")]
    if len(ids)!=1: raise SystemExit(f"expected exactly one {key} in source config")
    lines[ids[0]]=f"{key}: {value}"
Path(d).write_text("\n".join(lines)+"\n")
PY
fi

write_state preflight "GPU/port preflight passed"
SERVER_ENV=(env TMPDIR="$OUT/tmp" XDG_CACHE_HOME="$OUT/tmp/xdg" HF_HOME=/media/ruio/hdd/pace_pi05_eval_20261022/.hf-home OPENPI_DATA_HOME=/media/ruio/hdd/pace_pi05_eval_20261022/openpi_data JAX_COMPILATION_CACHE_DIR="$OUT/cache/jax" CUDA_VISIBLE_DEVICES="$SERVER_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION=0.60 PYTHONPATH="$TRAIN/openpi/src:$ROBOTWIN:$ROBOTWIN/XPolicyLab" WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1)
cd "$ROBOTWIN"
setsid "${SERVER_ENV[@]}" "$SERVER_PY" -u "$ROBOTWIN/XPolicyLab/setup_policy_server.py" --config_path "$SERVER_CONFIG" --host 127.0.0.1 --port "$PORT" >"$OUT/control/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" > "$OUT/control/server.pid"
for _ in $(seq 1 900); do
  kill -0 "$server_pid" 2>/dev/null || { tail -80 "$OUT/control/server.log"; exit 10; }
  port_open && break
  sleep 2
done
port_open || { echo "server readiness timeout" >&2; exit 11; }

SIM_ENV=(env TMPDIR="$OUT/tmp" XDG_CACHE_HOME="$OUT/tmp/xdg" WARP_CACHE_PATH="$OUT/tmp/warp" CUDA_VISIBLE_DEVICES="$SIM_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$ROBOTWIN" PYTHONDONTWRITEBYTECODE=1 ROBOTWIN_ROOT="$ROBOTWIN")
write_state smoke "one smoke seed 31000; excluded from formal cohort"
"${SIM_ENV[@]}" "$SIM_PY" -u "$EVALUATOR" --host 127.0.0.1 --port "$PORT" --seed-start 31000 --episodes 1 --frequency 30 --step-limit 400 --output-dir "$OUT/smoke/seed_31000" >"$OUT/control/smoke.log" 2>&1
python3 - "$OUT/smoke/seed_31000" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads((p/"evaluation/metadata.json").read_text());r=[json.loads(x) for x in (p/"evaluation/episodes.jsonl").read_text().splitlines() if x];t=json.loads((p/"evaluation/trace/episode_000/trace.json").read_text())
assert m["status"]=="complete" and len(r)==1 and t["complete"] and t["queries"],(m,r,t)
print(f"smoke ok success={r[0]['success']} steps={r[0]['steps']} queries={r[0]['queries']}")
PY

write_state formal "smoke passed; recording fresh seeds 31000-31049"
"${SIM_ENV[@]}" "$SIM_PY" -u "$EVALUATOR" --host 127.0.0.1 --port "$PORT" --seed-start 31000 --episodes 50 --frequency 30 --step-limit 400 --output-dir "$OUT/formal" >"$OUT/control/formal.log" 2>&1
python3 - "$OUT/formal" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads((p/"evaluation/metadata.json").read_text());r=[json.loads(x) for x in (p/"evaluation/episodes.jsonl").read_text().splitlines() if x]
assert m["status"]=="complete" and len(r)==50 and [x["scene_seed"] for x in r]==list(range(31000,31050))
assert all(x["queries"]>0 and x["infra_error"] is False for x in r)
print(f"formal complete: {sum(bool(x['success']) for x in r)}/50")
PY

if kill -0 "$server_pid" 2>/dev/null; then
  kill -TERM -- "-$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
  wait "$server_pid" || true
  server_pid=""
fi
write_state scoring "rollouts complete; scoring frozen 5-fold D on simulator GPU"
SCORER_ENV=(env CUDA_VISIBLE_DEVICES="$SIM_GPU" PYTHONPATH="$ROBOTWIN/scripts:$ROBOTWIN" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1)
"${SCORER_ENV[@]}" "$SERVER_PY" -u "$SCORER" --trace-root "$OUT/formal" --d-root "$DROOT" --r3m-asset "$R3M" --feature-manifest "$DROOT/r3m_feature_manifest.json" --output "$OUT/outcome_auc.json" --device cuda:0 >"$OUT/control/scoring.log" 2>&1
write_state complete "outcome AUC saved at $OUT/outcome_auc.json"
echo "COMPLETE output=$OUT"
