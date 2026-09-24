#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=${1:?usage: run_paired_seed_evaluation.sh RUN_ROOT}
REPO=${ROBOTWIN_HIL_ROOT:?set ROBOTWIN_HIL_ROOT to the RoboTwin-HIL checkout}
ROOT="$REPO/RoboTwin"
XPL_ROOT="$ROOT/XPolicyLab"
POLICY_DIR="$XPL_ROOT/policy/Pi_05_RobotTwin"
WAIT_HELPER="$XPL_ROOT/utils/wait_for_policy_server.sh"
EVAL_PY=${EVAL_PY:?set EVAL_PY to the robotwin_hil Python interpreter}
POLICY_PY="$POLICY_DIR/openpi/.venv/bin/python"
DEFAULT_ASSET_REPO_ID=${ASSET_REPO_ID:-}
BASELINE_ASSET_REPO_ID=${BASELINE_ASSET_REPO_ID:-$DEFAULT_ASSET_REPO_ID}
A_ASSET_REPO_ID=${A_ASSET_REPO_ID:-$DEFAULT_ASSET_REPO_ID}
C_ASSET_REPO_ID=${C_ASSET_REPO_ID:-$DEFAULT_ASSET_REPO_ID}
SOURCE_MANIFEST=${SOURCE_MANIFEST:?set SOURCE_MANIFEST to the fixed seed manifest}
BASELINE_CKPT=${BASELINE_CKPT:?set BASELINE_CKPT}
A_CKPT=${A_CKPT:?set A_CKPT}
C_CKPT=${C_CKPT:?set C_CKPT}
MANIFEST="$RUN_ROOT/seed_manifest.json"
PORT=${PORT:-18301}
RUN_ID=${RUN_ROOT##*/}
TASK_CONFIG_NAME="handover_to_tray_v2_promptfix_paired_${RUN_ID}"
TASK_CONFIG="$ROOT/env_cfg/task_config/$TASK_CONFIG_NAME.yml"
BASE_TASK_CONFIG="$ROOT/env_cfg/task_config/handover_to_tray_v2_promptfix.yml"
SERVER_PID=
CURRENT_GROUP=setup
CONFIGS_PATCHED=0
TASK_CONFIG_CREATED=0
CONFIG_BACKUP_DIR="$RUN_ROOT/config_backups"
CUROBO_LEFT="$ROOT/assets/embodiments/aloha-agilex/curobo_left.yml"
CUROBO_RIGHT="$ROOT/assets/embodiments/aloha-agilex/curobo_right.yml"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/orchestrator.log") 2>&1

write_status() {
  printf '%s %s\n' "$(date -Is)" "$*" > "$RUN_ROOT/status.txt"
}

cleanup_server() {
  local pid=${SERVER_PID:-}
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  SERVER_PID=
}

restore_configs() {
  if (( CONFIGS_PATCHED )); then
    cp -p "$CONFIG_BACKUP_DIR/curobo_left.yml.original" "$CUROBO_LEFT"
    cp -p "$CONFIG_BACKUP_DIR/curobo_right.yml.original" "$CUROBO_RIGHT"
    CONFIGS_PATCHED=0
    echo "Restored original cuRobo configs."
  fi
  if (( TASK_CONFIG_CREATED )); then
    rm -f -- "$TASK_CONFIG"
    TASK_CONFIG_CREATED=0
    echo "Removed temporary task config $TASK_CONFIG_NAME."
  fi
}

on_exit() {
  local rc=$?
  cleanup_server
  restore_configs
  if (( rc != 0 )); then
    write_status "FAILED exit_code=$rc group=${CURRENT_GROUP:-unknown} time=$(date -Is)"
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PATH="$(dirname "$EVAL_PY"):$PATH"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

for required in "$EVAL_PY" "$POLICY_PY" "$SOURCE_MANIFEST" "$BASE_TASK_CONFIG" "$WAIT_HELPER"; do
  [[ -e "$required" ]] || { echo "Required path missing: $required" >&2; exit 2; }
done
SEED_COUNT=$("$EVAL_PY" -c \
  'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["test_seeds"]))' \
  "$SOURCE_MANIFEST")
[[ "$SEED_COUNT" =~ ^[1-9][0-9]*$ ]] || {
  echo "Seed manifest has an invalid test_seeds count: $SEED_COUNT" >&2
  exit 2
}
for asset_id in "$BASELINE_ASSET_REPO_ID" "$A_ASSET_REPO_ID" "$C_ASSET_REPO_ID"; do
  [[ -n "$asset_id" ]] || {
    echo "Set ASSET_REPO_ID or all of BASELINE_ASSET_REPO_ID, A_ASSET_REPO_ID, and C_ASSET_REPO_ID." >&2
    exit 2
  }
done
if ss -ltn 2>/dev/null | grep -qE ":${PORT}[[:space:]]"; then
  echo "Port $PORT is already occupied; refusing to use or stop another service." >&2
  exit 2
fi
if [[ -e "$TASK_CONFIG" ]]; then
  echo "Temporary task config already exists: $TASK_CONFIG" >&2
  exit 2
fi

mkdir -p "$CONFIG_BACKUP_DIR"
cp "$SOURCE_MANIFEST" "$MANIFEST"
printf 'manifest_sha256=%s\nsource_manifest=%s\n' \
  "$(sha256sum "$MANIFEST" | awk '{print $1}')" "$SOURCE_MANIFEST" \
  > "$RUN_ROOT/manifest_info.txt"

# The Ubuntu checkout was relocated from /media/ruio/hdd to /hdd. Patch only
# these two cuRobo path prefixes during the run, then restore exact originals.
cp -p "$CUROBO_LEFT" "$CONFIG_BACKUP_DIR/curobo_left.yml.original"
cp -p "$CUROBO_RIGHT" "$CONFIG_BACKUP_DIR/curobo_right.yml.original"
CONFIGS_PATCHED=1
for config_file in "$CUROBO_LEFT" "$CUROBO_RIGHT"; do
  if ! grep -q '/media/ruio/hdd/robotwin-hil' "$config_file"; then
    echo "Expected relocated path not found in $config_file; refusing to patch." >&2
    exit 2
  fi
  sed 's#/media/ruio/hdd/robotwin-hil#/hdd/robotwin-hil#g' \
    "$config_file" > "$config_file.tmp.$$"
  chmod --reference="$config_file" "$config_file.tmp.$$"
  mv "$config_file.tmp.$$" "$config_file"
done
printf 'left_original_sha256=%s\nright_original_sha256=%s\n' \
  "$(sha256sum "$CONFIG_BACKUP_DIR/curobo_left.yml.original" | awk '{print $1}')" \
  "$(sha256sum "$CONFIG_BACKUP_DIR/curobo_right.yml.original" | awk '{print $1}')" \
  > "$CONFIG_BACKUP_DIR/original_sha256.txt"

cp "$BASE_TASK_CONFIG" "$TASK_CONFIG"
TASK_CONFIG_CREATED=1
sed -i 's/^eval_video_log: false$/eval_video_log: true/' "$TASK_CONFIG"
if ! grep -qx 'eval_video_log: true' "$TASK_CONFIG"; then
  echo "Could not enable video logging in temporary task config." >&2
  exit 2
fi

printf 'STARTED time=%s run_root=%s\n' "$(date -Is)" "$RUN_ROOT" > "$RUN_ROOT/status.txt"

run_one() {
  local name=$1
  local ckpt=$2
  local asset_id=$3
  local group_dir="$RUN_ROOT/$name"
  local rc
  local result_line

  CURRENT_GROUP=$name
  [[ -d "$ckpt/params" ]] || { echo "Checkpoint params missing: $ckpt" >&2; return 30; }
  [[ -s "$ckpt/assets/$asset_id/norm_stats.json" ]] || {
    echo "Matching norm_stats missing: $ckpt/assets/$asset_id/norm_stats.json" >&2
    return 31
  }
  mkdir -p "$group_dir"
  if [[ ! -e "$group_dir/assets" ]]; then
    ln -s "$ROOT/assets" "$group_dir/assets"
  fi
  find "$ckpt/params" -type f -print0 | sort -z | xargs -0 -r sha256sum \
    > "$group_dir/checkpoint_params.sha256"
  sha256sum "$ckpt/assets/$asset_id/norm_stats.json" \
    > "$group_dir/norm_stats.sha256"
  cat > "$group_dir/run_config.txt" <<EOF
name=$name
checkpoint=$ckpt
asset_repo_id=$asset_id
task_name=handover_to_tray
task_config=$TASK_CONFIG_NAME
evaluator_sha256=$(sha256sum "$ROOT/scripts/eval_policy_xpolicylab.py" | awk '{print $1}')
seed_manifest=$MANIFEST
seed_manifest_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')
seed_split=test
num_seeds=$SEED_COUNT
expert_check=false
instruction_source=scene_tags
instruction_type=seen
action_type=joint
frequency_hz=30
video_logging=true
video_fps=10
policy_server_port=$PORT
EOF
  write_status "RUNNING group=$name started=$(date -Is)"
  echo "[$(date -Is)] Starting $name; checkpoint=$ckpt"

  if ss -ltn 2>/dev/null | grep -qE ":${PORT}[[:space:]]"; then
    echo "Port $PORT became occupied before $name; refusing to connect to unknown service." >&2
    return 32
  fi

  setsid env \
    PATH="$PATH" \
    PYTHONPATH="$ROOT:$XPL_ROOT:$POLICY_DIR/openpi/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    "$POLICY_PY" "$XPL_ROOT/setup_policy_server.py" \
      --config_path "$POLICY_DIR/deploy.yml" \
      --overrides \
        port="$PORT" host=127.0.0.1 bench_name=RobotTwin \
        task_name=handover_to_tray ckpt_name="$ckpt" \
        env_cfg_type=aloha_agilex seed=0 policy_name=Pi_05_RobotTwin \
        action_type=joint action_dim=14 repo_id="$asset_id" \
    > "$group_dir/policy_server.log" 2>&1 &
  SERVER_PID=$!
  printf 'policy_server_pid=%s\n' "$SERVER_PID" >> "$group_dir/run_config.txt"
  if ! bash "$WAIT_HELPER" 127.0.0.1 "$PORT" "$SERVER_PID" "$name policy server" 1200; then
    echo "Policy server failed for $name; see $group_dir/policy_server.log" >&2
    tail -80 "$group_dir/policy_server.log" >&2 || true
    return 33
  fi

  local additional_info="ckpt_name=$ckpt,ckpt_setting=$ckpt,action_type=joint,evaluation_id=paired-$RUN_ID-$name,trial_id=paired-$RUN_ID-$name,action_case_id=paired-$RUN_ID-$name"
  set +e
  (
    cd "$group_dir"
    "$EVAL_PY" "$ROOT/scripts/eval_policy_xpolicylab.py" \
      --bench_name RobotTwin \
      --task_name handover_to_tray \
      --env_cfg_type aloha_agilex \
      --policy_name Pi_05_RobotTwin \
      --host 127.0.0.1 \
      --port "$PORT" \
      --protocol ws \
      --eval_batch false \
      --root_dir "$ROOT" \
      --device_id 0 \
      --seed 0 \
      --task_config "$TASK_CONFIG_NAME" \
      --instruction_type seen \
      --instruction_source scene_tags \
      --test_num "$SEED_COUNT" \
      --expert_check false \
      --frequency 30 \
      --seed_manifest "$MANIFEST" \
      --seed_split test \
      --additional_info "$additional_info"
  ) > "$group_dir/eval.log" 2>&1
  rc=$?
  set -e
  cleanup_server
  if (( rc != 0 )); then
    echo "Evaluation failed for $name (exit=$rc); last log lines:" >&2
    tail -100 "$group_dir/eval.log" >&2 || true
    return "$rc"
  fi

  result_line=$(grep -E '^Final success rate:' "$group_dir/eval.log" | tail -1 || true)
  if [[ -z "$result_line" ]] || ! grep -qE "^Final success rate: [0-9]+/$SEED_COUNT =" <<<"$result_line"; then
    echo "Missing or incomplete $SEED_COUNT-seed result for $name; see $group_dir/eval.log" >&2
    return 34
  fi
  printf '%s\n' "$result_line" | tee "$group_dir/summary.txt"
  grep -E '^Data has been saved to ' "$group_dir/eval.log" | tail -1 \
    >> "$group_dir/summary.txt" || true
  local result_path
  result_path=$(sed -n 's/^Data has been saved to \(.*\/_result.txt\)$/\1/p' \
    "$group_dir/eval.log" | tail -1)
  if [[ -z "$result_path" ]]; then
    echo "Could not resolve evaluation result path for $name." >&2
    return 35
  fi
  local result_dir="$ROOT/$(dirname "$result_path")"
  printf '%s\n' "$result_dir" > "$group_dir/eval_result_dir.txt"
  find "$result_dir" -maxdepth 1 -type f -name 'episode*.mp4' | wc -l \
    > "$group_dir/video_count.txt"
  write_status "COMPLETE group=$name finished=$(date -Is) result=$result_dir"
}

run_one baseline_sft9999 "$BASELINE_CKPT" "$BASELINE_ASSET_REPO_ID"
run_one A_sft_hil_original "$A_CKPT" "$A_ASSET_REPO_ID"
run_one C_sft_policy_hil_original "$C_CKPT" "$C_ASSET_REPO_ID"

printf 'ALL_COMPLETE time=%s\n' "$(date -Is)" > "$RUN_ROOT/status.txt"
echo "All paired evaluations completed."
find "$RUN_ROOT" -mindepth 2 -maxdepth 2 -name summary.txt -print -exec cat {} \;
