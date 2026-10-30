#!/usr/bin/env bash
set -Eeuo pipefail

# Paired headless evaluation for the four 100-step LoRA ablations.  The policy
# serving config, task/prompt, seeds, timing and evaluator are fixed across
# groups; only checkpoint and its matching norm_stats asset change.

REPO_ROOT=${ROBOTWIN_HIL_ROOT:-/hdd/robotwin-hil}
ROBOTWIN_ROOT="$REPO_ROOT/RoboTwin"
XPL_ROOT="$ROBOTWIN_ROOT/XPolicyLab"
POLICY_DIR="$XPL_ROOT/policy/Pi_05_RobotTwin"
POLICY_PY=${POLICY_PY:-"$POLICY_DIR/openpi/.venv/bin/python"}
EVAL_PY=${EVAL_PY:-/hdd/miniconda3/envs/robotwin_hil/bin/python}
DEPLOY_CONFIG=${DEPLOY_CONFIG:-"$XPL_ROOT/pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$REPO_ROOT/outputs/ablation_4way_20260923_eval/checkpoints/pi05_robotwin_handover_to_tray_v2_promptfix"}
RUN_ROOT=${RUN_ROOT:-"$REPO_ROOT/outputs/ablation_4way_headless_daggercfg_$(date +%Y%m%d_%H%M%S)"}
WAIT_FOR_STATUS_FILE=${WAIT_FOR_STATUS_FILE:-}
WAIT_FOR_PID_FILE=${WAIT_FOR_PID_FILE:-}
WAIT_FOR_OUTPUT_DIR=${WAIT_FOR_OUTPUT_DIR:-}
PORT=${PORT:-18301}
SEED_START=${SEED_START:-31000}
NUM_EPISODES=${NUM_EPISODES:-100}
STEP_LIMIT=${STEP_LIMIT:-900}
CONTROL_HZ=${CONTROL_HZ:-30}
RENDER_FREQ=${RENDER_FREQ:-0}
SAVE_FREQ=${SAVE_FREQ:-15}
TASK_CONFIG_NAME=handover_to_tray_v2_promptfix
ASSET_ORIGINAL=baseline_baseline_normstats_allframes_v1
ASSET_CORRECTED=sft_policy_hil_equal3_episode_safe_20260923

LEFT_CUROBO="$ROBOTWIN_ROOT/assets/embodiments/aloha-agilex/curobo_left.yml"
RIGHT_CUROBO="$ROBOTWIN_ROOT/assets/embodiments/aloha-agilex/curobo_right.yml"
WAIT_HELPER="$XPL_ROOT/utils/wait_for_policy_server.sh"
SERVER_PID=
CONFIGS_PATCHED=0
CURRENT_GROUP=setup
STATUS_FILE="$RUN_ROOT/status.txt"
CONFIG_BACKUP_DIR="$RUN_ROOT/config_backups"
ABLATION_GROUPS=(
  A_sft_hil_original
  B_sft_hil_corrected
  C_sft_policy_hil_original
  D_sft_policy_hil_corrected
)

fail() {
  echo "[ERROR] $*" >&2
  return 1
}

write_status() {
  printf '%s %s\n' "$(date -Is)" "$*" > "$STATUS_FILE"
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
    cp -p "$CONFIG_BACKUP_DIR/curobo_left.yml.original" "$LEFT_CUROBO"
    cp -p "$CONFIG_BACKUP_DIR/curobo_right.yml.original" "$RIGHT_CUROBO"
    CONFIGS_PATCHED=0
    echo "Restored original cuRobo config files."
  fi
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  cleanup_server
  restore_configs
  if (( rc == 0 )); then
    write_status "COMPLETE finished=$(date -Is)"
  else
    write_status "FAILED exit_code=$rc group=${CURRENT_GROUP:-unknown} time=$(date -Is)"
  fi
  exit "$rc"
}

wait_for_prior_eval() {
  [[ -n "$WAIT_FOR_STATUS_FILE" ]] || return 0
  echo "Waiting for prior evaluation to finish: $WAIT_FOR_STATUS_FILE"
  while true; do
    local prior_state=
    if [[ -s "$WAIT_FOR_STATUS_FILE" ]]; then
      prior_state=$(head -n 1 "$WAIT_FOR_STATUS_FILE")
      [[ "$prior_state" != *FAILED* ]] || {
        fail "Prior evaluation failed; refusing to start paired groups: $prior_state"
        return 1
      }
    fi
    local prior_running=0
    if [[ -n "$WAIT_FOR_PID_FILE" && -s "$WAIT_FOR_PID_FILE" ]]; then
      local prior_pid
      prior_pid=$(cat "$WAIT_FOR_PID_FILE")
      if [[ "$prior_pid" =~ ^[0-9]+$ && -r "/proc/$prior_pid/stat" ]]; then
        local prior_stat
        prior_stat=$(ps -o stat= -p "$prior_pid" 2>/dev/null || true)
        [[ "$prior_stat" == *Z* ]] || prior_running=1
      fi
    fi
    local port_in_use=0
    ss -ltn 2>/dev/null | grep -qE ":${PORT}[[:space:]]" && port_in_use=1 || true
    if (( prior_running == 0 && port_in_use == 0 )); then
      if [[ "$prior_state" == *COMPLETE* ]]; then
        echo "Prior evaluation completed: $prior_state"
        return 0
      fi
      if [[ -n "$WAIT_FOR_OUTPUT_DIR" ]]; then
        local records="$WAIT_FOR_OUTPUT_DIR/recorded/eval_records.jsonl"
        local record_count=0
        [[ -f "$records" ]] && record_count=$(wc -l < "$records")
        if (( record_count == NUM_EPISODES )) && compgen -G "$WAIT_FOR_OUTPUT_DIR/recorded/summary_*.json" >/dev/null; then
          echo "Prior evaluation finished with $record_count recorded episodes; port $PORT is free."
          return 0
        fi
        fail "Prior evaluation exited before a complete result (records=$record_count; expected=$NUM_EPISODES; state=${prior_state:-missing})."
        return 1
      fi
      if [[ -n "$WAIT_FOR_PID_FILE" ]]; then
        fail "Prior evaluation PID exited and port $PORT is free, but there is no output directory to verify completion."
        return 1
      fi
    fi
    sleep 20
  done
}

asset_for_group() {
  case "$1" in
    A_sft_hil_original|C_sft_policy_hil_original) printf '%s' "$ASSET_ORIGINAL" ;;
    B_sft_hil_corrected|D_sft_policy_hil_corrected) printf '%s' "$ASSET_CORRECTED" ;;
    *) fail "Unknown ablation group: $1" ;;
  esac
}

record_group_config() {
  local group="$1" checkpoint="$2" asset_id="$3" group_dir="$4"
  local data_factor stats_factor
  case "$group" in
    A_sft_hil_original)
      data_factor='SFT 50% + HIL 50%; no policy rollout'
      stats_factor='original SFT stats'
      ;;
    B_sft_hil_corrected)
      data_factor='SFT 50% + HIL 50%; no policy rollout'
      stats_factor='episode-safe equal-three-source stats shared across B/D'
      ;;
    C_sft_policy_hil_original)
      data_factor='SFT 50% + concat(policy rollout,HIL) 50%'
      stats_factor='original SFT stats'
      ;;
    D_sft_policy_hil_corrected)
      data_factor='SFT 50% + concat(policy rollout,HIL) 50%'
      stats_factor='episode-safe equal-three-source stats shared across B/D'
      ;;
  esac

  cat > "$group_dir/run_config.txt" <<EOF
group=$group
checkpoint=$checkpoint
asset_repo_id=$asset_id
data_factor=$data_factor
norm_stats_factor=$stats_factor
norm_stats_sha256=$(sha256sum "$checkpoint/assets/$asset_id/norm_stats.json" | awk '{print $1}')
seed_start=$SEED_START
episodes=$NUM_EPISODES
step_limit=$STEP_LIMIT
task_name=handover_to_tray
task_config=$TASK_CONFIG_NAME
prompt=Pass the red bar from the left arm to the right arm and place it in the blue tray.
expert_check=false
instruction_source=fixed_dagger_prompt
action_type=joint
policy_action_horizon=$ACTION_HORIZON
policy_raw_action_dim=$MODEL_ACTION_DIM
robotwin_joint_action_dim=14
frequency_hz=$CONTROL_HZ
render_freq=$RENDER_FREQ
save_freq=$SAVE_FREQ
save_videos=failure
policy_deploy_config=$DEPLOY_CONFIG
policy_deploy_config_sha256=$(sha256sum "$DEPLOY_CONFIG" | awk '{print $1}')
task_config_sha256=$(sha256sum "$ROBOTWIN_ROOT/env_cfg/task_config/$TASK_CONFIG_NAME.yml" | awk '{print $1}')
evaluator_sha256=$(sha256sum "$ROBOTWIN_ROOT/scripts/policy_eval_record.py" | awk '{print $1}')
adapter_sha256=$(sha256sum "$ROBOTWIN_ROOT/scripts/eval_policy_xpolicylab.py" | awk '{print $1}')
policy_server_port=$PORT
EOF
  find "$checkpoint/params" -type f -print0 | sort -z | xargs -0 -r sha256sum \
    > "$group_dir/checkpoint_params.sha256"
}

run_group() {
  local group="$1"
  local checkpoint="$CHECKPOINT_ROOT/$group/99"
  local asset_id group_dir eval_rc
  asset_id=$(asset_for_group "$group")
  group_dir="$RUN_ROOT/$group"
  mkdir -p "$group_dir"
  mkdir -p "$group_dir/recorded"
  CURRENT_GROUP="$group"

  [[ -d "$checkpoint/params" ]] || fail "Checkpoint params missing: $checkpoint"
  [[ -s "$checkpoint/assets/$asset_id/norm_stats.json" ]] || \
    fail "Checkpoint-matched norm_stats missing: $checkpoint/assets/$asset_id/norm_stats.json"
  ln -s "$ROBOTWIN_ROOT/assets" "$group_dir/assets"
  record_group_config "$group" "$checkpoint" "$asset_id" "$group_dir"

  if ss -ltn 2>/dev/null | grep -qE ":${PORT}[[:space:]]"; then
    fail "Port $PORT is occupied before $group; refusing to connect to an unknown service."
  fi

  write_status "RUNNING group=$group seed_start=$SEED_START episodes=$NUM_EPISODES started=$(date -Is)"
  echo "[$(date -Is)] Starting $group; checkpoint=$checkpoint asset=$asset_id"
  setsid env \
    PATH="$(dirname "$EVAL_PY"):$PATH" \
    PYTHONPATH="$ROBOTWIN_ROOT:$XPL_ROOT:$POLICY_DIR/openpi/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    "$POLICY_PY" -u "$XPL_ROOT/setup_policy_server.py" \
      --config_path "$DEPLOY_CONFIG" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --overrides "model_path=$checkpoint" "repo_id=$asset_id" \
    > "$group_dir/policy_server.log" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" > "$group_dir/policy_server.pid"
  if ! bash "$WAIT_HELPER" 127.0.0.1 "$PORT" "$SERVER_PID" "$group policy server" 1200; then
    tail -100 "$group_dir/policy_server.log" >&2 || true
    fail "Policy server failed for $group."
  fi

  set +e
  (
    cd "$ROBOTWIN_ROOT"
    PYTHONPATH="$ROBOTWIN_ROOT:$XPL_ROOT:$POLICY_DIR/openpi/src" \
      PYTHONUNBUFFERED=1 \
      PYTHONWARNINGS=ignore::UserWarning \
      CUDA_VISIBLE_DEVICES=0 \
      MUJOCO_GL=egl \
      "$EVAL_PY" "$ROBOTWIN_ROOT/scripts/policy_eval_record.py" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --policy-name Pi_05_RobotTwin \
        --ckpt-name "$checkpoint" \
        --task-config "$TASK_CONFIG_NAME" \
        --seed-start "$SEED_START" \
        --episodes "$NUM_EPISODES" \
        --frequency "$CONTROL_HZ" \
        --render-freq "$RENDER_FREQ" \
        --save-freq "$SAVE_FREQ" \
        --step-limit "$STEP_LIMIT" \
        --save-videos failure \
        --output-dir "$group_dir/recorded"
  ) > "$group_dir/eval.log" 2>&1
  eval_rc=$?
  set -e
  cleanup_server
  if (( eval_rc != 0 )); then
    tail -120 "$group_dir/eval.log" >&2 || true
    fail "Evaluation failed for $group (exit=$eval_rc); see $group_dir/eval.log."
  fi

  "$EVAL_PY" - "$group_dir/recorded/eval_records.jsonl" \
    "$group_dir/summary.txt" "$SEED_START" "$NUM_EPISODES" <<'PY'
import json
import sys
from pathlib import Path

records_path, summary_path = Path(sys.argv[1]), Path(sys.argv[2])
seed_start, expected = int(sys.argv[3]), int(sys.argv[4])
records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
expected_seeds = list(range(seed_start, seed_start + expected))
actual_seeds = [int(row["seed"]) for row in records]
if actual_seeds != expected_seeds:
    raise SystemExit(f"expected exact seeds {expected_seeds[0]}..{expected_seeds[-1]}, got {len(actual_seeds)} records: {actual_seeds[:5]}..{actual_seeds[-5:]}")
successes = sum(bool(row["success"]) for row in records)
summary_path.write_text(
    f"episodes={len(records)}\nsuccesses={successes}\nfailures={len(records)-successes}\nsuccess_rate={successes/len(records):.4f}\nseed_start={seed_start}\nseed_end={seed_start+expected-1}\n",
    encoding="utf-8",
)
print(summary_path.read_text(), end="")
PY
  write_status "GROUP_COMPLETE group=$group result=$group_dir/recorded/summary_*.json"
  echo "[$(date -Is)] Completed $group: $(tr '\n' ' ' < "$group_dir/summary.txt")"
}

[[ ! -e "$RUN_ROOT" ]] || fail "Refusing to overwrite existing output directory: $RUN_ROOT"
[[ "$NUM_EPISODES" == 100 ]] || fail "This formal paired runner requires NUM_EPISODES=100."
[[ "$RENDER_FREQ" == 0 ]] || fail "This runner is headless and requires RENDER_FREQ=0."
[[ -x "$POLICY_PY" && -x "$EVAL_PY" ]] || fail "Policy/evaluator Python interpreter missing."
[[ -s "$DEPLOY_CONFIG" ]] || fail "Dedicated policy deploy config missing: $DEPLOY_CONFIG"
[[ -s "$WAIT_HELPER" ]] || fail "Policy server readiness helper missing: $WAIT_HELPER"
[[ -f "$ROBOTWIN_ROOT/scripts/policy_eval_record.py" ]] || fail "policy_eval_record.py missing."
grep -qx 'train_config_name: pi05_robotwin_handover_to_tray_v2_promptfix' "$DEPLOY_CONFIG" || \
  fail "Refusing a generic deploy config; expected the v2_promptfix 10-step config."
grep -qx 'action_type: joint' "$DEPLOY_CONFIG" || fail "Deploy config must use joint actions."

read -r ACTION_HORIZON MODEL_ACTION_DIM < <(
  cd "$POLICY_DIR/openpi"
  PYTHONPATH=src "$POLICY_PY" -c \
    'from openpi.training import config; c=config.get_config("pi05_robotwin_handover_to_tray_v2_promptfix"); print(c.model.action_horizon, c.model.action_dim)'
)
[[ "$ACTION_HORIZON" == 10 ]] || fail "Expected policy action horizon 10, got $ACTION_HORIZON."

mkdir -p "$RUN_ROOT" "$CONFIG_BACKUP_DIR"
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_status "WAITING started=$(date -Is) run_root=$RUN_ROOT"

wait_for_prior_eval

for group in "${ABLATION_GROUPS[@]}"; do
  checkpoint="$CHECKPOINT_ROOT/$group/99"
  asset_id=$(asset_for_group "$group")
  [[ -d "$checkpoint/params" ]] || fail "Checkpoint params missing: $checkpoint"
  [[ -s "$checkpoint/assets/$asset_id/norm_stats.json" ]] || \
    fail "Checkpoint-matched norm_stats missing: $checkpoint/assets/$asset_id/norm_stats.json"
done

for required in \
  "$ROBOTWIN_ROOT/env_cfg/task_config/$TASK_CONFIG_NAME.yml" \
  "$LEFT_CUROBO" "$RIGHT_CUROBO"; do
  [[ -s "$required" ]] || fail "Required evaluation file missing: $required"
done
if ss -ltn 2>/dev/null | grep -qE ":${PORT}[[:space:]]"; then
  fail "Port $PORT is still occupied after the prior evaluation completed."
fi

python3 - "$RUN_ROOT/seed_manifest.json" "$SEED_START" "$NUM_EPISODES" <<'PY'
import json
import sys
from pathlib import Path

path, start, count = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
path.write_text(
    json.dumps({"test_seeds": list(range(start, start + count))}, indent=2) + "\n",
    encoding="utf-8",
)
PY
sha256sum "$RUN_ROOT/seed_manifest.json" > "$RUN_ROOT/seed_manifest.sha256"
printf 'repo_commit=%s\nrun_root=%s\nseed_manifest_sha256=%s\nprior_eval_status=%s\nprior_eval_pid_file=%s\nprior_eval_output_dir=%s\npolicy_action_horizon=%s\npolicy_raw_action_dim=%s\n' \
  "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)" \
  "$RUN_ROOT" \
  "$(awk '{print $1}' "$RUN_ROOT/seed_manifest.sha256")" \
  "${WAIT_FOR_STATUS_FILE:-none}" \
  "${WAIT_FOR_PID_FILE:-none}" \
  "${WAIT_FOR_OUTPUT_DIR:-none}" \
  "$ACTION_HORIZON" "$MODEL_ACTION_DIM" > "$RUN_ROOT/run_config.txt"

# This Ubuntu checkout was relocated from /media/ruio/hdd to /hdd. Temporarily
# rewrite only the two known cuRobo prefixes and restore exact backups on exit.
cp -p "$LEFT_CUROBO" "$CONFIG_BACKUP_DIR/curobo_left.yml.original"
cp -p "$RIGHT_CUROBO" "$CONFIG_BACKUP_DIR/curobo_right.yml.original"
CONFIGS_PATCHED=1
for config_file in "$LEFT_CUROBO" "$RIGHT_CUROBO"; do
  grep -q '/media/ruio/hdd/robotwin-hil' "$config_file" || \
    fail "Expected relocated cuRobo path not found in $config_file; refusing to patch."
  sed 's#/media/ruio/hdd/robotwin-hil#/hdd/robotwin-hil#g' \
    "$config_file" > "$config_file.tmp.$$"
  chmod --reference="$config_file" "$config_file.tmp.$$"
  mv "$config_file.tmp.$$" "$config_file"
done

export PATH="$(dirname "$EVAL_PY"):$PATH"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl

write_status "RUNNING all_groups=${ABLATION_GROUPS[*]} seed_start=$SEED_START episodes=$NUM_EPISODES action_horizon=$ACTION_HORIZON frequency=$CONTROL_HZ started=$(date -Is)"
for group in "${ABLATION_GROUPS[@]}"; do
  run_group "$group"
done

"$EVAL_PY" - "$RUN_ROOT" "${ABLATION_GROUPS[@]}" <<'PY'
import itertools
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
groups = sys.argv[2:]
results = {}
for group in groups:
    path = root / group / "recorded" / "eval_records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    results[group] = {int(row["seed"]): bool(row["success"]) for row in rows}
seed_sets = {tuple(sorted(values)) for values in results.values()}
if len(seed_sets) != 1:
    raise SystemExit("four groups do not share the exact same seed set")
paired = {}
for left, right in itertools.combinations(groups, 2):
    seeds = results[left]
    left_wins = sum(seeds[s] and not results[right][s] for s in seeds)
    right_wins = sum(results[right][s] and not seeds[s] for s in seeds)
    paired[f"{left}_vs_{right}"] = {
        "left_only_success": left_wins,
        "right_only_success": right_wins,
        "both_same": len(seeds) - left_wins - right_wins,
    }
summary = {
    "seed_start": min(next(iter(values)) for values in results.values()),
    "seed_count": len(next(iter(results.values()))),
    "groups": {
        name: {
            "successes": sum(values.values()),
            "episodes": len(values),
            "success_rate": sum(values.values()) / len(values),
        }
        for name, values in results.items()
    },
    "paired_outcomes": paired,
}
(root / "paired_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

write_status "COMPLETE finished=$(date -Is) summary=$RUN_ROOT/paired_summary.json"
echo "All four paired evaluations completed: $RUN_ROOT/paired_summary.json"
