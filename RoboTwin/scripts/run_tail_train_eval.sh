#!/usr/bin/env bash
set -euo pipefail

# Start a critic run in the background, evaluate the first requested checkpoint,
# and leave the training process SIGSTOP'ed for later continuation.
#
# This wrapper never starts a policy server and never executes the robot.

usage() {
  cat <<'EOF'
Usage:
  run_tail_train_eval.sh \
    --python /path/to/python \
    --sft-cache /path/to/cache/sft \
    --heldout-cache /path/to/cache/heldout \
    --hil-cache /path/to/cache/hil \
    --run-dir /path/to/runs/expert_bootstrap \
    --report-dir /path/to/reports/expert_bootstrap_step5000

The wrapper starts train_tail.py with --steps 10000 by default, waits until
step 5000 has been checkpointed, runs eval_tail_value.py on that checkpoint,
and then sends SIGSTOP to the still-running training process.  Continue later
with: kill -CONT <pid>
EOF
}

PYTHON=python
SFT_CACHE=
HELDOUT_CACHE=
HIL_CACHE=
RUN_DIR=
REPORT_DIR=
TARGET_STEP=5000
TOTAL_STEPS=10000
SUITE=all
DEVICE=cuda
POLL_SECONDS=10
TIMEOUT_SECONDS=86400
EVAL_EVERY=500

while (($#)); do
  case "$1" in
    --python) PYTHON=$2; shift 2 ;;
    --sft-cache) SFT_CACHE=$2; shift 2 ;;
    --heldout-cache) HELDOUT_CACHE=$2; shift 2 ;;
    --hil-cache) HIL_CACHE=$2; shift 2 ;;
    --run-dir) RUN_DIR=$2; shift 2 ;;
    --report-dir) REPORT_DIR=$2; shift 2 ;;
    --target-step) TARGET_STEP=$2; shift 2 ;;
    --total-steps) TOTAL_STEPS=$2; shift 2 ;;
    --suite) SUITE=$2; shift 2 ;;
    --device) DEVICE=$2; shift 2 ;;
    --poll-seconds) POLL_SECONDS=$2; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS=$2; shift 2 ;;
    --eval-every) EVAL_EVERY=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in SFT_CACHE HELDOUT_CACHE HIL_CACHE RUN_DIR REPORT_DIR; do
  if [[ -z "${!value_name}" ]]; then
    echo "Missing required argument for ${value_name}" >&2
    usage >&2
    exit 2
  fi
done

if (( TARGET_STEP < EVAL_EVERY || TOTAL_STEPS <= TARGET_STEP )); then
  echo "Require TOTAL_STEPS > TARGET_STEP >= EVAL_EVERY" >&2
  exit 2
fi
if (( TARGET_STEP % EVAL_EVERY != 0 )); then
  echo "TARGET_STEP must be a multiple of EVAL_EVERY so a checkpoint exists" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to reuse non-empty run directory: $RUN_DIR" >&2
  exit 2
fi
if [[ -e "$REPORT_DIR" ]] && [[ -n "$(find "$REPORT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty report directory: $REPORT_DIR" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROBOTWIN_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
TRAIN_LOG="${RUN_DIR}.train.log"
CHECKPOINT="$RUN_DIR/step_$(printf '%07d' "$TARGET_STEP").pt"
STATE_FILE="$RUN_DIR/orchestration_state.txt"
TRAIN_PID=

write_state() {
  local status=$1
  mkdir -p "$RUN_DIR"
  {
    printf 'status=%s\n' "$status"
    printf 'pid=%s\n' "${TRAIN_PID:-}"
    printf 'checkpoint=%s\n' "$CHECKPOINT"
    printf 'report_dir=%s\n' "$REPORT_DIR"
    printf 'continue_command=kill -CONT %s\n' "${TRAIN_PID:-<pid>}"
  } > "$STATE_FILE"
}

pause_if_running() {
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -STOP "$TRAIN_PID"
  fi
}

on_interrupt() {
  pause_if_running
  write_state interrupted
  echo "Training was paused after wrapper interruption; see $STATE_FILE" >&2
  exit 130
}
trap on_interrupt INT TERM

mkdir -p "$(dirname -- "$RUN_DIR")" "$(dirname -- "$REPORT_DIR")"
echo "Starting critic training in background; log: $TRAIN_LOG"
"$PYTHON" "$ROBOTWIN_ROOT/scripts/train_tail.py" \
  --cache-dir "$SFT_CACHE" \
  --output-dir "$RUN_DIR" \
  --steps "$TOTAL_STEPS" \
  --eval-every "$EVAL_EVERY" \
  --device "$DEVICE" \
  > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "training_pid=$TRAIN_PID"

started_at=$(date +%s)
while [[ ! -f "$CHECKPOINT" ]]; do
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    wait "$TRAIN_PID" || true
    write_state training_exited_before_target
    echo "Training exited before $CHECKPOINT; inspect $TRAIN_LOG" >&2
    exit 1
  fi
  now=$(date +%s)
  if (( now - started_at >= TIMEOUT_SECONDS )); then
    pause_if_running
    write_state timeout_paused
    echo "Timed out waiting for $CHECKPOINT; training paused. See $TRAIN_LOG" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

echo "Evaluating $CHECKPOINT"
if ! "$PYTHON" "$ROBOTWIN_ROOT/scripts/eval_tail_value.py" \
  --cache-dir "$SFT_CACHE" "$HELDOUT_CACHE" "$HIL_CACHE" \
  --checkpoint "$CHECKPOINT" \
  --suite "$SUITE" \
  --output-dir "$REPORT_DIR" \
  --device "$DEVICE"; then
  pause_if_running
  write_state evaluation_failed_paused
  echo "Evaluation failed; training was paused. See $REPORT_DIR and $TRAIN_LOG" >&2
  exit 1
fi

if kill -0 "$TRAIN_PID" 2>/dev/null; then
  kill -STOP "$TRAIN_PID"
  write_state evaluated_and_paused
  echo "Training paused: pid=$TRAIN_PID"
  echo "Continue with: kill -CONT $TRAIN_PID"
else
  write_state training_completed_before_pause
  echo "Training completed before it could be paused; checkpoint and report are valid."
fi
