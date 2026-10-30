#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname "$ROOT")"
PYTHON="${ROBOTWIN_PYTHON:-python}"
PORT=${HG_DAGGER_PORT:-18303}
EPISODES=${1:-20}
SEED_START=${2:-40000}
OUTPUT=${3:-$ROOT/data/handover_to_tray_hg_dagger_r1/handover_to_tray/aloha_agilex}
MIN_FREE_MIB=${HG_DAGGER_MIN_FREE_MIB:-7000}

if [[ -z "${DISPLAY:-}" ]]; then
  GNOME_PID=$(pgrep -u "$(id -u)" -x gnome-shell | head -1 || true)
  if [[ -n "$GNOME_PID" ]]; then
    DISPLAY=$(tr '\0' '\n' <"/proc/$GNOME_PID/environ" | sed -n 's/^DISPLAY=//p' | head -1)
    XAUTHORITY=$(tr '\0' '\n' <"/proc/$GNOME_PID/environ" | sed -n 's/^XAUTHORITY=//p' | head -1)
    export DISPLAY XAUTHORITY
  fi
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "No desktop DISPLAY found. Run this on the 5090 desktop session." >&2
  exit 4
fi
if ! ss -ltn | grep -q ":$PORT "; then
  echo "Policy tunnel/server port $PORT is not listening on the 5090." >&2
  exit 5
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && (( FREE_MIB < MIN_FREE_MIB )); then
    echo "Only ${FREE_MIB} MiB GPU memory is free; HG-DAgger collection needs at least ${MIN_FREE_MIB} MiB." >&2
    echo "Wait for the current RoboTwin evaluation workers to finish, then rerun this script." >&2
    exit 6
  fi
fi

echo "Starting $EPISODES HG-DAgger rollouts at seed $SEED_START on DISPLAY=$DISPLAY"
echo "Press i to hand control permanently to the scripted expert; press q to stop."

cd "$ROOT"
exec "$PYTHON" -u scripts/hg_dagger_handover.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --episodes "$EPISODES" \
  --seed-start "$SEED_START" \
  --output-dir "$OUTPUT" \
  --save-data true \
  --render-freq 5 \
  --save-freq 15
