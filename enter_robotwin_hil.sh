#!/usr/bin/env bash
set -euo pipefail

# Enter the HDD-backed RoboTwin HIL environment on the Ubuntu/4090 host.
# The host already mounts the disk at /hdd, while some existing tools use the
# historical /media/ruio/hdd path.  A user namespace gives this shell a
# private bind mount without requiring sudo or changing /etc/fstab.

if [[ "${ROBOTWIN_HIL_HDD_MOUNTED:-0}" != "1" ]]; then
  exec unshare -Urnm --mount-proc env ROBOTWIN_HIL_HDD_MOUNTED=1 "$0" "$@"
fi

if ! mountpoint -q /media/ruio/hdd; then
  mount --bind /hdd /media/ruio/hdd
fi

export ROBOTWIN_HIL_ROOT=/media/ruio/hdd/robotwin-hil
export ROBOTWIN_ROOT="$ROBOTWIN_HIL_ROOT/RoboTwin"
export VIRTUAL_ENV="$ROBOTWIN_HIL_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:/media/ruio/hdd/miniconda3/bin:$PATH"
export PYTHONPATH="$ROBOTWIN_ROOT:$ROBOTWIN_ROOT/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$ROBOTWIN_HIL_ROOT/outputs/lerobot_datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROBOTWIN_HIL_ROOT/.cache/pip}"
export HF_HOME="${HF_HOME:-$ROBOTWIN_HIL_ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROBOTWIN_HIL_ROOT/.cache/xdg}"
export TMPDIR="${TMPDIR:-$ROBOTWIN_HIL_ROOT/.cache/tmp}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

mkdir -p "$ROBOTWIN_HIL_ROOT"/{outputs/{logs,manifests,lerobot_datasets},.cache/{pip,huggingface,xdg,tmp}}
cd "$ROBOTWIN_ROOT"

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

echo "RoboTwin HIL environment ready"
echo "  root:    $ROBOTWIN_ROOT"
echo "  python:  $(command -v python)"
echo "  mount:   $(findmnt -no SOURCE,TARGET /media/ruio/hdd 2>/dev/null || echo '/hdd -> /media/ruio/hdd (user namespace)')"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unavailable)"
exec bash --noprofile --norc
