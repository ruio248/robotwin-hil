#!/usr/bin/env bash
set -euo pipefail

# Enter the HDD-backed RoboTwin HIL environment on the Ubuntu/4090 host.
# The host already mounts the disk at /hdd, while some existing tools use the
# historical /media/ruio/hdd path.  A user namespace gives this shell a
# private bind mount without requiring sudo or changing /etc/fstab.

if [[ "${ROBOTWIN_HIL_HDD_MOUNTED:-0}" != "1" ]]; then
  # Do not request --mount-proc: some SSH sessions allow user/mount
  # namespaces but reject the extra procfs mount.  RoboTwin does not need a
  # private procfs for this bind mount or for headless smoke tests.
  # Keep the host network namespace so a policy server on 127.0.0.1 can be
  # reached from the HIL process.  Only user + mount namespaces are needed.
  exec unshare -Ur -m env ROBOTWIN_HIL_HDD_MOUNTED=1 "$0" "$@"
fi

if ! mountpoint -q /media/ruio/hdd; then
  mount --bind /hdd /media/ruio/hdd
fi

export ROBOTWIN_HIL_ROOT=/media/ruio/hdd/robotwin-hil
export ROBOTWIN_ROOT="$ROBOTWIN_HIL_ROOT/RoboTwin"
export ROBOTWIN_OPENPI_ROOT="$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_05_RobotTwin/openpi"
export OPENPI_ROOT="$ROBOTWIN_OPENPI_ROOT"
export ROBOTWIN_CONDA_ENV="${ROBOTWIN_CONDA_ENV:-/media/ruio/hdd/miniconda3/envs/robotwin_hil}"
export VIRTUAL_ENV="$ROBOTWIN_CONDA_ENV"
export ROBOTWIN_PYTHON="$ROBOTWIN_CONDA_ENV/bin/python"
export PATH="$ROBOTWIN_CONDA_ENV/bin:/media/ruio/hdd/miniconda3/bin:$PATH"
export PYTHONPATH="$ROBOTWIN_ROOT:$ROBOTWIN_ROOT/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$ROBOTWIN_HIL_ROOT/outputs/lerobot_datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROBOTWIN_HIL_ROOT/.cache/pip}"
export HF_HOME="${HF_HOME:-$ROBOTWIN_HIL_ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROBOTWIN_HIL_ROOT/.cache/xdg}"
export TMPDIR="${TMPDIR:-$ROBOTWIN_HIL_ROOT/.cache/tmp}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROBOTWIN_HIL_ROOT/.cache/warp}"
export WARP_CACHE_ROOT="${WARP_CACHE_ROOT:-$WARP_CACHE_PATH}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$ROBOTWIN_HIL_ROOT/.cache/jax}"
export OPENPI_LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-$ROBOTWIN_HIL_ROOT/.cache/openpi}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

mkdir -p "$ROBOTWIN_HIL_ROOT"/{outputs/{logs,manifests,lerobot_datasets},.cache/{pip,huggingface,xdg,tmp,warp,jax,openpi}}
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
