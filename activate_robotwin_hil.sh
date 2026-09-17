#!/usr/bin/env bash

if [[ -z "${ROBOTWIN_HIL_ROOT:-}" ]]; then
  ROBOTWIN_HIL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
export ROBOTWIN_HIL_ROOT
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$ROBOTWIN_HIL_ROOT/RoboTwin}"
if [[ -z "${CONDA_PREFIX:-}" && -x "$ROBOTWIN_HIL_ROOT/conda/bin/python" ]]; then
  export CONDA_PREFIX="$ROBOTWIN_HIL_ROOT/conda"
fi
if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/bin" ]]; then
  export PATH="$CONDA_PREFIX/bin:$PATH"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDACXX=$CUDA_HOME/bin/nvcc
if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROBOTWIN_HIL_ROOT/.cache/pip}"
export HF_HOME="${HF_HOME:-$ROBOTWIN_HIL_ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROBOTWIN_HIL_ROOT/.cache/xdg}"
export TMPDIR="${TMPDIR:-$ROBOTWIN_HIL_ROOT/.cache/tmp}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
