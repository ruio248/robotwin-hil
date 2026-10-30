#!/usr/bin/env bash
# Isolated benchmark entry, retaining the working HIL namespace/runtime setup.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_root="${ROBOTWIN_ROOT:-$repo_root/outputs/robotwin_hil_runtime/RoboTwin}"
export ROBOTWIN_HIL_ROOT="$repo_root"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
phase="${1:?phase: policy|offline|branch|report}"
shift
if [[ "$phase" == policy ]]; then
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
  export PYTHONPATH="$runtime_root:$runtime_root/XPolicyLab/policy/Pi_05_RobotTwin/openpi/src"
  cd "$runtime_root"
  exec "${OPENPI_PYTHON:-/hdd/robotwin-hil/RoboTwin/XPolicyLab/policy/Pi_05_RobotTwin/openpi/.venv/bin/python}" -u \
    scripts/serve_coverage_benchmark_policy.py \
    --config_path "$runtime_root/XPolicyLab/pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml" --port 18304
fi
exec bash "$repo_root/robotwin_hil/enter_ubuntu.sh" python -u scripts/run_coverage_benchmark.py "$phase" \
  --config "$runtime_root/configs/coverage_benchmark_v1.json" --profile ubuntu \
  --output-dir "$repo_root/outputs/coverage_benchmark_v1_20261009/$phase" "$@"
