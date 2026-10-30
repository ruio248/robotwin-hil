#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROBOTWIN_PYTHON:-python}"
DATASET_ROOT=${1:-$ROOT/data/handover_to_tray_hg_dagger_r1/handover_to_tray/aloha_agilex}
EXPECTED=${2:-}

cd "$ROOT"
if [[ -n "$EXPECTED" ]]; then
  exec "$PYTHON" scripts/validate_hg_dagger_handover_dataset.py \
    --dataset-root "$DATASET_ROOT" \
    --expected-episodes "$EXPECTED"
fi
exec "$PYTHON" scripts/validate_hg_dagger_handover_dataset.py \
  --dataset-root "$DATASET_ROOT"
