#!/usr/bin/env bash
# Convert the first 450 validated expert episodes to the PI0.5 LeRobot train split.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname "$ROOT")"
ACTIVATE="$REPO_ROOT/activate_robotwin_hil.sh"
REPO_ID=ruio248/robotwin_handover_to_tray_v1
NATIVE_ROOT="$ROOT/data/handover_to_tray_v1/handover_to_tray/aloha_agilex"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/outputs/lerobot_datasets}"

source "$ACTIVATE"
cd "$ROOT"

python -c "import json, sys; r=json.load(open('$NATIVE_ROOT/validation_report.json')); print('native_valid', r['valid_episodes'], '/', r['episodes']); sys.exit(0 if r.get('ok') and r.get('episodes') == 500 else 2)"
test ! -e "$HF_LEROBOT_HOME/$REPO_ID" || {
  echo "Refusing to overwrite existing LeRobot dataset: $HF_LEROBOT_HOME/$REPO_ID"
  exit 1
}

python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "handover_to_tray_v1.handover_to_tray.aloha_agilex" \
  --repo_id "$REPO_ID" \
  --max_episode 450 \
  --resolution 240x320

python scripts/validate_robotwin_lerobot_dataset.py \
  --repo-id "$REPO_ID" \
  --expected-episodes 450
