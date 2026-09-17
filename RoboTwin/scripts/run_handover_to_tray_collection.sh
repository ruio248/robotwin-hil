#!/usr/bin/env bash
# Generate the scripted-expert handover-to-tray dataset only after smoke passes.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname "$ROOT")"
ACTIVATE="$REPO_ROOT/activate_robotwin_hil.sh"
SMOKE_REPORT=${SMOKE_REPORT:-$REPO_ROOT/outputs/logs/handover_to_tray_smoke_50_v1_left.json}
EVAL_SEED_MANIFEST=${EVAL_SEED_MANIFEST:-$REPO_ROOT/outputs/manifests/handover_to_tray_v1_eval_seeds.json}
DATA_ROOT="$ROOT/data/handover_to_tray_v1/handover_to_tray/aloha_agilex"

source "$ACTIVATE"
cd "$ROOT"

python -c "import json, sys; r=json.load(open('$SMOKE_REPORT')); print('smoke', r['successes'], '/', r['episodes'], 'rate=', r['success_rate']); sys.exit(0 if r.get('passed') else 2)"

# Select and persist disjoint scripted-expert-valid development and final test
# seeds before collecting any demonstrations. These seeds are never part of the
# 500 successful planner trajectories generated below.
python scripts/select_handover_to_tray_eval_seeds.py \
  --output "$EVAL_SEED_MANIFEST"

# The native collector first obtains 500 successful planner trajectories, then
# replays exactly those seeds at 15 Hz into HDF5. save_video=false in the task
# config means RGB stays in HDF5 and no bulk MP4 files are produced.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python scripts/collect_data.py handover_to_tray handover_to_tray_v1

python scripts/validate_handover_to_tray_dataset.py \
  --dataset-root "$DATA_ROOT" \
  --report "$DATA_ROOT/validation_report.json"

python - "$DATA_ROOT" "$EVAL_SEED_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
eval_manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
episodes = list(range(500))
manifest = {
    "dataset": "ruio248/robotwin_handover_to_tray_v1",
    "train_episodes": episodes[:450],
    "validation_episodes": episodes[450:],
    "development_seeds": eval_manifest["development_seeds"],
    "test_seeds": eval_manifest["test_seeds"],
    "notes": (
        "Development and test seeds are disjoint from training candidates and "
        "have been screened for scripted-planner executability."
    ),
}
(root / "split_manifest_v1.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
print(root / "split_manifest_v1.json")
PY

# Convert only after the native HDF5 validator and the fixed split manifest pass.
bash scripts/convert_handover_to_tray_lerobot.sh
