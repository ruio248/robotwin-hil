#!/usr/bin/env bash
set -Eeuo pipefail

# Generate same-observation expert/policy chunk pairs and the frozen R3M
# features needed by the established D-support screening trainer.  This is a
# separate, seed-1 cohort; it never overwrites the original ten-task run.
ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
ROBOTWIN=/media/ruio/hdd/robotwin-hil-enhanced-sampling/RoboTwin
CONTROL="$ROOT/control/d_pair_support_turn_switch_dump_bin_bigbin_20261029"
PAIR_ROOT="$ROOT/outputs/d_action_pairs_support_turn_switch_dump_bin_bigbin_20261029_seed1_n30"
PIPELINE="$ROOT/control/goal_d_pair_pipeline_20261026"
COLLECTOR="$PIPELINE/collect_matched_pairs_v2.py"
CLIENT_PYTHON=/hdd/miniconda3/envs/robotwin_hil/bin/python
SERVER_PYTHON="$OPENPI/.venv/bin/python"
R3M="$ROOT/control/d_pair_seed1_regeneration_20261026/assets/r3m18/pytorch_model.bin"
GPU=0
PORT=18493
TASKS=(turn_switch dump_bin_bigbin)

[[ "$(findmnt -n -o SOURCE --target "$ROOT")" == /dev/sda1 ]]
[[ -x "$CLIENT_PYTHON" && -x "$SERVER_PYTHON" && -f "$COLLECTOR" ]]
[[ -s "$R3M" ]]
[[ ! -e "$PAIR_ROOT" ]]

mkdir -p "$CONTROL/tasks"
STATE="$CONTROL/state.json"
LOG="$CONTROL/run.log"
[[ ! -e "$STATE" ]]
exec 9>"$CONTROL/runner.lock"
flock -n 9 || { echo "another D-pair runner holds $CONTROL/runner.lock" >&2; exit 3; }

write_state() {
  "$CLIENT_PYTHON" - "$STATE" "$1" "${2:-}" "${3:-}" "${4:-0}" "${5:-}" <<'PY'
import json, os, sys
from pathlib import Path
p, status, task, phase, completed, detail = sys.argv[1:]
value = {
    "status": status, "task": task or None, "phase": phase or None,
    "completed_tasks": int(completed), "scene_seed_split": 1,
    "target_expert_episodes": 30, "candidate_scenes": 40,
    "horizon": 10, "candidates_per_observation": 4,
    "policy_checkpoint_source": "public task-specific pi0.5 checkpoint",
    "pair_semantics": "same RGB/state/prompt; expert future H10 vs policy H10",
    "pair_root": "/media/ruio/hdd/pace_pi05_eval_20261022/outputs/d_action_pairs_support_turn_switch_dump_bin_bigbin_20261029_seed1_n30",
    "detail": detail or None,
}
tmp = p + ".tmp"
Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
os.replace(tmp, p)
PY
}

record() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
completed=0
current_task=""
current_stage="preflight"
on_error() {
  rc=$?
  write_state failed "$current_task" "$current_stage" "$completed" "exit=$rc command=$BASH_COMMAND" || true
  record "failed task=${current_task:-none} stage=$current_stage rc=$rc completed_tasks=$completed"
  exit "$rc"
}
trap on_error ERR

mkdir -p "$PAIR_ROOT"
write_state running "" preflight 0
[[ "$(findmnt -n -o SOURCE --target "$ROOT")" == /dev/sda1 ]]
# The released cache is tied to the unmodified task files.  Keep this cohort
# on that exact simulation rather than silently mixing the separate settling fix.
expected_dump_sha=3bd57fe9172d59731db1ccf1bc9605dfe3ef08a0f7224edc720ea4812c214611
actual_dump_sha=$(sha256sum "$ROOT/envs/dump_bin_bigbin.py" | awk '{print $1}')
[[ "$actual_dump_sha" == "$expected_dump_sha" ]]
for task in "${TASKS[@]}"; do
  [[ -d "$ROOT/checkpoints/$task/params" ]]
  [[ -s "$ROOT/checkpoints/$task/assets/robotwin/norm_stats.json" ]]
done

wait_for_slot() {
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if (( used <= 4096 )) && ! ss -ltnH 2>/dev/null | grep -q ":$PORT[[:space:]]"; then
      return 0
    fi
    record "waiting_for_slot gpu=$GPU memory_used_mib=$used port=$PORT"
    sleep 30
  done
}

for task in "${TASKS[@]}"; do
  current_task="$task"
  task_root="$PAIR_ROOT/${task}_seed1_n30"
  task_log="$CONTROL/tasks/${task}.log"
  [[ ! -e "$task_root" ]]
  wait_for_slot
  current_stage=collect_same_observation_pairs
  write_state running "$task" "$current_stage" "$completed"
  record "collect task=$task split=1 candidates=40 target_expert_successes=30 checkpoint=public_pi05_${task}"
  env PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH" \
      OPENPI_DATA_HOME="$ROOT/openpi_data" HF_HOME="$ROOT/.hf-home" \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
    "$CLIENT_PYTHON" -u "$COLLECTOR" \
      --task "$task" --checkpoint-root "$ROOT/checkpoints" \
      --checkpoint-id "public_pi05_${task}" --output "$task_root" \
      --pace-root "$ROOT/pace" --robotwin-root "$ROOT" \
      --openpi-root "$OPENPI" --server-python "$SERVER_PYTHON" \
      --scene-seed 1 --scene-offset 0 --episodes 40 --target-episodes 30 \
      --gpu "$GPU" --port "$PORT" --horizon 10 >"$task_log" 2>&1

  current_stage=validate_pairs
  "$CLIENT_PYTHON" - "$task_root" "$task" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root, task = Path(sys.argv[1]), sys.argv[2]
state = json.loads((root / "state.json").read_text())
summary = json.loads((root / "summary.json").read_text())
assert state.get("status") == "complete" and state.get("completed_episodes") == 30, state
assert state.get("scene_seed_split") == 1 and state.get("strictly_same_observation") is True, state
assert summary.get("complete") is True and summary.get("task") == task
assert summary.get("episodes") == 30 and summary.get("expert_successes") == 30
assert len(summary.get("scene_seeds", [])) == 30
assert len(set(summary.get("scene_seeds", []))) == 30
metas = sorted(root.glob("episode_*_scene_*/metadata.json"))
assert len(metas) == 30, len(metas)
for path in metas:
    meta = json.loads(path.read_text())
    assert meta.get("expert_success") is True and meta.get("complete_pairs", 0) > 0
    with np.load(path.parent / "pairs.npz", allow_pickle=False) as z:
        n = len(z["state"])
        assert z["expert_chunks"].shape == (n, 10, 14)
        assert z["policy_chunks"].shape == (n, 10, 14)
        assert all(np.isfinite(z[k]).all() for k in ("state", "expert_chunks", "policy_chunks"))
PY

  wait_for_slot
  current_stage=annotate_expert_subplan_stages
  write_state running "$task" "$current_stage" "$completed"
  env PYTHONPATH="$ROOT/pace/src:$ROOT:$ROOT/description/utils" \
    "$CLIENT_PYTHON" -u "$PIPELINE/coverage_benchmark/annotate_pair_stages.py" \
      --task "$task" --task-root "$task_root" --pace-root "$ROOT/pace" \
      --robotwin-root "$ROOT" --collector "$COLLECTOR" \
      >"$CONTROL/tasks/${task}_stages.log" 2>&1

  wait_for_slot
  current_stage=build_k4_policy_candidates
  write_state running "$task" "$current_stage" "$completed"
  env PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH" \
      PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src:$ROOT/pace/src:$ROBOTWIN/scripts" \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
    "$CLIENT_PYTHON" -u "$PIPELINE/coverage_benchmark/augment_matched_policy_candidates.py" \
      --task "$task" --input-root "$task_root" --checkpoint-root "$ROOT/checkpoints" \
      --openpi-root "$OPENPI" --server-python "$SERVER_PYTHON" \
      --gpu "$GPU" --port "$PORT" --candidates 4 \
      >"$CONTROL/tasks/${task}_k4.log" 2>&1

  wait_for_slot
  current_stage=encode_frozen_r3m_features
  write_state running "$task" "$current_stage" "$completed"
  env PYTHONPATH="$PIPELINE:$ROBOTWIN/scripts:$ROOT/pace/src" \
    "$SERVER_PYTHON" -u "$PIPELINE/coverage_benchmark/encode_matched_pairs_r3m.py" \
      --task-root "$task_root" --asset "$R3M" --device cuda:0 --batch-size 64 \
      >"$CONTROL/tasks/${task}_r3m.log" 2>&1

  completed=$((completed + 1))
  current_stage=task_complete
  write_state running "$task" "$current_stage" "$completed"
  record "complete task=$task completed_tasks=$completed output=$task_root"
done

current_task=""
current_stage=write_manifest
"$CLIENT_PYTHON" - "$PAIR_ROOT" "$PAIR_ROOT/postprocess_manifest.json" "${TASKS[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
tasks = sys.argv[3:]
rows = []
included = {"pairs.npz", "metadata.json", "stage_labels.npz", "stage_replay.json",
            "policy_candidates_k4.npz", "policy_candidates_k4.json",
            "features_r3m18.npz", "r3m_feature_manifest.json"}
for task in tasks:
    task_root = root / f"{task}_seed1_n30"
    state = json.loads((task_root / "state.json").read_text())
    summary = json.loads((task_root / "summary.json").read_text())
    feature = json.loads((task_root / "r3m_feature_manifest.json").read_text())
    assert state.get("status") == "complete" and summary.get("complete") is True
    assert summary.get("episodes") == 30 and summary.get("expert_successes") == 30
    assert feature.get("complete") is True and feature.get("episodes") == 30
    hashes = {}
    for path in sorted(task_root.rglob("*")):
        if path.is_file() and path.name in included:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[str(path.relative_to(task_root))] = digest
    rows.append({"task": task, "status": "complete", "episodes": 30,
                 "scene_seed_split": 1, "files_sha256": hashes})
manifest = {"complete": True, "protocol": "same_observation_expert_policy_H10_seed1_v1",
            "tasks": rows}
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

write_state complete "" done "$completed"
record "complete tasks=$completed pair_root=$PAIR_ROOT manifest=$PAIR_ROOT/postprocess_manifest.json"
