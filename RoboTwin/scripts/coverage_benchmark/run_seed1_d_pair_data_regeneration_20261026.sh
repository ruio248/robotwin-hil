#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh, non-overlapping same-observation expert/policy action-chunk pairs.
# The established seed0 cohort is read-only and is never overwritten.
ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
CONTROL="$ROOT/control/d_pair_seed1_regeneration_20261026"
PAIR_ROOT="$ROOT/outputs/d_action_pairs_screen_20261026_seed1_n30"
SCRIPT="$ROOT/control/goal_d_pair_pipeline_20261026/collect_matched_pairs_v2.py"
CLIENT_PYTHON=/hdd/miniconda3/envs/robotwin_hil/bin/python
SERVER_PYTHON="$OPENPI/.venv/bin/python"
GPU=0
PORT=18489
CANDIDATE_SCENES=40
TARGET_EPISODES=30
SCENE_SEED=1
TASKS=(
  beat_block_hammer
  click_alarmclock
  click_bell
  move_playingcard_away
  press_stapler
  place_empty_cup
  place_object_stand
  place_phone_stand
  open_microwave
  lift_pot
)

[[ "$(findmnt -n -o SOURCE --target "$ROOT")" == /dev/sda1 ]]
[[ -x "$CLIENT_PYTHON" && -x "$SERVER_PYTHON" && -f "$SCRIPT" ]]
[[ ! -e "$PAIR_ROOT" ]]
[[ ! -e "$CONTROL" ]]

mkdir -p "$CONTROL"
STATE="$CONTROL/state.json"
LOG="$CONTROL/orchestrator.log"
TASK_LOG_DIR="$CONTROL/tasks"
mkdir -p "$TASK_LOG_DIR"

write_state() {
  "$CLIENT_PYTHON" - "$STATE" "$1" "${2:-}" "${3:-}" "${4:-0}" <<'PY'
import json, os, sys
from pathlib import Path
p, status, task, phase, completed = sys.argv[1:]
payload = {
    "status": status,
    "task": task or None,
    "phase": phase or None,
    "completed_tasks": int(completed),
    "scene_seed_split": 1,
    "scene_offset": 0,
    "candidate_scenes_per_task": 40,
    "target_successful_expert_episodes": 30,
    "pair_semantics": "same stored RGB/state/prompt; expert future H10 vs policy H10 queried at that exact observation",
}
tmp = p + ".tmp"
Path(tmp).write_text(json.dumps(payload, indent=2) + "\n")
os.replace(tmp, p)
PY
}

record() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
completed=0
current_task=""
on_error() {
  rc=$?
  write_state failed "$current_task" "exit_${rc}" "$completed" || true
  record "failed task=${current_task:-none} rc=$rc completed_tasks=$completed"
  exit "$rc"
}
trap on_error ERR
write_state running "" "preflight" "$completed"

wait_for_slot() {
  local used attempt
  for attempt in $(seq 1 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if (( used <= 4096 )) && ! ss -ltnH 2>/dev/null | grep -q ":$PORT[[:space:]]"; then
      return 0
    fi
    record "waiting_for_slot gpu=$GPU memory_used_mib=$used port=$PORT attempt=$attempt"
    sleep 30
  done
  return 1
}

for task in "${TASKS[@]}"; do
  current_task="$task"
  task_out="$PAIR_ROOT/${task}_seed1_n30"
  task_log="$TASK_LOG_DIR/${task}.log"
  [[ ! -e "$task_out" ]]
  [[ -d "$ROOT/checkpoints/$task/params" ]]
  [[ -s "$ROOT/checkpoints/$task/assets/robotwin/norm_stats.json" ]]
  wait_for_slot
  write_state running "$task" "collecting_30_expert_success_pairs" "$completed"
  record "starting task=$task scene_seed=$SCENE_SEED offset=0 candidates=$CANDIDATE_SCENES target_successes=$TARGET_EPISODES gpu=$GPU port=$PORT"
  if env \
    PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH" \
    OPENPI_DATA_HOME="$ROOT/openpi_data" \
    HF_HOME="$ROOT/.hf-home" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
    "$CLIENT_PYTHON" -u "$SCRIPT" \
      --task "$task" \
      --checkpoint-root "$ROOT/checkpoints" \
      --checkpoint-id "public_pi05_${task}" \
      --output "$task_out" \
      --pace-root "$ROOT/pace" \
      --robotwin-root "$ROOT" \
      --openpi-root "$OPENPI" \
      --server-python "$SERVER_PYTHON" \
      --scene-seed "$SCENE_SEED" \
      --scene-offset 0 \
      --episodes "$CANDIDATE_SCENES" \
      --target-episodes "$TARGET_EPISODES" \
      --gpu "$GPU" \
      --port "$PORT" \
      --horizon 10 >"$task_log" 2>&1; then
    "$CLIENT_PYTHON" - "$task_out" "$task" <<'PY'
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
assert summary.get("pair_semantics") == "expert_next_50Hz_joint_samples_vs_policy_first_H_absolute_joint_actions"
seeds = summary.get("scene_seeds", [])
assert len(seeds) == 30 and len(set(seeds)) == 30
metas = sorted(root.glob("episode_*_scene_*/metadata.json"))
assert len(metas) == 30, len(metas)
for meta_path in metas:
    meta = json.loads(meta_path.read_text())
    assert meta.get("expert_success") is True and meta.get("complete_pairs", 0) > 0
    assert "exact sampled joint vector" in meta.get("pair_observation_check", "")
    with np.load(meta_path.parent / "pairs.npz", allow_pickle=False) as z:
        required = {"state", "images_head", "images_left", "images_right", "expert_chunks", "policy_chunks", "stage_id"}
        assert required.issubset(z.files), (meta_path, z.files)
        n = len(z["state"])
        assert n == meta["complete_pairs"]
        assert z["expert_chunks"].shape == (n, 10, 14)
        assert z["policy_chunks"].shape == (n, 10, 14)
        assert all(len(z[k]) == n for k in ("images_head", "images_left", "images_right", "stage_id"))
        assert all(np.isfinite(z[k]).all() for k in ("state", "expert_chunks", "policy_chunks"))
PY
  else
    rc=$?
    record "collector_failed task=$task rc=$rc log=$task_log"
    exit "$rc"
  fi
  completed=$((completed + 1))
  write_state running "$task" "task_complete" "$completed"
  record "complete task=$task completed_tasks=$completed pair_root=$task_out"
done

"$CLIENT_PYTHON" - "$PAIR_ROOT" "$CONTROL/collection_manifest.json" "${TASKS[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, output, *tasks = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
rows = []
for task in tasks:
    task_root = root / f"{task}_seed1_n30"
    summary = json.loads((task_root / "summary.json").read_text())
    file_hashes = {}
    for path in sorted(task_root.rglob("*")):
        if path.is_file() and path.name in {"pairs.npz", "metadata.json"}:
            file_hashes[str(path.relative_to(task_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append({"task": task, "status": "complete", "episodes": 30,
                 "scene_seed_split": 1, "unique_scene_seeds": summary["scene_seeds"],
                 "pair_count": summary["pairs"], "file_sha256": file_hashes})
output.write_text(json.dumps({"complete": True, "tasks": rows}, indent=2) + "\n")
PY
write_state complete "" "done" "$completed"
record "complete tasks=$completed pair_root=$PAIR_ROOT"
