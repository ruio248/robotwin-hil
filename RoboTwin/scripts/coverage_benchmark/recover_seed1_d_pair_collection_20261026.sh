#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
CONTROL="$ROOT/control/d_pair_seed1_regeneration_20261026"
PAIR_ROOT="$ROOT/outputs/d_action_pairs_screen_20261026_seed1_n30"
COLLECTOR="$ROOT/control/goal_d_pair_pipeline_20261026/collect_matched_pairs_recoverable.py"
PY=/hdd/miniconda3/envs/robotwin_hil/bin/python
SERVER_PY="$OPENPI/.venv/bin/python"
LOG="$CONTROL/recovery.log"
PORT=18489

record() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
write_queue_state() {
  "$PY" - "$CONTROL/state.json" "$1" "${2:-}" "${3:-0}" <<'PY'
import json, os, sys
from pathlib import Path
p, status, task, completed = sys.argv[1:]
d = json.loads(Path(p).read_text())
d.update(status=status, task=task or None, phase="recovered_seed1_collection",
         completed_tasks=int(completed), candidate_scenes_per_task={
             "open_microwave": 100, "other_tasks": 40})
tmp = p + ".tmp"
Path(tmp).write_text(json.dumps(d, indent=2) + "\n")
os.replace(tmp, p)
PY
}

[[ "$(findmnt -n -o SOURCE --target "$ROOT")" == /dev/sda1 ]]
[[ -x "$PY" && -x "$SERVER_PY" && -f "$COLLECTOR" ]]
[[ ! -e "$CONTROL/recovery_started.marker" ]]
[[ ! -e "$CONTROL/collection_manifest.json" ]]
cd "$ROOT"

"$PY" - "$ROOT" "$CONTROL" "$PAIR_ROOT" <<'PY'
import hashlib, json, os, shutil, sys, tempfile
from pathlib import Path
root, control, pair_root = map(Path, sys.argv[1:])
tasks = ["beat_block_hammer", "click_alarmclock", "click_bell",
         "move_playingcard_away", "press_stapler", "place_empty_cup",
         "place_object_stand", "place_phone_stand"]
queue = json.loads((control / "state.json").read_text())
assert queue.get("task") == "open_microwave" and queue.get("completed_tasks") == 8
for task in tasks:
    d = pair_root / f"{task}_seed1_n30"
    state = json.loads((d / "state.json").read_text())
    summary = json.loads((d / "summary.json").read_text())
    assert state.get("status") == "complete" and state.get("scene_seed_split") == 1
    assert summary.get("complete") is True and summary.get("episodes") == 30
    assert len(set(summary.get("scene_seeds", []))) == 30
target = pair_root / "open_microwave_seed1_n30"
state_path = target / "state.json"
state = json.loads(state_path.read_text())
assert state.get("status") == "failed"
assert state.get("error") == "AssertionError: target_pose cannot be None for move action."
assert state.get("candidate_scenes") == 40 and state.get("attempted_scenes") == 2
assert state.get("completed_episodes") == 2 and state.get("scene_seed_split") == 1
assert [x["scene_seed"] for x in state.get("episode_summaries", [])] == [200000, 200001]
assert not state.get("rejected_scenes")
assert len(list(target.glob("episode_*_scene_*/metadata.json"))) == 2
cache_path = root / "pace/configs/expert_scenes.json"
cache_bytes = cache_path.read_bytes()
scenes = json.loads(cache_bytes)["tasks"]["open_microwave"]["1"]
assert len(scenes) >= 100
assert [int(s["scene_seed"]) for s in scenes[:2]] == [200000, 200001]
backup = control / "open_microwave_seed1_state_before_recovery.json"
assert not backup.exists()
shutil.copy2(state_path, backup)
state["candidate_scenes"] = 100
state["candidate_pool_extension"] = {
    "from_candidates": 40, "to_candidates": 100,
    "preserved_prefix_scene_count": 2,
    "cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
    "reason": "expert fallback grasp pose can be unreachable; retain successful prefix and seek 30 valid expert episodes",
}
fd, tmp = tempfile.mkstemp(prefix="state.json.", dir=target)
try:
    with os.fdopen(fd, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    os.replace(tmp, state_path)
finally:
    if os.path.exists(tmp): os.unlink(tmp)
print(json.dumps({"preflight": "ok", "complete_tasks": 8,
                  "preserved_open_microwave_episodes": 2,
                  "pool": 100, "old_state_sha256": hashlib.sha256(backup.read_bytes()).hexdigest()}))
PY

: > "$CONTROL/recovery_started.marker"
record "recovery_started; preserved eight complete tasks and two microwave episodes"

wait_for_slot() {
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    if (( used <= 4096 )) && ! ss -ltnH 2>/dev/null | grep -q ":$PORT[[:space:]]"; then
      return 0
    fi
    record "waiting_for_slot gpu0_mib=$used port=$PORT"
    sleep 30
  done
}

export OPENPI_DATA_HOME="$ROOT/openpi_data" HF_HOME="$ROOT/.hf-home" XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH"
for task in open_microwave lift_pot; do
  if [[ "$task" == open_microwave ]]; then
    candidates=100
    extra=(--resume)
  else
    candidates=40
    extra=()
    [[ ! -e "$PAIR_ROOT/lift_pot_seed1_n30" ]]
  fi
  wait_for_slot
  write_queue_state running "$task" 8
  record "collecting task=$task candidates=$candidates"
  if ! "$PY" -u "$COLLECTOR" \
      --task "$task" --checkpoint-root "$ROOT/checkpoints" \
      --checkpoint-id "public_pi05_${task}" \
      --output "$PAIR_ROOT/${task}_seed1_n30" \
      --pace-root "$ROOT/pace" --robotwin-root "$ROOT" \
      --openpi-root "$OPENPI" --server-python "$SERVER_PY" \
      --scene-seed 1 --scene-offset 0 --episodes "$candidates" \
      --target-episodes 30 --gpu 0 --port "$PORT" --horizon 10 \
      "${extra[@]}" >> "$CONTROL/tasks/${task}_recovery.log" 2>&1; then
    write_queue_state failed "$task" 8
    record "collector_failed task=$task"
    exit 1
  fi
  "$PY" - "$PAIR_ROOT/${task}_seed1_n30" "$task" <<'PY'
import json, sys
from pathlib import Path
p, task = Path(sys.argv[1]), sys.argv[2]
s = json.loads((p / "state.json").read_text())
d = json.loads((p / "summary.json").read_text())
assert s.get("status") == "complete" and s.get("scene_seed_split") == 1
assert d.get("complete") is True and d.get("episodes") == 30 and d.get("expert_successes") == 30
assert len(set(d.get("scene_seeds", []))) == 30
assert len(list(p.glob("episode_*_scene_*/metadata.json"))) == 30
print(json.dumps({"task": task, "episodes": 30, "attempted": d.get("attempted_scenes"),
                  "rejected": len(d.get("rejected_scenes", [])), "pairs": d.get("pairs")}))
PY
  record "task_complete task=$task"
done

"$PY" - "$PAIR_ROOT" "$CONTROL/collection_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, output = map(Path, sys.argv[1:])
tasks = ["beat_block_hammer", "click_alarmclock", "click_bell",
         "move_playingcard_away", "press_stapler", "place_empty_cup",
         "place_object_stand", "place_phone_stand", "open_microwave", "lift_pot"]
rows = []
for task in tasks:
    p = root / f"{task}_seed1_n30"
    s = json.loads((p / "state.json").read_text())
    d = json.loads((p / "summary.json").read_text())
    assert s.get("status") == "complete" and d.get("complete") is True
    assert s.get("scene_seed_split") == 1 and d.get("episodes") == 30
    hashes = {}
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.name in {"pairs.npz", "metadata.json"}:
            hashes[str(f.relative_to(p))] = hashlib.sha256(f.read_bytes()).hexdigest()
    rows.append({"task": task, "status": "complete", "episodes": 30,
                 "scene_seed_split": 1, "candidate_scenes": s["candidate_scenes"],
                 "unique_scene_seeds": d["scene_seeds"], "pair_count": d["pairs"],
                 "file_sha256": hashes})
output.write_text(json.dumps({"complete": True, "tasks": rows}, indent=2) + "\n")
PY
write_queue_state complete "" 10
record "collection_complete; postprocess watcher may begin"
