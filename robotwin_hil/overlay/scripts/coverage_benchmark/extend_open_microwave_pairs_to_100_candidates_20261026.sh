#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
CONTROL="$ROOT/control/goal_d_pair_pipeline_20261026"
PAIR_ROOT="$ROOT/outputs/d_action_pairs_screen_20261026_v2"
OUTPUT="$PAIR_ROOT/open_microwave_seed0_n30"
COLLECTOR="$CONTROL/collect_matched_pairs_recoverable.py"
CLIENT_PYTHON=/hdd/miniconda3/envs/robotwin_hil/bin/python
SERVER_PYTHON="$OPENPI/.venv/bin/python"
PORT=18489
LOG="$CONTROL/open_microwave_extend_to_100_candidates.log"
RECOVERED_QUEUE="$CONTROL/screened_tasks_v2_recovered_20261026"
RESULT_ROOT="$ROOT/outputs/d_action_pairs_screen_20261026_v2/d_screening_results_recovered_20261026"
POSTPROCESS_DIR="$CONTROL/postprocess_recovered_20261026"
POSTPROCESS="$CONTROL/coverage_benchmark/run_d_pair_postprocess_recovered_20261026.sh"

mkdir -p "$CONTROL/audit"
record() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

wait_owned_process_gone() {
  local pid="$1" needle="$2" cmd
  while [[ -r "/proc/$pid/cmdline" ]]; do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    [[ "$cmd" == *"$needle"* ]] || return 0
    record "waiting_for_owned_process pid=$pid cmd=$cmd"
    sleep 20
  done
}

findmnt -n -o SOURCE /media/ruio/hdd | grep -qx /dev/sda1
wait_owned_process_gone 689828 recover_open_microwave_and_run_d_screening.sh
wait_owned_process_gone 691785 collect_matched_pairs_recoverable.py
wait_owned_process_gone 691832 serve_policy.py
wait_owned_process_gone 637845 run_d_pair_postprocess_20261026.sh

while ss -ltnH 2>/dev/null | grep -q ":$PORT[[:space:]]"; do
  record "waiting_for_policy_port port=$PORT"
  sleep 20
done
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  (( used <= 4096 )) && break
  record "waiting_for_gpu_slot gpu0_mib=$used"
  sleep 30
done

"$CLIENT_PYTHON" - "$ROOT" "$OUTPUT" "$CONTROL/audit/open_microwave_state_before_100_candidates.json" <<'PY'
import hashlib, json, os, pathlib, shutil, sys, tempfile
root, output, snapshot = map(pathlib.Path, sys.argv[1:])
state_path = output / "state.json"
state = json.loads(state_path.read_text())
if state.get("candidate_scenes") != 40 or state.get("attempted_scenes") != 40:
    raise SystemExit(f"unexpected resumable prefix: candidates={state.get('candidate_scenes')} attempted={state.get('attempted_scenes')}")
if state.get("expected_episodes") != 30 or state.get("completed_episodes", 0) >= 30:
    raise SystemExit("unexpected target or episode count; refusing to alter collection metadata")
if state.get("status") not in {"failed", "insufficient_successful_expert_episodes"}:
    raise SystemExit(f"collector is not in a resumable terminal state: {state.get('status')}")
cache_path = root / "pace/configs/expert_scenes.json"
cache_bytes = cache_path.read_bytes()
cache = json.loads(cache_bytes)
scenes = cache["tasks"]["open_microwave"]["0"]
if len(scenes) < 100:
    raise SystemExit(f"seed0 cache only has {len(scenes)} scenes, expected at least 100")
known = {int(s["scene_seed"]) for s in scenes[:40]}
recorded = {int(x["scene_seed"]) for x in state.get("episode_summaries", [])}
recorded.update(int(x["scene_seed"]) for x in state.get("rejected_scenes", []))
if recorded != known:
    raise SystemExit(f"40-scene prefix mismatch: recorded={len(recorded)} cache={len(known)}")
if snapshot.exists():
    raise SystemExit(f"audit snapshot already exists; refusing overwrite: {snapshot}")
shutil.copy2(state_path, snapshot)
state["candidate_scenes"] = 100
state["candidate_pool_extension"] = {
    "from_candidates": 40,
    "to_candidates": 100,
    "cache_file": str(cache_path),
    "cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
    "preserved_prefix_scene_count": 40,
    "reason": "the 40-candidate cap could not yield the unchanged target of 30 successful expert episodes",
}
fd, temp_name = tempfile.mkstemp(prefix="state.json.", dir=output)
try:
    with os.fdopen(fd, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    os.replace(temp_name, state_path)
finally:
    if os.path.exists(temp_name):
        os.unlink(temp_name)
print(json.dumps({"snapshot": str(snapshot), "old_state_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                  "cache_sha256": hashlib.sha256(cache_bytes).hexdigest(), "candidate_scenes": 100,
                  "preserved_prefix_scenes": 40, "previous_successes": state.get("completed_episodes")}))
PY

record "resume_open_microwave_same_seed0_prefix candidates=100 target_expert_successes=30"
PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH" \
OPENPI_DATA_HOME="$ROOT/openpi_data" \
HF_HOME="$ROOT/.hf-home" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
"$CLIENT_PYTHON" -u "$COLLECTOR" \
  --task open_microwave \
  --checkpoint-root "$ROOT/checkpoints" \
  --checkpoint-id public_pi05_open_microwave \
  --output "$OUTPUT" \
  --pace-root "$ROOT/pace" \
  --robotwin-root "$ROOT" \
  --openpi-root "$OPENPI" \
  --server-python "$SERVER_PYTHON" \
  --scene-seed 0 --scene-offset 0 --episodes 100 --target-episodes 30 \
  --gpu 0 --port "$PORT" --horizon 10 --resume \
  >> "$CONTROL/screened_tasks_v2/open_microwave_extension_100.log" 2>&1

"$CLIENT_PYTHON" - "$OUTPUT/state.json" "$OUTPUT/summary.json" <<'PY'
import json,sys
state=json.load(open(sys.argv[1])); summary=json.load(open(sys.argv[2]))
assert state.get("candidate_scenes")==100 and state.get("status")=="complete"
assert summary.get("complete") is True and summary.get("episodes")==30
seeds=summary.get("scene_seeds",[])
assert len(seeds)==30 and len(set(seeds))==30
assert summary.get("expert_successes")==30
print(json.dumps({"verified":True,"episodes":30,"candidate_scenes":100,
                  "attempted_scenes":summary.get("attempted_scenes"),"pairs":summary.get("pairs"),
                  "rejected":len(summary.get("rejected_scenes",[])),"scene_seeds":seeds}))
PY
record "open_microwave_complete_30_from_candidate_pool_100"

"$CLIENT_PYTHON" - "$ROOT" "$PAIR_ROOT" "$RECOVERED_QUEUE" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
root, pair_root, queue = map(pathlib.Path, sys.argv[1:])
tasks = ["beat_block_hammer", "click_alarmclock", "click_bell", "move_playingcard_away",
         "press_stapler", "place_empty_cup", "place_object_stand", "place_phone_stand",
         "open_microwave", "lift_pot"]
if queue.exists():
    raise SystemExit(f"refusing to overwrite recovered queue: {queue}")
rows = []
for task in tasks:
    folder = pair_root / f"{task}_seed0_n30"
    state = json.loads((folder / "state.json").read_text())
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text())
    seeds = summary.get("scene_seeds", [])
    if state.get("status") != "complete" or state.get("completed_episodes") != 30:
        raise SystemExit(f"pair state incomplete for {task}")
    if summary.get("complete") is not True or summary.get("episodes") != 30:
        raise SystemExit(f"pair summary incomplete for {task}")
    if len(seeds) != 30 or len(set(seeds)) != 30 or summary.get("expert_successes") != 30:
        raise SystemExit(f"pair summary has invalid positive expert cohort for {task}")
    if state.get("strictly_same_observation") is not True or state.get("policy_visited_states_included") is not False:
        raise SystemExit(f"pair semantics mismatch for {task}")
    if summary.get("pairs", 0) <= 0:
        raise SystemExit(f"no action pairs for {task}")
    rows.append({"task": task, "status": "complete", "output": str(folder),
                 "episodes": 30, "candidate_scenes": summary.get("candidate_scenes"),
                 "pairs": summary["pairs"], "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest()})
old_queue = queue.parent / "screened_tasks_v2"
old_state = old_queue / "orchestrator_state.json"
state = {"status": "complete", "tasks": rows, "episode_target_per_task": 30,
         "reconciled_from_pair_summaries": True, "source_queue_state": str(old_state),
         "source_queue_state_sha256": hashlib.sha256(old_state.read_bytes()).hexdigest(),
         "completed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
queue.mkdir(parents=True)
queue_state = queue / "orchestrator_state.json"
queue_state.write_text(json.dumps(state, indent=2) + "\n")
(queue / "reconciliation.json").write_text(json.dumps({
    "validation": "all ten task pair states and summaries verified complete with 30 unique successful expert episodes each",
    "tasks": rows,
    "prior_partial_failed_queue_state": str(old_state),
    "prior_partial_failed_queue_state_sha256": state["source_queue_state_sha256"]
}, indent=2) + "\n")
print(json.dumps({"recovered_queue": str(queue), "tasks": len(rows),
                  "candidate_scenes": {r["task"]: r["candidate_scenes"] for r in rows}}))
PY

if [[ ! -f "$POSTPROCESS" ]]; then
  record "missing_recovered_postprocess_script path=$POSTPROCESS"
  exit 1
fi
record "starting_D_postprocess output_root=$RESULT_ROOT"
D_PAIR_QUEUE="$RECOVERED_QUEUE" \
D_PAIR_RESULT_ROOT="$RESULT_ROOT" \
D_PAIR_POSTPROCESS_DIR="$POSTPROCESS_DIR" \
bash "$POSTPROCESS"
record "all_task_D_screening_postprocess_complete output_root=$RESULT_ROOT"
