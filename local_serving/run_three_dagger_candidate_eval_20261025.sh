#!/usr/bin/env bash
# Sequentially smoke-test and formally evaluate three released RoboTwin pi0.5
# task checkpoints on the existing Ubuntu evaluation environment.
set -Eeuo pipefail

ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
OPENPI=/media/ruio/hdd/Openpi-PACE-eval-20261022
CONTROL="$ROOT/control/three_dagger_candidate_eval_20261025"
DOWNLOAD_LOG="$ROOT/download_three_dagger_candidates_20261025.log"
DOWNLOAD_PID=487360
TASKS=(pick_dual_bottles put_bottles_dustbin stack_blocks_two)
PORT=18480
GPU=0

mkdir -p "$CONTROL"
STATE="$CONTROL/state_seed_disjoint.txt"
status() { printf '%s %s\n' "$(date '+%F %T %z')" "$*" | tee -a "$STATE"; }

fail_trap() {
  local rc=$?
  if (( rc != 0 )); then
    status "failed exit=$rc line=${BASH_LINENO[0]:-unknown}"
  fi
}
trap fail_trap EXIT

[[ "$(findmnt -n -o SOURCE /media/ruio/hdd)" == /dev/sda1 ]]
[[ -d "$OPENPI/pace/src/openpi_pace" ]]
[[ -f "$ROOT/checkpoints/manifest.json" ]]
[[ -f "$ROOT/pace/configs/expert_scenes.json" ]]

status "waiting_for_verified_download pid=$DOWNLOAD_PID"
for ((i=0; i<720; i++)); do
  if grep -q 'all selected task files verified' "$DOWNLOAD_LOG"; then
    break
  fi
  if ! ps -p "$DOWNLOAD_PID" -o args= 2>/dev/null | grep -Fq 'download_from_manifest.py'; then
    status "failed download_process_exited_without_verified_marker"
    exit 20
  fi
  sleep 30
done
grep -q 'all selected task files verified' "$DOWNLOAD_LOG" || {
  status "failed download_wait_timeout"; exit 21;
}

export PATH="/hdd/miniconda3/envs/robotwin_hil/bin:$PATH"
export OPENPI_DATA_HOME="$ROOT/openpi_data"
export HF_HOME="$ROOT/.hf-home"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
cd "$ROOT"

# Re-hash every downloaded file before the first policy is launched.
status "revalidating_three_task_files"
python3 "$ROOT/download_from_manifest.py" \
  --manifest "$ROOT/checkpoints/manifest.json" --output "$ROOT/checkpoints" \
  "${TASKS[@]}" >> "$CONTROL/checkpoint_revalidation.log" 2>&1

wait_for_slot() {
  local i used free
  for ((i=0; i<120; i++)); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if (( used <= 4096 && free >= 18000 )) && ! ss -ltnH 2>/dev/null | grep -q ":$PORT[[:space:]]"; then
      return 0
    fi
    sleep 30
  done
  status "failed resource_wait_timeout gpu=$GPU"
  return 1
}

verify_run() {
  local run_root="$1" task="$2" expected="$3" seed="$4"
  python3 - "$run_root/$task/fixed_h50/seed$seed" "$task" "$expected" "$seed" <<'PY'
import json, pathlib, sys
root, task, expected, seed = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
meta = json.loads((root / "metadata.json").read_text())
assert meta["status"] == "complete", meta
assert meta["task"] == task and meta["method"] == "fixed_h50" and meta["seed"] == seed, meta
assert meta["expert_check_mode"] == "cached_scenes", meta
rows = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines() if line]
assert len(rows) == expected, (task, len(rows), expected)
seeds = [row["scene_seed"] for row in rows]
assert len(set(seeds)) == expected, (task, seeds)
successes = sum(bool(row["success"]) for row in rows)
print(f"{task}: {successes}/{expected}; scenes={seeds[0]}..{seeds[-1]}")
PY
}

run_one() {
  local stage="$1" task="$2" episodes="$3" seed="$4"
  local logfile="$CONTROL/${stage}_${task}.log"
  wait_for_slot
  status "starting stage=$stage task=$task episodes=$episodes gpu=$GPU port=$PORT"
  bash "$ROOT/pace/scripts/eval_all.sh" \
    --method fixed --horizon 50 --seed "$seed" --episodes "$episodes" \
    --gpus "$GPU" --jobs-per-gpu 1 --tasks "$task" \
    --start-port "$PORT" --ultrafast > "$logfile" 2>&1
  local run_root
  run_root=$(awk -F': ' '/^Results: / {print $2; exit}' "$logfile")
  [[ -n "$run_root" && -d "$run_root/$task/fixed_h50/seed$seed" ]]
  verify_run "$run_root" "$task" "$episodes" "$seed"
  printf '%s %s %s\n' "$stage" "$task" "$run_root" >> "$CONTROL/run_paths.txt"
  status "complete stage=$stage task=$task run_root=$run_root"
}

status "smoke_start tasks=${TASKS[*]} protocol=fixed_h50 cached_scenes_seed2; excluded_from_formal_seed0"
for task in "${TASKS[@]}"; do run_one smoke "$task" 1 2; done

status "formal_start seed0_first_50_cached_scenes protocol=fixed_h50"
for task in "${TASKS[@]}"; do run_one formal "$task" 50 0; done

python3 - "$CONTROL/run_paths.txt" "$CONTROL/aggregate_three_tasks.json" <<'PY'
import json, pathlib, sys
paths_file, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = {}
for line in paths_file.read_text().splitlines():
    stage, task, run_root = line.split(" ", 2)
    if stage != "formal":
        continue
    root = pathlib.Path(run_root) / task / "fixed_h50" / "seed0"
    meta = json.loads((root / "metadata.json").read_text())
    episodes = [json.loads(x) for x in (root / "episodes.jsonl").read_text().splitlines() if x]
    seeds = [x["scene_seed"] for x in episodes]
    rows[task] = {
        "successes": sum(bool(x["success"]) for x in episodes),
        "episodes": len(episodes),
        "success_rate": sum(bool(x["success"]) for x in episodes) / len(episodes),
        "scene_seeds": seeds,
        "unique_scene_seeds": len(set(seeds)),
        "checkpoint_config": meta["server_metadata"]["config"],
        "task_config": meta["task_config"],
        "expert_check_mode": meta["expert_check_mode"],
        "evaluation_method": meta["method"],
        "run_root": str(root),
    }
assert set(rows) == {"pick_dual_bottles", "put_bottles_dustbin", "stack_blocks_two"}
assert all(x["episodes"] == 50 and x["unique_scene_seeds"] == 50 for x in rows.values())
out.write_text(json.dumps({"complete": True, "tasks": rows}, indent=2) + "\n")
PY
status "complete all_three_formal_50_seed_evaluations"
