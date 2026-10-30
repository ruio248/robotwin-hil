#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for Ubuntu pair generation, transfer the verified dataset directly to
# JG over SSH agent forwarding, launch the existing five-fold D trainer, and
# copy only the compact JSON reports back to this workstation.
LOCAL_REPO=/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling
UBUNTU_ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
PAIR_ROOT="$UBUNTU_ROOT/outputs/d_action_pairs_support_turn_switch_dump_bin_bigbin_20261029_seed1_n30"
PAIR_STATE="$UBUNTU_ROOT/control/d_pair_support_turn_switch_dump_bin_bigbin_20261029/state.json"
JG_HOST=100.99.60.71
JG_USER=ruihao
JG_BASE=/home/ruihao/d_pair_support_turn_switch_dump_bin_bigbin_20261029
TRAIN_SCRIPT="$LOCAL_REPO/RoboTwin/scripts/train_d_pair_support_two_tasks_jg_20261029.sh"
LOCAL_OUT="$LOCAL_REPO/outputs/d_pair_support_two_tasks_20261029"

last=""
while true; do
  snapshot=$(ssh -o BatchMode=yes -o ConnectTimeout=8 ubuntu "cat '$PAIR_STATE'" 2>/dev/null) || {
    echo "Ubuntu state temporarily unavailable; retrying" >&2
    sleep 30
    continue
  }
  read -r status task phase completed < <(printf '%s' "$snapshot" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status","?"),d.get("task") or "-",d.get("phase") or "-",d.get("completed_tasks",0))')
  current="$status task=$task phase=$phase completed=$completed"
  if [[ "$current" != "$last" ]]; then
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$current"
    last="$current"
  fi
  case "$status" in
    complete) break ;;
    failed|partial_failed)
      echo "Ubuntu D-pair pipeline failed; not transferring or starting training." >&2
      exit 2
      ;;
  esac
  sleep 60
done

# Do not reuse or overwrite any pre-existing JG experiment directory.
if ssh -o BatchMode=yes JG-ts "test ! -e '$JG_BASE'"; then
  ssh -o BatchMode=yes JG-ts "mkdir -p '$JG_BASE/data' '$JG_BASE/control'"
else
  echo "Refusing to overwrite existing JG experiment root: $JG_BASE" >&2
  exit 3
fi

remote_copy="rsync -a --checksum --partial --info=progress2 '$PAIR_ROOT/turn_switch_seed1_n30' '$PAIR_ROOT/dump_bin_bigbin_seed1_n30' '$PAIR_ROOT/postprocess_manifest.json' '$JG_USER@$JG_HOST:$JG_BASE/data/'"
ssh -A -o BatchMode=yes ubuntu "$remote_copy"

remote_dry="rsync -an --checksum --itemize-changes '$PAIR_ROOT/turn_switch_seed1_n30' '$PAIR_ROOT/dump_bin_bigbin_seed1_n30' '$PAIR_ROOT/postprocess_manifest.json' '$JG_USER@$JG_HOST:$JG_BASE/data/'"
dry=$(ssh -A -o BatchMode=yes ubuntu "$remote_dry")
if [[ -n "$dry" ]]; then
  printf 'Post-transfer checksum dry-run found differences:\n%s\n' "$dry" >&2
  exit 4
fi
echo "Ubuntu-to-JG rsync checksum dry-run: no differences"

scp -o BatchMode=yes "$TRAIN_SCRIPT" "JG-ts:$JG_BASE/control/train_d_pair_support_two_tasks_jg_20261029.sh"
ssh -o BatchMode=yes JG-ts "chmod 755 '$JG_BASE/control/train_d_pair_support_two_tasks_jg_20261029.sh'; test ! -e '$JG_BASE/control/state.json'; nohup bash '$JG_BASE/control/train_d_pair_support_two_tasks_jg_20261029.sh' > '$JG_BASE/control/launcher.log' 2>&1 < /dev/null & echo \$! > '$JG_BASE/control/launcher.pid'; sleep 3; cat '$JG_BASE/control/launcher.pid'; cat '$JG_BASE/control/launcher.log'"

last=""
while true; do
  snapshot=$(ssh -o BatchMode=yes -o ConnectTimeout=8 JG-ts "cat '$JG_BASE/control/state.json'" 2>/dev/null) || {
    echo "JG training state not yet available; retrying" >&2
    sleep 30
    continue
  }
  read -r status task phase completed < <(printf '%s' "$snapshot" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status","?"),d.get("task") or "-",d.get("phase") or "-",d.get("completed_tasks",0))')
  current="$status task=$task phase=$phase completed=$completed"
  if [[ "$current" != "$last" ]]; then
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$current"
    last="$current"
  fi
  case "$status" in
    complete) break ;;
    failed|partial_failed)
      echo "JG D training failed; results remain on JG and were not restarted." >&2
      exit 5
      ;;
  esac
  sleep 60
done

mkdir -p "$LOCAL_OUT"
scp -o BatchMode=yes \
  "JG-ts:$JG_BASE/results/aggregate_d_task_screening.json" \
  "JG-ts:$JG_BASE/results/turn_switch/metrics.json" \
  "JG-ts:$JG_BASE/results/turn_switch/manifest.json" \
  "JG-ts:$JG_BASE/results/dump_bin_bigbin/metrics.json" \
  "JG-ts:$JG_BASE/results/dump_bin_bigbin/manifest.json" \
  "$LOCAL_OUT/"
echo "D-support training complete; compact JSON reports synced to $LOCAL_OUT"
