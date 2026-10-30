#!/usr/bin/env bash
set -Eeuo pipefail

UBUNTU_ROOT=/media/ruio/hdd/pace_pi05_eval_20261022
UBUNTU_DATA="$UBUNTU_ROOT/outputs/d_action_pairs_support_turn_switch_dump_bin_bigbin_20261029_train50_test20"
UBUNTU_STATE="$UBUNTU_ROOT/control/d_pair_support_50train20test_two_tasks_20261029/state.json"
JG_HOST=100.99.60.71
JG_USER=ruihao
JG_BASE=/home/ruihao/d_pair_support_turn_switch_dump_bin_bigbin_20261029_50train20test
LOCAL_REPO=/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling
TRAINER="$LOCAL_REPO/RoboTwin/scripts/train_d_support_50train20test_two_tasks_jg_20261029.py"
LOCAL_OUT="$LOCAL_REPO/outputs/d_pair_support_two_tasks_50train20test_20261029"

last=""
while true; do
  snapshot=$(ssh -o BatchMode=yes -o ConnectTimeout=8 ubuntu "cat '$UBUNTU_STATE'" 2>/dev/null) || {
    echo "Ubuntu state temporarily unavailable; retrying" >&2
    sleep 30
    continue
  }
  read -r status task phase < <(printf '%s' "$snapshot" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status","?"),d.get("task") or "-",d.get("phase") or "-")')
  current="$status task=$task phase=$phase"
  if [[ "$current" != "$last" ]]; then
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$current"
    last="$current"
  fi
  case "$status" in
    complete) break ;;
    failed)
      echo "Ubuntu collection/postprocessing failed; not transferring or starting JG training." >&2
      exit 2
      ;;
  esac
  sleep 60
done

ssh -o BatchMode=yes JG-ts "test ! -e '$JG_BASE' && mkdir -p '$JG_BASE/data' '$JG_BASE/control'"
remote_copy="rsync -a --checksum --partial --info=progress2 '$UBUNTU_DATA/turn_switch_train50' '$UBUNTU_DATA/turn_switch_test20' '$UBUNTU_DATA/dump_bin_bigbin_train50' '$UBUNTU_DATA/dump_bin_bigbin_test20' '$UBUNTU_DATA/dataset_manifest.json' '$JG_USER@$JG_HOST:$JG_BASE/data/'"
ssh -A -o BatchMode=yes ubuntu "$remote_copy"

remote_dry="rsync -an --checksum --itemize-changes '$UBUNTU_DATA/turn_switch_train50' '$UBUNTU_DATA/turn_switch_test20' '$UBUNTU_DATA/dump_bin_bigbin_train50' '$UBUNTU_DATA/dump_bin_bigbin_test20' '$UBUNTU_DATA/dataset_manifest.json' '$JG_USER@$JG_HOST:$JG_BASE/data/'"
dry=$(ssh -A -o BatchMode=yes ubuntu "$remote_dry")
if [[ -n "$dry" ]]; then
  printf 'Post-transfer checksum dry-run found differences:\n%s\n' "$dry" >&2
  exit 3
fi
echo "Ubuntu-to-JG data transfer verified by checksum dry-run."

scp -o BatchMode=yes "$TRAINER" "JG-ts:$JG_BASE/control/train_d_support_50train20test_two_tasks_jg_20261029.py"
ssh -o BatchMode=yes JG-ts "chmod 755 '$JG_BASE/control/train_d_support_50train20test_two_tasks_jg_20261029.py'; test ! -e '$JG_BASE/control/state.json'; test ! -e '$JG_BASE/results'; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0; nohup /home/ruihao/train_critic/.venv/bin/python -u '$JG_BASE/control/train_d_support_50train20test_two_tasks_jg_20261029.py' > '$JG_BASE/control/launcher.log' 2>&1 < /dev/null & echo \$! > '$JG_BASE/control/launcher.pid'; sleep 3; cat '$JG_BASE/control/launcher.pid'; cat '$JG_BASE/control/launcher.log'"

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
    failed)
      echo "JG D training failed; it will not be retried automatically." >&2
      exit 4
      ;;
  esac
  sleep 60
done

mkdir -p "$LOCAL_OUT"
scp -o BatchMode=yes \
  "JG-ts:$JG_BASE/results/aggregate_d_support_50train20test.json" \
  "JG-ts:$JG_BASE/results/turn_switch/metrics.json" \
  "JG-ts:$JG_BASE/results/turn_switch/manifest.json" \
  "JG-ts:$JG_BASE/results/dump_bin_bigbin/metrics.json" \
  "JG-ts:$JG_BASE/results/dump_bin_bigbin/manifest.json" \
  "$LOCAL_OUT/"
echo "Both D functions trained on 50 episodes/task and evaluated on 20 independent episodes/task; compact reports synced to $LOCAL_OUT"
