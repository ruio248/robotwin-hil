#!/usr/bin/env bash
set -Eeuo pipefail
OUT=/media/ruio/hdd/turn_switch_d_auc_own5k_20261030
CHECKPOINT="$OUT/checkpoint/5000"
mkdir -p "$OUT/control"
exec 8>"$OUT/control/queue.lock"
flock -n 8 || { echo "another queue owns the lock" >&2; exit 3; }
trap 'rc=$?; if (( rc != 0 )); then echo "failed_rc=$rc" >"$OUT/control/queue_state.txt"; fi' EXIT
echo waiting_checkpoint >"$OUT/control/queue_state.txt"
deadline=$((SECONDS + 7200))
while true; do
  if (( SECONDS > deadline )); then echo "checkpoint transfer timed out" >&2; exit 5; fi
  if [[ -f "$CHECKPOINT/params/manifest.ocdbt" ]]; then
    read -r count bytes < <(find "$CHECKPOINT" -type f -printf '%s\n' | awk '{total+=$1; count++} END {printf "%d %.0f\n", count, total}')
    manifest_sha=$(sha256sum "$CHECKPOINT/params/manifest.ocdbt" | cut -d' ' -f1)
    if [[ "$count" == 37 && "$bytes" == 12438807635 &&
          "$manifest_sha" == 503dcd0e832ca7090725599f1d5f9bf5a1057b180fcbfc92abeeb26cc20796c0 ]]; then
      sleep 30
      read -r count bytes < <(find "$CHECKPOINT" -type f -printf '%s\n' | awk '{total+=$1; count++} END {printf "%d %.0f\n", count, total}')
      if [[ "$count" == 37 && "$bytes" == 12438807635 ]]; then break; fi
    fi
  fi
  sleep 30
done
echo checkpoint_verified >"$OUT/control/queue_state.txt"
if [[ "$(findmnt -n -o SOURCE /media/ruio/hdd)" != /dev/sda1 ]]; then
  echo "wrong data mount" >&2
  exit 4
fi
echo running >"$OUT/control/queue_state.txt"
bash "$OUT/run_turn_switch_d_outcome_auc_20261030.sh"
echo complete >"$OUT/control/queue_state.txt"
