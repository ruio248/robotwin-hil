#!/usr/bin/env bash
# Stream the staged bf16 checkpoint from the JG/5090 host to this Ubuntu box
# over the LAN and extract it into the local policy checkpoint tree.
set -euo pipefail

DEST=/hdd/robotwin-hil/RoboTwin/XPolicyLab/policy/Pi_05_RobotTwin/checkpoints/pi05_robotwin_handover_to_tray_v2_promptfix/robotwin_handover_to_tray_v2_promptfix_bf16_inference
SRC_JG=/home/ruihao/ckpt_stage/pi05_v2_bf16

mkdir -p "$DEST"

ssh -i /home/ruio/.ssh/id_ed25519_JG \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new \
  ruihao@192.168.101.11 \
  "tar -C '$SRC_JG' -cf - 9999" | tar -C "$DEST" -xf -

echo STAGE_B_DONE
