# robotwin-hil

Clean private snapshot of the RoboTwin human-in-the-loop and HG-DAgger work
from the JG 5090 machine.

This repository keeps source code, task definitions, robot assets, policy
adapters, and reproducible scripts. It intentionally excludes the machine
local conda environment, Python toolchain, caches, logs, checkpoints, videos,
generated datasets, evaluation outputs, backups, and nested Git metadata.

## Layout

RoboTwin/ contains the benchmark code and the handover_to_tray task.

RoboTwin/scripts/hg_dagger_handover.py is the human-gated DAgger entry point.
The run_hg_dagger scripts provide collection, acceptance, and validation.

RoboTwin/XPolicyLab/policy/Pi_05_RobotTwin contains the Pi0.5/OpenPI policy
adapter, server/client setup, and deployment configuration.

RoboTwin/XPolicyLab is a vendored XPolicyLab source snapshot used by the
RoboTwin integration.

RoboTwin/env_cfg/task_config contains the handover_to_tray smoke, v1, and
prompt-fix configurations.

RoboTwin/assets/embodiments contains robot embodiment and cuRobo assets.

## Quick start

    source ./activate_robotwin_hil.sh
    cd "$ROBOTWIN_ROOT"

The activation script uses the repository directory as its root. If a bundled
environment is not present, set ROBOTWIN_PYTHON or activate an external
environment before running the scripts.

The default HG-DAgger collection uses policy action chunks until the operator
presses i. It then discards the remaining policy chunk and attempts a scripted
expert recovery. Only accepted recovery data is written.

## Upstream sources

- RoboTwin: https://github.com/robotwin-Platform/RoboTwin
- XPolicyLab: https://github.com/XPolicyLab/XPolicyLab
- OpenPI: https://github.com/Physical-Intelligence/openpi

This is a private working snapshot. Check the upstream licenses under
RoboTwin/ and the vendored policy directories before redistributing it.
