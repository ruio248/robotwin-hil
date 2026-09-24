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

The large, ignored RoboTwin asset snapshot is machine-local rather than part
of the Git history. On the configured Ubuntu host it is hydrated at
`/hdd/robotwin-hil/RoboTwin/assets` and is exposed as
`/media/ruio/hdd/robotwin-hil/RoboTwin/assets` inside the launcher namespace.

## Quick start

    source ./activate_robotwin_hil.sh
    cd "$ROBOTWIN_ROOT"

The activation script uses the repository directory as its root. If a bundled
environment is not present, set ROBOTWIN_PYTHON or activate an external
environment before running the scripts.

The default HG-DAgger collection uses policy action chunks until the operator
presses i. It then discards the remaining policy chunk and attempts a scripted
expert recovery. Only accepted recovery data is written.

## Ubuntu/4090 HDD environment

On the Ubuntu host where the data disk is mounted at `/hdd`, enter the
HDD-backed environment with:

    bash ./enter_robotwin_hil.sh

The launcher creates a private user-namespace bind mount from `/hdd` to the
historical `/media/ruio/hdd` path, selects the HDD-backed `robotwin_hil`
conda clone, sets the RoboTwin, LeRobot, Warp, JAX, and OpenPI cache/output
locations on the HDD, and opens a shell in `RoboTwin/`. It does not require
`sudo` and does not modify `/etc/fstab`.

The RoboTwin evaluation environment is the conda clone selected by the
launcher. The Pi0.5 policy server uses its separate OpenPI environment:

    source "$ROBOTWIN_OPENPI_ROOT/.venv/bin/activate"

For the handover-to-tray checkpoint, start the policy server after placing the
checkpoint under
`RoboTwin/XPolicyLab/policy/Pi_05_RobotTwin/checkpoints/`:

    bash XPolicyLab/policy/Pi_05_RobotTwin/setup_eval_policy_server.sh \
      RoboTwin handover_to_tray v2_promptfix_9999 aloha_agilex joint 40000 0 uv 18300 127.0.0.1

Then, in another terminal on the same host, run the human-gated client:

    python scripts/hg_dagger_handover.py --host 127.0.0.1 --port 18300 \
      --policy-name Pi_05_RobotTwin --ckpt-name v2_promptfix_9999 \
      --task-config handover_to_tray_v2_promptfix --seed-start 40000 \
      --render-freq 5 --frequency 30

The repository intentionally does not contain model weights. The checkpoint
directory must contain either an OpenPI `params/` tree or a PyTorch
`model.safetensors` file, plus the normalization file expected by this
adapter:

    assets/ruio248/robotwin_handover_to_tray_v1/norm_stats.json

If the checkpoint is stored elsewhere, pass its absolute directory path as
`ckpt_name` to both the server and client commands. The server resolves a
checkpoint path before trying the conventional
`<bench>-<ckpt>-<env>-<action>-<seed>` directory name.

## Offline tail coverage value

The optional frozen-policy coverage critic workflow is documented in
[TAIL_VALUE_README.md](TAIL_VALUE_README.md). It adds `tail_data.py`,
`train_tail.py`, and `eval_tail_value.py` under `RoboTwin/scripts/`, with
mean/farthest candidate ablations and separate demonstration/HIL diagnostics.
This is an approximate-transition research prototype, not a success predictor
or an online recovery benchmark. Existing HIL collection is unchanged.

## Coverage-guided enhanced sampling

Fixed-window MC coverage-guided action-chunk sampling and its matched Vanilla
control are documented in [ENHANCED_SAMPLING.md](ENHANCED_SAMPLING.md). The
existing evaluation entry point opts in with `--es-mode enhanced`; default
evaluation remains `--es-mode off`.

## Upstream sources

- RoboTwin: https://github.com/robotwin-Platform/RoboTwin
- XPolicyLab: https://github.com/XPolicyLab/XPolicyLab
- OpenPI: https://github.com/Physical-Intelligence/openpi

This is a private working snapshot. Check the upstream licenses under
RoboTwin/ and the vendored policy directories before redistributing it.
