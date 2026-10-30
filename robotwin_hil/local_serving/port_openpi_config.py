#!/usr/bin/env python3
"""Port the RoboTwin 14-DoF data config and Pi0.5 TrainConfig into an OpenPI checkout.

Idempotent: running it twice is a no-op. A ``.bak`` copy is written next to the
target file before the first modification.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


IMPORT_ANCHOR = "import openpi.policies.libero_policy as libero_policy\n"
IMPORT_LINE = "import openpi.policies.robotwin_policy as robotwin_policy\n"

CLASS_ANCHOR = (
    "@dataclasses.dataclass(frozen=True)\n"
    "class RLDSDroidDataConfig(DataConfigFactory):\n"
)

CLASS_BLOCK = '''@dataclasses.dataclass(frozen=True)
class LeRobotRoboTwinDataConfig(DataConfigFactory):
    """PI0.5 data path for the three-camera 14-DoF RoboTwin task."""

    action_dim: int = 14
    has_left_wrist: bool = True
    has_right_wrist: bool = True
    default_prompt: str | None = None
    use_delta_joint_actions: bool = True
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_structure = {
            "observation/image": "observation.images.cam_high",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        if self.has_left_wrist:
            repack_structure["observation/left_wrist_image"] = "observation.images.cam_left_wrist"
        if self.has_right_wrist:
            repack_structure["observation/right_wrist_image"] = "observation.images.cam_right_wrist"

        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])
        data_transforms = _transforms.Group(
            inputs=[
                robotwin_policy.RoboTwinInputs(
                    model_type=model_config.model_type,
                    has_left_wrist=self.has_left_wrist,
                    has_right_wrist=self.has_right_wrist,
                )
            ],
            outputs=[robotwin_policy.RoboTwinOutputs(action_dim=self.action_dim)],
        )
        if self.use_delta_joint_actions:
            if self.action_dim != 14:
                raise ValueError(f"RoboTwin action_dim must be 14, got {self.action_dim}")
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


'''

CONFIGS_END_ANCHOR = "\n]\n\nif len({config.name for config in _CONFIGS}) != len(_CONFIGS):"

CONFIG_BLOCK = '''
    # RoboTwin long-horizon handover-to-tray v2 (prompt-fixed) Pi0.5 baseline.
    TrainConfig(
        name="pi05_robotwin_handover_to_tray_v2_promptfix",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotRoboTwinDataConfig(
            repo_id="ruio248/robotwin_handover_to_tray_v2_promptfix",
            action_dim=14,
            has_left_wrist=True,
            has_right_wrist=True,
            use_delta_joint_actions=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        num_workers=8,
        ema_decay=0.999,
        num_train_steps=10_000,
        save_interval=1_000,
        keep_period=5_000,
        wandb_enabled=False,
    ),
'''


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: port_openpi_config.py <path/to/openpi/training/config.py>")
        return 2

    path = Path(sys.argv[1]).resolve()
    text = path.read_text(encoding="utf-8")
    original = text

    if "LeRobotRoboTwinDataConfig" not in text:
        if IMPORT_ANCHOR not in text:
            raise SystemExit(f"import anchor not found in {path}")
        text = text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_LINE, 1)
        if CLASS_ANCHOR not in text:
            raise SystemExit(f"class anchor not found in {path}")
        text = text.replace(CLASS_ANCHOR, CLASS_BLOCK + CLASS_ANCHOR, 1)

    if "pi05_robotwin_handover_to_tray_v2_promptfix" not in text:
        if CONFIGS_END_ANCHOR not in text:
            raise SystemExit(f"_CONFIGS end anchor not found in {path}")
        text = text.replace(CONFIGS_END_ANCHOR, CONFIG_BLOCK + CONFIGS_END_ANCHOR, 1)

    if text == original:
        print(f"no changes needed: {path}")
        return 0

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path} (backup: {backup})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
