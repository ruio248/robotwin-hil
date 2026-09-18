#!/usr/bin/env python3
"""Record one scripted-expert demo of the native handover_block base task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))
if str(ROBOTWIN_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT / "scripts"))

from eval_policy_xpolicylab import class_decorator, load_task_args, safe_close_env


PROMPT = "Pass the red block from the left to the right and place it on the blue pad."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", default="handover_to_tray_v2_promptfix")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--render-freq", type=int, default=5)
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument("--render-max-num-materials", type=int, default=128)
    parser.add_argument("--render-max-num-textures", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    cli.output_dir = cli.output_dir.expanduser().resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    user_args = {
        "task_name": "handover_block",
        "task_config": cli.task_config,
        "policy_name": "Pi_05",
        "ckpt_name": "unused",
    }
    args, _ = load_task_args(user_args)
    args["eval_instruction"] = "seen"
    args["eval_mode"] = True
    args["render_freq"] = int(cli.render_freq)
    args["need_plan"] = True
    args["save_data"] = True
    args["save_video"] = True
    args["save_path"] = str(cli.output_dir)
    args["save_freq"] = int(cli.save_freq)
    args["render_max_num_materials"] = int(cli.render_max_num_materials)
    args["render_max_num_textures"] = int(cli.render_max_num_textures)

    task_env = class_decorator("handover_block")
    task_env.setup_demo(now_ep_num=0, seed=int(cli.seed), is_test=False, **args)
    task_env.set_instruction(PROMPT)
    task_env.ep_num = 0

    if hasattr(task_env, "_update_render"):
        task_env._update_render()
    viewer = getattr(task_env, "viewer", None)
    if viewer is not None:
        viewer.render()

    task_env.play_once()
    task_env.merge_pkl_to_hdf5_video(instructions=[PROMPT])
    safe_close_env(task_env)
    print(f"video_dir={cli.output_dir / 'video'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
