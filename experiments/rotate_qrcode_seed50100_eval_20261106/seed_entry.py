"""Run the pinned SFT evaluator with only its derived scene-seed start changed."""
from __future__ import annotations

import os

import eval_policy_xpolicylab_official_trace as official
from test_render import Sapien_TEST


def main() -> None:
    start = int(os.environ["EVAL_SCENE_SEED_START"])
    if start != 50100:
        raise ValueError(f"Unexpected scene seed start: {start}")
    args = official.parse_args()
    if args.get("task_name") != "rotate_qrcode" or args.get("seed") != 0:
        raise ValueError("Expected rotate_qrcode --seed 0")
    if args.get("eval_batch") or not args.get("expert_check") or args.get("instruction_type") != "seen":
        raise ValueError("Expected single-process expert-checked seen evaluation")
    original = official.eval_remote_policy

    def with_scene_seed(task_name, task_env, task_args, model_client, st_seed, **kwargs):
        if st_seed != 100000:
            raise ValueError(f"Unexpected official derived start seed: {st_seed}")
        print(f"[SEED-OVERRIDE] official evaluator scene start {st_seed} -> {start}", flush=True)
        return original(task_name, task_env, task_args, model_client, start, **kwargs)

    official.eval_remote_policy = with_scene_seed
    Sapien_TEST()
    official.main(args)


if __name__ == "__main__":
    main()
