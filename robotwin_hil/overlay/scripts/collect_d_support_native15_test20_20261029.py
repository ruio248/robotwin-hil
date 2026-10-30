#!/usr/bin/env python3
"""Generate a separate 20-episode scripted-expert test cohort at native 15 Hz.

This wraps RoboTwin's own collect_data.py in an isolated simulator workspace.
The two task-specific 5k checkpoints and the original 50 SFT demonstrations are
not used to generate these expert trajectories.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path

import h5py


PROMPTS = {
    "turn_switch": "Turn the switch to its final position.",
    "dump_bin_bigbin": "Pick up the small bin and pour all the balls into the big bin.",
}


class StopAfterCandidateRange(BaseException):
    pass


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(PROMPTS))
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path,
                        help="Isolated cwd with env_cfg/ and a complete assets/ tree")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scene-seed-start", type=int, default=500000)
    parser.add_argument("--max-candidate-scenes", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--smoke-only", action="store_true",
                        help="Collect one non-counting expert demo to validate the isolated simulator")
    args = parser.parse_args()
    if ((args.smoke_only and args.episodes != 1)
            or (not args.smoke_only and args.episodes != 20)
            or args.max_candidate_scenes < args.episodes):
        parser.error("formal protocol requires 20 successes; smoke-only requires exactly one")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    robotwin = args.robotwin_root.resolve()
    workspace = args.workspace.resolve()
    source_script = robotwin / "scripts/collect_data.py"
    if not source_script.is_file():
        raise FileNotFoundError(source_script)
    for required in (workspace / "assets/objects", workspace / "assets/embodiments",
                     workspace / "env_cfg/task_config/demo_clean.yml"):
        if not required.exists():
            raise FileNotFoundError(f"isolated simulator workspace lacks {required}")

    args.output_root.mkdir(parents=True)
    state_path = args.output_root / "state.json"
    atomic_json(state_path, {
        "status": "starting", "task": args.task,
        "requested_successful_episodes": args.episodes,
        "smoke_only": args.smoke_only,
        "candidate_seed_start": args.scene_seed_start,
        "max_candidate_scenes": args.max_candidate_scenes,
        "frequency_hz": 15, "task_config": "demo_clean",
        "source": "RoboTwin scripted-expert native HDF5 data collector",
        "diagnostic_only": True, "accepted_for_training": False,
    })

    os.chdir(workspace)
    sys.path.insert(0, str(robotwin))
    spec = importlib.util.spec_from_file_location("robotwin_collect_data_sft_test", source_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load RoboTwin/scripts/collect_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prompt = PROMPTS[args.task]

    def fixed_instructions(config, episode_info, episode_index):
        target = Path(config["save_path"]) / "instruction" / f"episode_{episode_index:07d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        item = {"seen": [prompt], "unseen": [prompt], "episode_index": int(episode_index)}
        target.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n")
        return item

    module.build_episode_instructions = fixed_instructions
    original_run = module.run
    generated = {"scene_seeds": [], "output_dir": None}

    def run_isolated(task_env, config):
        config["episode_num"] = args.episodes
        config["use_seed"] = False
        config["collect_data"] = True
        config["save_freq"] = 15
        config["render_freq"] = 0
        config["save_path"] = str(
            args.output_root / config["task_config"] / config["task_name"]
            / re.sub(r"[^A-Za-z0-9_]+", "_", str(config["embodiment"][0])).lower()
        )
        output = Path(config["save_path"])
        if output.exists():
            raise FileExistsError(f"refusing existing output {output}")
        generated["output_dir"] = str(output)

        original_setup = task_env.setup_demo

        def setup_with_disjoint_seed(*setup_args, **setup_kwargs):
            base_seed = int(setup_kwargs["seed"])
            if base_seed >= args.max_candidate_scenes:
                raise StopAfterCandidateRange(
                    f"candidate seed range exhausted at {base_seed}/{args.max_candidate_scenes}"
                )
            actual_seed = args.scene_seed_start + base_seed
            setup_kwargs["seed"] = actual_seed
            generated["last_seed"] = actual_seed
            return original_setup(*setup_args, **setup_kwargs)

        task_env.setup_demo = setup_with_disjoint_seed
        try:
            original_run(task_env, config)
        except StopAfterCandidateRange as error:
            raise RuntimeError(str(error)) from error

        seed_file = output / "seed.txt"
        raw_seeds = [int(value) for value in seed_file.read_text().split()]
        actual_seeds = [args.scene_seed_start + value for value in raw_seeds]
        if len(actual_seeds) != args.episodes:
            raise RuntimeError(
                f"only {len(actual_seeds)}/{args.episodes} successful expert scenes "
                f"within {args.max_candidate_scenes} candidates"
            )
        seed_file.write_text(" ".join(map(str, actual_seeds)) + "\n")
        hdf5_files = sorted((output / "data").glob("episode_*.hdf5"))
        if len(hdf5_files) != args.episodes:
            raise RuntimeError(f"expected 20 native HDF5 files, found {len(hdf5_files)}")
        for path in hdf5_files:
            with h5py.File(path, "r") as h5:
                frequency = int(h5["additional_info/frequency"][()])
                if frequency != 15:
                    raise ValueError(f"{path}: expected native 15 Hz, got {frequency}")
        generated["scene_seeds"] = actual_seeds
        atomic_json(output / "scene_seed_manifest.json", {
            "complete": True, "task": args.task,
            "scene_seed_split": "smoke_only" if args.smoke_only else "independent_generated",
            "scene_seeds": actual_seeds, "prompt": prompt, "frequency_hz": 15,
            "smoke_only": args.smoke_only,
            "episodes": [{"episode_index": i, "scene_seed": seed,
                          "hdf5": f"data/episode_{i:07d}.hdf5"}
                         for i, seed in enumerate(actual_seeds)],
        })
        atomic_json(output / "summary.json", {
            "complete": True, "task": args.task, "episodes": args.episodes,
            "expert_successes": args.episodes, "scene_seeds": actual_seeds,
            "frequency_hz": 15, "task_config": "demo_clean",
            "prompt": prompt, "diagnostic_only": True, "accepted_for_training": False,
            "smoke_only": args.smoke_only,
        })

    module.run = run_isolated
    try:
        module.main(task_name=args.task, task_config="demo_clean")
        output_dirs = list(args.output_root.glob(f"demo_clean/{args.task}/*"))
        if len(output_dirs) != 1 or not (output_dirs[0] / "summary.json").is_file():
            raise RuntimeError("RoboTwin collector returned without a complete test summary")
        atomic_json(state_path, {
            "status": "smoke_complete" if args.smoke_only else "complete",
            "task": args.task, "completed_episodes": args.episodes,
            "scene_seeds": generated["scene_seeds"], "output_dir": str(output_dirs[0]),
            "frequency_hz": 15, "diagnostic_only": True, "accepted_for_training": False,
        })
        print(f"COMPLETE {args.task} native15 expert test20: {output_dirs[0]}", flush=True)
    except BaseException as error:
        atomic_json(state_path, {
            "status": "failed", "task": args.task,
            "detail": f"{type(error).__name__}: {error}",
            "last_scene_seed": generated.get("last_seed"),
            "output_dir": generated.get("output_dir"),
        })
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
