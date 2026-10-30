#!/usr/bin/env python3
"""Prepare paired scripted-expert and HIL datasets for one-stage IWR.

The one-stage HG-DAgger collector deliberately starts recording only after the
scripted stages 1--5.  Therefore the ``policy`` prefix present in a saved HIL
episode is *not* the scripted expert prefix and must never be relabelled as
expert data.  This utility uses the first N valid saved HIL episodes to:

1. freeze an auditable episode/seed manifest;
2. export only frames labelled ``hil`` from the saved collection; and
3. replay the same seeds and record the native scripted expert through stage 5.

Both sources are written as independent native RoboTwin HDF5 datasets.  The
training loader can then concatenate the two LeRobot conversions, reproducing
the historical C-group sampling semantics without mixing their provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))
if str(ROBOTWIN_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT / "scripts"))

from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video  # noqa: E402
from eval_policy_xpolicylab import (  # noqa: E402
    class_decorator,
    load_task_args,
    safe_close_env,
)
from one_stage_dagger import handoff_record  # noqa: E402
from policy_eval_record import PROMPT  # noqa: E402


MANIFEST_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("manifest", "export-hil", "collect-expert", "validate"),
    )
    parser.add_argument("--hil-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--task-config", default="handover_to_tray_v2_promptfix"
    )
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument("--render-max-num-materials", type=int, default=128)
    parser.add_argument("--render-max-num-textures", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-new-expert-episodes", type=int, default=0,
        help="Stop expert replay after this many newly collected episodes (0=unlimited); resume in a fresh simulator process.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_path(output_root: Path) -> Path:
    return output_root / "selection_manifest.json"


def discover_source_episodes(hil_root: Path, count: int) -> list[dict[str, Any]]:
    raw_root = hil_root / "raw"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"missing raw episode directory: {raw_root}")
    episode_dirs = sorted(
        path
        for path in raw_root.glob("episode_*")
        if path.is_dir() and (path / "episode.json").is_file()
    )
    if len(episode_dirs) < count:
        raise ValueError(f"requested {count} episodes, found {len(episode_dirs)}")

    selected: list[dict[str, Any]] = []
    for output_index, episode_dir in enumerate(episode_dirs[:count]):
        meta_path = episode_dir / "episode.json"
        meta = load_json(meta_path)
        control_mask = [str(value).lower() for value in meta.get("control_mask", [])]
        frame_paths = sorted(
            (episode_dir / "frames").glob("*.pkl"),
            key=lambda path: int(path.stem),
        )
        expected_indices = list(range(len(frame_paths)))
        actual_indices = [int(path.stem) for path in frame_paths]
        if actual_indices != expected_indices:
            raise ValueError(f"non-contiguous frames in {episode_dir}")
        if len(frame_paths) != len(control_mask):
            raise ValueError(
                f"frame/mask mismatch in {episode_dir}: "
                f"{len(frame_paths)} != {len(control_mask)}"
            )
        if str(meta.get("supervisor_label", "")).lower() != "success":
            raise ValueError(f"episode is not a successful HIL recovery: {episode_dir}")
        hil_indices = [i for i, source in enumerate(control_mask) if source == "hil"]
        if len(hil_indices) < 2:
            raise ValueError(f"episode has fewer than two HIL frames: {episode_dir}")
        source_index = int(meta.get("episode_index", episode_dir.name.split("_")[-1]))
        selected.append(
            {
                "output_episode_index": output_index,
                "source_episode_index": source_index,
                "seed": int(meta["seed"]),
                "source_episode_dir": str(episode_dir.resolve()),
                "source_metadata": str(meta_path.resolve()),
                "source_metadata_sha256": sha256(meta_path),
                "all_frames": len(control_mask),
                "policy_frames": sum(value == "policy" for value in control_mask),
                "hil_frames": len(hil_indices),
                "hil_source_frame_indices": hil_indices,
                "supervisor_label": meta.get("supervisor_label"),
                "save_freq": int(meta.get("save_freq", 15) or 15),
            }
        )
    seeds = [item["seed"] for item in selected]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds in selected episodes")
    return selected


def create_manifest(cli: argparse.Namespace) -> dict[str, Any]:
    cli.output_root.mkdir(parents=True, exist_ok=True)
    path = manifest_path(cli.output_root)
    selected = discover_source_episodes(cli.hil_root, cli.episodes)
    session_reports = sorted(cli.hil_root.glob("session_*.json"))
    value = {
        "manifest_version": MANIFEST_VERSION,
        "created_at_utc": utc_now(),
        "complete": False,
        "selection_rule": "first N saved valid HIL episodes in raw episode-index order",
        "episodes_requested": cli.episodes,
        "task_config": cli.task_config,
        "instruction": PROMPT,
        "frequency_hz": cli.save_freq,
        "source_hil_root": str(cli.hil_root.resolve()),
        "source_session_report": str(session_reports[-1].resolve()) if session_reports else None,
        "source_session_report_sha256": sha256(session_reports[-1]) if session_reports else None,
        "expert_source_definition": "same-seed scripted stages 1-5 replay",
        "hil_source_definition": "saved frames whose control_mask equals 'hil'",
        "excluded_definition": "saved stage-6 policy frames are excluded from both sources",
        "episodes": selected,
    }
    if path.exists():
        current = load_json(path)
        immutable = (
            "manifest_version",
            "episodes_requested",
            "task_config",
            "instruction",
            "frequency_hz",
            "source_hil_root",
            "source_session_report_sha256",
            "expert_source_definition",
            "hil_source_definition",
            "excluded_definition",
            "episodes",
        )
        if any(current.get(key) != value.get(key) for key in immutable):
            raise ValueError(f"existing manifest differs from current selection: {path}")
        print(f"[MANIFEST] verified existing {path}", flush=True)
        return current
    atomic_json(path, value)
    print(f"[MANIFEST] wrote {path}", flush=True)
    return value


def require_manifest(cli: argparse.Namespace) -> dict[str, Any]:
    path = manifest_path(cli.output_root)
    if not path.is_file():
        raise FileNotFoundError(f"run the manifest command first: {path}")
    value = load_json(path)
    if int(value.get("manifest_version", -1)) != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version in {path}")
    if int(value.get("episodes_requested", -1)) != cli.episodes:
        raise ValueError("--episodes differs from frozen manifest")
    return value


def hdf5_summary(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        state_lengths = [len(handle["state"][name]) for name in handle["state"]]
        action_lengths = [len(handle["action"][name]) for name in handle["action"]]
        if not state_lengths or len(set(state_lengths + action_lengths)) != 1:
            raise ValueError(f"inconsistent state/action lengths: {path}")
        # The native file also stores two 7-D end-effector poses.  Pi0.5's
        # state/action extractor intentionally uses only the four joint fields,
        # so validate that same 14-D contract rather than summing every dataset
        # in the HDF5 groups.
        joint_fields = (
            "left_arm_joint_states",
            "left_ee_joint_states",
            "right_arm_joint_states",
            "right_ee_joint_states",
        )
        missing_state = [name for name in joint_fields if name not in handle["state"]]
        missing_action = [name for name in joint_fields if name not in handle["action"]]
        if missing_state or missing_action:
            raise ValueError(
                f"missing joint fields in {path}: "
                f"state={missing_state}, action={missing_action}"
            )
        state_dim = sum(int(handle["state"][name].shape[1]) for name in joint_fields)
        action_dim = sum(int(handle["action"][name].shape[1]) for name in joint_fields)
        frequency = int(handle["additional_info"]["frequency"][()])
        seeds = (
            sorted(set(int(value) for value in handle["additional_info"]["seed"][:]))
            if "seed" in handle["additional_info"]
            else []
        )
        stages = (
            sorted(set(int(value) for value in handle["additional_info"]["stage_id"][:]))
            if "stage_id" in handle["additional_info"]
            else []
        )
        return {
            "transitions": state_lengths[0],
            "state_dim": state_dim,
            "action_dim": action_dim,
            "frequency_hz": frequency,
            "seeds": seeds,
            "stage_ids": stages,
        }


def verify_existing_hdf5(
    path: Path, *, seed: int, transitions: int | None, frequency: int
) -> dict[str, Any]:
    summary = hdf5_summary(path)
    if summary["state_dim"] != 14 or summary["action_dim"] != 14:
        raise ValueError(f"expected 14-D state/action in {path}: {summary}")
    if summary["frequency_hz"] != frequency:
        raise ValueError(f"frequency mismatch in {path}: {summary}")
    if summary["seeds"] != [seed]:
        raise ValueError(f"seed mismatch in {path}: {summary}")
    if transitions is not None and summary["transitions"] != transitions:
        raise ValueError(f"transition count mismatch in {path}: {summary}")
    return summary


def export_hil(cli: argparse.Namespace, manifest: dict[str, Any]) -> None:
    output_dir = cli.output_root / "native" / "hil"
    data_dir = output_dir / "data"
    staging_root = output_dir / ".staging"
    data_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for item in manifest["episodes"]:
        output_index = int(item["output_episode_index"])
        seed = int(item["seed"])
        target = data_dir / f"episode_{output_index:07d}.hdf5"
        expected_transitions = int(item["hil_frames"]) - 1
        if target.exists():
            if not cli.resume:
                raise FileExistsError(f"existing output requires --resume: {target}")
            summary = verify_existing_hdf5(
                target,
                seed=seed,
                transitions=expected_transitions,
                frequency=cli.save_freq,
            )
            records.append({**item, **summary, "hdf5": str(target)})
            print(f"[HIL] verified episode={output_index} seed={seed}", flush=True)
            continue

        stage = staging_root / f"episode_{output_index:07d}"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        source_frames = Path(item["source_episode_dir"]) / "frames"
        try:
            for new_index, old_index in enumerate(item["hil_source_frame_indices"]):
                source = source_frames / f"{int(old_index)}.pkl"
                target_frame = stage / f"{new_index}.pkl"
                try:
                    os.link(source, target_frame)
                except OSError:
                    shutil.copy2(source, target_frame)
            process_folder_to_hdf5_video(
                str(stage),
                str(target),
                str(output_dir / "video" / f"episode_{output_index:07d}.mp4"),
                instructions=[PROMPT],
                frequency=cli.save_freq,
                episode_metadata={
                    "source": "hil",
                    "source_episode_index": int(item["source_episode_index"]),
                    "seed": seed,
                    "source_metadata_sha256": item["source_metadata_sha256"],
                },
                save_video=False,
            )
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        summary = verify_existing_hdf5(
            target,
            seed=seed,
            transitions=expected_transitions,
            frequency=cli.save_freq,
        )
        records.append({**item, **summary, "hdf5": str(target)})
        print(
            f"[HIL] exported episode={output_index} seed={seed} "
            f"transitions={summary['transitions']}",
            flush=True,
        )
    shutil.rmtree(staging_root, ignore_errors=True)
    atomic_json(
        output_dir / "export_manifest.json",
        {"complete": True, "created_at_utc": utc_now(), "episodes": records},
    )


def expert_runtime_args(cli: argparse.Namespace) -> dict[str, Any]:
    user_args = {
        "task_name": "handover_to_tray",
        "task_config": cli.task_config,
        "policy_name": "scripted_expert",
        "ckpt_setting": "none",
    }
    args, _ = load_task_args(user_args)
    args.update(
        {
            "eval_instruction": "seen",
            "eval_mode": True,
            "render_freq": 0,
            "need_plan": True,
            "save_data": True,
            "save_video": False,
            "save_freq": cli.save_freq,
            "render_max_num_materials": cli.render_max_num_materials,
            "render_max_num_textures": cli.render_max_num_textures,
        }
    )
    return args


def collect_expert(cli: argparse.Namespace, manifest: dict[str, Any]) -> None:
    output_dir = cli.output_root / "native" / "expert"
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    args = expert_runtime_args(cli)
    args["save_path"] = str(output_dir)
    records: list[dict[str, Any]] = []
    new_episodes = 0

    for item in manifest["episodes"]:
        output_index = int(item["output_episode_index"])
        seed = int(item["seed"])
        target = data_dir / f"episode_{output_index:07d}.hdf5"
        if target.exists():
            if not cli.resume:
                raise FileExistsError(f"existing output requires --resume: {target}")
            summary = verify_existing_hdf5(
                target, seed=seed, transitions=None, frequency=cli.save_freq
            )
            if not summary["stage_ids"] or not set(summary["stage_ids"]).issubset(
                {1, 2, 3, 4, 5}
            ):
                raise ValueError(f"unexpected expert stages in {target}: {summary}")
            records.append({**item, **summary, "hdf5": str(target)})
            print(f"[EXPERT] verified episode={output_index} seed={seed}", flush=True)
            continue

        task_env = class_decorator("handover_to_tray")
        try:
            task_env.setup_demo(
                now_ep_num=output_index,
                seed=seed,
                is_test=False,
                **args,
            )
            task_env.set_instruction(PROMPT)
            task_env.current_control_source = "expert_prefix"
            prefix_ok = bool(task_env.play_prefix_until_placement())
            handoff = handoff_record(task_env)
            if not prefix_ok or not handoff.get("ready"):
                raise RuntimeError(
                    f"same-seed expert prefix failed for seed={seed}: {handoff}"
                )
            if int(task_env.FRAME_IDX) < 2:
                raise RuntimeError(f"too few expert frames for seed={seed}")
            task_env.episode_metadata.update(
                {
                    "source": "scripted_expert_prefix",
                    "source_stages": [1, 2, 3, 4, 5],
                    "source_hil_episode_index": int(item["source_episode_index"]),
                    "source_hil_metadata_sha256": item["source_metadata_sha256"],
                    "control_mask": list(task_env.control_mask),
                    "handoff": handoff,
                }
            )
            task_env.merge_pkl_to_hdf5_video(instructions=[PROMPT])
            task_env.remove_data_cache()
        except Exception:
            cache = output_dir / ".cache" / f"episode{output_index}"
            shutil.rmtree(cache, ignore_errors=True)
            if target.exists():
                target.unlink()
            raise
        finally:
            safe_close_env(task_env, clear_cache=((output_index + 1) % 5 == 0))

        summary = verify_existing_hdf5(
            target, seed=seed, transitions=None, frequency=cli.save_freq
        )
        if not summary["stage_ids"] or not set(summary["stage_ids"]).issubset(
            {1, 2, 3, 4, 5}
        ):
            raise ValueError(f"unexpected expert stages in {target}: {summary}")
        records.append({**item, **summary, "hdf5": str(target)})
        new_episodes += 1
        print(
            f"[EXPERT] collected episode={output_index} seed={seed} "
            f"transitions={summary['transitions']} stages={summary['stage_ids']}",
            flush=True,
        )
        if (cli.max_new_expert_episodes and new_episodes >= cli.max_new_expert_episodes
                and output_index + 1 < len(manifest["episodes"])):
            print(f"[EXPERT] process limit reached after {new_episodes} new episodes; resume to continue", flush=True)
            return

    atomic_json(
        output_dir / "collection_manifest.json",
        {"complete": True, "created_at_utc": utc_now(), "episodes": records},
    )


def validate_native(cli: argparse.Namespace, manifest: dict[str, Any]) -> None:
    result: dict[str, Any] = {
        "complete": False,
        "validated_at_utc": utc_now(),
        "episodes": cli.episodes,
        "sources": {},
    }
    expected_seeds = [int(item["seed"]) for item in manifest["episodes"]]
    for source in ("expert", "hil"):
        source_dir = cli.output_root / "native" / source
        files = sorted((source_dir / "data").glob("episode_*.hdf5"))
        if len(files) != cli.episodes:
            raise ValueError(
                f"{source}: expected {cli.episodes} HDF5 files, found {len(files)}"
            )
        source_records = []
        for index, (path, seed) in enumerate(zip(files, expected_seeds)):
            expected_transitions = None
            if source == "hil":
                expected_transitions = int(manifest["episodes"][index]["hil_frames"]) - 1
            summary = verify_existing_hdf5(
                path,
                seed=seed,
                transitions=expected_transitions,
                frequency=cli.save_freq,
            )
            stages = set(summary["stage_ids"])
            if source == "expert" and (not stages or not stages.issubset({1, 2, 3, 4, 5})):
                raise ValueError(f"expert stage leakage in {path}: {summary}")
            # The intervention is requested during stage 6, but a successful
            # expert recovery may deliberately restart from an earlier stage
            # (for example stage 4 after losing the receiver grasp).  Source
            # purity is guaranteed by selecting control_mask == "hil" above;
            # do not discard valid recovery actions merely because their
            # semantic recovery stage is earlier than 6.
            if source == "hil" and (
                not stages or not stages.issubset({1, 2, 3, 4, 5, 6})
            ):
                raise ValueError(f"invalid HIL recovery stage in {path}: {summary}")
            source_records.append(
                {
                    "episode_index": index,
                    "seed": seed,
                    "path": str(path),
                    "sha256": sha256(path),
                    **summary,
                }
            )
        result["sources"][source] = {
            "episodes": len(source_records),
            "transitions": sum(item["transitions"] for item in source_records),
            "records": source_records,
        }
    result["sampling_contract"] = {
        "outer_batch": "50% original SFT + 50% concatenated expert/HIL source",
        "inner_composite": "ConcatDataset(expert, HIL), sampled uniformly by transition",
        "norm_stats": "use original SFT checkpoint norm_stats; do not recompute mixed stats",
    }
    result["complete"] = True
    atomic_json(cli.output_root / "native_manifest.json", result)
    manifest["complete"] = True
    manifest["completed_at_utc"] = utc_now()
    manifest["native_manifest"] = str((cli.output_root / "native_manifest.json").resolve())
    atomic_json(manifest_path(cli.output_root), manifest)
    print(
        "[VALIDATE] complete "
        f"expert={result['sources']['expert']['transitions']} transitions, "
        f"hil={result['sources']['hil']['transitions']} transitions",
        flush=True,
    )


def main() -> int:
    cli = parse_args()
    cli.hil_root = cli.hil_root.expanduser().resolve()
    cli.output_root = cli.output_root.expanduser().resolve()
    if cli.episodes < 1 or cli.save_freq < 1 or cli.max_new_expert_episodes < 0:
        raise ValueError("--episodes and --save-freq must be positive; --max-new-expert-episodes cannot be negative")
    if cli.command == "manifest":
        create_manifest(cli)
        return 0
    manifest = require_manifest(cli)
    if cli.command == "export-hil":
        export_hil(cli, manifest)
    elif cli.command == "collect-expert":
        collect_expert(cli, manifest)
    elif cli.command == "validate":
        validate_native(cli, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
