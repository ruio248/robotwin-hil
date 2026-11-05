"""Raw-frame persistence shared by interactive DAgger collectors."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)

def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def training_eligible(episode_dir: Path, meta: dict[str, Any]) -> bool:
    """Preserve the existing strict episode gate in every offline exporter."""
    if (episode_dir / "DIAGNOSTIC_ONLY.json").exists():
        return False
    for source in (meta, meta.get("episode_metadata") or {}, meta.get("record") or {}):
        if str(source.get("diagnostic_only", False)).lower() in {"1", "true", "yes", "y", "on"}:
            return False
        if "accepted_for_training" in source and str(source["accepted_for_training"]).lower() not in {
            "1", "true", "yes", "y", "on",
        }:
            return False
    return True

def append_segment(
    segments: list[dict[str, Any]],
    source: str,
    start_step: int,
    end_step: int,
) -> None:
    """Record a contiguous control segment when it has at least one step."""
    if int(end_step) > int(start_step):
        segments.append(
            {
                "source": source,
                "start_step": int(start_step),
                "end_step": int(end_step),
            }
        )

def finalize_episode(
    task_env,
    output_dir: Path,
    episode_index: int,
    instruction: str,
    *,
    save: bool,
    supervisor_label: str,
    segments: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Persist or discard the raw recorded trajectory after supervisor review.

    Collection deliberately stops at the raw stage: frames stay as the native
    per-frame pkl cache and metadata stays in ``episode.json``. HDF5, MP4 and
    LeRobot conversion happen later in ``export_hg_dagger_dataset.py`` so the
    control loop never pays for encoding.
    """
    task_env.episode_metadata["control_mask"] = list(task_env.control_mask)
    task_env.episode_metadata["segments"] = segments
    task_env.episode_metadata["supervisor_label"] = supervisor_label
    task_env.episode_metadata["saved"] = bool(save)
    task_env.info["task_metadata"] = task_env.episode_metadata
    task_env.info["control_mask"] = list(task_env.control_mask)
    task_env.info["segments"] = segments
    task_env.info["supervisor_label"] = supervisor_label
    task_env.info["saved"] = bool(save)

    if not save:
        discard_recovery_cache(task_env)
        return None

    if int(task_env.FRAME_IDX) < 2:
        raise RuntimeError("Episode produced fewer than two recorded frames; cannot save.")

    raw_dir = output_dir / "raw" / f"episode_{episode_index:07d}"
    if raw_dir.exists():
        raise FileExistsError(f"Refusing to overwrite saved raw episode: {raw_dir}")
    frames_dir = raw_dir / "frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = getattr(task_env, "folder_path", {}).get("cache")
    if cache and Path(cache).exists():
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        shutil.move(str(cache), str(frames_dir))

    write_json(
        raw_dir / "episode.json",
        {
            "episode_index": int(episode_index),
            "instruction": instruction,
            "seed": int(getattr(task_env, "episode_seed", -1)),
            "supervisor_label": supervisor_label,
            "save_freq": int(task_env.save_freq or 15),
            "frequency": int(task_env.save_freq or 15),
            "control_mask": list(task_env.control_mask),
            "segments": segments,
            "episode_metadata": task_env.episode_metadata,
            "info": task_env.info,
            "record": extra or {},
        },
    )

    append_jsonl(
        output_dir / "episodes.jsonl",
        {
            "episode_index": episode_index,
            "saved": True,
            "supervisor_label": supervisor_label,
            "control_mask": list(task_env.control_mask),
            "segments": segments,
            "raw_dir": str(raw_dir),
            "hil_frames": (extra or {}).get("hil_frames"),
            "rollout_seconds": (extra or {}).get("rollout_seconds"),
            "expert_success": ((extra or {}).get("expert_result") or {}).get("success"),
            "expert_branch": ((extra or {}).get("expert_result") or {}).get("branch"),
            "intervention_count": (extra or {}).get("intervention_count"),
        },
    )
    return raw_dir

def discard_recovery_cache(task_env) -> None:
    cache = getattr(task_env, "folder_path", {}).get("cache")
    if cache and Path(cache).exists():
        task_env.remove_data_cache()
