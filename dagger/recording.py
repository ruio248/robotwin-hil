"""Raw-frame persistence shared by interactive DAgger collectors."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


# Both offline HIL pipelines require two within-controller action transitions.
MIN_HIL_SEGMENT_TRANSITIONS = 2


def assess_operator_hil_segments(control_mask: list[str], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conservatively accept only recorded, successful expert-motion runs.

    A run with a failed/unknown motion is never partially promoted. This is
    intentionally opt-in for cup/QR; old raw keeps its existing eligibility.
    """
    spans = []
    start = None
    for index, source in enumerate([*control_mask, "policy"]):
        if str(source).lower() == "hil" and start is None:
            start = index
        elif str(source).lower() != "hil" and start is not None:
            stop = index
            overlapping = [e for e in events if e.get("phase") == "stage"
                           and e.get("frame_start") is not None
                           and e.get("frame_end_exclusive") is not None
                           and start <= int(e["frame_start"]) < stop]
            covered = any(e.get("substage") in {"grasp", "lift", "place_and_release", "release", "retreat"}
                          and int(e["frame_end_exclusive"]) > int(e["frame_start"])
                          for e in overlapping)
            good = bool(stop - start - 1 >= MIN_HIL_SEGMENT_TRANSITIONS
                        and covered and all(e.get("ok") and e.get("joints_legal") for e in overlapping))
            spans.append({"start": start, "end_exclusive": stop, "accepted": good,
                          "motion_events": len(overlapping),
                          "reason": None if good else "short_missing_or_failed_expert_motion"})
            start = None
    return spans


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
    """Keep legacy compatibility, but require explicit acceptance for new strict runs."""
    if (episode_dir / "DIAGNOSTIC_ONLY.json").exists():
        return False
    record = meta.get("record") or {}
    if record.get("formal_training_schema") in {1, 2} and record.get("accepted_for_training") is not True:
        return False
    for source in (meta, meta.get("episode_metadata") or {}, meta.get("record") or {}):
        if str(source.get("diagnostic_only", False)).lower() in {"1", "true", "yes", "y", "on"}:
            return False
        if "accepted_for_training" in source and str(source["accepted_for_training"]).lower() not in {
            "1", "true", "yes", "y", "on",
        }:
            return False
    return True


def accepted_hil_spans(meta: dict[str, Any], candidate_runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Schema-2 exports use exact audited spans, never all HIL runs by default."""
    record = meta.get("record") or {}
    if record.get("formal_training_schema") != 2:
        return candidate_runs
    accepted = {(int(row["start"]), int(row["end_exclusive"]))
                for row in record.get("accepted_hil_segments") or [] if row.get("accepted") is True}
    if not accepted.issubset(set(candidate_runs)):
        raise ValueError("Accepted HIL spans do not match recorded control_mask")
    if record.get("accepted_for_training") is True and not accepted:
        raise ValueError("Schema-2 accepted episode has no accepted HIL span")
    return [span for span in candidate_runs if span in accepted]


def has_explicit_training_acceptance(meta: dict[str, Any]) -> bool:
    return any("accepted_for_training" in source for source in
               (meta, meta.get("episode_metadata") or {}, meta.get("record") or {}))


def summarize_takeover_metrics(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count distinct operator requests, expert attempts, and final outcomes."""
    requests = [event for record in records for event in record.get("takeover_request_events", [])]
    interventions = [item for record in records for item in record.get("interventions", [])]
    attempts = [result for item in interventions for result in
                (item.get("attempt_results") or ([item["expert_result"]] if item.get("expert_result") else []))
                if isinstance(result, dict)]
    return {
        "takeover_requests": len(requests),
        "takeover_requests_accepted": sum(event.get("accepted") is True for event in requests),
        "stage_selection_rejections": sum(len(record.get("recovery_selection_rejections", [])) for record in records),
        "expert_attempts_with_result": len(attempts),
        "expert_planning_successes": sum(result.get("plan_success") is True for result in attempts),
        "expert_execution_successes": sum(result.get("success") is True for result in attempts),
        "final_task_successes": sum(record.get("final_check_success") is True for record in records),
    }

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
