"""Recover completed HG-DAgger collection progress from durable raw records."""

from __future__ import annotations

import fcntl
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


RAW_NAME = re.compile(r"episode_(\d+)$")
SAMPLING_NAME = re.compile(r"episode_(\d+)_seed_(\d+)\.jsonl$")


@dataclass
class CollectionProgress:
    next_rollout_index: int
    next_episode_index: int
    saved_valid_hil: int
    aborted_rollouts: int
    records: list[dict[str, Any]]
    used_seeds: set[int]
    prior_seconds: float
    source_report: str | None


def collection_lock(output_dir: Path):
    """Hold this file object until collection exits to prevent two writers."""
    handle = (output_dir / ".collection.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"Another collector is using {output_dir}") from exc
    return handle


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _raw_episodes(output_dir: Path) -> dict[int, dict[str, Any]]:
    episodes: dict[int, dict[str, Any]] = {}
    for path in sorted((output_dir / "raw").glob("episode_*")):
        match = RAW_NAME.fullmatch(path.name)
        if not match or not path.is_dir():
            raise ValueError(f"Unexpected raw episode path: {path}")
        index = int(match.group(1))
        if index in episodes:
            raise ValueError(f"Duplicate raw episode index {index}")
        metadata = path / "episode.json"
        if not metadata.is_file():
            raise ValueError(f"Incomplete raw episode; inspect before resuming: {path}")
        data = _read_object(metadata)
        frames = path / "frames"
        if int(data.get("episode_index", -1)) != index or not frames.is_dir() or not any(frames.iterdir()):
            raise ValueError(f"Invalid raw episode metadata or missing frames: {path}")
        record = data.get("record", {})
        if not isinstance(record, dict) or record.get("save_decision") is not True:
            raise ValueError(f"Raw episode lacks a confirmed save decision: {path}")
        episodes[index] = data
    return episodes


def _saved_index(output_dir: Path) -> set[int]:
    path = output_dir / "episodes.jsonl"
    if not path.is_file():
        return set()
    indexes: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        entry = json.loads(line)
        index = int(entry["episode_index"])
        if entry.get("saved") is not True or index in indexes:
            raise ValueError(f"Invalid or duplicate saved episode at {path}:{line_number}")
        indexes.add(index)
    return indexes


def _sampling_logs(output_dir: Path):
    for path in sorted((output_dir / "sampling").glob("episode_*.jsonl")):
        match = SAMPLING_NAME.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected sampling log: {path}")
        outcome = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A hard interruption may leave the last JSONL line partial.
                break
            if row.get("event") == "hil_outcome":
                outcome = row
        yield int(match.group(1)), int(match.group(2)), outcome


def inspect_collection(output_dir: Path, *, target_mode: str, expected: dict[str, Any] | None = None,
                       resume: bool = False) -> CollectionProgress:
    """Inspect durable state without changing files; refuse ambiguous recovery."""
    raw = _raw_episodes(output_dir)
    saved_index = _saved_index(output_dir)
    reports = sorted(output_dir.glob("session_*.json"))
    logs = list(_sampling_logs(output_dir))
    if not resume:
        unfinished = any((output_dir / ".cache").glob("episode*"))
        if raw or saved_index or reports or logs or unfinished:
            raise ValueError(f"Collection data already exists in {output_dir}; use --resume")
        return CollectionProgress(0, 0, 0, 0, [], set(), 0.0, None)
    if not reports:
        raise ValueError(f"No completed session report in {output_dir}; cannot restore rollout history")
    if set(raw) != saved_index:
        raise ValueError("raw episodes and episodes.jsonl disagree; inspect before resuming")

    report_path = reports[-1]
    report = _read_object(report_path)
    for name, value in (expected or {}).items():
        if report.get(name) != value:
            raise ValueError(f"Resume configuration mismatch for {name}: {report.get(name)!r} != {value!r}")
    records = list(report.get("records", []))
    if not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Invalid records in {report_path}")
    by_rollout = {int(row["rollout_index"]): row for row in records}
    if len(by_rollout) != len(records):
        raise ValueError(f"Duplicate rollout index in {report_path}")
    used_seeds = {int(row["seed"]) for row in records if row.get("seed") is not None}

    valid_hil = 0
    for episode in raw.values():
        record = episode["record"]
        index = int(record["rollout_index"])
        existing = by_rollout.get(index)
        if existing is not None and existing.get("save_decision") is not True:
            raise ValueError(f"Saved raw episode conflicts with rollout {index}")
        if existing is not None and int(existing.get("seed", -1)) != int(episode["seed"]):
            raise ValueError(f"Seed mismatch in saved raw episode for rollout {index}")
        if existing is None:
            by_rollout[index] = record
        used_seeds.add(int(episode["seed"]))
        hil_frames = int(record.get("hil_frames") or sum(
            str(source).lower() == "hil" for source in episode.get("control_mask", [])))
        if hil_frames > 0 and (target_mode != "expert" or
                               bool((record.get("expert_result") or {}).get("success"))):
            valid_hil += 1

    max_log_index = -1
    for index, seed, outcome in logs:
        max_log_index = max(max_log_index, index)
        used_seeds.add(seed)
        if index not in by_rollout and outcome and outcome.get("status") in {"completed", "aborted"}:
            by_rollout[index] = {
                "rollout_index": index, "seed": seed, "rollout_status": outcome["status"],
                "save_decision": False, "policy_steps": int(outcome.get("policy_steps", 0)),
                "manual_takeover_requests": int(outcome.get("manual_takeover_requests", 0)),
                "intervention_count": int(outcome.get("interventions", 0)),
                "reconstructed_from_log": True,
            }
    if int(report.get("saved_episodes", 0)) > len(raw):
        raise ValueError("Session report references saved episodes missing from raw data")
    if int(report.get("saved_hil_episodes", 0)) > valid_hil:
        raise ValueError("Session report references valid HIL episodes missing from raw data")
    hdf5_indexes = [int(match.group(1)) for path in (output_dir / "data").glob("episode_*.hdf5")
                    if (match := RAW_NAME.fullmatch(path.stem))]
    next_episode = max([*raw, *hdf5_indexes], default=-1) + 1
    next_rollout = max([*by_rollout, max_log_index], default=-1) + 1
    all_records = [by_rollout[index] for index in sorted(by_rollout)]
    aborted = max(int(report.get("aborted_rollouts", 0)),
                  sum(row.get("rollout_status") == "aborted" for row in all_records))
    return CollectionProgress(next_rollout, next_episode, valid_hil, aborted, all_records,
                              used_seeds, float(report.get("total_seconds", 0.0)), str(report_path))


def preserve_incomplete_cache(output_dir: Path) -> list[Path]:
    """Move unfinished frame caches aside without deleting or counting them."""
    preserved = []
    cache_root = output_dir / ".cache"
    for path in sorted(cache_root.glob("episode*")):
        if not path.is_dir():
            continue
        destination_root = output_dir / "interrupted_cache"
        destination_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = destination_root / f"{path.name}_{stamp}"
        suffix = 1
        while destination.exists():
            destination = destination_root / f"{path.name}_{stamp}_{suffix}"
            suffix += 1
        path.rename(destination)
        preserved.append(destination)
    return preserved
