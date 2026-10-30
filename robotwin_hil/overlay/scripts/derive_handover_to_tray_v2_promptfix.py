"""Create a prompt-corrected LeRobot derivative without copying payload data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


OLD_TASK = "Pass the red bar from the the left arm arm to the the right arm arm and place it in the blue tray."
NEW_TASK = "Pass the red bar from the left arm to the right arm and place it in the blue tray."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_files(root: Path) -> list[Path]:
    return [
        path
        for directory in ("data", "videos")
        for path in sorted((root / directory).rglob("*"))
        if path.is_file()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    args = parser.parse_args()

    source = args.source_root.resolve()
    destination = args.destination_root.resolve()
    temporary = destination.with_name(destination.name + ".creating")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {destination} or {temporary}")
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError("source and destination must be on the same filesystem")

    tasks_path = source / "meta" / "tasks.jsonl"
    episodes_path = source / "meta" / "episodes.jsonl"
    task_rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line]
    episode_rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    if task_rows != [{"task_index": 0, "task": OLD_TASK}]:
        raise ValueError("source tasks.jsonl does not match the expected v1 prompt")
    if not episode_rows or any(row.get("tasks") != [OLD_TASK] for row in episode_rows):
        raise ValueError("source episode metadata does not uniformly use the v1 prompt")

    temporary.mkdir(parents=True)
    for directory in ("data", "videos"):
        shutil.copytree(source / directory, temporary / directory, copy_function=os.link)
    shutil.copytree(source / "meta", temporary / "meta")

    (temporary / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": NEW_TASK}) + "\n", encoding="utf-8"
    )
    with (temporary / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in episode_rows:
            row["tasks"] = [NEW_TASK]
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    source_payload = payload_files(source)
    hardlinked = 0
    payload_bytes = 0
    for source_path in source_payload:
        target_path = temporary / source_path.relative_to(source)
        source_stat = source_path.stat()
        target_stat = target_path.stat()
        if (source_stat.st_dev, source_stat.st_ino) != (target_stat.st_dev, target_stat.st_ino):
            raise RuntimeError(f"payload file is not hard-linked: {source_path}")
        hardlinked += 1
        payload_bytes += source_stat.st_size

    metadata_names = ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source),
        "source_metadata_sha256": {
            name: sha256(source / "meta" / name) for name in metadata_names
        },
        "old_task": OLD_TASK,
        "new_task": NEW_TASK,
        "changed_files": ["meta/tasks.jsonl", "meta/episodes.jsonl"],
        "hardlinked_payload_files": hardlinked,
        "hardlinked_payload_bytes": payload_bytes,
    }
    (temporary / "DERIVATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    os.rename(temporary, destination)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
