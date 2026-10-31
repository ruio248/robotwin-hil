#!/usr/bin/env python3
"""Collect handover_block experts with the pinned, repaired RoboTwin instructions.

The native collector and simulator are reused without editing the shared RoboTwin
checkout. Only instruction selection and its provenance metadata are overridden.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


COLLECTOR = Path(
    "/media/ruio/hdd/place_empty_cup_demo_source_ablation_20261028/"
    "code/collect_full_experts.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_complete(prompt: str) -> None:
    text = prompt.casefold()
    if not all(part in text for part in ("left", "right", "blue pad")):
        raise ValueError(f"Instruction omits a required stage: {prompt!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if args.output.parent == args.old_manifest.parent or args.output in args.old_manifest.parents:
        raise ValueError("New output must be separate from the old expert dataset")

    template = json.loads(args.instructions.read_text())
    prompts = template["seen"]
    if len(prompts) != 50 or len(set(prompts)) != 50:
        raise ValueError("Expected exactly 50 distinct official seen templates")
    for prompt in prompts + template["unseen"]:
        require_complete(prompt)
    if len(template["unseen"]) != 10:
        raise ValueError("Expected 10 official unseen templates")
    old = json.loads(args.old_manifest.read_text())
    if not old.get("complete") or len(old["episodes"]) != 50:
        raise ValueError("Old dataset manifest is not complete")

    root = args.robotwin.resolve()
    for path in (root, root / "pace/src", root / "description/utils"):
        sys.path.insert(0, str(path))
    from pace import robotwin as adapter

    source = importlib.util.spec_from_file_location("native_expert_collector", COLLECTOR)
    if source is None or source.loader is None:
        raise RuntimeError(f"Cannot load {COLLECTOR}")
    collector = importlib.util.module_from_spec(source)
    source.loader.exec_module(collector)
    collector.TASK = "handover_block"
    instruction_sha = sha256(args.instructions)
    prompt_calls: list[str] = []

    def fixed_seen_instruction(task: str, episode_info: dict, count: int, split: str) -> str:
        # The frozen collector requests 'unseen'. This *experiment* deliberately
        # substitutes the official fixed 'seen' split, one distinct template per
        # accepted episode. The metadata below records the actual split.
        if task != "handover_block" or count != 1 or split != "unseen":
            raise ValueError((task, count, split))
        index = len(prompt_calls)
        if index >= 50:
            raise ValueError("More than 50 accepted episodes")
        prompt = prompts[index]
        require_complete(prompt)
        prompt_calls.append(prompt)
        return prompt

    adapter.sample_instruction = fixed_seen_instruction
    original_save = collector.save_episode

    def save_with_provenance(folder, frames, scene, split, index, prompt, info, stages):
        if index != len(prompt_calls) - 1 or prompt != prompts[index]:
            raise ValueError("Prompt-to-episode index mismatch")
        record = original_save(folder, frames, scene, split, index, prompt, info, stages)
        metadata_path = Path(folder) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["prompt_scheme"] = "RoboTwin PR #510 fixed seen; template index = episode index"
        metadata["prompt_split"] = "seen"
        metadata["prompt_template_index"] = index
        metadata["prompt_template_sha256"] = instruction_sha
        collector.write(metadata_path, metadata)
        checksums_path = Path(folder) / "files.sha256.json"
        checksums = json.loads(checksums_path.read_text())
        checksums["metadata.json"] = sha256(metadata_path)
        collector.write(checksums_path, checksums)
        record["manifest_sha256"] = sha256(checksums_path)
        return record

    collector.save_episode = save_with_provenance
    sys.argv = [str(COLLECTOR), "--robotwin", str(root), "--output", str(args.output)]
    if args.smoke:
        sys.argv.append("--smoke")
    collector.main()

    manifest = json.loads((args.output / "manifest.json").read_text())
    expected = 1 if args.smoke else 50
    records = manifest["episodes"]
    if not manifest.get("complete") or len(records) != expected:
        raise ValueError("Collector did not complete the expected number of episodes")
    seen = set()
    distribution = Counter()
    for index, record in enumerate(records):
        folder = Path(record["folder"])
        metadata = json.loads((folder / "metadata.json").read_text())
        prompt = metadata["prompt"]
        require_complete(prompt)
        if prompt != prompts[index] or metadata["prompt_template_index"] != index:
            raise ValueError(f"Prompt mismatch in episode {index}")
        if metadata["prompt_split"] != "seen" or not metadata["expert_success"]:
            raise ValueError(f"Invalid provenance or expert outcome in episode {index}")
        seed = int(metadata["scene_seed"])
        if seed in seen or seed in old["excluded_test_scene_seeds"]:
            raise ValueError(f"Duplicate or held-out scene seed: {seed}")
        seen.add(seed)
        distribution[prompt] += 1
    if not args.smoke and set(distribution.values()) != {1}:
        raise ValueError("The 50 official seen prompts are not represented once each")
    old_seeds = {int(record["scene_seed"]) for record in old["episodes"]}
    report = {
        "complete": True,
        "task": "handover_block",
        "episodes": len(records),
        "unique_scene_seeds": len(seen),
        "expert_successes": len(records),
        "complete_prompt_count": sum(distribution.values()),
        "unique_prompts": len(distribution),
        "prompt_split": "official_fixed_seen",
        "official_fix_pr": "https://github.com/RoboTwin-Platform/RoboTwin/pull/510",
        "official_instruction_sha256": instruction_sha,
        "old_manifest_sha256": sha256(args.old_manifest),
        "new_manifest_sha256": sha256(args.output / "manifest.json"),
        "old_seed_overlap": sorted(seen & old_seeds),
        "held_out_test_seed_overlap": sorted(seen & set(old["excluded_test_scene_seeds"])),
        "distribution": dict(distribution),
        "scene_seeds": sorted(seen),
    }
    report_name = "smoke_validation.json" if args.smoke else "validation.json"
    collector.write(args.output.parent / "control" / report_name, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
