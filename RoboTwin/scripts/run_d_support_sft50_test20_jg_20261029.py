#!/usr/bin/env python3
"""Run the established 10k ChunkCoverageCritic trainer on a validated 50/20 set."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> None:
    base = Path(os.environ["D_SUPPORT_BASE"]).resolve()
    code = Path(os.environ["D_SUPPORT_CODE"]).resolve()
    trainer_path = Path(os.environ["D_SUPPORT_TRAINER"]).resolve()
    if not (base / "data/dataset_manifest.json").is_file():
        raise FileNotFoundError(base / "data/dataset_manifest.json")
    if not trainer_path.is_file():
        raise FileNotFoundError(trainer_path)
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location("d_support_trainer_module", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trainer {trainer_path}")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original_read_episode = trainer.base.read_episode

    def read_episode_with_verified_stage_status(root, meta_path):
        episode = original_read_episode(root, meta_path)
        if episode["stage_verified_anchors"] == 0:
            episode["stage_annotation_mode"] = "unverified_alignment_no_source_stage_labels"
        return episode

    trainer.base.read_episode = read_episode_with_verified_stage_status
    trainer.BASE = base
    trainer.DATA = base / "data"
    trainer.RESULTS = base / "results"
    trainer.CONTROL = base / "control"
    trainer.CODE = code
    trainer.PY = Path(sys.executable)
    trainer.main()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        traceback.print_exc()
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}), flush=True)
        raise
