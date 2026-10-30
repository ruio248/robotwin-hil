#!/usr/bin/env python3
"""Read-only offline/online parity audit on trusted saved RGB/chunk observations.

No policy RPCs, simulator, or live collection. Only the requested new JSON report
is written; existing reports are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from coverage_sampling.calibration import file_sha256
from coverage_sampling.core import DecisionSampler, config_record
from coverage_sampling.critic import create_online_scorer
from coverage_sampling.options import config_from_args
from coverage_benchmark.common import weights as offline_weights
from coverage_benchmark.scorers import NeuralScorer
from tail_value.cache import CAMERAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--encoder-weights", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.observations < 1: raise ValueError("Require at least one observation")
    torch.set_num_threads(2)
    config_args = dict(es_mode="enhanced", task_name="handover_to_tray", es_window_start=0, es_window_end=0,
                       es_scorer="discriminator_chunk", es_critic=args.fit_root / "disc_s42.pt",
                       es_encoder_weights=args.encoder_weights, es_device=args.device,
                       es_calibration=args.fit_root / "calibration.json", es_calibration_key="disc_s42")
    cfg = config_from_args(config_args)
    online = create_online_scorer(config_args, cfg)
    offline = NeuralScorer(config_args["es_critic"], args.device)
    calibration = json.loads(config_args["es_calibration"].read_text())["disc_s42"]
    manifest = json.loads((args.candidate_cache / "manifest.json").read_text())
    if manifest.get("complete") is not True: raise ValueError("Candidate cache is incomplete")
    sources = manifest["observations"]
    selected = np.linspace(0, len(sources) - 1, min(args.observations, len(sources)), dtype=int)
    if not len(selected): raise ValueError("No saved observations")
    encoder = online.encoder
    encode_calls = [0]
    def counted_encoder(images):
        encode_calls[0] += 1
        return encoder(images)
    online.encoder = counted_encoder
    rows = []
    for index in selected:
        source = sources[int(index)]
        cache = args.candidate_cache / source["cache"]
        if file_sha256(cache) != manifest["completed"][source["cache"]]: raise ValueError("Changed candidate cache")
        with np.load(cache, allow_pickle=False) as d:
            chunks, state, order = d["candidates"], d["state"], d["subset_order"]
        frame = Path(source["frame_path"])
        if frame.stat().st_size != source["source_size"]: raise ValueError("Changed source observation")
        with frame.open("rb") as stream: obs = pickle.load(stream)  # trusted project collection
        np.testing.assert_allclose(obs["joint_action"]["vector"], state, atol=1e-6, rtol=0)
        images = {key: obs["observation"][camera]["rgb"] for key, camera in zip(
            CAMERAS, ("head_camera", "left_camera", "right_camera"))}
        features = encoder(images)
        expected = offline.score(features, state, chunks)
        before = encode_calls[0]
        bound = online.bind_observation(obs)
        actual = bound(chunks)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
        np.testing.assert_array_equal(bound(chunks), actual)
        np.testing.assert_allclose(bound(chunks[::-1]), actual[::-1], rtol=0, atol=1e-5)
        if encode_calls[0] - before != 1: raise AssertionError("Repeated scoring re-encoded the observation")
        candidates = chunks[order[:4]]
        before = encode_calls[0]
        blend, record = DecisionSampler(cfg, None, 42).select(
            0, lambda: None, sample_candidates=lambda k: candidates,
            score_candidates=online.bind_observation(obs))
        if encode_calls[0] - before != 1: raise AssertionError("Candidates/blend must share one encoding")
        candidate_scores = offline.score(features, state, candidates)
        expected_weights = offline_weights(candidate_scores, calibration)
        np.testing.assert_allclose(record["weights"], expected_weights, rtol=0, atol=1e-7)
        expected_blend = np.tensordot(expected_weights, candidates.astype(np.float64), axes=(0, 0)).astype(np.float32)
        np.testing.assert_array_equal(blend, expected_blend)
        expected_blend_score = float(offline.score(features, state, blend[None])[0])
        np.testing.assert_allclose(record["blended_chunk_score"], expected_blend_score, rtol=0, atol=1e-5)
        rows.append(dict(cache=source["cache"], source_sha256=file_sha256(frame),
                         max_score_error=float(np.max(np.abs(actual - expected))),
                         weights=record["weights"], ess_fraction=record["ess_fraction"],
                         score_range=record["score_range"], blended_chunk_score=record["blended_chunk_score"],
                         weighted_candidate_score=record["weighted_candidate_score"],
                         joint_rms_shift_from_uniform=record["joint_rms_shift_from_uniform"],
                         selection_seconds=record["selection_seconds"], visual_encodings_per_decision=1))
    report = dict(complete=True, observations=len(rows), device=args.device, atol=1e-5,
                  sampling_config=config_record(cfg), scorer=online.metadata, rows=rows,
                  code_sha256={name: file_sha256(Path(__file__).parent / name) for name in (
                      "verify_online_chunk_discriminator.py", "coverage_sampling/core.py",
                      "coverage_sampling/critic.py", "coverage_sampling/options.py", "coverage_sampling/calibration.py")},
                  candidate_manifest_sha256=file_sha256(args.candidate_cache / "manifest.json"),
                  interpretation="Parity audit only; no policy inference, action execution, or success-rate trial")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"complete": True, "observations": len(rows),
                      "max_score_error": max(r["max_score_error"] for r in rows), "output": str(args.output)}))


if __name__ == "__main__": main()
