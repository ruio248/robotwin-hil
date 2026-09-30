"""Sampling logic independent of SAPIEN, Torch and the policy server."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    mode: str
    window_start: int
    window_end: int
    num_candidates: int = 4
    horizon: int = 10
    beta: float = 10.0
    seed: int = 42
    replay_atol: float = 1e-4
    low_threshold: float | None = None
    action_aggregation: str = "weighted_mean"

    def __post_init__(self):
        if self.mode not in {"vanilla", "enhanced"}:
            raise ValueError("mode must be vanilla or enhanced")
        for name in ("window_start", "window_end", "num_candidates", "horizon", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.window_start < 0 or self.window_end < self.window_start:
            raise ValueError("Require 0 <= window_start <= window_end (inclusive, zero-based)")
        if not 2 <= self.num_candidates <= 8 or self.horizon < 1 or self.seed < 0:
            raise ValueError("Require 2 <= N <= 8, H >= 1 and seed >= 0")
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if not np.isfinite(self.replay_atol) or self.replay_atol <= 0:
            raise ValueError("replay_atol must be finite and positive")
        if self.low_threshold is not None and not np.isfinite(self.low_threshold):
            raise ValueError("low_threshold must be finite")
        if self.action_aggregation != "weighted_mean":
            raise ValueError("Only weighted_mean action aggregation is supported")

    def active(self, decision):
        return self.window_start <= decision <= self.window_end


class ManualSamplingWindow:
    """Arm a bounded sampling window at a real policy decision with a key press."""

    def __init__(self, duration: int):
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
            raise ValueError("Manual sampling duration must be a positive integer")
        self.duration = duration
        self.start: int | None = None

    def active(self, decision: int) -> bool:
        return self.start is not None and self.start <= decision < self.start + self.duration

    def toggle(self, decision: int) -> bool:
        """Return True when this press arms the window, False when it cancels it."""
        if self.active(decision):
            self.start = None
            return False
        self.start = decision
        return True


def coverage_weights(scores, beta):
    """Self-normalized importance weights; the proposal is already the policy."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Require a nonempty finite score vector")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if beta == 0:
        return np.full(len(scores), 1.0 / len(scores))
    # Common shift cancels algebraically. Negative infinity is valid here:
    # an extremely large positive gap simply receives zero mass.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-beta * (scores - scores.min()))
    return weights / weights.sum()


def validated_chunk(chunk, horizon):
    chunk = np.array(chunk, dtype=np.float32, copy=True)
    if chunk.shape != (horizon, 14) or not np.isfinite(chunk).all():
        raise ValueError(f"Expected finite absolute-joint chunk [{horizon},14], got {chunk.shape}")
    return chunk


class DecisionSampler:
    """Draw policy chunks and blend them using same-observation action scores.

    The optional rollout backend remains for reproducing older diagnostics;
    online callers provide score_candidates to score only each first action.
    """
    def __init__(self, config, rollout, episode_seed):
        self.config, self.rollout = config, rollout
        self.checked_replay = False

    def select(self, decision, sample_chunk, *, sample_candidates, score_candidates=None,
               active_override=None):
        start = time.monotonic()
        cfg = self.config
        active = cfg.active(decision) if active_override is None else bool(active_override)
        count = cfg.num_candidates if active else 1
        if active:
            candidates = np.asarray(sample_candidates(count), dtype=np.float32)
            if candidates.shape != (count, cfg.horizon, 14) or not np.isfinite(candidates).all():
                raise ValueError(f"Expected finite batched candidates [{count},{cfg.horizon},14], "
                                 f"got {candidates.shape}")
            candidates = candidates.copy()
        else:
            candidates = np.stack([validated_chunk(sample_chunk(), cfg.horizon)])
        sampling_seconds = time.monotonic() - start
        record = {"decision": int(decision), "active": active, "mode": cfg.mode,
                  "num_candidates": count, "candidates": candidates.tolist(),
                  "candidate_generation": "batched" if active else "single",
                  "policy_inference_requests": 1,
                  "sampling_seconds": sampling_seconds}
        if active:
            if score_candidates is not None:
                # C(o_t, a_t) already contains the TD-trained continuation
                # estimate. No hypothetical physics transition is needed.
                scores = np.asarray(score_candidates(candidates[:, 0, :].copy()), dtype=np.float64)
                if scores.shape != (count,) or not np.isfinite(scores).all():
                    raise ValueError(f"Expected {count} finite current-action coverage scores")
                score_basis, branches, replay = "current_observation_first_action", [], None
            else:
                if self.rollout is None:
                    raise ValueError("Active sampling requires a current-action scorer or legacy rollout backend")
                branches, replay = self.rollout.evaluate(candidates, verify=not self.checked_replay,
                                                         atol=cfg.replay_atol)
                self.checked_replay = True
                if len(branches) != count:
                    raise ValueError("Rollout returned the wrong candidate count")
                scores = []
                for branch in branches:
                    curve = np.asarray(branch["coverage"], dtype=np.float64)
                    if curve.ndim != 1 or not 1 <= len(curve) <= cfg.horizon or not np.isfinite(curve).all():
                        raise ValueError("Every branch needs 1..H finite coverage scores")
                    scores.append(float(curve.min()))
                score_basis = "legacy_branch_min"
            scoring_seconds = time.monotonic() - start - sampling_seconds
            beta = cfg.beta if cfg.mode == "enhanced" else 0.0
            weights = coverage_weights(scores, beta)
            # Keep one weight per whole chunk. Only the first action is scored;
            # later chunk actions are carried along, not evaluated in advance.
            action_chunk = np.tensordot(weights, candidates.astype(np.float64),
                                        axes=(0, 0)).astype(np.float32)
            record.update(score_basis=score_basis, score_action_index=0,
                          scoring_seconds=scoring_seconds, branches=branches,
                          scores=np.asarray(scores).tolist(), weights=weights.tolist(),
                          action_aggregation="weighted_mean", selected=None,
                          selected_score=None,
                          weighted_candidate_score=float(np.dot(weights, scores)),
                          blended_chunk=action_chunk.tolist(),
                          uniform_expected_score=float(np.mean(scores)),
                          effective_sample_size=float(1.0 / np.square(weights).sum()),
                          replay_check=replay)
            if score_basis == "legacy_branch_min":
                record["weighted_candidate_min"] = record["weighted_candidate_score"]
        else:
            action_chunk = candidates[0].copy()
            record.update(action_aggregation="single", selected=0,
                          weights=[1.0], branches=[], scores=[])
        record["selection_seconds"] = time.monotonic() - start
        return action_chunk, record


class EpisodeLog:
    """Append decisions before execution; flush real steps so failures are visible."""
    def __init__(self, path, config, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8")
        self.config = config
        self.active_decisions = 0
        self.active_scores = []
        self.write({"event": "episode_start", "config": asdict(config), **metadata})

    def write(self, record):
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()

    def decision(self, record):
        self.active_decisions += int(record["active"])
        self.write({"event": "decision", **record})

    def executed(self, decision, action_index, coverage, active, action, **extra):
        if coverage is None and active and action_index == 0:
            raise ValueError("Active sampling decisions require a real coverage score for the first action")
        coverage = float(coverage) if coverage is not None else None
        if active and coverage is not None:
            self.active_scores.append(coverage)
        self.write({"event": "executed", "decision": decision, "action_index": action_index,
                    "coverage": coverage, "active": active, "action": np.asarray(action).tolist(), **extra})

    def finish(self, success, error, rollout_steps):
        try:
            scores = self.active_scores
            threshold = self.config.low_threshold
            self.write({"event": "episode_end", "success": bool(success), "error": error,
                        "rollout_steps": rollout_steps, "active_decisions": self.active_decisions,
                        "window_reached": self.active_decisions > 0,
                        "active_executed_scores": len(scores),
                        "active_min": min(scores) if scores else None,
                        "active_mean": float(np.mean(scores)) if scores else None,
                        "low_threshold": threshold,
                        "active_low_fraction": float(np.mean(np.array(scores) < threshold))
                        if scores and threshold is not None else None,
                        "active_low_hit": bool(min(scores) < threshold)
                        if scores and threshold is not None else None})
        finally:
            self.stream.close()
