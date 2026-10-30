"""Sampling logic independent of SAPIEN, Torch and the policy server."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np


ARM_JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
GRIPPER_INDICES = np.array([6, 13], dtype=np.int64)


@dataclass(frozen=True)
class SamplingConfig:
    mode: str
    window_start: int
    window_end: int
    num_candidates: int = 4
    policy_microbatch: int = 4
    horizon: int = 10
    beta: float = 10.0
    seed: int = 42
    replay_atol: float = 1e-4
    low_threshold: float | None = None
    action_aggregation: str = "weighted_mean"
    score_representation: str = "first_action"
    score_scale: float = 1.0
    uniform_weights: bool = False
    calibration_sha256: str | None = None
    calibration_key: str | None = None
    gaussian_sigma: float | None = None

    def __post_init__(self):
        if self.mode not in {"vanilla", "enhanced", "gaussian"}:
            raise ValueError("mode must be vanilla, enhanced or gaussian")
        for name in ("window_start", "window_end", "num_candidates", "policy_microbatch", "horizon", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.window_start < 0 or self.window_end < self.window_start:
            raise ValueError("Require 0 <= window_start <= window_end (inclusive, zero-based)")
        min_candidates = 1 if self.mode == "gaussian" else 2
        if (not min_candidates <= self.num_candidates <= 32 or not 2 <= self.policy_microbatch <= 32
                or self.horizon < 1 or self.seed < 0):
            raise ValueError(
                f"Require {min_candidates} <= N <= 32, 2 <= policy_microbatch <= 32, "
                "H >= 1 and seed >= 0"
            )
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if not np.isfinite(self.replay_atol) or self.replay_atol <= 0:
            raise ValueError("replay_atol must be finite and positive")
        if self.low_threshold is not None and not np.isfinite(self.low_threshold):
            raise ValueError("low_threshold must be finite")
        if self.action_aggregation != "weighted_mean":
            raise ValueError("Only weighted_mean action aggregation is supported")
        if self.score_representation not in {"first_action", "full_chunk"}:
            raise ValueError("Unknown score representation")
        if not np.isfinite(self.score_scale) or self.score_scale < 0 or (
                self.score_scale == 0 and not self.uniform_weights):
            raise ValueError("Require a positive score scale unless calibration is degenerate")
        if self.score_representation == "full_chunk" and self.horizon != 10:
            raise ValueError("The full-chunk discriminator requires horizon=10")
        if self.mode == "gaussian":
            if (self.gaussian_sigma is None or not np.isfinite(self.gaussian_sigma)
                    or self.gaussian_sigma < 0):
                raise ValueError("Gaussian mode requires a finite --es-gaussian-sigma >= 0")
        elif self.gaussian_sigma is not None:
            raise ValueError("gaussian_sigma is only valid in gaussian mode")

    @property
    def score_basis(self):
        if self.mode == "gaussian":
            return "none_output_action_isotropic_gaussian"
        return ("current_observation_full_chunk" if self.score_representation == "full_chunk"
                else "current_observation_first_action")

    @property
    def requires_scores(self):
        return self.mode in {"vanilla", "enhanced"}

    def active(self, decision):
        return self.window_start <= decision <= self.window_end


def config_record(config):
    """Keep legacy default records identical, including resumable collections."""
    record = asdict(config)
    if config.mode == "gaussian":
        record["action_aggregation"] = (
            "single_candidate_plus_isotropic_gaussian" if config.num_candidates == 1
            else "uniform_mean_plus_isotropic_gaussian"
        )
        record["gaussian_noise_space"] = "absolute_joint_chunk_output"
        record["gaussian_noise_grippers"] = False
        record["gaussian_covariance"] = "sigma^2 * I over H*12 arm-joint entries"
    if (config.score_representation == "first_action" and config.score_scale == 1
            and not config.uniform_weights and config.calibration_sha256 is None
            and config.calibration_key is None):
        for key in ("score_representation", "score_scale", "uniform_weights",
                    "calibration_sha256", "calibration_key"):
            record.pop(key)
    # Old K<=4 collections used one request for the whole candidate batch; the
    # default microbatch remains behavior-compatible and need not break resume.
    if config.policy_microbatch == 4 and config.num_candidates <= 4:
        record.pop("policy_microbatch")
    if config.gaussian_sigma is None:
        record.pop("gaussian_sigma")
    return record


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


def coverage_weights(scores, beta, *, scale=1.0, uniform=False):
    """Self-normalized importance weights; the proposal is already the policy."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Require a nonempty finite score vector")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not np.isfinite(scale) or scale < 0 or (scale == 0 and not uniform):
        raise ValueError("Invalid calibration scale")
    if beta == 0 or uniform:
        return np.full(len(scores), 1.0 / len(scores))
    # Common shift cancels algebraically. Negative infinity is valid here:
    # an extremely large positive gap simply receives zero mass.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-beta * (scores - scores.min()) / scale)
    return weights / weights.sum()


def validated_chunk(chunk, horizon):
    chunk = np.array(chunk, dtype=np.float32, copy=True)
    if chunk.shape != (horizon, 14) or not np.isfinite(chunk).all():
        raise ValueError(f"Expected finite absolute-joint chunk [{horizon},14], got {chunk.shape}")
    return chunk


class DecisionSampler:
    """Draw policy chunks and blend them using same-observation action scores.

    The optional rollout backend remains for reproducing older diagnostics;
    Online scoring uses either legacy first actions or complete chunks, explicitly
    selected by config. No hypothetical simulator transitions are required.
    """
    def __init__(self, config, rollout, episode_seed):
        self.config, self.rollout = config, rollout
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, (int, np.integer)):
            raise ValueError("episode_seed must be an integer")
        self.episode_seed = int(episode_seed)
        self.checked_replay = False

    def _isotropic_gaussian_noise(self, decision):
        """Deterministic per-(experiment, episode, decision) output-space noise.

        The 120 arm-joint entries of a H=10 chunk are iid N(0, sigma^2).
        Grippers are deliberately unchanged so their different units and binary-like
        semantics do not contaminate the joint-space ablation.
        """
        cfg = self.config
        seed = np.random.SeedSequence([cfg.seed, self.episode_seed, int(decision)])
        rng = np.random.default_rng(seed)
        noise = np.zeros((cfg.horizon, 14), dtype=np.float32)
        noise[:, ARM_JOINT_INDICES] = rng.normal(
            0.0, cfg.gaussian_sigma, size=(cfg.horizon, len(ARM_JOINT_INDICES))
        ).astype(np.float32)
        return noise

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
                  "candidate_generation": ("microbatched" if active and count > cfg.policy_microbatch
                                           else "batched" if active else "single"),
                  "policy_inference_requests": ((count + cfg.policy_microbatch - 1) // cfg.policy_microbatch
                                                if active else 1),
                  "sampling_seconds": sampling_seconds}
        if active:
            if cfg.mode == "gaussian":
                # Keep the same K-candidate policy proposal and inference budget as
                # enhanced sampling.  Only the direction-selection mechanism changes:
                # uniform candidate mean plus iid action-space Gaussian noise.
                weights = np.full(count, 1.0 / count, dtype=np.float64)
                uniform_chunk = np.tensordot(
                    weights, candidates.astype(np.float64), axes=(0, 0)
                ).astype(np.float32)
                noise = self._isotropic_gaussian_noise(decision)
                action_chunk = uniform_chunk + noise
                arm_noise = noise[:, ARM_JOINT_INDICES].astype(np.float64)
                positive_weights = weights[weights > 0]
                record.update(
                    score_basis=cfg.score_basis,
                    score_action_index=None,
                    scoring_seconds=0.0,
                    branches=[], replay_check=None,
                    output_scoring_seconds=0.0,
                    blended_chunk_score=None,
                    score_scale=None, beta=0.0, uniform_weights=True,
                    score_range=None, score_std=None, scores=[], weights=weights.tolist(),
                    action_aggregation=(
                        "single_candidate_plus_isotropic_gaussian" if count == 1
                        else "uniform_mean_plus_isotropic_gaussian"
                    ),
                    selected=None, selected_score=None,
                    weighted_candidate_score=None,
                    blended_chunk=action_chunk.tolist(),
                    uniform_expected_score=None,
                    effective_sample_size=float(1.0 / np.square(weights).sum()),
                    ess_fraction=float(1.0 / (count * np.square(weights).sum())),
                    max_weight=float(weights.max()),
                    candidate_weight_kl_to_uniform=float(
                        np.sum(positive_weights * np.log(count * positive_weights))
                    ),
                    joint_rms_shift_from_uniform=float(np.sqrt(np.mean(np.square(arm_noise)))),
                    gripper_rms_shift_from_uniform=0.0,
                    gaussian_sigma=float(cfg.gaussian_sigma),
                    gaussian_noise_arm_mean=float(np.mean(arm_noise)),
                    gaussian_noise_arm_std=float(np.std(arm_noise)),
                    gaussian_noise_arm_rms=float(np.sqrt(np.mean(np.square(arm_noise)))),
                    gaussian_noise_arm_max_abs=float(np.max(np.abs(arm_noise))),
                    gaussian_noise_grippers=False,
                    gaussian_noise_space="absolute_joint_chunk_output",
                    gaussian_covariance="sigma^2 * I over H*12 arm-joint entries",
                    gaussian_rng_key=[int(cfg.seed), int(self.episode_seed), int(decision)],
                )
                record["selection_seconds"] = time.monotonic() - start
                return action_chunk, record
            if score_candidates is not None:
                inputs = candidates if cfg.score_representation == "full_chunk" else candidates[:, 0, :]
                scores = np.asarray(score_candidates(inputs.copy()), dtype=np.float64)
                if scores.shape != (count,) or not np.isfinite(scores).all():
                    raise ValueError(f"Expected {count} finite coverage scores")
                score_basis, branches, replay = cfg.score_basis, [], None
            else:
                if cfg.score_representation == "full_chunk":
                    raise ValueError("Full-chunk scoring requires a direct scorer, not branch replay")
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
            weights = coverage_weights(scores, beta, scale=cfg.score_scale, uniform=cfg.uniform_weights)
            # One scalar weight per candidate, shared by every time step and joint.
            action_chunk = np.tensordot(weights, candidates.astype(np.float64),
                                        axes=(0, 0)).astype(np.float32)
            blended_score = None
            if cfg.score_representation == "full_chunk":
                # Evaluate the actual float32 output, not E_w[B(A)]. The bound
                # scorer reuses the same observation features as the candidates.
                blended = np.asarray(score_candidates(action_chunk[None].copy()), dtype=np.float64)
                if blended.shape != (1,) or not np.isfinite(blended).all():
                    raise ValueError("Expected one finite blended chunk score")
                blended_score = float(blended[0])
            output_scoring_seconds = time.monotonic() - start - sampling_seconds - scoring_seconds
            shift = action_chunk.astype(np.float64) - candidates.astype(np.float64).mean(axis=0)
            positive_weights = weights[weights > 0]
            record.update(score_basis=score_basis,
                          score_action_index=None if cfg.score_representation == "full_chunk" else 0,
                          scoring_seconds=scoring_seconds, branches=branches,
                          output_scoring_seconds=output_scoring_seconds,
                          blended_chunk_score=blended_score,
                          score_scale=cfg.score_scale, beta=beta,
                          uniform_weights=cfg.uniform_weights,
                          score_range=float(np.ptp(scores)), score_std=float(np.std(scores)),
                          scores=np.asarray(scores).tolist(), weights=weights.tolist(),
                          action_aggregation="weighted_mean", selected=None,
                          selected_score=None,
                          weighted_candidate_score=float(np.dot(weights, scores)),
                          blended_chunk=action_chunk.tolist(),
                          uniform_expected_score=float(np.mean(scores)),
                          effective_sample_size=float(1.0 / np.square(weights).sum()),
                          ess_fraction=float(1.0 / (count * np.square(weights).sum())),
                          max_weight=float(weights.max()),
                          candidate_weight_kl_to_uniform=float(np.sum(positive_weights * np.log(count * positive_weights))),
                          joint_rms_shift_from_uniform=float(np.sqrt(np.mean(shift[:, ARM_JOINT_INDICES] ** 2))),
                          gripper_rms_shift_from_uniform=float(np.sqrt(np.mean(shift[:, GRIPPER_INDICES] ** 2))),
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
        self.write({"event": "episode_start", "config": config_record(config), **metadata})

    def write(self, record):
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()

    def decision(self, record):
        self.active_decisions += int(record["active"])
        self.write({"event": "decision", **record})

    def executed(self, decision, action_index, coverage, active, action, **extra):
        if coverage is None and active and action_index == 0 and self.config.requires_scores:
            raise ValueError("Active sampling decisions require a real coverage score at execution start")
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
