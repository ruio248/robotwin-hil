"""DART Gaussian supervisor noise for 12 arm joints (radians).

This module deliberately does not perturb a learned policy.  DART perturbs
the *executed expert* action while training on the expert's clean intention.
Callers must obtain aligned expert/learner actions at the same observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARM_DIMS = 12
FULL_DIMS = 14
ARM_INDICES = np.array([*range(6), *range(7, 13)])


def _as_arm_actions(actions: np.ndarray, name: str) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] not in (ARM_DIMS, FULL_DIMS):
        raise ValueError(f"{name} must have shape [N,12] or [N,14]")
    if not np.isfinite(actions).all():
        raise ValueError(f"{name} contains nonfinite values")
    return actions[:, ARM_INDICES] if actions.shape[1] == FULL_DIMS else actions


def fit_covariance(
    learner_actions: np.ndarray,
    expert_intended_actions: np.ndarray,
    *,
    episode_offsets: np.ndarray | None = None,
    shrinkage: float = 0.0,
    variance_floor: float = 0.0,
    alpha_per_step: float | None = None,
) -> np.ndarray:
    """Estimate E[(learner - expert)(learner - expert)^T].

    If offsets are provided, average each episode covariance equally, as in
    the official DART experiment implementation. Without offsets, pool steps.
    ``alpha_per_step`` is the desired trace in rad^2 per supervisor step.
    """
    learner = _as_arm_actions(learner_actions, "learner_actions")
    expert = _as_arm_actions(expert_intended_actions, "expert_intended_actions")
    if learner.shape != expert.shape or learner.shape[0] == 0:
        raise ValueError("expert and learner actions must be aligned and nonempty")
    if not 0 <= shrinkage <= 1 or variance_floor < 0:
        raise ValueError("invalid shrinkage or variance_floor")
    residual = learner - expert
    if episode_offsets is None:
        covariance = residual.T @ residual / len(residual)
    else:
        offsets = np.asarray(episode_offsets)
        if (offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0
                or offsets[-1] != len(residual) or np.any(np.diff(offsets) <= 0)):
            raise ValueError("episode_offsets must start at 0, end at N, and increase")
        covariance = np.mean(
            [part.T @ part / len(part) for part in np.split(residual, offsets[1:-1])],
            axis=0,
        )
    covariance = (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
    covariance = (covariance + covariance.T) / 2
    if variance_floor:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.maximum(eigenvalues, variance_floor)) @ eigenvectors.T
    if alpha_per_step is not None:
        if not np.isfinite(alpha_per_step) or alpha_per_step <= 0:
            raise ValueError("alpha_per_step must be positive")
        trace = np.trace(covariance)
        if trace <= 0:
            raise ValueError("zero residual covariance cannot be trace-scaled")
        covariance *= alpha_per_step / trace
    return (covariance + covariance.T) / 2


def noisy_expert_action(
    expert_intended_action: np.ndarray, covariance: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw one DART supervisor action; never perturb the two grippers."""
    clean = np.asarray(expert_intended_action, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if clean.shape not in ((ARM_DIMS,), (FULL_DIMS,)) or not np.isfinite(clean).all():
        raise ValueError("expert_intended_action must be a finite [12] or [14] vector")
    if cov.shape != (ARM_DIMS, ARM_DIMS) or not np.isfinite(cov).all():
        raise ValueError("covariance must be finite [12,12]")
    if not np.allclose(cov, cov.T, atol=1e-10) or np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError("covariance must be symmetric positive semidefinite")
    noise = rng.multivariate_normal(np.zeros(ARM_DIMS), cov, check_valid="raise")
    result = clean.copy()
    result[ARM_INDICES if len(clean) == FULL_DIMS else slice(None)] += noise
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit DART noise from paired same-observation actions")
    parser.add_argument("--pairs", type=Path, required=True, help="NPZ with learner_actions and expert_intended_actions")
    parser.add_argument("--output", type=Path, required=True, help="output NPZ path")
    parser.add_argument("--shrinkage", type=float, default=0.0)
    parser.add_argument("--variance-floor", type=float, default=0.0, help="rad^2")
    parser.add_argument("--alpha-per-step", type=float, default=None, help="desired covariance trace, rad^2")
    args = parser.parse_args()
    with np.load(args.pairs, allow_pickle=False) as pairs:
        offsets = pairs["episode_offsets"] if "episode_offsets" in pairs else None
        covariance = fit_covariance(
            pairs["learner_actions"], pairs["expert_intended_actions"],
            episode_offsets=offsets, shrinkage=args.shrinkage,
            variance_floor=args.variance_floor, alpha_per_step=args.alpha_per_step,
        )
        count = len(pairs["learner_actions"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, covariance=covariance)
    print(json.dumps({"output": str(args.output), "pairs": count,
                      "trace_rad2": float(np.trace(covariance)),
                      "min_eigenvalue_rad2": float(np.linalg.eigvalsh(covariance).min())}))


if __name__ == "__main__":
    main()
