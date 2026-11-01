"""Import-only compatibility for older HIL helpers; this is not an evaluator.

New SFT evaluations must run ``eval_policy_xpolicylab_official_trace.py``.
The former combined evaluator (fixed seeds, enhanced sampling, trace hooks)
was retired because it changed the official control protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from eval_policy_xpolicylab_official_trace import (
    absolute_joint_chunk_from_response,
    build_instruction,
    build_policy_client,
    class_decorator,
    close_policy_client,
    is_episode_end,
    load_task_args,
    normalize_action_chunk,
    notify_trial_end,
    prepare_policy_case,
    reset_policy,
    robotwin_obs_to_xpolicylab,
    safe_close_env,
    xpolicylab_action_to_robotwin,
)


def sample_absolute_joint_chunk(
    model_client: Any, observation: Mapping[str, Any], action_type: str,
) -> np.ndarray:
    """Compatibility helper for a single absolute-joint policy chunk."""
    raw = model_client.call(func_name="get_action")
    return absolute_joint_chunk_from_response(
        normalize_action_chunk(raw), observation, action_type,
    )


def sample_absolute_joint_candidates(
    model_client: Any,
    observation: Mapping[str, Any],
    action_type: str,
    num_candidates: int,
    microbatch_size: int = 4,
) -> np.ndarray:
    """Compatibility helper for K same-observation policy chunks."""
    if isinstance(num_candidates, bool) or not isinstance(num_candidates, int) or not 1 <= num_candidates <= 32:
        raise ValueError("num_candidates must be an integer between 1 and 32")
    if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int) or not 2 <= microbatch_size <= 32:
        raise ValueError("microbatch_size must be an integer between 2 and 32")
    chunks = []
    remaining = num_candidates
    while remaining:
        batch_size = min(microbatch_size, remaining)
        raw_batch = (
            [model_client.call(func_name="get_action")]
            if batch_size == 1 else
            model_client.call(func_name="get_action_candidates", obs=batch_size)
        )
        if not isinstance(raw_batch, (list, tuple, np.ndarray)) or len(raw_batch) != batch_size:
            raise ValueError(f"policy must return exactly {batch_size} candidate chunks")
        chunks.extend(
            absolute_joint_chunk_from_response(
                normalize_action_chunk(candidate), observation, action_type,
            )
            for candidate in raw_batch
        )
        remaining -= batch_size
    if len({chunk.shape for chunk in chunks}) != 1:
        raise ValueError("candidate chunks must have the same shape")
    return np.stack(chunks)


if __name__ == "__main__":
    raise SystemExit(
        "The combined evaluator was retired. Run "
        "scripts/eval_policy_xpolicylab_official_trace.py with --trace_dir."
    )
