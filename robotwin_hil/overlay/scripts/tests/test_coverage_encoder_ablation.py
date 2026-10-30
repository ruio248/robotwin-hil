from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coverage_encoder_ablation import normalize_views, r3m_convnet_state
from coverage_encoder_hil_score import aggregate
from report_coverage_encoder_ablation import paired_difference


def test_view_normalization_and_dino_padding():
    original = np.tile(np.arange(1, 385, dtype=np.float32), (2, 3))
    features = normalize_views(original, 384)
    assert features.shape == (2, 1536)
    blocks = features.reshape(2, 3, 512)
    np.testing.assert_allclose(np.linalg.norm(blocks, axis=2), 1, rtol=1e-6)
    assert not np.any(blocks[:, :, 384:])


def test_reject_degenerate_visual_block():
    with pytest.raises(ValueError, match="Zero-length"):
        normalize_views(np.zeros((1, 1536), dtype=np.float32), 512)


def test_r3m_mapping_requires_complete_matching_state():
    expected = {"conv1.weight": torch.ones((2, 2)), "bn1.bias": torch.zeros(2)}
    packed = {"r3m": {"module.convnet." + key: value for key, value in expected.items()}}
    mapped = r3m_convnet_state(packed, expected)
    assert set(mapped) == set(expected)
    with pytest.raises(ValueError, match="complete"):
        r3m_convnet_state({"r3m": {"module.convnet.conv1.weight": torch.ones((2, 2))}}, expected)


def test_paired_rank_uses_episode_identity():
    base = {"rank": {"episodes": {str(i): .5 for i in range(30)}}}
    changed = {"rank": {"episodes": {str(i): .6 for i in range(30)}}}
    result = paired_difference(base, changed)
    assert result["paired_episodes"] == 30
    assert result["mean_delta"] == pytest.approx(.1)
    assert result["episode_bootstrap_ci95"] == pytest.approx([.1, .1])


def test_takeover_aggregation_respects_independent_episodes():
    episodes = [{"relative_frame": list(range(-20, 0)),
                 "candidate_mean": [0.] * 10 + [-1.] * 10,
                 "candidate_range": [0.2] * 20} for _ in range(3)]
    result = aggregate(episodes, 1.)
    assert result["episodes"] == 3
    assert result["pre_takeover_observations"] == 60
    assert result["late_vs_early"]["episodes_with_both_windows"] == 3
    assert result["late_vs_early"]["late_minus_early_mean"] == pytest.approx(-1.)
