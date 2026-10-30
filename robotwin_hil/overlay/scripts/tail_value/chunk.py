"""Ten-action macro-transition critic, separate from the 14D first-action critic."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .cache import ACTION_DIM, FEATURE_DIM, read_json, sha256
from .model import finite_guard

HORIZON = 10


def macro_reward(gamma: float, horizon: int = HORIZON) -> float:
    """Discounted sum of one unit of demonstrated support per recorded step."""
    if not 0 <= gamma < 1 or horizon < 1:
        raise ValueError("Invalid gamma or horizon")
    return sum(gamma ** i for i in range(horizon))


def validate_chunk_episode(data: dict, *, horizon: int, candidates: int) -> None:
    n = len(data["state"])
    shapes = {"features": (n, FEATURE_DIM), "state": (n, ACTION_DIM),
              "expert_chunks": (n, horizon, ACTION_DIM),
              "policy_chunks": (n, candidates, horizon, ACTION_DIM),
              "frame_index": (n,), "valid_transition": (n,)}
    for key, shape in shapes.items():
        if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key}; expected {shape}")
    if n < 2 or not np.array_equal(data["valid_transition"], np.arange(n) < n - 1):
        raise ValueError("An episode needs real next macro observations; no fabricated terminal")
    if data["valid_transition"].dtype != np.bool_ or not np.array_equal(
            np.diff(data["frame_index"]), np.full(n - 1, horizon)):
        raise ValueError("Macro anchors must be consecutive, horizon-spaced recorded frames")


class ChunkTransitionTable:
    """Episode-safe macro transitions; source features are the frozen old-cache features."""
    def __init__(self, root: Path, manifest: dict, split: str):
        self.horizon = int(manifest["config"]["horizon"])
        self.candidates = int(manifest["config"]["num_candidates"])
        entries = [e for e in manifest["episodes"] if e["split"] == split]
        if not entries:
            raise ValueError(f"Empty {split} split")
        fields = {k: [] for k in ("features", "state", "expert_chunks", "policy_chunks")}
        indices, offset = [], 0
        for entry in entries:
            path = Path(root) / entry["file"]
            if not path.is_file() or sha256(path) != entry.get("sha256"):
                raise ValueError(f"Missing/corrupt chunk cache: {path}")
            with np.load(path, allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
            validate_chunk_episode(data, horizon=self.horizon, candidates=self.candidates)
            indices.extend((np.flatnonzero(data["valid_transition"]) + offset).tolist())
            for key in fields:
                fields[key].append(data[key].astype(np.float32))
            offset += len(data["state"])
        self.data = {k: torch.from_numpy(np.concatenate(v)) for k, v in fields.items()}
        self.indices = torch.tensor(indices, dtype=torch.long)
        if not len(self.indices):
            raise ValueError("No macro transitions")

    def __len__(self):
        return len(self.indices)

    def batch(self, positions, device):
        idx = self.indices[positions]
        data = {key: value[idx].to(device) for key, value in self.data.items()}
        for key in ("features", "state", "policy_chunks"):
            data["next_" + key] = self.data[key][idx + 1].to(device)
        return data

    def normalization(self):
        state = self.data["state"][self.indices]
        actions = self.data["expert_chunks"][self.indices].reshape(-1, ACTION_DIM)
        return {"state_mean": state.mean(0), "state_scale": state.std(0, unbiased=False).clamp_min(.05),
                "action_mean": actions.mean(0), "action_scale": actions.std(0, unbiased=False).clamp_min(.05)}


class ChunkCoverageCritic(nn.Module):
    def __init__(self, normalization: dict, horizon: int = HORIZON, width: int = 256):
        super().__init__()
        if horizon < 1 or width < 1:
            raise ValueError("Invalid horizon or width")
        self.horizon, self.width = horizon, width
        for name in ("state_mean", "state_scale", "action_mean", "action_scale"):
            value = torch.as_tensor(normalization[name], dtype=torch.float32).clone()
            if value.shape != (ACTION_DIM,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid normalization {name}")
            if name.endswith("scale") and (value < .05 - 1e-7).any():
                raise ValueError(f"Invalid normalization scale {name}")
            self.register_buffer(name, value)
        layers, input_dim = [], FEATURE_DIM + ACTION_DIM + horizon * ACTION_DIM
        for _ in range(3):
            layers.extend([nn.Linear(input_dim, width), nn.LayerNorm(width), nn.SiLU()])
            input_dim = width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features, state, chunk):
        if chunk.shape[-2:] != (self.horizon, ACTION_DIM):
            raise ValueError("Expected full physical [H,14] chunk")
        normalized = (chunk - self.action_mean) / self.action_scale
        inputs = torch.cat([features, (state - self.state_mean) / self.state_scale,
                            normalized.flatten(start_dim=-2)], dim=-1)
        return self.net(inputs).squeeze(-1)

    def score_many(self, features, state, chunks):
        k = chunks.shape[1]
        return self(features[:, None].expand(-1, k, -1),
                    state[:, None].expand(-1, k, -1), chunks)


def chunk_critic_loss(model, target_model, batch, *, gamma, alpha, score_limit=1e4):
    horizon = model.horizon
    with torch.no_grad():
        q_next = target_model.score_many(batch["next_features"], batch["next_state"],
                                         batch["next_policy_chunks"])
        td_target = macro_reward(gamma, horizon) + gamma ** horizon * q_next.mean(1)
    q_expert = model(batch["features"], batch["state"], batch["expert_chunks"])
    q_policy = model.score_many(batch["features"], batch["state"], batch["policy_chunks"])
    finite_guard({"q_expert": q_expert, "q_policy": q_policy,
                  "q_next": q_next, "td_target": td_target}, score_limit)
    td = (q_expert - td_target.detach()).square().mean()
    conservative = (q_policy.mean(1) - q_expert).mean()
    loss = td + alpha * conservative
    finite_guard({"loss": loss})
    metrics = {"td": td, "conservative": conservative,
               "weighted_conservative": alpha * conservative}
    values = {"q_expert": q_expert.detach(), "q_policy": q_policy.detach(),
              "q_next": q_next.detach(), "td_target": td_target.detach(),
              "expert_minus_policy": (q_expert - q_policy.mean(1)).detach()}
    return loss, metrics, values


def load_chunk_manifest(root: Path, *, complete=True) -> dict:
    manifest = read_json(Path(root) / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest["config"].get("representation") != "absolute_joint_chunk":
        raise ValueError("Unsupported chunk-cache schema")
    if complete and manifest.get("complete") is not True:
        raise ValueError("Chunk cache is incomplete")
    return manifest
