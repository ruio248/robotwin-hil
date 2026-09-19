"""Support Q critic and explicit mean/farthest conservative objectives."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn

from .cache import ACTION_DIM, FEATURE_DIM, load_episode


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_guard(values, limit=None):
    for name, value in values.items():
        value = torch.as_tensor(value)
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite {name}")
        if limit is not None and value.numel() and value.abs().max().item() > limit:
            raise FloatingPointError(f"{name} exceeded absolute score limit {limit}")


class TransitionTable:
    """CPU frame storage; next frames are indexed, not copied or synthesized."""
    def __init__(self, root, manifest, split):
        entries = [e for e in manifest["episodes"] if e["split"] == split]
        if not entries or any(e["kind"] != "demo" for e in entries):
            raise ValueError(f"{split} requires demonstration episodes; never train on HIL")
        fields = {k: [] for k in ("features", "state", "a_demo", "a_pi")}
        indices, offset = [], 0
        for entry in entries:
            arrays = load_episode(Path(root) / entry["file"], "demo", manifest["config"]["compatibility"]["num_candidates"])
            indices.extend((np.flatnonzero(arrays["valid_transition"]) + offset).tolist())
            for key in fields:
                fields[key].append(arrays[key].astype(np.float32))
            offset += len(arrays["state"])
        self.data = {key: torch.from_numpy(np.concatenate(parts)) for key, parts in fields.items()}
        self.indices = torch.tensor(indices, dtype=torch.long)
        if not len(self.indices):
            raise ValueError("No valid transitions")

    def __len__(self):
        return len(self.indices)

    def batch(self, positions, device):
        idx = self.indices[positions]
        result = {k: v[idx].to(device) for k, v in self.data.items()}
        result.update({"next_" + k: self.data[k][idx + 1].to(device) for k in ("features", "state", "a_pi")})
        return result

    def normalization(self):
        result = {}
        for key in ("state", "a_demo"):
            value = self.data[key][self.indices]
            prefix = "action" if key == "a_demo" else "state"
            result[prefix + "_mean"] = value.mean(0)
            result[prefix + "_scale"] = value.std(0, unbiased=False).clamp_min(0.05)
        return result


class CoverageCritic(nn.Module):
    def __init__(self, normalization, width=256):
        super().__init__()
        self.width = width
        for name in ("state_mean", "state_scale", "action_mean", "action_scale"):
            value = torch.as_tensor(normalization[name], dtype=torch.float32).clone()
            if value.shape != (ACTION_DIM,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid normalization {name}")
            if name.endswith("scale") and (value < 0.05 - 1e-7).any():
                raise ValueError("Normalization scale below 0.05")
            self.register_buffer(name, value)
        layers, input_dim = [], FEATURE_DIM + 2 * ACTION_DIM
        for _ in range(3):
            layers.extend([nn.Linear(input_dim, width), nn.LayerNorm(width), nn.SiLU()])
            input_dim = width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features, state, action):
        return self.net(torch.cat([features, (state - self.state_mean) / self.state_scale,
                                   (action - self.action_mean) / self.action_scale], dim=-1)).squeeze(-1)

    def score_many(self, features, state, actions):
        k = actions.shape[1]
        return self(features[:, None].expand(-1, k, -1), state[:, None].expand(-1, k, -1), actions)


def action_distances(candidates, expert, scale):
    """Normalized RMS distance in the shared absolute joint/qpos space.

    Both inputs are absolute actions.  Subtracting the same current state
    from both would produce the identical difference, so no delta conversion
    is needed for coverage ranking.
    """
    return (((candidates - expert[:, None]) / scale).square().mean(-1)).sqrt()


def farthest_indices(candidates, expert, scale):
    return action_distances(candidates, expert, scale).argmax(dim=1)


def loss_from_scores(q_demo, q_pi, target, farthest, *, alpha, output_reg, reduction, mode):
    if reduction not in ("mean", "farthest") or mode not in ("original", "stabilized"):
        raise ValueError("Unknown objective mode")
    conservative_value = q_pi.mean(1) if reduction == "mean" else q_pi.gather(1, farthest[:, None]).squeeze(1)
    td = (q_demo - target.detach()).square().mean()
    conservative = (conservative_value - q_demo).mean()
    penalty = 0.5 * (q_demo.square() + q_pi.square().mean(1)).mean()
    regularizer = output_reg * penalty if mode == "stabilized" else penalty * 0.0
    return td + alpha * conservative + regularizer, {
        "td": td, "conservative": conservative, "output_penalty": penalty, "regularizer": regularizer,
    }


def critic_loss(model, target_model, batch, *, gamma, alpha, output_reg, reduction, mode, score_limit=1e4):
    with torch.no_grad():
        q_next = target_model.score_many(batch["next_features"], batch["next_state"], batch["next_a_pi"])
        target = 1.0 + gamma * q_next.mean(1)
    q_demo = model(batch["features"], batch["state"], batch["a_demo"])
    q_pi = model.score_many(batch["features"], batch["state"], batch["a_pi"])
    finite_guard({"q_demo": q_demo, "q_pi": q_pi, "q_next": q_next, "target": target}, score_limit)
    farthest = farthest_indices(batch["a_pi"], batch["a_demo"], model.action_scale)
    loss, metrics = loss_from_scores(q_demo, q_pi, target, farthest, alpha=alpha,
                                     output_reg=output_reg, reduction=reduction, mode=mode)
    finite_guard({"loss": loss})
    return loss, metrics, {"q_demo": q_demo.detach(), "q_pi": q_pi.detach(), "target": target}


@torch.no_grad()
def soft_update(target, model, tau):
    for dst, src in zip(target.parameters(), model.parameters(), strict=True):
        dst.lerp_(src, tau)


def make_target(model):
    return copy.deepcopy(model).requires_grad_(False).eval()


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampler": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["sampler"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    # Checkpoints include optimizer/Python RNG state. Load only trusted local files.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return payload


def models_from_checkpoint(payload, device):
    model = CoverageCritic(payload["normalization"], payload["config"]["width"]).to(device)
    model.load_state_dict(payload["model"])
    target = make_target(model)
    target.load_state_dict(payload["target_model"])
    return model, target
