"""Online frozen scorer using exactly the tail training visual preprocessing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tail_value.cache import CAMERAS, sha256
from tail_value.model import CoverageCritic, finite_guard
from tail_value.prepare import ENCODER_SPEC, FrozenEncoder


class OnlineCoverage:
    def __init__(self, checkpoint, encoder_weights, device="cpu"):
        checkpoint, encoder_weights = Path(checkpoint), Path(encoder_weights)
        # Explicit local artifacts prevent accidental model downloads in eval.
        if not checkpoint.is_file() or not encoder_weights.is_file():
            raise FileNotFoundError("Provide a trusted critic checkpoint and local ResNet18 weights")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload["config"]
        self.device = torch.device(device)
        self.model = CoverageCritic(payload["normalization"], width=int(config["width"]))
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).requires_grad_(False).eval()
        self.encoder = FrozenEncoder(self.device, weights_path=encoder_weights)
        self.limit = float(config.get("score_limit", 1e4))
        self.metadata = {"critic_checkpoint": str(checkpoint.resolve()), "critic_sha256": sha256(checkpoint),
                         "critic_step": payload.get("step"), "critic_alpha": config.get("alpha"),
                         "critic_bootstrap": config.get("bootstrap"), "encoder": ENCODER_SPEC,
                         "encoder_weights": str(encoder_weights.resolve()), "encoder_sha256": sha256(encoder_weights),
                         "action_semantics": "executed absolute joint target [left6,gripper,right6,gripper]",
                         "state_semantics": "RoboTwin joint_action.vector (same drive-target state as training)"}

    @torch.inference_mode()
    def score_actions(self, observation, actions):
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if state.shape != (14,) or actions.ndim != 2 or actions.shape[1] != 14 or not len(actions):
            raise ValueError("Coverage critic requires 14D state and [K,14] absolute joint actions")
        cameras = ("head_camera", "left_camera", "right_camera")
        images = {key: observation["observation"][camera]["rgb"] for key, camera in zip(CAMERAS, cameras)}
        features = self.encoder(images)
        tensors = [torch.as_tensor(x, device=self.device)[None] for x in (features, state, actions)]
        finite_guard(dict(zip(("features", "state", "action"), tensors)))
        scores = self.model.score_many(*tensors)[0]
        finite_guard({"coverage": scores}, self.limit)
        return scores.cpu().numpy().astype(np.float64)

    def __call__(self, observation, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,):
            raise ValueError("Coverage critic requires one 14D absolute joint action")
        return float(self.score_actions(observation, action[None])[0])

    def bind_observation(self, observation):
        return lambda actions: self.score_actions(observation, actions)


class OnlineChunkDiscriminator:
    """Frozen ImageNet/ResNet18 + full-chunk BCE logit; no sigmoid or L2 transform.

    A binding belongs to one policy decision. Visual features are computed lazily
    once, then reused for candidate and blended-chunk scoring at that observation.
    """
    def __init__(self, checkpoint, encoder_weights, *, device="cpu"):
        from tail_value.chunk import ChunkCoverageCritic

        checkpoint, encoder_weights = Path(checkpoint), Path(encoder_weights)
        if not checkpoint.is_file() or not encoder_weights.is_file():
            raise FileNotFoundError("Provide local discriminator and ImageNet ResNet18 weights")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload["config"]
        if (payload.get("representation") != "absolute_joint_chunk" or config.get("horizon") != 10
                or config.get("objective") != "balanced_source_bce"):
            raise ValueError("Expected an H=10 absolute-joint balanced_source_bce discriminator")
        self.device = torch.device(device)
        self.model = ChunkCoverageCritic(payload["normalization"], 10, int(config.get("width", 256)))
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).requires_grad_(False).eval()
        self.encoder = FrozenEncoder(self.device, weights_path=encoder_weights)
        self.metadata = {
            "scorer_kind": "discriminator_chunk", "score_semantics": "raw BCE logit; high means expert support",
            "critic_checkpoint": str(checkpoint.resolve()), "critic_sha256": sha256(checkpoint),
            "critic_step": payload.get("step"), "critic_objective": config["objective"],
            "critic_seed": config.get("seed"), "critic_horizon": 10,
            "encoder": ENCODER_SPEC, "visual_feature_normalization": "raw frozen features; no per-view L2",
            "encoder_weights": str(encoder_weights.resolve()), "encoder_sha256": sha256(encoder_weights),
            "action_semantics": "full absolute joint chunk [10,14]; [left6,gripper,right6,gripper]",
            "state_semantics": "RoboTwin joint_action.vector (same drive-target state as training)",
        }

    def bind_observation(self, observation):
        state = np.array(observation["joint_action"]["vector"], dtype=np.float32, copy=True)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Discriminator requires a finite 14D state")
        features = None

        @torch.inference_mode()
        def score_chunks(chunks):
            nonlocal features
            chunks = np.ascontiguousarray(chunks, dtype=np.float32)
            if chunks.ndim != 3 or chunks.shape[1:] != (10, 14) or not len(chunks) or not np.isfinite(chunks).all():
                raise ValueError("Discriminator requires finite full chunks [K,10,14]")
            if features is None:
                cameras = ("head_camera", "left_camera", "right_camera")
                images = {key: observation["observation"][camera]["rgb"] for key, camera in zip(CAMERAS, cameras)}
                features = self.encoder(images)
            tensors = [torch.as_tensor(x, dtype=torch.float32, device=self.device)[None]
                       for x in (features, state, chunks)]
            finite_guard(dict(zip(("features", "state", "chunks"), tensors)))
            scores = self.model.score_many(*tensors)[0]
            finite_guard({"discriminator": scores})
            return scores.cpu().numpy().astype(np.float64)

        return score_chunks

    def score_chunks(self, observation, chunks):
        return self.bind_observation(observation)(chunks)


def create_online_scorer(args, config):
    if not config.requires_scores:
        raise ValueError(f"Sampling mode {config.mode!r} does not use an online scorer")
    if config.score_representation == "first_action":
        return OnlineCoverage(args["es_critic"], args["es_encoder_weights"], device=args.get("es_device", "cpu"))
    from .calibration import load_discriminator_calibration

    calibration, digest = load_discriminator_calibration(
        args["es_calibration"], config.calibration_key, args["es_critic"])
    if digest != config.calibration_sha256:
        raise ValueError("Calibration changed after sampling configuration was created")
    scorer = OnlineChunkDiscriminator(args["es_critic"], args["es_encoder_weights"], device=args.get("es_device", "cpu"))
    scorer.metadata.update(calibration_file=str(Path(args["es_calibration"]).resolve()),
                           calibration_sha256=digest, calibration_key=config.calibration_key,
                           calibration=calibration, effective_beta=config.beta,
                           weighting_semantics="softmax(-beta * B / scale); low support gets more weight",
                           kl_semantics="candidate categorical KL is diagnostic, not an executed-policy KL bound")
    return scorer
