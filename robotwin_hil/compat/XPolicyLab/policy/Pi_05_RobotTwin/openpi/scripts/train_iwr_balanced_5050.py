#!/usr/bin/env python3
"""Train RoboTwin Pi0.5 on exact 50:50 baseline/DAgger batches.

This is the policy-training side of the Ubuntu HG-DAgger experiment.  The
baseline and DAgger LeRobot roots are kept as separate datasets, and every
batch contains exactly half of each.  The loss is the ordinary mean over the
combined batch, so both source losses have coefficient 1.0.  In particular,
this entry point does not apply ``piperx.sample_weight`` or infer a weight from
the HIL label.

The action data remain 14-D absolute joint targets.  The RoboTwin OpenPI data
config performs its existing internal arm-joint delta transform and restores
absolute targets for inference; this script does not introduce another action
conversion.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import multiprocessing
import platform
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import train as base_train


DEFAULT_CONFIG = "pi05_robotwin_handover_to_tray_v2_promptfix"
DEFAULT_BASELINE_REPO = "baseline"
DEFAULT_DAGGER_REPO = "dagger"
DEFAULT_NORM_STATS_ASSET_ID = "iwr_hg_dagger_balanced_5050"


def validate_mix_manifest(path: str | None) -> None:
    if path is None:
        return
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mix = manifest.get("mix", {})
    expected = {
        "baseline_fraction": 0.5,
        "dagger_fraction": 0.5,
        "baseline_loss_weight": 1.0,
        "dagger_loss_weight": 1.0,
    }
    for key, value in expected.items():
        if float(mix.get(key, float("nan"))) != value:
            raise ValueError(f"{manifest_path}: {key} must be {value}")
    if manifest.get("action_semantics") != "14d_absolute_joint_target":
        raise ValueError(f"{manifest_path}: expected 14d_absolute_joint_target")


class BalancedDataLoader:
    """Two independent LeRobot loaders concatenated into exact 50:50 batches."""

    def __init__(
        self,
        data_config: _config.DataConfig,
        baseline_dataset,
        dagger_dataset,
        *,
        source_batch_size: int,
        sharding_: jax.sharding.Sharding | None,
        shuffle: bool,
        num_batches: int | None,
        baseline_workers: int,
        dagger_workers: int,
        seed: int,
    ):
        self._data_config = data_config
        self._source_batch_size = source_batch_size
        self._sharding = sharding_
        self._shuffle = shuffle
        self._num_batches = num_batches
        self._seed = seed
        self.last_source_counts: dict[str, int] = {}
        self._baseline_loader = self._make_loader(
            baseline_dataset, source_batch_size, shuffle=shuffle, workers=baseline_workers, seed=seed
        )
        self._dagger_loader = self._make_loader(
            dagger_dataset, source_batch_size, shuffle=shuffle, workers=dagger_workers, seed=seed + 9973
        )

    @staticmethod
    def _make_loader(dataset, batch_size: int, *, shuffle: bool, workers: int, seed: int):
        if len(dataset) < batch_size:
            raise ValueError(f"source dataset has {len(dataset)} items, smaller than batch {batch_size}")
        generator = torch.Generator().manual_seed(seed)
        mp_context = multiprocessing.get_context("spawn") if workers > 0 else None
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            multiprocessing_context=mp_context,
            persistent_workers=workers > 0,
            collate_fn=_data_loader._collate_fn,
            worker_init_fn=_data_loader._worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[tuple[_model.Observation, _model.Actions]]:
        baseline_iter = iter(self._baseline_loader)
        dagger_iter = iter(self._dagger_loader)
        emitted = 0
        while self._num_batches is None or emitted < self._num_batches:
            try:
                baseline = next(baseline_iter)
            except StopIteration:
                baseline_iter = iter(self._baseline_loader)
                baseline = next(baseline_iter)
            try:
                dagger = next(dagger_iter)
            except StopIteration:
                dagger_iter = iter(self._dagger_loader)
                dagger = next(dagger_iter)

            batch = jax.tree.map(
                lambda left, right: np.concatenate([np.asarray(left), np.asarray(right)], axis=0),
                baseline,
                dagger,
            )
            source_ids = np.concatenate(
                [
                    np.zeros(self._source_batch_size, dtype=np.int32),
                    np.ones(self._source_batch_size, dtype=np.int32),
                ]
            )
            if self._shuffle:
                permutation = np.random.default_rng(self._seed + emitted).permutation(len(source_ids))
                batch = jax.tree.map(lambda value: value[permutation], batch)
                source_ids = source_ids[permutation]
            self.last_source_counts = {
                "baseline": int(np.sum(source_ids == 0)),
                "dagger": int(np.sum(source_ids == 1)),
            }
            expected = {"baseline": self._source_batch_size, "dagger": self._source_batch_size}
            if self.last_source_counts != expected:
                raise AssertionError(f"50:50 source contract violated: {self.last_source_counts}")
            if self._sharding is not None:
                batch = jax.tree.map(lambda value: jax.make_array_from_process_local_data(self._sharding, value), batch)
            emitted += 1
            yield _model.Observation.from_dict(batch), batch["actions"]


def _data_config(config: _config.TrainConfig, repo_id: str, norm_stats_asset_id: str) -> _config.DataConfig:
    assets = replace(config.data.assets, asset_id=norm_stats_asset_id)
    return replace(config.data, repo_id=repo_id, assets=assets).create(config.assets_dirs, config.model)


def _assert_same_norm_stats(left: dict, right: dict) -> None:
    if left is None or right is None:
        raise ValueError("mixed norm_stats are required for both sources")
    if set(left) != set(right):
        raise ValueError(f"norm_stats keys differ: {sorted(left)} vs {sorted(right)}")
    for key in sorted(left):
        for field in ("mean", "std", "q01", "q99"):
            left_value = getattr(left[key], field)
            right_value = getattr(right[key], field)
            if left_value is None or right_value is None:
                if left_value is not right_value:
                    raise ValueError(f"norm_stats[{key}].{field} differs by None-ness")
                continue
            if not np.allclose(np.asarray(left_value), np.asarray(right_value), rtol=1e-6, atol=1e-8):
                raise ValueError(f"norm_stats[{key}].{field} differs")


def create_balanced_data_loader(
    config: _config.TrainConfig,
    *,
    baseline_repo_id: str,
    dagger_repo_id: str,
    norm_stats_asset_id: str,
    sharding_: jax.sharding.Sharding | None,
    shuffle: bool,
    num_batches: int | None = None,
) -> BalancedDataLoader:
    if config.batch_size <= 0 or config.batch_size % 2:
        raise ValueError(f"batch-size must be a positive even number, got {config.batch_size}")
    source_batch_size = config.batch_size // 2
    baseline_config = _data_config(config, baseline_repo_id, norm_stats_asset_id)
    dagger_config = _data_config(config, dagger_repo_id, norm_stats_asset_id)
    _assert_same_norm_stats(baseline_config.norm_stats, dagger_config.norm_stats)
    baseline_dataset = _data_loader.create_torch_dataset(baseline_config, config.model.action_horizon, config.model)
    baseline_dataset = _data_loader.transform_dataset(baseline_dataset, baseline_config)
    dagger_dataset = _data_loader.create_torch_dataset(dagger_config, config.model.action_horizon, config.model)
    dagger_dataset = _data_loader.transform_dataset(dagger_dataset, dagger_config)
    baseline_workers = config.num_workers // 2
    dagger_workers = config.num_workers - baseline_workers
    logging.info(
        "balanced_batch baseline=%d dagger=%d loss_weights=1:1 workers=%d/%d",
        source_batch_size,
        source_batch_size,
        baseline_workers,
        dagger_workers,
    )
    return BalancedDataLoader(
        baseline_config,
        baseline_dataset,
        dagger_dataset,
        source_batch_size=source_batch_size,
        sharding_=sharding_,
        shuffle=shuffle,
        num_batches=num_batches,
        baseline_workers=baseline_workers,
        dagger_workers=dagger_workers,
        seed=config.seed,
    )


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions):
        # Exact batch balance makes this the desired 1:1 source loss.
        # Deliberately do not apply piperx.sample_weight or a HIL multiplier.
        return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, value: value.value.ndim > 1,
        ),
    )
    return new_state, {"loss": loss, "grad_norm": optax.global_norm(grads), "param_norm": optax.global_norm(kernel_params)}


def _shape(value) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(value).shape)


def run_loader_smoke(args: argparse.Namespace, config: _config.TrainConfig) -> None:
    loader = create_balanced_data_loader(
        config,
        baseline_repo_id=args.baseline_repo_id,
        dagger_repo_id=args.dagger_repo_id,
        norm_stats_asset_id=args.norm_stats_asset_id,
        sharding_=None,
        shuffle=args.shuffle,
        num_batches=args.loader_smoke_batches,
    )
    for index, (observation, actions) in enumerate(loader):
        print(
            f"batch={index} source_counts={loader.last_source_counts} "
            f"state={_shape(observation.state)} actions={_shape(actions)} "
            f"high={_shape(observation.images['base_0_rgb'])} "
            f"left={_shape(observation.images['left_wrist_0_rgb'])} "
            f"right={_shape(observation.images['right_wrist_0_rgb'])}",
            flush=True,
        )


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    config = _config.get_config(args.config_name)
    model = config.model
    if args.lora:
        # A full Pi0.5 update needs an A100/H100-class memory budget.  The
        # Ubuntu HIL host is a 24 GB RTX 4090, so its supported training path
        # is LoRA: load the existing full checkpoint and optimize only the
        # low-rank adapters (plus the parameters not covered by the model's
        # freeze filter).
        model = replace(
            model,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
    if args.lora and args.lora_only:
        # Keep the 24 GB path genuinely parameter efficient.  The upstream
        # model freeze filter can leave vision/projection weights trainable;
        # that is useful on larger GPUs but leaves too little headroom on the
        # Ubuntu RTX 4090 once the Pi0.5 activations are compiled.
        freeze_filter = nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*lora.*")))
    elif args.lora:
        freeze_filter = model.get_freeze_filter()
    else:
        freeze_filter = config.freeze_filter
    assets = replace(config.data.assets, asset_id=args.norm_stats_asset_id)
    data_factory = replace(config.data, repo_id=args.baseline_repo_id, assets=assets)
    updates: dict[str, object] = {
        "model": model,
        "freeze_filter": freeze_filter,
        "ema_decay": None if args.no_ema else config.ema_decay,
        "data": data_factory,
        "exp_name": args.exp_name,
        "assets_base_dir": args.assets_base_dir,
        "checkpoint_base_dir": args.checkpoint_base_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_train_steps": args.num_train_steps,
        "save_interval": args.save_interval,
        "keep_period": args.keep_period,
        "overwrite": args.overwrite,
        "resume": args.resume,
        "wandb_enabled": args.wandb_enabled,
        "fsdp_devices": args.fsdp_devices,
        "log_interval": args.log_interval,
    }
    if args.weight_loader_params:
        updates["weight_loader"] = _weight_loaders.CheckpointWeightLoader(args.weight_loader_params)
    return replace(config, **updates)


def train(args: argparse.Namespace, config: _config.TrainConfig) -> None:
    base_train.init_logging()
    logging.info("Running RoboTwin HIL 50:50 IWR training on %s", platform.node())
    logging.info("baseline_repo_id=%s dagger_repo_id=%s loss_weighting=uniform_1_to_1", args.baseline_repo_id, args.dagger_repo_id)
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch-size {config.batch_size} must be divisible by device_count={jax.device_count()}")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=config.resume
    )
    base_train.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    data_loader = create_balanced_data_loader(
        config,
        baseline_repo_id=args.baseline_repo_id,
        dagger_repo_id=args.dagger_repo_id,
        norm_stats_asset_id=args.norm_stats_asset_id,
        sharding_=data_sharding,
        shuffle=args.shuffle,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info("Initialized balanced loader: %s", data_loader.last_source_counts)
    train_state, train_state_sharding = base_train.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(range(start_step, config.num_train_steps), initial=start_step, total=config.num_train_steps, dynamic_ncols=True)
    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            logging.info("Step %d: %s source_counts=%s", step, ", ".join(f"{key}={value:.4f}" for key, value in reduced.items()), data_loader.last_source_counts)
            wandb.log(reduced, step=step)
            infos = []
        batch = next(data_iter)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
    checkpoint_manager.wait_until_finished()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-repo-id", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--dagger-repo-id", default=DEFAULT_DAGGER_REPO)
    parser.add_argument("--norm-stats-asset-id", default=DEFAULT_NORM_STATS_ASSET_ID)
    parser.add_argument("--mix-manifest")
    parser.add_argument("--exp-name", default="iwr_hg_dagger_balanced_5050")
    parser.add_argument("--assets-base-dir", default="./assets")
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument("--weight-loader-params")
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use Pi0.5 LoRA variants; recommended/required on a 24 GB RTX 4090.",
    )
    parser.add_argument(
        "--lora-only",
        action="store_true",
        help="Freeze every non-LoRA parameter (lowest-memory LoRA mode).",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable the EMA copy to reduce memory; useful for 24 GB LoRA runs.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-train-steps", type=int, default=2500)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-period", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--loader-smoke-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_mix_manifest(args.mix_manifest)
    config = build_config(args)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.loader_smoke_batches > 0:
        logging.basicConfig(level=logging.INFO)
        run_loader_smoke(args, config)
    else:
        train(args, config)


if __name__ == "__main__":
    main()
