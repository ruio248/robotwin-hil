"""One-factor H50 ablation: original disc_s42 ImageNet encoder, unchanged trainer.

Read-only access to the existing paired data, R3M baseline, and frozen outcomes.
All new artifacts live in --output. No rollout, policy service, or test-label fit.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights

from encode_pairs_h50 import image_tensor as current_image_tensor
from score_outcome_h50 import sha256
from tail_value.chunk import ChunkCoverageCritic
from tail_value.prepare import FrozenEncoder, image_tensor


TASK = "stack_blocks_two"


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def encode(encoder, images, device):
    blocks = []
    for start in range(0, len(images), 64):
        batch = torch.stack([image_tensor(im) for im in images[start:start + 64]]).to(device)
        with torch.inference_mode():
            blocks.append(encoder(batch).float().cpu().numpy())
    return np.concatenate(blocks).astype(np.float32)


def prepare(args, encoder, device, baseline_protocol):
    episodes = []
    cache = args.feature_cache or (args.output / "features_imagenet18")
    if args.feature_cache:
        previous = json.loads((cache.parent / "protocol.json").read_text())
        assert previous["complete"] and previous["encoder_sha256"] == sha256(args.encoder_asset)
        assert previous["baseline_checkpoint_sha256"] == sha256(args.source / "models" / TASK / "model.pt")
    else:
        cache.mkdir()
    source_map = baseline_protocol["source_sha256"]
    for split in (0, 1):
        root = args.source / "data" / TASK / f"seed{split}_n25"
        collection = json.loads((root / "state.json").read_text())
        assert collection["status"] == "complete" and collection["expert_chunk_horizon"] == 50
        for meta_path in sorted(root.glob("episode_*_scene_*/metadata.json")):
            meta = json.loads(meta_path.read_text())
            assert meta["task"] == TASK and meta["expert_success"] and meta["chunk_horizon"] == 50
            scene = int(meta["scene_seed"])
            pair_path = meta_path.parent / "pairs.npz"
            pair_sha = sha256(pair_path)
            assert pair_sha == source_map[str(scene)]["pairs"]
            with np.load(pair_path, allow_pickle=False) as raw:
                state = raw["state"].astype(np.float32)
                expert = raw["expert_chunks"].astype(np.float32)
                policy = raw["policy_chunks"].astype(np.float32)
                if args.feature_cache:
                    dest = cache / f"scene_{scene}.npz"
                    assert sha256(dest) == previous["features"][str(scene)]
                    assert pair_sha == previous["source_pairs"][str(scene)]
                    with np.load(dest, allow_pickle=False) as encoded:
                        features = encoded["features"].astype(np.float32)
                else:
                    blocks = []
                    for name in ("images_head", "images_left", "images_right"):
                        images = raw[name]
                        assert images.dtype == np.uint8 and len(images) == len(state)
                        # Verify this experiment changes weights, not resize/padding/RGB normalization.
                        assert torch.equal(image_tensor(images[0]), current_image_tensor(images[0]))
                        blocks.append(encode(encoder, images, device))
                    features = np.concatenate(blocks, axis=1)
            n = len(state)
            assert state.shape == (n, 14) and expert.shape == policy.shape == (n, 50, 14)
            assert features.shape == (n, 1536) and meta["complete_pairs"] == n
            assert all(np.isfinite(a).all() for a in (features, state, expert, policy))
            if not args.feature_cache:
                dest = cache / f"scene_{scene}.npz"
                np.savez_compressed(dest, features=features)
            episodes.append(dict(scene_seed=scene, features=features, state=state,
                                 expert=expert, policy=policy, pair_sha256=pair_sha,
                                 feature_sha256=sha256(dest)))
            print(f"ENCODED {len(episodes)}/50 scene={scene} anchors={n}", flush=True)
            save_json(args.output / "state.json", dict(stage="encoding", episodes=len(episodes)))
    assert len(episodes) == len({e["scene_seed"] for e in episodes}) == 50
    assert sorted(e["scene_seed"] for e in episodes) == baseline_protocol["scene_seeds"]
    return episodes


def train(args, episodes, device, baseline):
    # Exactly the current H50 model's episode-uniform sampler and optimizer;
    # do NOT also switch to historical disc_s42's observation-uniform sampler.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    states = np.concatenate([e["state"] for e in episodes])
    actions = np.concatenate([e["expert"].reshape(-1, 14) for e in episodes])
    norm = {"state_mean": torch.from_numpy(states.mean(0)),
            "state_scale": torch.from_numpy(np.maximum(states.std(0), .05)),
            "action_mean": torch.from_numpy(actions.mean(0)),
            "action_scale": torch.from_numpy(np.maximum(actions.std(0), .05))}
    assert all(torch.equal(value, baseline["normalization"][key]) for key, value in norm.items())
    model = ChunkCoverageCritic(norm, horizon=50, width=256).to(device)
    assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == {
        k: tuple(v.shape) for k, v in baseline["model"].items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(42)
    sizes = np.asarray([len(e["state"]) for e in episodes])
    for step in range(1, args.steps + 1):
        ids = rng.integers(len(episodes), size=256)
        rows = [int(rng.integers(sizes[i])) for i in ids]
        picked = [episodes[i] for i in ids]
        batch = {key: torch.as_tensor(np.stack([e[key][j] for e, j in zip(picked, rows)]), device=device)
                 for key in ("features", "state", "expert", "policy")}
        optimizer.zero_grad(set_to_none=True)
        e = model(batch["features"], batch["state"], batch["expert"])
        p = model(batch["features"], batch["state"], batch["policy"])
        loss = .5 * (F.softplus(-e).mean() + F.softplus(p).mean())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite training loss, step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 500 == 0 or step == args.steps:
            row = dict(step=step, loss=float(loss.detach()),
                       expert_logit=float(e.detach().mean()), policy_logit=float(p.detach().mean()))
            with (args.output / "training_metrics.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            save_json(args.output / "state.json", dict(stage="training", **row))
            print("TRAIN", json.dumps(row), flush=True)
    torch.save({"protocol": "disc_s42_encoder_only_h50_v1", "task": TASK, "horizon": 50,
                "encoder": "ImageNet ResNet18 IMAGENET1K_V1 raw pooled features (no L2)",
                "steps": args.steps, "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "normalization": norm, "scene_seeds": sorted(e["scene_seed"] for e in episodes)},
               args.output / "model.pt")
    model.eval()
    ranked, count = 0., 0
    bce = []
    with torch.inference_mode():
        for episode in episodes:
            batch = {k: torch.as_tensor(episode[k], device=device)
                     for k in ("features", "state", "expert", "policy")}
            e = model(batch["features"], batch["state"], batch["expert"])
            p = model(batch["features"], batch["state"], batch["policy"])
            ranked += float((e > p).sum() + .5 * (e == p).sum())
            count += len(e)
            bce.extend((.5 * (F.softplus(-e) + F.softplus(p))).cpu().tolist())
    return model, dict(final_minibatch_loss=float(loss.detach()), anchors=count,
                       in_sample_expert_ranking=ranked / count,
                       in_sample_bce=float(np.mean(bce)),
                       warning="Training-set diagnostics only; not held-out expert-ranking accuracy.")


def score(args, model, encoder, device):
    evaluation = args.source / "outcome/formal" / TASK / "evaluation"
    candidates = args.source / "outcome_candidates/stack_blocks_two_h50_k4"
    meta = json.loads((evaluation / "metadata.json").read_text())
    records = [json.loads(line) for line in (evaluation / "episodes.jsonl").read_text().splitlines()]
    manifest = json.loads((candidates / "manifest.json").read_text())
    baseline_metrics = json.loads((args.source / "metrics/stack_blocks_two_h50_k4.json").read_text())
    assert meta["status"] == "complete" and meta["task"] == TASK and meta["method"] == "fixed_h50"
    assert len(records) == len({r["scene_seed"] for r in records}) == 50
    assert manifest["complete"] and manifest["candidate_count"] == 4 and manifest["action_horizon"] == 50
    assert manifest["trace_metadata_sha256"] == sha256(evaluation / "metadata.json")
    assert baseline_metrics["candidate_manifest_sha256"] == sha256(candidates / "manifest.json")
    trained = torch.load(args.output / "model.pt", map_location="cpu", weights_only=False)
    assert not set(trained["scene_seeds"]).intersection(r["scene_seed"] for r in records)
    rows = []
    for record in records:
        assert not record.get("infra_error") and not record.get("error")
        folder = evaluation / "trace" / f"episode_{record['episode']:03d}"
        trace = json.loads((folder / "trace.json").read_text())
        assert trace["complete"] and trace["scene_seed"] == record["scene_seed"]
        assert trace["success"] == record["success"]
        name = f"episode_{record['episode']:03d}_scene_{record['scene_seed']}.npz"
        assert sha256(candidates / name) == manifest["files"][name]["sha256"]
        with np.load(candidates / name, allow_pickle=False) as raw:
            actions = raw["candidates"].astype(np.float32)
        count = int(record["queries"])
        assert actions.shape == (count, 4, 50, 14) and len(trace["queries"]) == count
        assert np.isfinite(actions).all()
        states = []
        for i in range(count):
            with np.load(folder / f"query_{i:04d}.npz", allow_pickle=False) as raw:
                state = raw["state"].astype(np.float32)
                assert state.shape == (14,) and np.array_equal(actions[i, 0], raw["policy_h50"])
                states.append(state)
        blocks = []
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            images = []
            for i in range(count):
                with Image.open(folder / f"query_{i:04d}_{camera}.png") as image:
                    images.append(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
            blocks.append(encode(encoder, images, device))
        features = np.concatenate(blocks, axis=1)
        with torch.inference_mode():
            logits = model.score_many(torch.as_tensor(features, device=device),
                                      torch.as_tensor(np.stack(states), device=device),
                                      torch.as_tensor(actions, device=device)).float().cpu().numpy()
        assert logits.shape == (count, 4) and np.isfinite(logits).all()
        scores = logits.mean(1)
        rows.append(dict(scene_seed=record["scene_seed"], success=record["success"], queries=count,
                         candidate_logits=logits.tolist(), per_query_mean=scores.tolist(),
                         score_full=float(scores.mean())))
        print(f"SCORED {len(rows)}/50 scene={record['scene_seed']}", flush=True)
    identity = lambda r: (r["scene_seed"], r["success"], r["queries"])
    assert sorted(map(identity, rows)) == sorted(map(identity, baseline_metrics["per_episode"]))
    result = dict(complete=True, task=TASK, candidate_count=4, action_horizon=50,
                  d_checkpoint_sha256=sha256(args.output / "model.pt"),
                  candidate_manifest_sha256=sha256(candidates / "manifest.json"),
                  per_episode=rows)
    save_json(args.output / "outcome_scores.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--encoder-asset", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or bool(args.feature_cache) != bool(args.encoder_asset):
        parser.error("positive steps required; feature cache and encoder asset must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    baseline_path = args.source / "models" / TASK / "model.pt"
    protocol_path = baseline_path.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["status"] == "complete" and protocol["seed"] == 42
    assert protocol["training_steps"] == 10000 and protocol["batch_size"] == 256
    assert sha256(baseline_path) == "8019dd5893514283e44e17cf6984fcc3b97f2f505ef0a96e6b66eea070a3da05"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    try:
        url = ResNet18_Weights.IMAGENET1K_V1.url
        if args.encoder_asset:
            asset = args.encoder_asset
            assert asset.name == url.rsplit("/", 1)[-1] and asset.is_file()
        else:
            assets = args.output / "assets"
            assets.mkdir()
            save_json(args.output / "state.json", dict(stage="download_imagenet_encoder"))
            torch.hub.load_state_dict_from_url(url, model_dir=str(assets), check_hash=True, progress=False)
            asset = assets / url.rsplit("/", 1)[-1]
        encoder = FrozenEncoder(device, weights_path=asset).model
        episodes = prepare(args, encoder, device, protocol)
        model, train_metrics = train(args, episodes, device, baseline)
        save_json(args.output / "state.json", dict(stage="scoring_frozen_outcomes"))
        result = score(args, model, encoder, device)
        protocol_new = dict(complete=True, task=TASK, baseline_checkpoint_sha256=sha256(baseline_path),
                            baseline_protocol_sha256=sha256(protocol_path),
                            changed=("Only frozen visual encoder weights: R3M18 -> original disc_s42 ImageNet18"
                                     if args.steps == 10000 else
                                     f"ImageNet18 encoder, {args.steps} updates; compare steps against the ImageNet18 10000-update run"),
                            preserved=["same 50 expert episodes", "same 7872 paired observations", "same K1 train policy chunks",
                                       "H50 entire action input", "same normalization", "same MLP width256 / 3 layers",
                                       "same episode-uniform sampler/seed42/batch256", f"AdamW lr=wd=1e-4 / {args.steps} updates",
                                       "same frozen 50 test trajectories and K4 candidates"],
                            not_full_historical_replication=["Historical disc_s42 used H10, K4 train negatives and observation-uniform sampling.",
                                                             "Here H50/K1/episode-uniform are retained to isolate the encoder change."],
                            encoder_url=url, encoder_sha256=sha256(asset), preprocessing_equality_verified=True,
                            feature_cache_source=str(args.feature_cache) if args.feature_cache else None,
                            training_steps=args.steps,
                            head_sha256=sha256(args.source / "code/tail_value/chunk.py"),
                            script_sha256=sha256(Path(__file__)), training=train_metrics,
                            source_pairs={str(e["scene_seed"]): e["pair_sha256"] for e in episodes},
                            features={str(e["scene_seed"]): e["feature_sha256"] for e in episodes},
                            d_checkpoint_sha256=result["d_checkpoint_sha256"],
                            candidate_manifest_sha256=result["candidate_manifest_sha256"],
                            result_sha256=sha256(args.output / "outcome_scores.json"))
        save_json(args.output / "protocol.json", protocol_new)
        save_json(args.output / "state.json", dict(stage="complete", training=train_metrics))
        print("COMPLETE", json.dumps(train_metrics), flush=True)
    except Exception as exc:
        save_json(args.output / "failure.json", dict(error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
