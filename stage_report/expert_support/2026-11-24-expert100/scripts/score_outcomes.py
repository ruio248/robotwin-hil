#!/usr/bin/env python3
"""Score saved SFT policy queries with a frozen expert-source discriminator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'robotwin_hil/overlay/scripts'))
from coverage_benchmark.encode_matched_pairs_r3m import image_tensor, r3m_state
from tail_value.chunk import ChunkCoverageCritic
from train import digest
from coverage_benchmark.train_screened_task_discriminators import atomic_json


def source_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    comparison = positive[:, None] - negative[None, :]
    return float(((comparison > 0).sum() + .5 * (comparison == 0).sum()) / comparison.size)


def load_encoder(asset: Path, kind: str, device: torch.device):
    encoder = resnet18(weights=None)
    encoder.fc = torch.nn.Identity()
    payload = torch.load(asset, map_location='cpu', weights_only=True)
    weights = (r3m_state(payload, encoder.state_dict()) if kind == 'r3m'
               else {key: value for key, value in payload.items() if not key.startswith('fc.')})
    encoder.load_state_dict(weights, strict=True)
    return encoder.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def encode_views(encoder, images: list[list[np.ndarray]], device):
    views = []
    for camera_images in images:
        vectors = []
        for start in range(0, len(camera_images), 32):
            batch = torch.stack([image_tensor(frame) for frame in camera_images[start:start + 32]]).to(device)
            vectors.append(encoder(batch).float().cpu().numpy())
        views.append(np.concatenate(vectors))
    return np.concatenate(views, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('click_alarmclock', 'place_dual_shoes', 'place_empty_cup'), required=True)
    parser.add_argument('--trace-manifest', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--encoder-asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    report = json.loads((args.training_root / args.task / 'report.json').read_text())
    if report['task'] != args.task or report['test_episodes'] != 20:
        raise ValueError('Training report mismatch')
    expected_encoder_sha = ('f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
                            if args.task == 'place_empty_cup' else
                            '58d7bfcda80d81a4c54582e0dd3b0a73aaa2cafff111325eff729419ea989c1e')
    if digest(args.encoder_asset) != expected_encoder_sha:
        raise ValueError('Encoder weight SHA mismatch')
    encoder = load_encoder(args.encoder_asset,
                           'imagenet' if args.task == 'place_empty_cup' else 'r3m', device)
    checkpoint = args.training_root / args.task / report.get('checkpoint_relative', 'final_full50/last.pt')
    if digest(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Discriminator checkpoint SHA mismatch')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = ChunkCoverageCritic(payload['normalization'], horizon=report['source']['horizon'], width=256)
    model.load_state_dict(payload['model'], strict=True)
    model = model.to(device).eval()
    trace = json.loads(args.trace_manifest.read_text())
    if (trace['counted_official_episodes'] != 50 or trace['infra_error_episodes'] != 0
            or trace['query_covered_official_episodes'] != 50):
        raise ValueError('Expected 50 complete, query-covered SFT episodes')
    rows = []
    for episode in trace['episodes']:
        if not episode['counted_in_official_result']:
            continue
        states, actions = [], []
        images = [[], [], []]
        for query in episode['queries']:
            with np.load(args.trace_manifest.parent / episode['directory'] / query['file']) as z:
                states.append(np.concatenate([z['state_left_arm_joint_state'], z['state_left_ee_joint_state'],
                                              z['state_right_arm_joint_state'], z['state_right_ee_joint_state']]))
                actions.append(z['absolute_joint_chunk'])
                for camera, name in enumerate(('rgb_cam_head', 'rgb_cam_left_wrist', 'rgb_cam_right_wrist')):
                    images[camera].append(z[name])
        if not actions or any(a.shape != (report['source']['horizon'], 14) for a in actions):
            raise ValueError(f'Invalid query actions for seed {episode["scene_seed"]}')
        features = encode_views(encoder, images, device)
        with torch.inference_mode():
            scores = model(torch.as_tensor(features, device=device),
                           torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
                           torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device)).cpu().numpy()
        rows.append({'scene_seed': episode['scene_seed'], 'success': episode['success'],
                     'queries': len(scores), 'score_mean': float(scores.mean()),
                     'score_min': float(scores.min())})
        print(json.dumps({'episode': len(rows), 'seed': episode['scene_seed']}), flush=True)
    positive = np.asarray([r['score_mean'] for r in rows if r['success']])
    negative = np.asarray([r['score_mean'] for r in rows if not r['success']])
    if len(rows) != 50 or len(positive) == 0 or len(negative) == 0:
        raise ValueError('Incomplete success/failure groups')
    rng = np.random.default_rng(42)
    bootstraps = [source_auc(rng.choice(positive, len(positive), replace=True),
                             rng.choice(negative, len(negative), replace=True)) for _ in range(10000)]
    result = {'task': args.task, 'episodes': rows, 'successes': len(positive), 'failures': len(negative),
              'success_score_mean': float(positive.mean()), 'failure_score_mean': float(negative.mean()),
              'success_auc': source_auc(positive, negative),
              'auc_ci95': np.quantile(bootstraps, [.025, .975]).tolist(),
              'checkpoint_sha256': report['checkpoint_sha256'],
              'trace_manifest_sha256': digest(args.trace_manifest),
              'encoder_sha256': expected_encoder_sha}
    atomic_json(args.output, result)


if __name__ == '__main__':
    main()
