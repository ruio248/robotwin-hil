#!/usr/bin/env python3
"""Train an episode-disjoint source discriminator on a frozen train50/test20 cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'robotwin_hil/overlay/scripts'))
from coverage_benchmark import train_screened_task_discriminators as base
from tail_value.chunk import ChunkCoverageCritic


def digest(path: Path) -> str:
    hash_value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hash_value.update(block)
    return hash_value.hexdigest()


def load_cohort(data_root: Path, task: str, candidates: int = 4) -> tuple[list[dict], list[dict], dict]:
    manifest_path = data_root / task / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    horizon = 50 if task == 'click_alarmclock' else 10
    if not manifest.get('complete') or manifest.get('cohorts') != {'train': 50, 'test': 20}:
        raise ValueError('Expected complete train50/test20 manifest')
    rows = {'train': [], 'test': []}
    for entry in manifest['episodes']:
        if task == 'click_alarmclock':
            paths = entry['files']
            with np.load(data_root / task / paths['pairs.npz']) as z:
                state = z['state']
                expert = z['expert_chunks']
            with np.load(data_root / task / paths['features_r3m18.npz']) as z:
                features = z['features']
            with np.load(data_root / task / paths['policy_candidates_k4.npz']) as z:
                policy = z['policy_chunks']
            seed = entry['scene_seed']
            file_hash = hashlib.sha256(''.join(
                digest(data_root / task / paths[name]) for name in
                ('pairs.npz', 'features_r3m18.npz', 'policy_candidates_k4.npz')
            ).encode()).hexdigest()
        else:
            file = data_root / task / entry['file']
            with np.load(file) as z:
                state, expert = z['state'], z['expert_chunks']
                features, policy = z['features'], z['policy_chunks']
            seed = entry.get('scene_seed', entry.get('id'))
            file_hash = digest(file)
        n = len(state)
        if (state.shape != (n, 14) or expert.shape != (n, horizon, 14)
                or features.shape != (n, 1536) or policy.shape != (n, candidates, horizon, 14)
                or not all(np.isfinite(x).all() for x in (state, expert, features, policy))):
            raise ValueError(f'Invalid paired cohort row: {entry}')
        rows[entry['split']].append({
            'episode': str(seed), 'scene_seed': str(seed),
            'state': state.astype(np.float32), 'expert': expert.astype(np.float32),
            'features': features.astype(np.float32), 'policy': policy.astype(np.float32),
            'sha256': file_hash,
        })
    if len(rows['train']) != 50 or len(rows['test']) != 20:
        raise ValueError('Expected exactly 50 train and 20 test episodes')
    return rows['train'], rows['test'], {'sha256': digest(manifest_path), 'horizon': horizon}


def score_rows(model, rows, device, calibration):
    reports = []
    for episode in rows:
        expert, policy = base.score_episode(model, episode, device)
        scale = max(calibration['scale'], 1e-12)
        weights = np.exp(-(policy - policy.min(axis=1, keepdims=True)) / scale)
        weights /= weights.sum(axis=1, keepdims=True)
        uniform = episode['policy'].mean(axis=1)
        weighted = np.einsum('nk,nkhd->nhd', weights, episode['policy'])
        shift = weighted - uniform
        ranks = (expert[:, None] > policy).mean(axis=1) + .5 * (expert[:, None] == policy).mean(axis=1)
        reports.append({
            'episode': episode['episode'], 'anchors': len(expert),
            'expert_rank': float(ranks.mean()),
            'expert_logit_mean': float(expert.mean()),
            'policy_logit_mean': float(policy.mean()),
            'candidate_pair_gap_mean': float(np.abs(policy[:, :, None] - policy[:, None, :])[:, np.triu_indices(4, 1)[0], np.triu_indices(4, 1)[1]].mean()),
            'score_range_mean': float(np.ptp(policy, axis=1).mean()),
            'score_std_mean': float(policy.std(axis=1).mean()),
            'ess_over_k_beta1': float((1 / (4 * np.square(weights).sum(axis=1))).mean()),
            'max_weight_beta1': float(weights.max(axis=1).mean()),
            'weighted_joint_rms_mrad_beta1': float(np.sqrt(np.square(shift[:, :, [0,1,2,3,4,5,7,8,9,10,11,12]]).mean(axis=(1,2))).mean() * 1000),
            'weighted_gripper_rms_beta1': float(np.sqrt(np.square(shift[:, :, [6,13]]).mean(axis=(1,2))).mean()),
        })
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('click_alarmclock', 'place_dual_shoes', 'place_empty_cup'), required=True)
    parser.add_argument('--data-root', type=Path, default=REPO / 'data')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    train, test, source = load_cohort(args.data_root, args.task)
    print(json.dumps({'task': args.task, 'train': len(train), 'test': len(test), 'source': source}), flush=True)
    if args.verify_only:
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    output = args.output_root / args.task
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    groups = [list(x) for x in np.array_split(rng.permutation(50), 5)]
    oof_expert, oof_policy = [], []
    folds = []
    for fold in range(5):
        validation = groups[fold]
        fit = [i for i in range(50) if i not in set(validation)]
        base.seed_all(42 + fold)
        norm = base.normalization(train, fit)
        model = ChunkCoverageCritic(norm, horizon=source['horizon'], width=256).to(device)
        fold_dir = output / f'fold_{fold}'
        base.train_fold(model, norm, train, fit, fold_dir, steps=args.steps,
                        batch_size=args.batch_size, seed=42 + fold, save_every=1000)
        for index in validation:
            expert, policy = base.score_episode(model, train[index], device)
            oof_expert.append(expert)
            oof_policy.append(policy)
        folds.append({'fold': fold, 'fit': [train[i]['episode'] for i in fit],
                      'validation': [train[i]['episode'] for i in validation],
                      'checkpoint_sha256': digest(fold_dir / 'last.pt')})
        base.atomic_json(output / 'progress.json', {'completed_folds': len(folds), 'folds': folds})
    calibration = base.calibrate(np.concatenate(oof_expert), np.concatenate(oof_policy), target=.5, beta_max=100.)
    base.seed_all(1042)
    norm = base.normalization(train, list(range(50)))
    model = ChunkCoverageCritic(norm, horizon=source['horizon'], width=256).to(device)
    final = output / 'final_full50'
    base.train_fold(model, norm, train, list(range(50)), final, steps=args.steps,
                    batch_size=args.batch_size, seed=1042, save_every=1000)
    report = {'task': args.task, 'train_episodes': 50, 'test_episodes': 20,
              'candidate_count': 4, 'steps': args.steps, 'batch_size': args.batch_size,
              'source': source, 'calibration': calibration,
              'train_episode_sha256': {row['episode']: row['sha256'] for row in train},
              'test_episode_sha256': {row['episode']: row['sha256'] for row in test},
              'folds': folds, 'test20': score_rows(model, test, device, calibration),
              'checkpoint_sha256': digest(final / 'last.pt'),
              'beta1_score': 'raw logit divided by train-only OOF scale'}
    base.atomic_json(output / 'report.json', report)
    print(json.dumps({'complete': True, 'task': args.task, 'report': str(output / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
