#!/usr/bin/env python3
"""Paired offline diagnostics for the H=10 coverage critic alpha sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coverage_benchmark.common import JOINTS, atomic_json, calibrate, sha256
from coverage_benchmark.scorers import NeuralScorer


ALPHAS = (0, 1, 3, 10)
STEPS = (1000, 5000, 10000, 20000)


def load_validation(path):
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    expected = {'cal': 430, 'diag': 844}
    if {name: int(np.sum(data['split'] == name)) for name in expected} != expected:
        raise ValueError('Changed calibration/diagnostic split')
    if len(np.unique(data['episode'][data['split'] == 'diag'])) != 30:
        raise ValueError('Expected 30 diagnostic episodes')
    if data['expert_chunks'].shape != (1274, 10, 14) or data['policy_chunks'].shape != (1274, 4, 10, 14):
        raise ValueError('Expected complete 10-action chunks')
    return data


def load_hil(k32_dir, hil_dir):
    k32_dir, hil_dir = Path(k32_dir), Path(hil_dir)
    km = json.loads((k32_dir / 'manifest.json').read_text())
    hm = json.loads((hil_dir / 'candidate_manifest.json').read_text())
    if not km.get('complete') or len(km['observations']) != 1260:
        raise ValueError('K32 cache is incomplete')
    episodes = hm['identity']['episodes']
    if not hm.get('complete') or len(episodes) != 50:
        raise ValueError('HIL cohort is incomplete')
    selected = {int(e['rollout_index']) for e in episodes}
    rows = [row for row in km['observations'] if int(row['rollout_index']) in selected]
    if len(rows) != 1233:
        raise ValueError(f'Expected 1233 saved-HIL pre-takeover observations, got {len(rows)}')
    needed = {(int(r['rollout_index']), int(Path(r['frame_path']).stem)) for r in rows}
    lookup = {}
    for episode in episodes:
        idx = int(episode['rollout_index'])
        name = f'rollout_{idx:04d}'
        file = hil_dir / f'{name}.npz'
        if sha256(file) != hm['completed'][name]:
            raise ValueError(f'Changed HIL features: {file}')
        with np.load(file, allow_pickle=False) as archive:
            frames, features, states = archive['frame_index'], archive['features'], archive['state']
            for frame, feature, state in zip(frames, features, states):
                key = (idx, int(frame))
                if key in needed:
                    lookup[key] = (feature.copy(), state.copy())
    arrays = {key: [] for key in ('features', 'state', 'candidates', 'order', 'episode', 'policy_step', 'progress')}
    for row in rows:
        idx = int(row['rollout_index'])
        frame = int(Path(row['frame_path']).stem)
        feature, state = lookup[(idx, frame)]
        file = k32_dir / row['cache']
        if sha256(file) != km['completed'][row['cache']]:
            raise ValueError(f'Changed candidate cache: {file}')
        with np.load(file, allow_pickle=False) as archive:
            candidates, order, cache_state = (archive[k] for k in ('candidates', 'subset_order', 'state'))
        if candidates.shape != (32, 10, 14) or sorted(order.tolist()) != list(range(32)):
            raise ValueError(f'Invalid candidate shape or order: {file}')
        np.testing.assert_allclose(state, cache_state, atol=1e-6, rtol=0)
        for key, value in dict(features=feature, state=state, candidates=candidates, order=order,
                               episode=idx, policy_step=row['policy_step'], progress=row['progress']).items():
            arrays[key].append(value)
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    return arrays, {'k32_manifest_sha256': sha256(k32_dir / 'manifest.json'),
                    'hil_manifest_sha256': sha256(hil_dir / 'candidate_manifest.json')}


def score_batches(scorer, features, state, chunks, batch_size=64):
    outputs = []
    model = scorer.model
    device = scorer.device
    with torch.inference_mode():
        for start in range(0, len(state), batch_size):
            part = slice(start, start + batch_size)
            f = torch.as_tensor(features[part], dtype=torch.float32, device=device)
            s = torch.as_tensor(state[part], dtype=torch.float32, device=device)
            c = torch.as_tensor(chunks[part], dtype=torch.float32, device=device)
            if c.ndim == 3:
                value = model(f, s, c)
            else:
                value = model.score_many(f, s, c)
            outputs.append(value.cpu().numpy().astype(np.float64))
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite coverage scores')
    return result


def weighted_metrics(scores, chunks, calibration):
    n, k = scores.shape
    scale, beta = calibration['scale'], calibration['beta']
    if calibration['degenerate'] or beta == 0:
        w = np.full((n, k), 1 / k)
    else:
        z = -beta * (scores - scores.min(axis=1, keepdims=True)) / scale
        w = np.exp(z - z.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
    weighted = np.einsum('nk,nkhd->nhd', w, chunks.astype(np.float64))
    uniform = chunks.mean(axis=1, dtype=np.float64)
    shift_mrad = np.sqrt(np.mean((weighted[:, :, JOINTS] - uniform[:, :, JOINTS]) ** 2,
                                      axis=(1, 2))) * 1000
    return {'range': np.ptp(scores, axis=1),
            'normalized_range': np.ptp(scores, axis=1) / scale if scale > 1e-12 else np.zeros(n),
            'ess_fraction': 1 / np.sum(w * w, axis=1) / k,
            'max_weight': w.max(axis=1), 'tilted_joint_shift_mrad': shift_mrad}


def quantiles(values):
    values = np.asarray(values, np.float64)
    return {'mean': float(np.mean(values)), 'q05': float(np.quantile(values, .05)),
            'median': float(np.median(values)), 'q95': float(np.quantile(values, .95))}


def expert_rank(expert, candidates, episodes):
    ranks = ((expert[:, None] > candidates).sum(axis=1)
             + .5 * (expert[:, None] == candidates).sum(axis=1)) / candidates.shape[1]
    by_episode = np.asarray([ranks[episodes == ep].mean() for ep in np.unique(episodes)])
    rng = np.random.default_rng(42)
    draws = by_episode[rng.integers(len(by_episode), size=(10000, len(by_episode)))].mean(axis=1)
    return {'episode_balanced_mean': float(by_episode.mean()),
            'episode_bootstrap_ci95': np.quantile(draws, [.025, .975]).tolist(),
            'observation_weighted_mean': float(ranks.mean()),
            'episodes': len(by_episode), 'observations': len(ranks)}


def paired_rank_differences(checkpoints, validation):
    """Compare alphas on the same 30 episodes, at matched optimizer steps."""
    v = validation
    diag = v['split'] == 'diag'
    features, states = v['features'][diag], v['state'][diag]
    expert, candidates = v['expert_chunks'][diag], v['policy_chunks'][diag]
    episodes = v['episode'][diag]
    unique = np.unique(episodes)
    scores = {}
    for alpha in ALPHAS:
        for step in STEPS:
            scorer = NeuralScorer(checkpoints / f'alpha_{alpha}_step_{step}.pt')
            e = score_batches(scorer, features, states, expert)
            p = score_batches(scorer, features, states, candidates)
            rank = ((e[:, None] > p).sum(axis=1) + .5 * (e[:, None] == p).sum(axis=1)) / 4
            scores[(alpha, step)] = np.asarray([rank[episodes == ep].mean() for ep in unique])
    rng = np.random.default_rng(42)
    ix = rng.integers(len(unique), size=(10000, len(unique)))
    result = {}
    for step in STEPS:
        baseline = scores[(1, step)]
        result[str(step)] = {}
        for alpha in (0, 3, 10):
            diff = scores[(alpha, step)] - baseline
            result[str(step)][str(alpha)] = {'alpha_minus_alpha1_episode_mean': float(diff.mean()),
                'paired_bootstrap_ci95': np.quantile(diff[ix].mean(axis=1), [.025, .975]).tolist(),
                'positive_episode_fraction': float(np.mean(diff > 0))}
    return result


def analyze_one(path, validation, hil):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    scorer = NeuralScorer(path)
    v = validation
    e = score_batches(scorer, v['features'], v['state'], v['expert_chunks'])
    p = score_batches(scorer, v['features'], v['state'], v['policy_chunks'])
    cal, diag = v['split'] == 'cal', v['split'] == 'diag'
    calibration = calibrate(e[cal], p[cal], beta_max=100.)
    diag_metrics = weighted_metrics(p[diag], v['policy_chunks'][diag], calibration)
    h = hil
    h_scores = score_batches(scorer, h['features'], h['state'], h['candidates'], batch_size=16)
    h_scores = np.take_along_axis(h_scores, h['order'], axis=1)
    h_chunks = np.take_along_axis(h['candidates'], h['order'][:, :, None, None], axis=1)
    hil_metrics = weighted_metrics(h_scores, h_chunks, calibration)
    result = {'checkpoint_sha256': sha256(path), 'step': int(payload['step']),
              'alpha': float(payload['config']['alpha']),
              'cache_fingerprint': payload['cache_fingerprint'],
              'validation_td_mse': float(payload['validation']['td']),
              'validation_expert_mean': float(payload['validation']['q_expert']['mean']),
              'validation_policy_mean': float(payload['validation']['q_policy']['mean']),
              'calibration': calibration,
              'diagnostic_expert_rank': expert_rank(e[diag], p[diag], v['episode'][diag]),
              'diagnostic_expert_minus_policy': quantiles(e[diag] - p[diag].mean(axis=1)),
              'diagnostic_candidate_metrics': {name: quantiles(value) for name, value in diag_metrics.items()},
              'hil_k32_candidate_metrics': {name: quantiles(value) for name, value in hil_metrics.items()},
              'hil_observations': len(h_scores), 'hil_episodes': len(np.unique(h['episode']))}
    return result


def make_plots(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {0: '#777777', 1: '#286E9A', 3: '#CF7E2F', 10: '#884EA0'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    panels = (
        ('validation_expert_mean', 'Validation expert score'),
        ('validation_td_mse', 'Validation TD MSE'),
        ('diagnostic_expert_rank', 'Expert rank against K=4 candidates'),
        ('hil_k32_candidate_metrics', 'HIL K=32 ESS / K'),
    )
    for ax, (key, title) in zip(axes.flat, panels):
        for alpha in ALPHAS:
            rows = [results[f'alpha_{alpha}_step_{step}'] for step in STEPS]
            if key == 'diagnostic_expert_rank':
                y = [r[key]['episode_balanced_mean'] for r in rows]
            elif key == 'hil_k32_candidate_metrics':
                y = [r[key]['ess_fraction']['median'] for r in rows]
            else:
                y = [r[key] for r in rows]
            ax.plot(STEPS, y, marker='o', color=colors[alpha], label=f'alpha={alpha}')
        ax.set(xlabel='Training steps', title=title)
        if key in ('diagnostic_expert_rank', 'hil_k32_candidate_metrics'):
            ax.set_ylim(0, 1.03)
        ax.grid(alpha=.18)
    axes[0, 0].legend()
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'alpha_training_and_offline.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    for ax, (metric, title) in zip(axes, (
        ('range', 'HIL K=32 score range'),
        ('normalized_range', 'HIL K=32 range / calibration SD'),
        ('tilted_joint_shift_mrad', 'Weighted vs uniform chunk RMS (mrad)'),
    )):
        for alpha in ALPHAS:
            y = [results[f'alpha_{alpha}_step_{step}']['hil_k32_candidate_metrics'][metric]['median']
                 for step in STEPS]
            ax.plot(STEPS, y, marker='o', color=colors[alpha], label=f'alpha={alpha}')
        ax.set(xlabel='Training steps', title=title, yscale='log')
        ax.grid(alpha=.18)
    axes[0].legend()
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'alpha_hil_candidate_effect.{ext}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints-dir', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--k32', type=Path, required=True)
    parser.add_argument('--hil', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = load_validation(args.validation)
    hil, hil_sources = load_hil(args.k32, args.hil)
    sources = {'validation_sha256': sha256(args.validation), **hil_sources}
    results = {}
    for alpha in ALPHAS:
        for step in STEPS:
            name = f'alpha_{alpha}_step_{step}'
            path = args.checkpoints_dir / f'{name}.pt'
            cached = args.output_dir / f'{name}.json'
            if cached.exists():
                old = json.loads(cached.read_text())
                if old['checkpoint_sha256'] != sha256(path):
                    raise ValueError(f'Changed checkpoint: {path}')
                result = old
            else:
                result = analyze_one(path, validation, hil)
                if result['alpha'] != alpha or result['step'] != step:
                    raise ValueError(f'Checkpoint identity mismatch: {path}')
                atomic_json(cached, result)
            results[name] = result
            print(name, 'expert_rank', result['diagnostic_expert_rank']['episode_balanced_mean'],
                  'hil_ess', result['hil_k32_candidate_metrics']['ess_fraction']['median'], flush=True)
    fingerprints = {r['cache_fingerprint'] for r in results.values()}
    if len(fingerprints) != 1:
        raise ValueError('Training caches differ across alpha groups')
    make_plots(results, args.output_dir)
    paired = paired_rank_differences(args.checkpoints_dir, validation)
    atomic_json(args.output_dir / 'metrics.json', {
        'scope': '45 internal SFT validation episodes (15 calibration, 30 diagnostic) and 50 saved HIL trajectories',
        'sources': sources, 'cache_fingerprint': fingerprints.pop(), 'results': results,
        'paired_expert_rank_differences': paired,
        'interpretation': 'Offline diagnostics only; expert rank is not a success label; HIL scores are not independent heldout.',
        'calibration_limit': 'beta<=100; methods may not attain ESS/K=0.5',
    })
    files = ['metrics.json', 'alpha_training_and_offline.png', 'alpha_training_and_offline.pdf',
             'alpha_hil_candidate_effect.png', 'alpha_hil_candidate_effect.pdf']
    files += [f'alpha_{alpha}_step_{step}.json' for alpha in ALPHAS for step in STEPS]
    atomic_json(args.output_dir / 'manifest.json', {'complete': True, 'sources': sources,
               'files': {name: sha256(args.output_dir / name) for name in files}})


if __name__ == '__main__':
    main()
