#!/usr/bin/env python3
"""Verify and export inference-only checkpoints for the H=10 alpha sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tail_value.cache import atomic_json, sha256
from tail_value.model import atomic_checkpoint

ALPHAS = (0, 1, 3, 10)
STEPS = (1000, 5000, 10000, 20000)


def source_path(alpha, step, sweep, baseline):
    if alpha == 1:
        name = 'run_full_h10_alpha1_10000' if step <= 10000 else 'run_full_h10_alpha1_20000'
        return baseline / name / f'step_{step:07d}.pt'
    return sweep / f'alpha_{alpha}' / f'step_{step:07d}.pt'


def first_model_hash(path):
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get('event') == 'start' and record.get('step') == 0:
            return record['model_sha256']
    raise ValueError(f'Missing initial model hash: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    starts = {1: first_model_hash(args.baseline_root / 'run_full_h10_alpha1_10000' / 'metrics.jsonl')}
    for alpha in (0, 3, 10):
        starts[alpha] = first_model_hash(args.sweep_root / f'alpha_{alpha}' / 'metrics.jsonl')
    if len(set(starts.values())) != 1:
        raise ValueError(f'Model initializations differ: {starts}')

    checkpoint_sources = {f'alpha_{alpha}_step_{step}.pt': source_path(alpha, step, args.sweep_root, args.baseline_root)
                          for alpha in ALPHAS for step in STEPS}
    for path in checkpoint_sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    reference, fingerprint, normalization = None, None, None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, source in checkpoint_sources.items():
        alpha = int(name.split('_')[1])
        step = int(name.split('_')[3].removesuffix('.pt'))
        payload = torch.load(source, map_location='cpu', weights_only=False)
        if payload['representation'] != 'absolute_joint_chunk' or payload['step'] != step or payload['config']['alpha'] != alpha:
            raise ValueError(f'Wrong representation, step, or alpha: {source}')
        if int(payload['config']['horizon']) != 10 or not payload.get('validation'):
            raise ValueError(f'Missing H=10 validation: {source}')
        config = dict(payload['config'])
        config.pop('alpha')
        if reference is None:
            reference, fingerprint, normalization = config, payload['cache_fingerprint'], payload['normalization']
        elif config != reference or payload['cache_fingerprint'] != fingerprint:
            raise ValueError(f'Training configurations or caches differ: {source}')
        for key in normalization:
            torch.testing.assert_close(torch.as_tensor(payload['normalization'][key]),
                                       torch.as_tensor(normalization[key]), rtol=0, atol=0)
        dest = args.output_dir / name
        export = {key: payload[key] for key in ('schema_version', 'representation', 'step', 'config',
                                                'normalization', 'model', 'cache_fingerprint', 'validation')}
        export['source_sha256'] = sha256(source)
        if dest.exists():
            previous = torch.load(dest, map_location='cpu', weights_only=False)
            if previous['source_sha256'] != export['source_sha256']:
                raise ValueError(f'Existing export differs: {dest}')
        else:
            atomic_checkpoint(dest, export)
        files[name] = {'sha256': sha256(dest), 'source': str(source), 'source_sha256': export['source_sha256']}
        print(name, 'val_td', payload['validation']['td'], flush=True)
    atomic_json(args.output_dir / 'manifest.json', {
        'complete': True, 'alphas': ALPHAS, 'steps': STEPS, 'seed': 42,
        'initial_model_sha256': next(iter(starts.values())),
        'cache_fingerprint': fingerprint, 'config_except_alpha': reference, 'files': files,
    })


if __name__ == '__main__':
    main()
