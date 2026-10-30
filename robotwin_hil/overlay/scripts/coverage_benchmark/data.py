from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from tail_value.chunk import load_chunk_manifest, validate_chunk_episode
from .common import sha256, atomic_json


def sft_data(root, expected=True):
    root = Path(root); manifest = load_chunk_manifest(root)
    entries = sorted(manifest['episodes'], key=lambda e:e['file'])
    val = [e['file'] for e in entries if e['split']=='val']
    rng = np.random.default_rng(42); val = [val[i] for i in rng.permutation(len(val))]
    calibration = set(val[:15])
    arrays = {k:[] for k in ('features','state','expert_chunks','policy_chunks','episode','frame_index','split')}
    sources = {}
    for eid, entry in enumerate(entries):
        path = root/entry['file']
        if sha256(path) != entry['sha256']: raise ValueError(f'Changed SFT cache: {path}')
        with np.load(path, allow_pickle=False) as f: d = dict(f)
        validate_chunk_episode(d,horizon=10,candidates=4)
        keep = d['valid_transition']
        split = 'train' if entry['split']=='train' else ('cal' if entry['file'] in calibration else 'diag')
        for key in ('features','state','expert_chunks','policy_chunks','frame_index'): arrays[key].append(d[key][keep])
        arrays['episode'].append(np.full(keep.sum(),eid))
        arrays['split'].append(np.full(keep.sum(),split))
        sources[str(eid)] = dict(entry, analysis_split=split)
    arrays = {k:np.concatenate(v) for k,v in arrays.items()}
    if expected and (sum(arrays['split']=='train') != 11461 or sum(arrays['split']!='train') != 1274
                     or len(calibration)!=15 or len(val)!=45):
        raise ValueError('SFT cohort/count changed')
    return arrays, {'manifest_sha256':sha256(root/'manifest.json'), 'episodes':sources,
                    'calibration_episodes':sorted(calibration), 'diagnostic_episodes':sorted(set(val)-calibration)}


def subset(data, split): return {k:v[data['split']==split] for k,v in data.items()}
