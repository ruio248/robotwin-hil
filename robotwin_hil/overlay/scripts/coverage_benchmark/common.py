from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tail_value.cache import atomic_json, atomic_npz, sha256

JOINTS = np.r_[0:6, 7:13]
KS = (4, 8, 16, 32)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_config(path, profile):
    config = json.loads(Path(path).read_text())
    config['paths'] = config['profiles'][profile]
    config['protocol_sha256'] = digest({k: v for k, v in config.items()
                                      if k not in ('paths', 'profiles')})
    return config


def identity(output, config, resume):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'identity.json'
    value = {'protocol_sha256': config['protocol_sha256'], 'protocol': config['protocol']}
    if path.exists():
        if not resume or json.loads(path.read_text()) != value:
            raise ValueError('Existing output requires --resume and identical protocol')
    else:
        atomic_json(path, value)


def verify_fit(root, protocol_sha256=None):
    """Fail closed on partial transfers or changed frozen calibration assets."""
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    if not manifest['complete'] or (protocol_sha256 is not None and manifest['protocol_sha256']!=protocol_sha256):
        raise ValueError('Fit incomplete or protocol mismatch')
    if json.loads((root/'models.json').read_text())!=manifest['models']:
        raise ValueError('Scorer index differs from frozen fit manifest')
    expected={r['file']:r['sha256'] for r in manifest['models'].values()}
    expected.update({name+'.json':manifest[name+'_sha256'] for name in ('split','calibration')})
    expected['validation.npz']=manifest['validation_sha256']
    for name,digest in expected.items():
        if sha256(root/name)!=digest:raise ValueError(f'Incomplete or changed fit asset: {name}')
    return manifest


def validate_chunks(chunks):
    a = np.asarray(chunks)
    if a.ndim != 3 or a.shape[1:] != (10, 14) or not np.isfinite(a).all():
        raise ValueError('Expected finite physical absolute joint chunks [K,10,14]')
    return a


def weights(scores, calibration):
    b = np.asarray(scores, np.float64)
    if b.ndim != 1 or not np.isfinite(b).all() or len(b) < 1:
        raise ValueError('Invalid scores')
    if calibration['degenerate'] or calibration['beta'] == 0:
        return np.full(len(b), 1 / len(b))
    logits = -calibration['beta'] * (b - b.min()) / calibration['scale']
    w = np.exp(logits - logits.max())
    return w / w.sum()


def calibrate(expert, policy, target=.5, beta_max=100.):
    """One global balanced-class scale; never normalize within an observation."""
    e, p = np.asarray(expert, np.float64), np.asarray(policy, np.float64)
    mean = .5 * (e.mean() + p.mean())
    scale = float(np.sqrt(.5 * (np.mean((e-mean)**2) + np.mean((p-mean)**2))))
    c = dict(mean=float(mean), scale=scale, beta=0., target_ess_fraction=target,
             beta_max=beta_max, degenerate=bool(scale < 1e-12), temperature_fit_k=p.shape[1])
    if c['degenerate'] or np.max(np.ptp(p, axis=1)) < 1e-12:
        c.update(degenerate=True, achieved_ess_fraction=1., reason='zero score or candidate variance')
        return c
    def ess(beta):
        z = -beta * (p-p.min(1, keepdims=True)) / scale
        w = np.exp(z-z.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        return float(np.median(1/(w*w).sum(1)/p.shape[1]))
    if ess(beta_max) > target:
        # Keep bounded weak discrimination, explicitly report target unattainable.
        c.update(beta=beta_max, achieved_ess_fraction=ess(beta_max), reason='target unattainable within preregistered beta cap')
        return c
    lo, hi = 0., beta_max
    for _ in range(60):
        mid = (lo+hi)/2
        if ess(mid) > target: lo = mid
        else: hi = mid
    c.update(beta=hi, achieved_ess_fraction=ess(hi), reason='calibrated')
    return c


def action_metrics(chunks, w=None):
    a = validate_chunks(chunks).astype(np.float64)
    i, j = np.triu_indices(len(a), 1)
    delta = a[i]-a[j]
    arms = delta[:, :, JOINTS]
    f = np.sqrt(np.sum(arms**2, axis=(1, 2)))
    out = {'joint_f_mean': float(f.mean()), 'joint_f_max': float(f.max()),
           'joint_rms_mean': float(f.mean()/np.sqrt(120)),
           'joint_rms_max': float(f.max()/np.sqrt(120)),
           'first_joint_rms_mean': float(np.sqrt(np.mean(arms[:, 0]**2, axis=1)).mean()),
           'last_joint_rms_mean': float(np.sqrt(np.mean(arms[:, -1]**2, axis=1)).mean()),
           'gripper_rms_mean': float(np.sqrt(np.mean(delta[:, :, [6, 13]]**2, axis=(1,2))).mean())}
    if w is not None:
        shift = np.tensordot(w, a, axes=(0,0))-a.mean(0)
        out.update(ess_fraction=float(1/np.sum(w*w)/len(w)), max_weight=float(max(w)),
                   tilted_joint_rms=float(np.sqrt(np.mean(shift[:, JOINTS]**2))),
                   tilted_gripper_rms=float(np.sqrt(np.mean(shift[:, [6,13]]**2))))
    return out


def paired_bootstrap(values, seed=42, repeats=10000):
    """Input already reduced to ONE value per independent scene."""
    x = np.asarray(values, np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Need finite per-scene values')
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(len(x), size=(repeats,len(x)))].mean(1)
    return {'mean': float(x.mean()), 'ci95': np.quantile(samples,[.025,.975]).tolist(), 'scenes': len(x)}


class RunLock:
    """Kernel lock released after crashes, unlike stale PID-file-only locks."""
    def __init__(self, path): self.path = Path(path)
    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+')
        fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.stream.seek(0); self.stream.truncate(); self.stream.write(str(os.getpid())); self.stream.flush()
        return self
    def __exit__(self, *args): self.stream.close()
