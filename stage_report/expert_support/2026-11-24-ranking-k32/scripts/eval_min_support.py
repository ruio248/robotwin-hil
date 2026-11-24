#!/usr/bin/env python3
"""Run the frozen official scene evaluator with K32 minimum-support policy calls.

Pass the ordinary eval_fixed100 arguments after ``--``. The wrapper changes
only the action returned at each policy query; environment stepping, expert
screening, scene manifests, and success accounting remain in eval_fixed100.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from score_outcomes import load_encoder, encode_views
from train import digest


class MinSupportSelector:
    def __init__(self, task: str, training_root: Path, encoder_asset: Path, log: Path, device: str,
                 microbatch: int = 1):
        from tail_value.chunk import ChunkCoverageCritic

        report = json.loads((training_root / task / 'report.json').read_text())
        checkpoint = training_root / task / 'final_full50/last.pt'
        if report['checkpoint_sha256'] != digest(checkpoint):
            raise ValueError('Discriminator checkpoint hash mismatch')
        expected = ('f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
                    if task == 'place_empty_cup' else
                    '58d7bfcda80d81a4c54582e0dd3b0a73aaa2cafff111325eff729419ea989c1e')
        if digest(encoder_asset) != expected:
            raise ValueError('Encoder asset hash mismatch')
        self.device = torch.device(device)
        self.encoder = load_encoder(encoder_asset, 'imagenet' if task == 'place_empty_cup' else 'r3m', self.device)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model = ChunkCoverageCritic(payload['normalization'], horizon=report['source']['horizon'], width=256)
        self.model.load_state_dict(payload['model'], strict=True)
        self.model.to(self.device).eval()
        self.reference = None
        if report.get('rank_lambda'):
            root = training_root.parent / 'matched_sft_models' / task
            path = root / 'final_full50/last.pt'
            if digest(path) != json.loads((root / 'report.json').read_text())['checkpoint_sha256']:
                raise ValueError('Reference BCE hash mismatch')
            ref = torch.load(path, map_location='cpu', weights_only=False)
            self.reference = ChunkCoverageCritic(ref['normalization'], horizon=report['source']['horizon'], width=256)
            self.reference.load_state_dict(ref['model'])
            self.reference.to(device).eval()
        self.horizon = report['source']['horizon']
        if microbatch < 1 or 32 % microbatch:
            raise ValueError('microbatch must divide 32')
        self.microbatch = microbatch
        self.scale = float(report['calibration']['scale'])
        self.log = log
        self.log.parent.mkdir(parents=True, exist_ok=True)
        if self.log.exists():
            raise FileExistsError(self.log)
        self.obs = None
        self.seed = None
        self.decision = 0
        self.log.write_text('')

    def record(self, value):
        with self.log.open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')

    def choose(self, original_call, xpl_action_to_joint):
        if self.obs is None:
            raise RuntimeError('Policy requested an action before update_obs')
        obs = self.obs
        state = obs['state']
        joint_state = np.concatenate([state['left_arm_joint_state'], state['left_ee_joint_state'],
                                      state['right_arm_joint_state'], state['right_ee_joint_state']]).astype(np.float32)
        source = {'joint_action': {'left_arm': state['left_arm_joint_state'],
                                   'left_gripper': float(state['left_ee_joint_state'][0]),
                                   'right_arm': state['right_arm_joint_state'],
                                   'right_gripper': float(state['right_ee_joint_state'][0])}}
        if self.microbatch == 1:
            raw = [original_call(func_name='get_action') for _ in range(32)]
        else:
            raw = []
            for _ in range(32 // self.microbatch):
                batch = original_call(func_name='get_action_candidates', obs=self.microbatch)
                if not isinstance(batch, (list, tuple)) or len(batch) != self.microbatch:
                    raise ValueError('Candidate server returned the wrong microbatch size')
                raw.extend(batch)
        candidates = np.stack([
            np.stack([xpl_action_to_joint(action, action_type='joint', current_observation=source)[0]
                      for action in chunk]) for chunk in raw
        ]).astype(np.float32)
        if candidates.shape != (32, self.horizon, 14) or not np.isfinite(candidates).all():
            raise ValueError(f'Expected finite K32 H{self.horizon} 14D chunks, got {candidates.shape}')
        cameras = obs['vision']
        features = encode_views(self.encoder, [[cameras[name]['color']] for name in
                                               ('cam_head', 'cam_left_wrist', 'cam_right_wrist')], self.device)
        with torch.inference_mode():
            inputs = (
                torch.as_tensor(np.repeat(features, 32, axis=0), dtype=torch.float32, device=self.device),
                torch.as_tensor(np.repeat(joint_state[None], 32, axis=0), dtype=torch.float32, device=self.device),
                torch.as_tensor(candidates, dtype=torch.float32, device=self.device),
            )
            scores = self.model(*inputs).cpu().numpy().astype(np.float64)
            extra = {} if self.reference is None else {'bce_scores': self.reference(*inputs).cpu().tolist()}
        if scores.shape != (32,) or not np.isfinite(scores).all():
            raise ValueError('Invalid expert support logits')
        index = int(np.argmin(scores))
        weights = np.exp(-(scores - scores.min()) / max(self.scale, 1e-12))
        weights /= weights.sum()
        shift = candidates[index] - candidates.mean(axis=0)
        self.record({**extra, 'scene_seed': self.seed, 'decision': self.decision, 'selected': index,
                     'scores': scores.tolist(), 'score_range': float(np.ptp(scores)),
                     'ess_over_k_beta1': float(1 / (32 * np.square(weights).sum())),
                     'joint_rms_mrad': float(np.sqrt(np.square(shift[:, [0,1,2,3,4,5,7,8,9,10,11,12]]).mean()) * 1000),
                     'gripper_rms': float(np.sqrt(np.square(shift[:, [6,13]]).mean())),
                     'min_support_score': float(scores[index]),
                     'candidate_score_mean': float(scores.mean())})
        self.decision += 1
        return raw[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--encoder-asset', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--microbatch', type=int, default=1,
                        help='Use 1 for Pi_05; Pi_05_RobotTwin supports batched candidates')
    parser.add_argument('--evaluator', type=Path, required=True)
    args, eval_argv = parser.parse_known_args()
    if eval_argv and eval_argv[0] == '--':
        eval_argv = eval_argv[1:]
    selector = MinSupportSelector(args.task, args.training_root, args.encoder_asset, args.log,
                                  args.device, args.microbatch)
    spec = importlib.util.spec_from_file_location('eval_fixed100_support', args.evaluator)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from client_server.ws.model_client import WsModelClient
    original_call = WsModelClient.call

    def intercepted_call(client, func_name=None, obs=None, **kwargs):
        if func_name == 'update_obs':
            selector.obs = obs
        elif func_name == 'prepare_case':
            selector.seed = obs.get('seed') if isinstance(obs, dict) else None
            selector.decision = 0
        elif func_name == 'get_action':
            return selector.choose(lambda **kw: original_call(client, **kw), module.xpolicylab_action_to_robotwin)
        return original_call(client, func_name=func_name, obs=obs, **kwargs)

    WsModelClient.call = intercepted_call
    sys.argv = [str(args.evaluator), *eval_argv]
    from test_render import Sapien_TEST
    Sapien_TEST()
    parsed = module.parse_args()
    if parsed['task_name'] != args.task or parsed.get('eval_batch'):
        raise ValueError('Task mismatch or batch evaluation requested')
    module.main(parsed)


if __name__ == '__main__':
    main()
