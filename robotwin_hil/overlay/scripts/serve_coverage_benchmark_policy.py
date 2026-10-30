#!/usr/bin/env python3
"""Dedicated benchmark-only noise-explicit policy service. Never use HIL port 18300."""
import asyncio
import os
import numpy as np
from serve_tail_chunk_policy import enable_legacy_openpi_batch_images, adapter, get_action_candidates
from XPolicyLab.setup_policy_server import parse_args_and_config
from client_server.ws.model_server import PolicyServer, PolicyServerConfig


def benchmark_info(self):
    m=self.policy._model
    config=getattr(m,'config',None)
    horizon=getattr(m,'action_horizon',getattr(config,'action_horizon',None))
    dim=getattr(m,'action_dim',getattr(config,'action_dim',None))
    if horizon is None or dim is None:raise ValueError('Cannot determine latent diffusion noise shape')
    return {'noise_shape':[int(horizon),int(dim)],'pid':os.getpid(),'api':'benchmark_fixed_noise_v1'}


def benchmark_action(self,request):
    if self.observation_window is None or self.observation_window['state'].shape[0]!=1:
        raise ValueError('Need one current observation')
    noise=np.asarray(request['noise'],dtype=np.float32)
    shape=tuple(benchmark_info(self)['noise_shape'])
    if noise.ndim!=3 or noise.shape[1:]!=shape or len(noise) not in (1,4) or not np.isfinite(noise).all():
        raise ValueError(f'Need [1 or 4,{shape}] finite noise; got {noise.shape}')
    one=adapter.slice_stacked_obs(self.observation_window,0)
    if len(noise)==1:
        raw=self.policy.infer(one,noise=noise[0])['actions'][None]
    else:
        raw=self.policy.infer(adapter.stack_obs([one]*len(noise)),noise=noise)['actions']
    actions=np.asarray(raw,dtype=np.float32)
    if actions.shape!=(len(noise),10,14) or not np.isfinite(actions).all():raise ValueError('Invalid physical output chunk')
    return actions


def main():
    c=parse_args_and_config()
    if int(c['port'])!=18304 or c.get('action_type')!='joint' or c.get('policy_name')!='Pi_05_RobotTwin':
        raise ValueError('Benchmark service restricted to dedicated 18304 physical-joint baseline')
    enable_legacy_openpi_batch_images()
    adapter.Model.benchmark_info=benchmark_info
    adapter.Model.benchmark_action=benchmark_action
    adapter.Model.get_action_candidates=get_action_candidates
    model=adapter.Model(c)
    print('BENCHMARK_INFO',benchmark_info(model),flush=True)
    asyncio.run(PolicyServer(model,PolicyServerConfig(host='127.0.0.1',port=18304)).serve_forever())


if __name__=='__main__':main()
