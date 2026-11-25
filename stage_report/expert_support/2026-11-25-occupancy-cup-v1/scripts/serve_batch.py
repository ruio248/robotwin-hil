"""Expose batched candidate sampling on the existing Pi_05 SFT server."""
import runpy
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

def install():
    from openpi.models import model as openpi_model
    from XPolicyLab.policy.Pi_05.model import Model, slice_stacked_obs, unpack_robot_state

    def get_action_candidates(self, count):
        policy=self.policy
        raw=slice_stacked_obs(self.observation_window,0)
        transformed=policy._input_transform(jax.tree.map(lambda x:x,raw))
        batched=jax.tree.map(lambda x:jnp.broadcast_to(jnp.asarray(x),(count,*np.shape(x))),transformed)
        policy._rng, key=jax.random.split(policy._rng)
        sample=policy._sample_actions(key,openpi_model.Observation.from_dict(batched),**policy._sample_kwargs)
        result=[]
        for actions in np.asarray(sample):
            physical=policy._output_transform({'state':np.asarray(transformed['state']), 'actions':actions})['actions']
            result.append(unpack_robot_state(physical,self.action_type,self.robot_action_dim_info,source_type='obs'))
        return result
    Model.get_action_candidates=get_action_candidates

path = Path(sys.argv.pop(1))
install()
runpy.run_path(str(path),run_name='__main__')
