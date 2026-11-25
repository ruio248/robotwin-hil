"""Score and select K32 candidates using the fixed occupancy critic."""
import json
import os
import numpy as np
import torch
from train_occupancy import ROOT, SPEC, DEVICE, CAMERAS, STATE, qscore, critic, tensor
from score_outcomes import load_encoder, encode_views
from run import sha

def install(module,mode,output):
    from client_server.ws.model_client import WsModelClient
    torch.set_num_threads(4)
    original_call=WsModelClient.call
    report=json.loads((ROOT/'training.json').read_text())
    assert sha(ROOT/'q.pt')==report['q_sha256'] and sha(SPEC['encoder'])==report['encoder_sha256']
    q=torch.load(ROOT/'q.pt',weights_only=False)
    model=critic(q['normalization']).eval()
    model.load_state_dict(q['model'])
    encoder=load_encoder(SPEC['encoder'],'imagenet',DEVICE)
    current={}
    def intercept(client,func_name=None,obs=None,**kwargs):
        if func_name=='prepare_case':
            current.update(seed=obs['seed'],decision=0,rng=np.random.default_rng([42,int(obs['seed'])]))
        if func_name=='update_obs':
            current['obs']=obs
        if func_name!='get_action':
            return original_call(client,func_name=func_name,obs=obs,**kwargs)
        observation=current['obs']
        state=observation['state']
        source={'joint_action':dict(left_arm=state[STATE[0]],left_gripper=float(state[STATE[1]][0]),
                                     right_arm=state[STATE[2]],right_gripper=float(state[STATE[3]][0]))}
        if os.getenv('OCCUPANCY_SERIAL'):
            raw=[original_call(client,func_name='get_action') for _ in range(32)]
        else:
            raw=[]
            for _ in range(4):
                raw.extend(original_call(client,func_name='get_action_candidates',obs=8))
        raw=[module.normalize_action_chunk(chunk) for chunk in raw]
        chunks=np.stack([module.absolute_joint_chunk_from_response(a,source,'joint') for a in raw])
        assert chunks.shape==(32,10,14) and np.isfinite(chunks).all()
        features=encode_views(encoder,[[observation['vision'][cam]['color']] for cam in CAMERAS],DEVICE)
        steps=observation['additional_info']['occupancy_policy_steps']
        limit=observation['additional_info']['occupancy_step_limit']
        with torch.inference_mode():
            scores=qscore(model,tensor(features).expand(32,-1),
                tensor(np.concatenate([state[key] for key in STATE])).expand(32,-1),
                tensor(chunks),tensor([[1-steps/limit]]).expand(32,1)).cpu().numpy()
        assert np.isfinite(scores).all()
        index=int(np.argmax(scores) if mode=='max' else np.argmin(scores) if mode=='min' else current['rng'].integers(32))
        weights=np.exp(scores-scores.max())
        weights/=weights.sum()
        shift=chunks[index]-chunks.mean(0)
        with (output/'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(scene=current['seed'],step=steps,index=index,scores=scores.tolist(),
                range=float(np.ptp(scores)),
                pair_gap=float(abs(scores[:,None]-scores[None,:])[np.triu_indices(32,1)].mean()),
                ess_over_k=float(1/(32*np.square(weights).sum())),
                joint_rms_mrad=float(1000*np.sqrt(np.square(shift[:,[0,1,2,3,4,5,7,8,9,10,11,12]]).mean())),
                gripper_rms=float(np.sqrt(np.square(shift[:,[6,13]]).mean()))))+'\n')
        current['decision']+=1
        return raw[index]
    WsModelClient.call=intercept
