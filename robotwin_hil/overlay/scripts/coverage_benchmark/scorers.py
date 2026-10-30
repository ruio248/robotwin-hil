from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from tail_value.chunk import ChunkCoverageCritic
from .common import validate_chunks


def normalization(train):
    a=train['expert_chunks'].reshape(-1,14); s=train['state']
    return dict(state_mean=s.mean(0),state_scale=s.std(0).clip(.05),action_mean=a.mean(0),action_scale=a.std(0).clip(.05))


def obs_features(f,s,norm):
    # Three unit-L2 visual blocks, one state RMS block; each block has equal weight.
    v=np.asarray(f,np.float64).reshape(-1,3,512)
    v=v/np.linalg.norm(v,axis=2,keepdims=True).clip(1e-12)
    z=(np.asarray(s).reshape(-1,14)-norm['state_mean'])/norm['state_scale']/np.sqrt(14)
    return np.concatenate([v.reshape(-1,1536),z],axis=1)/2


def act_features(a,norm):
    return ((np.asarray(a,np.float64)-norm['action_mean'])/norm['action_scale']).reshape(-1,140)/np.sqrt(140)


class NeuralScorer:
    def __init__(self,path,device='cpu'):
        d=torch.load(path,map_location='cpu',weights_only=False)
        if d['representation']!='absolute_joint_chunk' or d['config']['horizon']!=10: raise ValueError('Wrong critic representation')
        self.model=ChunkCoverageCritic(d['normalization'],10,d['config'].get('width',256))
        self.model.load_state_dict(d['model']); self.model.eval().requires_grad_(False).to(device)
        self.device=device
    def score(self,features,state,chunks):
        a=validate_chunks(chunks)
        with torch.inference_mode():
            f=torch.as_tensor(np.asarray(features),dtype=torch.float32,device=self.device)[None]
            s=torch.as_tensor(np.asarray(state),dtype=torch.float32,device=self.device)[None]
            x=torch.as_tensor(a,dtype=torch.float32,device=self.device)[None]
            return self.model.score_many(f,s,x)[0].cpu().numpy().astype(np.float64)


class SupportScorer:
    def __init__(self,path,kind):
        self.kind=kind
        with np.load(path,allow_pickle=False) as d: self.data=dict(d)
        self.norm={k:self.data[k] for k in ('state_mean','state_scale','action_mean','action_scale')}
    def score(self,features,state,chunks):
        a=validate_chunks(chunks); z=obs_features(features,state,self.norm)
        if self.kind=='svm':
            x=np.concatenate([np.repeat(z,len(a),axis=0),act_features(a,self.norm)],axis=1)/np.sqrt(2)
            kernel=np.exp(-float(self.data['svm_gamma'])*cdist(x,self.data['svm_support'],'sqeuclidean'))
            return (kernel@self.data['svm_dual'].reshape(-1)+float(self.data['svm_intercept'].item())).reshape(-1)
        d=cdist(z,self.data['observations'],'sqeuclidean')[0]
        ix=np.argsort(d,kind='stable')[:32]
        if self.kind=='observation':
            return np.full(len(a),np.exp(-d[ix]/(2*float(self.data['obs_bandwidth'])**2)).mean())
        delta=cdist(act_features(a,self.norm),self.data['actions'][ix],'sqeuclidean')
        return np.exp(-delta/(2*float(self.data['action_bandwidth'])**2)).mean(1)


def load_scorers(output,device='cpu'):
    import json
    root=Path(output); models=json.loads((root/'models.json').read_text())
    return {name:(NeuralScorer(root/record['file'],device) if record['kind']=='neural'
                  else SupportScorer(root/record['file'],record['kind'])) for name,record in models.items()}
