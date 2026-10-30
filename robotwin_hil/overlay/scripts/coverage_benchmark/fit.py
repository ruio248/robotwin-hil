from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.svm import OneClassSVM
from tail_value.chunk import ChunkCoverageCritic
from tail_value.model import atomic_checkpoint, seed_all
from .common import atomic_json, atomic_npz, sha256, calibrate
from .data import sft_data, subset
from .scorers import normalization, obs_features, act_features, load_scorers


def export_neural(source,dest):
    d=torch.load(source,map_location='cpu',weights_only=False)
    result={k:d[k] for k in ('representation','step','config','model','normalization')}
    result['normalization']={k:torch.as_tensor(v).cpu() for k,v in d['normalization'].items()}
    result['model']={k:v.cpu() for k,v in d['model'].items()}
    result['source_sha256']=sha256(source)
    atomic_checkpoint(dest,result)


def train_discriminator(train, norm, directory, seed, steps):
    directory.mkdir(parents=True,exist_ok=True); last=directory/'last.pt'
    seed_all(seed); model=ChunkCoverageCritic(norm,10,256).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(seed); start=0
    if last.exists():
        d=torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(d['model']); opt.load_state_dict(d['optimizer'])
        gen.set_state(d['generator']); torch.set_rng_state(d['torch_rng']); start=d['step']
    data={k:torch.as_tensor(train[k],dtype=torch.float32,device='cuda') for k in ('features','state','expert_chunks','policy_chunks')}
    for step in range(start+1,steps+1):
        ix=torch.randint(len(train['state']),(256,),generator=gen).cuda()
        opt.zero_grad(set_to_none=True)
        e=model(data['features'][ix],data['state'][ix],data['expert_chunks'][ix])
        p=model.score_many(data['features'][ix],data['state'][ix],data['policy_chunks'][ix])
        loss=.5*(torch.nn.functional.softplus(-e).mean()+torch.nn.functional.softplus(p).mean())
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite discriminator loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
        if step%500==0 or step==steps:
            record={'step':step,'loss':float(loss),'expert_logit':float(e.mean()),'policy_logit':float(p.mean()),'seed':seed}
            print('DISCRIMINATOR',json.dumps(record),flush=True)
            with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            atomic_checkpoint(last,dict(representation='absolute_joint_chunk',step=step,
                config={'horizon':10,'width':256,'seed':seed,'objective':'balanced_source_bce'},
                normalization=norm,model=model.state_dict(),optimizer=opt.state_dict(),
                generator=gen.get_state(),torch_rng=torch.get_rng_state()))
    return last


def run(config,output,resume):
    torch.set_num_threads(4)
    paths=config['paths']; output=Path(output); output.mkdir(parents=True,exist_ok=True)
    data,provenance=sft_data(paths['sft_cache']); train=subset(data,'train'); cal=subset(data,'cal')
    atomic_json(output/'split.json',provenance)
    # Small portable internal-validation archive for Ubuntu; no training leakage.
    atomic_npz(output/'validation.npz',{k:v[data['split']!='train'] for k,v in data.items()})
    norm=normalization(train); models={}
    support=output/'support.npz'
    if not support.exists():
        z=obs_features(train['features'],train['state'],norm); a=act_features(train['expert_chunks'],norm)
        distances=[]; observation_distances=[]
        for start in range(0,len(z),128):
            d=cdist(z[start:start+128],z,'sqeuclidean')
            d[train['episode'][start:start+128,None]==train['episode'][None,:]]=np.inf
            nearest=np.argsort(d,axis=1)[:,:32]
            dd=a[start:start+len(nearest),None,:]-a[nearest]
            distances.extend(np.sqrt(np.sum(dd*dd,axis=2)).reshape(-1))
            observation_distances.extend(np.sqrt(np.take_along_axis(d,nearest,axis=1)).reshape(-1))
        bandwidth=max(float(np.median(distances)),1e-8); obs_band=max(float(np.median(observation_distances)),1e-8)
        joint=np.concatenate([z,a],axis=1)/np.sqrt(2)
        rng=np.random.default_rng(42); ix=rng.integers(len(z),size=(100000,2)); ix=ix[ix[:,0]!=ix[:,1]]
        # Bounded intermediate memory, all model fitting anchors retained.
        d2=np.concatenate([np.sum((joint[p[:,0]]-joint[p[:,1]])**2,axis=1) for p in np.array_split(ix,100)])
        gamma=1/max(2*float(np.median(d2)),1e-12)
        print('FITTING_SVM',len(joint),'gamma',gamma,flush=True)
        svm=OneClassSVM(nu=.05,gamma=gamma,kernel='rbf',cache_size=1024).fit(joint)
        atomic_npz(support,dict(**norm,observations=z.astype('float32'),actions=a.astype('float32'),
            action_bandwidth=np.asarray(bandwidth),obs_bandwidth=np.asarray(obs_band),
            svm_gamma=np.asarray(gamma),svm_support=svm.support_vectors_,svm_dual=svm.dual_coef_,
            svm_intercept=svm.intercept_,episode=train['episode']))
    for name in ('knn','svm','observation'): models[name]={'kind':name,'file':'support.npz','sha256':sha256(support)}
    for seed in (42,43,44):
        for step in (1000,10000,20000):
            dest=output/f'td_s{seed}_{step}.pt'
            if dest.exists(): continue
            if seed==42:
                source=Path(paths['td_root'])/f'step_{step:07d}_inference.pt'
            else:
                run=output/f'td_seed{seed}'; source=run/f'step_{step:07d}.pt'
                if not source.exists():
                    cmd=[sys.executable,str(Path(__file__).parents[1]/'train_tail_chunk.py'),
                        '--cache-dir',paths['sft_cache'],'--output-dir',str(run),'--steps','20000','--seed',str(seed)]
                    if (run/'last.pt').exists():cmd += ['--resume',str(run/'last.pt')]
                    with (output/f'td_seed{seed}.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            export_neural(source,dest)
        for step in (1000,10000,20000):
            name=f'td_s{seed}_{step}'; models[name]={'kind':'neural','file':name+'.pt','sha256':sha256(output/(name+'.pt'))}
        dest=output/f'disc_s{seed}.pt'
        if not dest.exists(): export_neural(train_discriminator(train,norm,output/f'disc_seed{seed}',seed,10000),dest)
        models[f'disc_s{seed}']={'kind':'neural','file':dest.name,'sha256':sha256(dest)}
        atomic_json(output/'models.json',models)
    scorers=load_scorers(output); calibration={}
    for name,scorer in scorers.items():
        e=[];p=[]
        for f,s,a,c in zip(cal['features'],cal['state'],cal['expert_chunks'],cal['policy_chunks']):
            values=scorer.score(f,s,np.concatenate([a[None],c]));e.append(values[0]);p.append(values[1:])
        calibration[name]=calibrate(e,p,beta_max=config['protocol']['beta_max'])
        print('CALIBRATION',name,calibration[name],flush=True)
    atomic_json(output/'calibration.json',calibration)
    atomic_json(output/'manifest.json',{'complete':True,'protocol_sha256':config['protocol_sha256'],
        'train_anchors':len(train['state']),'calibration_anchors':len(cal['state']),
        'diagnostic_anchors':int(sum(data['split']=='diag')),'models':models,
        'split_sha256':sha256(output/'split.json'),'validation_sha256':sha256(output/'validation.npz'),
        'calibration_sha256':sha256(output/'calibration.json')})
