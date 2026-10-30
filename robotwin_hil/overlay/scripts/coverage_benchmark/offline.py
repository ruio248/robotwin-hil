from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from .common import KS, JOINTS, weights, action_metrics, atomic_json, atomic_npz, sha256, verify_fit
from .scorers import load_scorers


def correlation(x,y):
    if np.ptp(x)<1e-12 or np.ptp(y)<1e-12:return None
    value=float(spearmanr(x,y).statistic)
    return value if np.isfinite(value) else None


def analyze(f,s,chunks,scorers,calibration,order=None):
    order=np.arange(len(chunks)) if order is None else np.asarray(order)
    a=chunks[order]; result={'scores':{},'metrics':{}}
    for name,scorer in scorers.items():
        values=scorer.score(f,s,chunks)
        if not np.isfinite(values).all():raise FloatingPointError(name)
        result['scores'][name]=values[order].tolist()
        stats={}
        for k in KS:
            if k>len(a):continue
            b=values[order[:k]];w=weights(b,calibration[name]); c=calibration[name]
            stats[str(k)]={**action_metrics(a[:k],w),'b_range':float(np.ptp(b)),
                'b_std':float(np.std(b)),'b_mean':float(np.mean(b)),
                'b_range_normalized':float(np.ptp(b)/c['scale']) if c['scale']>1e-12 else 0.,
                'action_gap_spearman':correlation(pdist(a[:k,:,JOINTS].reshape(k,-1)),pdist(b[:,None])),
                'weights':w.tolist()}
            perm=np.random.default_rng(42+k).permutation(k)
            stats[str(k)]['permuted_tilted_joint_rms']=action_metrics(a[:k],w[perm])['tilted_joint_rms']
        result['metrics'][name]=stats
    return result


def verify(scorers,f,s,a):
    order=np.random.default_rng(42).permutation(len(a));checks={}
    for name,scorer in scorers.items():
        b=scorer.score(f,s,a);repeat=scorer.score(f,s,a);permuted=scorer.score(f,s,a[order])
        np.testing.assert_allclose(b,repeat,atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(b[order],permuted,atol=1e-6,rtol=1e-6)
        checks[name]={'repeat_max_error':float(max(abs(b-repeat))),
                      'permutation_max_error':float(max(abs(b[order]-permuted)))}
    return checks


def run(config,output,resume):
    torch.set_num_threads(4); paths=config['paths']; root=Path(paths['fit']);output=Path(output)
    fit_manifest=verify_fit(root,config['protocol_sha256'])
    scorers=load_scorers(root); calibration=json.loads((root/'calibration.json').read_text())
    score_dir=output/'scores';score_dir.mkdir(parents=True,exist_ok=True)
    completed={}; sources={}
    def process(name,arrays,metadata):
        metadata={**metadata,'fit_manifest_sha256':sha256(root/'manifest.json')}
        dest=score_dir/(name+'.json')
        if dest.exists():
            old=json.loads(dest.read_text())
            if old['metadata']!=metadata:raise ValueError('Changed offline input')
            completed[name]=sha256(dest);return
        rows=[]
        for i,(f,s,a) in enumerate(zip(arrays['features'],arrays['state'],arrays['candidates'])):
            item=analyze(f,s,a,scorers,calibration,arrays.get('order',[None]*len(arrays['state']))[i])
            item.update(frame_index=int(arrays['frame_index'][i]))
            if 'progress' in arrays:item.update(progress=float(arrays['progress'][i]),policy_step=int(arrays['policy_step'][i]))
            if 'expert_chunks' in arrays:item['expert_scores']={n:float(m.score(f,s,arrays['expert_chunks'][i:i+1])[0]) for n,m in scorers.items()}
            rows.append(item)
        atomic_json(dest,{'metadata':metadata,'rows':rows});completed[name]=sha256(dest)
        print('OFFLINE',name,'observations',len(rows),flush=True)
        atomic_json(output/'progress.json',{'completed':len(completed),'last':name})
    with np.load(root/'validation.npz',allow_pickle=False) as f:v=dict(f)
    first=np.flatnonzero(v['split']=='diag')[0]
    atomic_json(output/'invariance.json',verify(scorers,v['features'][first],v['state'][first],v['policy_chunks'][first]))
    for episode in np.unique(v['episode'][v['split']=='diag']):
        ix=(v['episode']==episode)&(v['split']=='diag')
        a={k:x[ix] for k,x in v.items()};a['candidates']=a['policy_chunks']
        process(f'validation_{episode}',a,{'cohort':'internal_diagnostic','episode':int(episode),
            'source_sha256':fit_manifest['validation_sha256']})
    if config.get('_validation_only'):
        atomic_json(output/'manifest.json',{'complete':False,'internal_validation_complete':True,
            'completed':completed,'protocol_sha256':config['protocol_sha256'],
            'fit_manifest_sha256':sha256(root/'manifest.json'),'reason':'internal-validation-only; HIL and historical cohorts not scored'})
        from .report import validation_report
        validation_report(config,output,output)
        return
    hil_root=Path(paths['hil']);hm=json.loads((hil_root/'candidate_manifest.json').read_text())
    if not hm['complete'] or len(hm['identity']['episodes'])!=50:raise ValueError('HIL cohort incomplete')
    feature_lookup={};hil_episodes={}
    for e in hm['identity']['episodes']:
        name=f"rollout_{e['rollout_index']:04d}";file=hil_root/(name+'.npz')
        if sha256(file)!=hm['completed'][name]:raise ValueError('Changed HIL cache')
        with np.load(file,allow_pickle=False) as f:a=dict(f)
        process('hil_'+name,a,{'cohort':'hil_full','episode':e['rollout_index'],'takeover':e['takeover'],'source_sha256':sha256(file)})
        for i,frame in enumerate(a['frame_index']):feature_lookup[(e['rollout_index'],int(frame))]=(a['features'][i],a['state'][i])
        hil_episodes[e['rollout_index']]=e
    kroot=Path(paths['k32']); km=json.loads((kroot/'manifest.json').read_text())
    if not km['complete'] or len(km['observations'])!=1260:raise ValueError('K32 cache incomplete')
    grouped=defaultdict(list)
    for row in km['observations']:
        if row['rollout_index'] in hil_episodes:grouped[row['rollout_index']].append(row)
    total=0
    for episode,rows in sorted(grouped.items()):
        arrays=defaultdict(list);digests=[]
        for row in rows:
            frame=int(Path(row['frame_path']).stem);f,s=feature_lookup[(episode,frame)]
            file=kroot/row['cache'];expected=km['completed'][row['cache']]
            if sha256(file)!=expected:raise ValueError('Changed K32 cache')
            with np.load(file,allow_pickle=False) as ar:a=dict(ar)
            np.testing.assert_allclose(s,a['state'],atol=1e-6,rtol=0)
            for k,value in dict(features=f,state=s,candidates=a['candidates'],order=a['subset_order'],
                frame_index=frame,progress=row['progress'],policy_step=row['policy_step']).items():arrays[k].append(value)
            digests.append(expected)
        arrays={k:np.asarray(v) for k,v in arrays.items()};total+=len(rows)
        process(f'k32_{episode}',arrays,{'cohort':'hil_pre_k32','episode':episode,'source_sha256':digests})
    if total!=1233:raise ValueError(f'Expected 1233 pre-HIL observations, found {total}')
    rr=Path(paths['rollouts']);rm=json.loads((rr/'candidate_manifest.json').read_text())
    labels={Path(r['hdf5_path']).stem:bool(r['success']) for r in map(json.loads,Path(paths['rollout_records']).read_text().splitlines())}
    if len(labels)!=100 or sum(labels.values())!=31 or not rm['complete']:raise ValueError('Historical rollout cohort changed')
    for name,label in sorted(labels.items()):
        file=rr/(name+'.npz')
        if sha256(file)!=rm['completed'][name]:raise ValueError('Changed historical cache')
        with np.load(file,allow_pickle=False) as f:a=dict(f)
        process('historical_'+name,a,{'cohort':'historical_rollout','episode':name,'success':label,'source_sha256':sha256(file)})
    for key in ('hil','rollouts','k32'):sources[key]=sha256(Path(paths[key])/('manifest.json' if key=='k32' else 'candidate_manifest.json'))
    atomic_json(output/'manifest.json',{'complete':True,'completed':completed,'sources':sources,
        'protocol_sha256':config['protocol_sha256'],'fit_manifest_sha256':sha256(root/'manifest.json'),
        'pre_hil_observations':total,'hil_episodes':50,'historical_rollouts':100})
