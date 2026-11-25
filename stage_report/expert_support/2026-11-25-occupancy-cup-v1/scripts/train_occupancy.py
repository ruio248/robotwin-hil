"""Fit query-level occupancy D and native SARSA critics on real SFT rollouts."""
import copy
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

from run import ROOT, SPEC, sha, write
from train import ChunkCoverageCritic, load_cohort, base
from score_outcomes import encode_views, load_encoder

CAMERAS = ('cam_head','cam_left_wrist','cam_right_wrist')
STATE = ('left_arm_joint_state','left_ee_joint_state','right_arm_joint_state','right_ee_joint_state')
GAMMA = .9
CAP = 5.
DEVICE = 'cuda'
CHECKPOINTS = (1000,3000,5000,10000)

def tensor(x):
    return torch.as_tensor(x,device=DEVICE,dtype=torch.float32)

def mlp(dim):
    layers=[]
    for _ in range(3):
        layers.extend((nn.Linear(dim,256),nn.LayerNorm(256),nn.SiLU()))
        dim=256
    return nn.Sequential(*layers,nn.Linear(dim,1))

def state_input(features,state,norm):
    return torch.cat((features,(state-tensor(norm['state_mean']))/tensor(norm['state_scale'])),dim=-1)

def critic(norm):
    model=ChunkCoverageCritic(norm,horizon=10)
    model.net[0]=nn.Linear(model.net[0].in_features+1,256)
    return model.to(DEVICE)

def qscore(model,features,state,actions,time):
    return model(torch.cat((features,time),dim=-1),state,actions)

def load_episodes(mode,encoder):
    directory=ROOT/mode/'trace'
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['status']=='complete' and not manifest['infra_error_episodes']
    episodes=[]
    for ep in manifest['episodes']:
        queries=ep['queries']
        folder=directory/ep['directory']
        arrays=[]
        for name in [q['file'] for q in queries]+['terminal.npz']:
            with np.load(folder/name) as z:
                arrays.append({key:z[key] for key in z.files})
        terminal=arrays[-1]
        limit=int(terminal['step_limit'])
        steps=np.array([q['policy_steps_before_query'] for q in queries]+[ep['rollout_steps']])
        executed=np.array([q['executed_actions'] for q in queries])
        actions=np.stack([z['absolute_joint_chunk'] for z in arrays[:-1]])
        assert terminal['done'] and terminal['success']==ep['success']
        assert steps[0]==0 and np.array_equal(np.diff(steps),executed)
        assert actions.shape==(len(queries),10,14) and np.isfinite(actions).all()
        assert np.all(executed[:-1]==10) and 0<executed[-1]<=10 and steps[-1]<=limit
        state=np.stack([np.concatenate([z['state_'+key] for key in STATE]) for z in arrays])
        features=encode_views(encoder,[[z['rgb_'+cam] for z in arrays] for cam in CAMERAS],DEVICE)
        poses=np.stack([[z.get('state_'+side+'_tcp_pose',np.full(7,np.nan))[:3] for side in ('left','right')] for z in arrays])
        episodes.append(dict(scene=ep['scene_seed'],success=ep['success'],state=state,features=features,
            time=(1-steps/limit).astype(np.float32)[:,None],actions=actions,poses=poses,
            steps=steps,executed=executed,limit=limit,reason=str(terminal['reason'])))
    return episodes

def summarize_d(model,ep,policy,norm):
    expert=torch.cat([state_input(tensor(e['features']),tensor(e['state']),norm) for e in ep])
    negative=torch.cat([state_input(tensor(e['features'][:-1]),tensor(e['state'][:-1]),norm) for e in policy])
    with torch.no_grad():
        a,b=(model(x).flatten() for x in (expert,negative))
    logits=torch.cat((a,b)).cpu().numpy()
    return dict(bce=float(.5*(F.softplus(-a).mean()+F.softplus(b).mean())),
        auc=roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],logits),
        expert_queries=len(a),sft_queries=len(b),expert_p_mean=float(a.sigmoid().mean()),
        policy_p_mean=float(b.sigmoid().mean()),
        saturation=float(np.mean(abs(logits)>np.log(99))))

@torch.no_grad()
def rewards(model,e,norm):
    logits=model(state_input(tensor(e['features']),tensor(e['state']),norm)).flatten()
    return F.softplus(-logits).clamp_max(CAP).cpu().numpy()

def train_d(experts,episodes,norm):
    positive=torch.cat([state_input(tensor(e['features']),tensor(e['state']),norm) for e in experts[0]])
    negative=torch.cat([state_input(tensor(e['features'][:-1]),tensor(e['state'][:-1]),norm) for e in episodes[0]])
    model=mlp(1550).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    curve=[]
    best=float('inf')
    for step in range(1,1001):
        p=positive[torch.randint(len(positive),(128,),device=DEVICE)]
        n=negative[torch.randint(len(negative),(128,),device=DEVICE)]
        loss=.5*(F.softplus(-model(p)).mean()+F.softplus(model(n)).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (50,100,250,500,1000):
            a,b=[summarize_d(model,e,s,norm) for e,s in zip(experts,episodes)]
            curve.append(dict(step=step,train=a,validation=b))
            if b['bce']<best:
                best=b['bce']
                torch.save(dict(model=model.state_dict(),normalization=norm,step=step,cap=CAP),ROOT/'d.pt')
    payload=torch.load(ROOT/'d.pt',weights_only=False)
    model.load_state_dict(payload['model'])
    return model.eval(),curve,payload['step']

def transition_batch(episodes,model,norm):
    fields={k:[] for k in ('features','state','actions','time','reward','next_index','done','return')}
    offset=0
    for e in episodes:
        reward=rewards(model,e,norm)[1:]
        returns=reward.copy()
        for i in range(len(returns)-2,-1,-1):
            returns[i]+=GAMMA*returns[i+1]
        n=len(reward)
        for k in ('features','state','time'):
            fields[k].append(e[k][:-1])
        fields['actions'].append(e['actions'])
        fields['reward'].append(reward)
        fields['return'].append(returns)
        fields['next_index'].append(np.minimum(np.arange(1,n+1),n-1)+offset)
        fields['done'].append(np.arange(n)==n-1)
        offset+=n
    batch={k:tensor(np.concatenate(v)) for k,v in fields.items()}
    batch['next_index']=batch['next_index'].long()
    return batch

def train_value(train,validation,norm,use_action):
    model=critic(norm) if use_action else mlp(1551).to(DEVICE)
    target=copy.deepcopy(model).requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    def predict(net,batch,index):
        f,s,a,t=(batch[k][index] for k in ('features','state','actions','time'))
        if use_action:
            return qscore(net,f,s,a,t)
        return net(torch.cat((state_input(f,s,norm),t),dim=-1)).flatten()
    name='q' if use_action else 'v'
    curve=[]
    best=float('inf')
    for step in range(1,10001):
        idx=torch.randint(len(train['reward']),(256,),device=DEVICE)
        with torch.no_grad():
            y=train['reward'][idx]+GAMMA*(1-train['done'][idx])*predict(target,train,train['next_index'][idx])
        loss=F.mse_loss(predict(model,train,idx),y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            for dst,src in zip(target.parameters(),model.parameters()):
                dst.lerp_(src,.005)
        if step in CHECKPOINTS:
            row={'step':step,'td_loss':float(loss)}
            with torch.no_grad():
                for split,b in [('train',train),('validation',validation)]:
                    p=predict(model,b,slice(None))
                    row[split]=dict(mc_rmse=float(F.mse_loss(p,b['return']).sqrt()),mean=float(p.mean()),
                                    std=float(p.std()),min=float(p.min()),max=float(p.max()))
            curve.append(row)
            payload=dict(model=model.state_dict(),normalization=norm,step=step)
            torch.save(payload,ROOT/f'{name}_{step}.pt')
            if row['validation']['mc_rmse']<best:
                best=row['validation']['mc_rmse']
                torch.save(payload,ROOT/f'{name}.pt')
            print(name,row,flush=True)
    return curve

def main():
    torch.set_num_threads(4)
    base.seed_all(1042)
    assert sha(SPEC['encoder'])=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    encoder=load_encoder(SPEC['encoder'],'imagenet',DEVICE)
    raw=load_episodes('collect',encoder)
    experts,_,source=load_cohort(ROOT.parent/'support_three_tasks_20261123/matched_sft_data','place_empty_cup')
    rng=np.random.default_rng(42)
    ei=rng.permutation(50)
    pi=rng.permutation(20)
    e=[ [experts[i] for i in ix] for ix in (ei[:40],ei[40:]) ]
    s=[ [raw[i] for i in ix] for ix in (pi[:16],pi[16:]) ]
    norm=base.normalization(e[0],list(range(40)))
    d,curve,step=train_d(e,s,norm)
    train,validation=[transition_batch(group,d,norm) for group in s]
    decomposition=[]
    for k,group in zip(('train','validation'),s):
        for ep in group:
            values=rewards(d,ep,norm)
            decomposition.append(dict(split=k,scene=ep['scene'],success=ep['success'],
                initial=float(values[0]),prefix5_mean=float(values[:5].mean()),
                middle=float(values[len(values)//2]),terminal=float(values[-1]),mean=float(values[1:].mean())))
    prefix={}
    for label,experts_group,sft_group in zip(('train','validation'),e,s):
        prefix[label]={}
        for count in (1,5):
            pos=np.concatenate([rewards(d,ep,norm)[:count] for ep in experts_group])
            neg=np.concatenate([rewards(d,ep,norm)[:count] for ep in sft_group])
            prefix[label][str(count)]=dict(expert_reward_mean=float(pos.mean()),sft_reward_mean=float(neg.mean()),
                auc=float(roc_auc_score(np.r_[np.zeros(len(pos)),np.ones(len(neg))],np.r_[pos,neg])))
    # Define low coverage against the training expert reward distribution.
    threshold=float(np.quantile(np.concatenate([rewards(d,ep,norm) for ep in e[0]]),.9))
    report=dict(expert_source=source,encoder_sha256=sha(SPEC['encoder']),seed=42,training_seed=1042,
        expert=[[row['episode'] for row in group] for group in e],sft=[[row['scene'] for row in group] for group in s],
        collect_sha256=sha(ROOT/'collect/trace/manifest.json'),d_step=step,d_curve=curve,
        d_sha256=sha(ROOT/'d.pt'),prefix_diagnostics=prefix,sft_reward_profiles=decomposition,threshold=threshold,cap=CAP,gamma=GAMMA,
        q_train_transitions=len(train['reward']),q_validation_transitions=len(validation['reward']))
    write(ROOT/'discriminator.json',report)
    report['q_curve']=train_value(train,validation,norm,True)
    report['v_curve']=train_value(train,validation,norm,False)
    report['q_sha256']=sha(ROOT/'q.pt')
    report['v_sha256']=sha(ROOT/'v.pt')
    write(ROOT/'training.json',report)

if __name__=='__main__':
    main()
