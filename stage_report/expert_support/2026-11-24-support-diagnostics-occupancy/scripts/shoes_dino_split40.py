"""Episode-disjoint 40/10 validation inside the original shoes expert50 cohort."""
from train_click_rank import ROOT, base, load_cohort, ChunkCoverageCritic, distance, I, J
from train import digest
import json
import numpy as np
import torch
import torch.nn.functional as F

def evaluate(model, rows, norm, delta):
    records = []
    for row in rows:
        expert, scores = base.score_episode(model.eval(), row, 'cuda')
        d = distance(row, norm['action_scale'].numpy())
        gaps = d[:,J] - d[:,I]
        mask = abs(gaps) > delta
        margins = ((scores[:,I] - scores[:,J]) * np.sign(gaps))[mask]
        records.append(dict(episode=row['episode'], anchors=len(expert),
            expert_wins=float(((expert[:,None]>scores)+.5*(expert[:,None]==scores)).sum()),
            rank_correct=float((margins>0).sum()+.5*(margins==0).sum()), pairs=int(mask.sum())))
    return dict(expert_rank=sum(r['expert_wins'] for r in records)/(4*sum(r['anchors'] for r in records)),
                candidate_rank=sum(r['rank_correct'] for r in records)/sum(r['pairs'] for r in records),
                episodes=records)

def main():
    torch.set_num_threads(4)
    rows, _, source = load_cohort(ROOT/'matched_sft_data','place_dual_shoes')
    entries = json.loads((ROOT/'matched_sft_data/place_dual_shoes/manifest.json').read_text())['episodes']
    for row, entry in zip(rows,[e for e in entries if e['split']=='train']):
        with np.load(ROOT/'shoes_dinov2_10k'/(entry['id']+'.npz')) as z:
            assert str(z['source'])==row['sha256']
            features=z['features']
        row['features']=np.pad(features,((0,0),(0,0),(0,128))).reshape(len(features),1536)
    indices=np.random.default_rng(42).permutation(50)
    tr=[rows[i] for i in indices[:40]]
    va=[rows[i] for i in indices[40:]]
    assert not {r['episode'] for r in tr}&{r['episode'] for r in va}
    dest=ROOT/'shoes_dinov2_split40_100k'
    dest.mkdir(exist_ok=True)
    base.atomic_json(dest/'split.json',{s:[dict(episode=r['episode'],sha256=r['sha256'],anchors=len(r['state'])) for r in group] for s,group in [('train',tr),('validation',va)]})
    norm=base.normalization(tr,list(range(40)))
    base.seed_all(1042)
    model=ChunkCoverageCritic(norm,horizon=10,width=256).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rng=np.random.default_rng(1042)
    sizes=np.array([len(r['state']) for r in tr])
    offsets=np.r_[0,np.cumsum(sizes)[:-1]]
    def tensor(a):
        return torch.as_tensor(a,device='cuda')
    feats,states,exp,pol=[tensor(np.concatenate([r[k] for r in tr])) for k in ['features','state','expert','policy']]
    distances=np.concatenate([distance(r,norm['action_scale'].numpy()) for r in tr])
    gaps=distances[:,J]-distances[:,I]
    delta=float(np.quantile(abs(gaps),.25))
    dif=tensor(gaps)
    curve=[]
    best=-1
    for step in range(1,100001):
        groups=rng.integers(40,size=256)
        idx=tensor(offsets[groups]+rng.integers(sizes[groups]))
        neg=tensor(rng.integers(4,size=256))
        f,s=feats[idx],states[idx]
        pos=model(f,s,exp[idx])
        scores=model(f.repeat_interleave(4,0),s.repeat_interleave(4,0),pol[idx].reshape(-1,10,14)).reshape(-1,4)
        bce=.5*(F.softplus(-pos).mean()+F.softplus(scores[torch.arange(256,device='cuda'),neg]).mean())
        gap=dif[idx]
        mask=gap.abs()>delta
        counts=mask.sum(1)
        losses=F.softplus(-gap.sign()*(scores[:,I]-scores[:,J]))
        rank=((losses*mask).sum(1)/counts.clamp_min(1)).sum()/(counts>0).sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        (bce+rank).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if step%1000==0:
            point=dict(step=step,bce=bce.item(),rank_loss=rank.item(),
                       train=evaluate(model,tr,norm,delta),validation=evaluate(model,va,norm,delta))
            curve.append(point)
            base.atomic_json(dest/'curve.json',curve)
            payload=dict(model=model.state_dict(),normalization=norm,step=step,optimizer=opt.state_dict())
            torch.save(payload,dest/'last.pt')
            if point['validation']['candidate_rank']>best:
                best=point['validation']['candidate_rank']
                torch.save(payload,dest/'best_validation.pt')
            model.train()
            print(step,point['train']['expert_rank'],point['validation']['expert_rank'],point['train']['candidate_rank'],point['validation']['candidate_rank'],flush=True)
    base.atomic_json(dest/'report.json',dict(source=source,split_seed=42,training_seed=1042,
        train_episodes=40,validation_episodes=10,steps=100000,batch_size=256,lr=1e-4,rank_lambda=1,delta=delta,
        encoder='frozen DINOv2 ViT-S/14 CLS',encoder_sha256=digest(ROOT/'shoes_dinov2_10k/encoder.pt'),
        checkpoint_sha256=digest(dest/'last.pt'),final=curve[-1],at_10000=curve[9],
        best_validation=max(curve,key=lambda p:p['validation']['candidate_rank'])))
if __name__=='__main__':
    main()
