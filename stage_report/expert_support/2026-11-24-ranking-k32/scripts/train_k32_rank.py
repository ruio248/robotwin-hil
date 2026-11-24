"""K32 candidate coverage ablation, matched normalized expert-distance labels."""
from train_click_rank import ROOT,distance,base,ChunkCoverageCritic,load_cohort,digest
import argparse,json
import numpy as np
import torch
import torch.nn.functional as F

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    task=args.task
    train,test,source=load_cohort(ROOT/'matched_sft_k32',task,candidates=32)
    old_train,old_test,_=load_cohort(ROOT/'matched_sft_data',task)
    for a,b in zip(train+test,old_train+old_test):
        for key in ['state','expert','features']:
            assert np.array_equal(a[key],b[key])
        assert np.array_equal(a['policy'][:,:4],b['policy'])
    out=ROOT/'rank_k32_10000'/task
    out.mkdir(parents=True,exist_ok=True)
    norm=base.normalization(train,list(range(50)))
    i,j=np.triu_indices(32,1)
    distances=np.concatenate([distance(row,norm['action_scale'].numpy()) for row in train])
    gaps=distances[:,j]-distances[:,i]
    delta=float(np.quantile(abs(gaps),.25))
    base.seed_all(1042)
    model=ChunkCoverageCritic(norm,horizon=10,width=256).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rng=np.random.default_rng(1042)
    sizes=np.array([len(e['state']) for e in train]);offset=np.r_[0,np.cumsum(sizes)[:-1]]
    data={k:torch.as_tensor(np.concatenate([r[k] for r in train]),device='cuda') for k in ['features','state','expert','policy']}
    gap_all=torch.as_tensor(gaps,device='cuda')
    for step in range(1,10001):
        groups=rng.integers(50,size=256)
        idx=torch.as_tensor(offset[groups]+rng.integers(sizes[groups]),device='cuda')
        neg=torch.as_tensor(rng.integers(32,size=256),device='cuda')
        f,s=data['features'][idx],data['state'][idx]
        pos=model(f,s,data['expert'][idx])
        scores=model(f.repeat_interleave(32,0),s.repeat_interleave(32,0),data['policy'][idx].reshape(-1,10,14)).reshape(-1,32)
        bce=.5*(F.softplus(-pos).mean()+F.softplus(scores[torch.arange(256,device='cuda'),neg]).mean())
        gap=gap_all[idx];mask=gap.abs()>delta;counts=mask.sum(1)
        loss=F.softplus(-gap.sign()*(scores[:,i]-scores[:,j]))
        rank=((loss*mask).sum(1)/counts.clamp_min(1)).sum()/(counts>0).sum().clamp_min(1)
        opt.zero_grad(set_to_none=True);(bce+rank).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%1000==0:
            print(step,bce.item(),rank.item(),flush=True)
            final=out/'final_full50';final.mkdir(exist_ok=True)
            torch.save(dict(model=model.state_dict(),normalization=norm,step=step,optimizer=opt.state_dict(),rng=rng.bit_generator.state),final/'last.pt')
    # Fixed train-derived labels and thresholds for every evaluated model.
    report=dict(task=task,source=source,steps=10000,batch_size=256,train_episodes=50,test_episodes=20,candidate_count=32,
                rank_lambda=1,delta=delta,checkpoint_sha256=digest(out/'final_full50/last.pt'),
                calibration=json.loads((ROOT/'matched_sft_models'/task/'report.json').read_text())['calibration'])
    metrics={}
    for label,folder in [('bce_k4','matched_sft_models'),('rank_k4_10k','rank_distance_v1'),('rank_k4_100k','rank_distance_100000'),('rank_k32_10k','rank_k32_10000')]:
        ck=torch.load(ROOT/folder/task/'final_full50/last.pt',map_location='cpu',weights_only=False)
        m=ChunkCoverageCritic(ck['normalization'],horizon=10,width=256).cuda();m.load_state_dict(ck['model']);m.eval()
        metrics[label]={}
        for split,rows in [('train',train),('test',test)]:
            correct=total=0; switched=anchors=0
            for e in rows:
                _,scores=base.score_episode(m,e,'cuda',batch_size=16)
                d=distance(e,norm['action_scale'].numpy());gap=d[:,j]-d[:,i]
                margins=((scores[:,i]-scores[:,j])*np.sign(gap))[abs(gap)>delta]
                correct+=(margins>0).sum()+.5*(margins==0).sum();total+=len(margins);anchors+=len(d)
            metrics[label][split]=dict(correct=float(correct),pairs=total,accuracy=float(correct/total),anchors=anchors)
    report['shared_k32_metrics']=metrics
    base.atomic_json(out/'report.json',report)
    print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
