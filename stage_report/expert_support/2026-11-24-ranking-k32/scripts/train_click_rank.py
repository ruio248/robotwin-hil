"""Click candidate ranking ablation on the frozen matched-SFT cohort."""
from train import REPO, base, ChunkCoverageCritic, load_cohort, score_rows, digest
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = REPO / 'outputs/support_three_tasks_20261123'
import argparse

def experiment_task(with_steps=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='click_alarmclock', choices=('click_alarmclock','place_dual_shoes','place_empty_cup'))
    if with_steps:
        parser.add_argument('--steps', type=int, default=10000)
        return parser.parse_args()
    return parser.parse_args().task
JOINTS = [0,1,2,3,4,5,7,8,9,10,11,12]
I,J = np.triu_indices(4,1)

def distance(row, scale):
    error = ((row['policy'] - row['expert'][:,None]) / scale)**2
    return .5*(error[...,JOINTS].mean((2,3)) + error[...,[6,13]].mean((2,3)))

def main():
    args = experiment_task(with_steps=True)
    task = args.task
    extended = args.steps > 10000
    if args.steps < 10000:
        raise ValueError('Expected at least 10000 steps')
    dest = ROOT / 'rank_distance_v1' / task
    original = dest
    if extended:
        dest = ROOT / f'rank_distance_{args.steps}' / task
    torch.set_num_threads(4)
    tr,te,source = load_cohort(ROOT/'matched_sft_data', task)
    old_dir = ROOT/'matched_sft_models'/task
    old_report = json.loads((old_dir/'report.json').read_text())
    assert source == old_report['source']
    for split,rows in [('train',tr),('test',te)]:
        assert {e['episode']:e['sha256'] for e in rows} == old_report[split+'_episode_sha256']
    base.seed_all(1042)
    norm = base.normalization(tr, list(range(50)))
    model = ChunkCoverageCritic(norm,horizon=source['horizon'],width=256).cuda()
    opt = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rng = np.random.default_rng(1042)
    sizes=np.array([len(r['state']) for r in tr])
    offsets=np.r_[0,np.cumsum(sizes)[:-1]]
    def tensor(a):
        return torch.as_tensor(a,device='cuda')
    feats,states,exp,pol=[tensor(np.concatenate([r[k] for r in tr])) for k in ['features','state','expert','policy']]
    distances=np.concatenate([distance(r,norm['action_scale'].numpy()) for r in tr])
    gaps=distances[:,J]-distances[:,I]
    delta=float(np.quantile(abs(gaps),.25))
    dif=tensor(gaps)
    final=dest/'final_full50'
    final.mkdir(parents=True,exist_ok=True)
    curve = []
    for step in range(1,args.steps+1):
        groups=rng.integers(50,size=256)
        idx=tensor(offsets[groups]+rng.integers(sizes[groups]))
        neg=tensor(rng.integers(4,size=256))
        f,s=feats[idx],states[idx]
        pos=model(f,s,exp[idx])
        scores=model(f.repeat_interleave(4,0),s.repeat_interleave(4,0),pol[idx].reshape(-1,source['horizon'],14)).reshape(-1,4)
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
            print(step,bce.item(),rank.item(),flush=True)
            if extended and step == 10000:
                path = original/'final_full50/last.pt'
                assert digest(path) == json.loads((original/'report.json').read_text())['checkpoint_sha256']
                reference = torch.load(path, weights_only=False)['model']
                assert all(torch.equal(v,reference[k]) for k,v in model.state_dict().items()), '10k replay mismatch'
            if extended and step >= 10000:
                point = {'step':step}
                for split,rows in [('train',tr),('test',te)]:
                    correct = total = 0
                    for episode in rows:
                        logits = base.score_episode(model.eval(),episode,'cuda')[1]
                        d = distance(episode,norm['action_scale'].numpy())
                        gap = d[:,J] - d[:,I]
                        margin = ((logits[:,I]-logits[:,J])*np.sign(gap))[abs(gap)>delta]
                        correct += (margin>0).sum() + .5*(margin==0).sum()
                        total += len(margin)
                    point[split] = correct/total
                model.train()
                curve.append(point)
                base.atomic_json(dest/'curve.json',curve)
            torch.save(dict(model=model.state_dict(),normalization=norm,step=step),final/'last.pt')
    old=ChunkCoverageCritic(norm,horizon=source['horizon'],width=256).cuda()
    ckpt=old_dir/'final_full50/last.pt'
    assert digest(ckpt)==old_report['checkpoint_sha256']
    old.load_state_dict(torch.load(ckpt,weights_only=False)['model'])
    diagnostics=[]
    for row in te:
        d=distance(row,norm['action_scale'].numpy())
        gap=d[:,J]-d[:,I]
        mask=abs(gap)>delta
        scores=[base.score_episode(m.eval(),row,'cuda')[1] for m in (old,model)]
        record=dict(scene=row['episode'],anchors=len(d),pairs=int(mask.sum()),possible=mask.size,
                    switched=int((scores[0].argmin(1)!=scores[1].argmin(1)).sum()))
        for name,sc in zip(('bce','rank'),scores):
            margin=(sc[:,I]-sc[:,J])*np.sign(gap)
            record[name+'_correct']=float(((margin[mask]>0)+.5*(margin[mask]==0)).sum())
        diagnostics.append(record)
    report=dict(task=task,source=source,steps=args.steps,batch_size=256,train_episodes=50,test_episodes=20,
                rank_lambda=1,delta=delta,threshold='training absolute pair gap 25th percentile',
                calibration=old_report['calibration'],scale_source='fixed BCE scale for comparison',
                checkpoint_sha256=digest(final/'last.pt'),paired_diagnostics=diagnostics,
                test20=score_rows(model,te,'cuda',old_report['calibration']))
    base.atomic_json(dest/'report.json',report)

if __name__=='__main__':
    main()
