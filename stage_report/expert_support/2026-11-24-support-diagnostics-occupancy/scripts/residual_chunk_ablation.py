"""Shoes complete-H10 input representation ablation with width-matched control."""
from train import *
from train_click_rank import ROOT,distance,I,J
import torch.nn.functional as F

class ResidualChunkCritic(ChunkCoverageCritic):
    def __init__(self, normalization, horizon=10, width=256, residual_enabled=True):
        super().__init__(normalization,horizon,width)
        self.residual_enabled=residual_enabled
        self.net[0]=torch.nn.Linear(1536+14+3*horizon*14,width)

    def forward(self, features, state, chunk):
        if chunk.shape[-2:] != (self.horizon,14):
            raise ValueError('Expected complete physical chunk')
        absolute=(chunk-self.action_mean)/self.action_scale
        relative=(chunk-state.unsqueeze(-2))/self.action_scale
        # First transition starts at the observed current joints.
        previous=torch.cat([state.unsqueeze(-2),chunk[...,:-1,:]],dim=-2)
        differences=(chunk-previous)/self.action_scale
        if not self.residual_enabled:
            relative=torch.zeros_like(relative)
            differences=torch.zeros_like(differences)
        inputs=torch.cat([features,(state-self.state_mean)/self.state_scale,
                          absolute.flatten(-2),relative.flatten(-2),differences.flatten(-2)],dim=-1)
        return self.net(inputs).squeeze(-1)

def main():
    torch.set_num_threads(4)
    task='place_dual_shoes'
    train,test,source=load_cohort(ROOT/'matched_sft_data',task)
    old=json.loads((ROOT/'rank_distance_v1'/task/'report.json').read_text())
    assert old['source']==source
    data_report=json.loads((ROOT/'matched_sft_models'/task/'report.json').read_text())
    for split,rows in [('train',train),('test',test)]:
        assert {e['episode']:e['sha256'] for e in rows}==data_report[split+'_episode_sha256']
    baseline=torch.load(ROOT/'rank_distance_v1'/task/'final_full50/last.pt',weights_only=False,map_location='cpu')
    norm=baseline['normalization'];delta=old['delta']
    sizes=np.array([len(r['state']) for r in train]);offsets=np.r_[0,np.cumsum(sizes)[:-1]]
    tensors={k:torch.as_tensor(np.concatenate([r[k] for r in train]),device='cuda') for k in ['state','features','expert','policy']}
    ds=np.concatenate([distance(r,norm['action_scale'].numpy()) for r in train])
    gaps=torch.as_tensor(ds[:,J]-ds[:,I],device='cuda')
    for enabled in (False,True):
        mode='absolute_zero_control' if not enabled else 'absolute_relative_differences'
        dest=ROOT/'residual_ablation_20261124'/mode/task
        dest.mkdir(parents=True,exist_ok=True)
        base.seed_all(1042)
        model=ResidualChunkCritic(norm,residual_enabled=enabled).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        rng=np.random.default_rng(1042);curve=[]
        for step in range(1,100001):
            groups=rng.integers(50,size=256)
            idx=torch.as_tensor(offsets[groups]+rng.integers(sizes[groups]),device='cuda')
            neg=torch.as_tensor(rng.integers(4,size=256),device='cuda')
            f,s=tensors['features'][idx],tensors['state'][idx]
            pos=model(f,s,tensors['expert'][idx])
            logits=model(f.repeat_interleave(4,0),s.repeat_interleave(4,0),tensors['policy'][idx].reshape(-1,10,14)).reshape(-1,4)
            bce=.5*(F.softplus(-pos).mean()+F.softplus(logits[torch.arange(256,device='cuda'),neg]).mean())
            g=gaps[idx];mask=g.abs()>delta;counts=mask.sum(1)
            rank=((F.softplus(-g.sign()*(logits[:,I]-logits[:,J]))*mask).sum(1)/counts.clamp_min(1)).sum()/(counts>0).sum().clamp_min(1)
            opt.zero_grad(set_to_none=True);(bce+rank).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step%1000==0:
                point=dict(step=step,bce=bce.item(),rank=rank.item())
                for split,rows in [('train',train),('test',test)]:
                    correct=total=0
                    for e in rows:
                        scores=base.score_episode(model.eval(),e,'cuda')[1]
                        d=distance(e,norm['action_scale'].numpy());gap=d[:,J]-d[:,I]
                        margin=((scores[:,I]-scores[:,J])*np.sign(gap))[abs(gap)>delta]
                        correct+=(margin>0).sum()+.5*(margin==0).sum();total+=len(margin)
                    point[split]=float(correct/total)
                model.train();curve.append(point);print(mode,point,flush=True)
                base.atomic_json(dest/'curve.json',curve)
                if step in (10000,100000):
                    stage=dest/str(step);final=stage/'final_full50';final.mkdir(parents=True,exist_ok=True)
                    torch.save(dict(model=model.state_dict(),normalization=norm,step=step,
                                    optimizer=opt.state_dict(),rng=rng.bit_generator.state),final/'last.pt')
                    report=dict(task=task,steps=step,batch_size=256,train_episodes=50,test_episodes=20,
                                source=source,model_type='residual_chunk',residual_enabled=enabled,
                                parameters=sum(p.numel() for p in model.parameters()),delta=delta,
                                rank_lambda=1,train_accuracy=point['train'],test_accuracy=point['test'],
                                checkpoint_sha256=digest(final/'last.pt'),calibration=old['calibration'],
                                test20=score_rows(model.eval(),test,'cuda',old['calibration']))
                    model.train()
                    base.atomic_json(stage/'report.json',report)
if __name__=='__main__':main()
