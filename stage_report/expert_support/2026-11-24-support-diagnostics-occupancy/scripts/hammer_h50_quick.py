"""Episode-disjoint 40/10 H50 K4 hammer distance-ranking diagnostic."""
from run_online20 import *
from train_click_rank import distance,JOINTS,I,J
from train import base,ChunkCoverageCritic,digest
from resample_cup_sft import decode_selected,atomic_npz,CAMERAS
from score_outcomes import load_encoder,encode_views
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

TASK='beat_block_hammer'
DEST=OUT/'hammer_h50_quick'
DATA=Path('/hdd/official_nine_task_sft_20261101/lerobot/beat_block_hammer_official50_v30_20261101')
def main():
    torch.set_num_threads(4)
    DEST.mkdir(parents=True,exist_ok=True)
    table=pq.read_table(DATA/'data/chunk-000/file-000.parquet').to_pydict()
    metas=pq.read_table(DATA/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    tasks=pq.read_table(DATA/'meta/tasks.parquet').to_pydict()
    prompts=dict(zip(tasks['task_index'],tasks['__index_level_0__']))
    selected=set(); anchors=[]
    for e in metas:
        a,b=e['dataset_from_index'],e['dataset_to_index']
        ids=np.arange(a,b-50,5)
        assert len(ids)>0
        anchors.append(ids);selected.update(ids.tolist())
    images={c:decode_selected(DATA/f'videos/observation.images.{c}/chunk-000/file-000.mp4',selected) for c in CAMERAS}
    spec=dict(port=18833,checkpoint=Path('/hdd/official_nine_task_pi05_sft_5000_20261101/checkpoints/beat_block_hammer/4999'),
              config='pi05_beat_block_hammer_official50_h50_state_5000_20261101',asset='beat_block_hammer_official50_v30_20261101')
    sys.path.insert(0,str(RUNTIME/'XPolicyLab'))
    from client_server.ws import WsModelClient
    proc=start_server(TASK,spec,DEST)
    client=None
    try:
        client=WsModelClient(url='ws://127.0.0.1:18833',evaluation_id='hammer-h50',trial_id='hammer-h50',action_case_id='hammer-h50')
        for n,(meta,ids) in enumerate(zip(metas,anchors)):
            target=DEST/f'episode_{n:03}.npz'
            if target.exists():continue
            policy=[]; state=np.asarray(table['observation.state'],dtype=np.float32)
            client.call(func_name='prepare_case',obs=dict(task_name=TASK,seed=n,instruction=prompts[table['task_index'][ids[0]]],action_type='joint'))
            client.call(func_name='reset')
            for idx in ids:
                client.call(func_name='update_obs',obs=dict(state=state[idx],instruction=prompts[table['task_index'][idx]],
                    images={c:images[c][int(idx)].transpose(2,0,1).copy() for c in CAMERAS}))
                chunks=client.call(func_name='get_action_candidates',obs=4)
                policy.append(np.asarray([[np.concatenate([a['left_arm_joint_state'],a['left_ee_joint_state'],a['right_arm_joint_state'],a['right_ee_joint_state']]) for a in chunk] for chunk in chunks],dtype=np.float32))
            atomic_npz(target,dict(state=state[ids],expert=np.stack([state[k+1:k+51] for k in ids]),policy=np.stack(policy)))
            print('sample',n+1,50,flush=True)
    finally:
        if client is not None and hasattr(client,'close'):client.close()
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)
    encoder=load_encoder(OUT/'assets/r3m18_pytorch_model.bin','r3m',torch.device('cuda'))
    episodes=[]
    for n,ids in enumerate(anchors):
        with np.load(DEST/f'episode_{n:03}.npz') as z: e={k:z[k] for k in z.files}
        assert e['policy'].shape==(len(ids),4,50,14)
        e['features']=encode_views(encoder,[[images[c][int(k)] for k in ids] for c in CAMERAS],torch.device('cuda'))
        e['episode']=str(n)
        episodes.append(e)
    del encoder,images
    perm=np.random.default_rng(42).permutation(50);fit=perm[:40].tolist();test=perm[40:].tolist()
    norm=base.normalization(episodes,fit)
    # Fixed labels shared by both objectives.
    delta=float(np.quantile(np.abs(np.concatenate([distance(episodes[k],norm['action_scale'].numpy()) for k in fit])[:,J]-
                                   np.concatenate([distance(episodes[k],norm['action_scale'].numpy()) for k in fit])[:,I]),.25))
    result=dict(task=TASK,horizon=50,candidates=4,train=fit,test=test,delta=delta,
                data_source=str(DATA),data_sha256=digest(DATA/'data/chunk-000/file-000.parquet'),
                semantics='next50 recorded qpos; source recordings 15Hz; no padding; episode-disjoint support split within SFT expert50')
    for name,weight in [('bce',0),('bce_rank',1)]:
        base.seed_all(1042)
        model=ChunkCoverageCritic(norm,horizon=50,width=256).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        rng=np.random.default_rng(1042)
        data={k:torch.as_tensor(np.concatenate([episodes[x][k] for x in fit]),device='cuda') for k in ['state','expert','policy','features']}
        ds=np.concatenate([distance(episodes[x],norm['action_scale'].numpy()) for x in fit])
        gaps=torch.as_tensor(ds[:,J]-ds[:,I],device='cuda')
        sizes=np.array([len(episodes[x]['state']) for x in fit]);offset=np.r_[0,np.cumsum(sizes)[:-1]]
        curve=[]
        for step in range(1,10001):
            groups=rng.integers(40,size=256);idx=torch.as_tensor(offset[groups]+rng.integers(sizes[groups]),device='cuda')
            neg=torch.as_tensor(rng.integers(4,size=256),device='cuda');f,s=data['features'][idx],data['state'][idx]
            pos=model(f,s,data['expert'][idx]);scores=model(f.repeat_interleave(4,0),s.repeat_interleave(4,0),data['policy'][idx].reshape(-1,50,14)).reshape(-1,4)
            bce=.5*(F.softplus(-pos).mean()+F.softplus(scores[torch.arange(256,device='cuda'),neg]).mean())
            g=gaps[idx];mask=g.abs()>delta;cnt=mask.sum(1)
            rank=((F.softplus(-g.sign()*(scores[:,I]-scores[:,J]))*mask).sum(1)/cnt.clamp_min(1)).sum()/(cnt>0).sum().clamp_min(1)
            opt.zero_grad(set_to_none=True);(bce+weight*rank).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step%1000==0:
                record=dict(step=step,bce=bce.item(),rank=rank.item())
                for split,indices in [('train',fit),('test',test)]:
                    correct=total=0
                    for k in indices:
                        e=episodes[k];_,scores=base.score_episode(model.eval(),e,'cuda')
                        ds=distance(e,norm['action_scale'].numpy());g=ds[:,J]-ds[:,I];margin=((scores[:,I]-scores[:,J])*np.sign(g))[abs(g)>delta]
                        correct+=(margin>0).sum()+.5*(margin==0).sum();total+=len(margin)
                    record[split]=dict(accuracy=float(correct/total),pairs=total)
                model.train();curve.append(record);print(name,record,flush=True)
        torch.save(dict(model=model.state_dict(),normalization=norm,step=10000),DEST/(name+'.pt'))
        result[name]=curve
        base.atomic_json(DEST/'report.json',result)
if __name__=='__main__':main()
