"""Fixed-SFT expert-state coverage expansion, nested 50->100."""
from run_online20 import *
from train import base,load_cohort,ChunkCoverageCritic,digest
from train_click_rank import distance,I,J
from resample_cup_sft import atomic_npz,joint_chunk,CAMERAS
from score_outcomes import load_encoder,encode_views
import torch
import torch.nn.functional as F
from types import SimpleNamespace

EXP=OUT/'expert100_20261124'
def collect(task):
    import pace.robotwin as adapter
    expert_root=Path('/hdd/pace_pi05_eval_20261022')
    adapter.ROBOTWIN_ROOT=expert_root
    sys.path.insert(0,str(expert_root))
    sys.path.insert(0,str(expert_root/"description/utils"))
    os.chdir(expert_root)
    from coverage_benchmark.collect_matched_pairs_recoverable import _StepTap,_qpos
    from client_server.ws import WsModelClient
    spec=SPECS[task];dest=EXP/task;dest.mkdir(parents=True,exist_ok=True)
    tr,te,_=load_cohort(OUT/'matched_sft_data',task)
    spec=dict(spec)
    if task=='place_empty_cup':spec['port']=18816
    proc=None;client=None;records=[]
    try:
        proc=start_server(task,spec,dest)
        client=WsModelClient(url=f'ws://127.0.0.1:{spec["port"]}',evaluation_id='expert100',trial_id='expert100',action_case_id='expert100')
        for seed in range(9200000,9201000):
            if len(records)==50:break
            target=dest/f'new_seed_{seed}.npz'
            if target.exists():
                records.append(dict(seed=seed,file=target.name,sha256=digest(target)));continue
            env,args=adapter.load_environment(task)
            # Images/state follow the 15Hz source expert recordings; capture every fifth recorded state.
            args.update(save_data=False,render_freq=0)
            states=[];anchors=[];count=0;last=-1
            try:
                env.setup_demo(now_ep_num=len(records),seed=seed,is_test=True,**args)
                scene=env.scene;dt=scene.get_timestep()
                def tap():
                    nonlocal count,last
                    count+=1
                    tick=int(count*dt*15+1e-8)
                    if tick<=last:return
                    last=tick
                    q=_qpos(env);states.append(q)
                    if (len(states)-1)%5==0:
                        obs=env.get_obs();assert np.allclose(obs['joint_action']['vector'],q,atol=1e-5)
                        anchors.append((len(states)-1,[np.asarray(obs['observation'][c]['rgb']).copy() for c in ['head_camera','left_camera','right_camera']]))
                env.scene=_StepTap(scene,tap)
                info=env.play_once()
                success=bool(env.plan_success and env.check_success())
                if not success:
                    print(task,'reject',seed,flush=True);continue
                prompt=adapter.sample_instruction(task,info,1,'seen')
                valid=[(k,img) for k,img in anchors if k+10<len(states)]
                if not valid:continue
                client.call(func_name='prepare_case',obs=dict(task_name=task,seed=seed,instruction=prompt,action_type='joint'))
                client.call(func_name='reset')
                policy=[]
                for k,img in valid:
                    client.call(func_name='update_obs',obs=dict(state=states[k],images={c:a.transpose(2,0,1).copy() for c,a in zip(CAMERAS,img)},instruction=prompt))
                    raw=([client.call(func_name='get_action') for _ in range(4)] if task=='place_empty_cup'
                         else client.call(func_name='get_action_candidates',obs=4))
                    policy.append(np.stack([joint_chunk(a) for a in raw]))
                arrays=dict(state=np.stack([states[k] for k,_ in valid]),expert=np.stack([states[k+1:k+11] for k,_ in valid]),
                            policy=np.stack(policy),images=np.stack([img for _,img in valid]),frame_index=np.array([k for k,_ in valid]),seed=np.array(seed),instruction=np.array(prompt))
                atomic_npz(target,arrays)
                records.append(dict(seed=seed,file=target.name,sha256=digest(target),anchors=len(valid)))
                print(task,'accepted',len(records),seed,len(valid),flush=True)
                write(dest/'progress.json',dict(episodes=records,complete=False))
            except Exception as error:
                print(task,'expert_error',seed,type(error).__name__,str(error)[:180],flush=True)
                # Preserve sampling or server errors rather than silently rejecting them.
                if len(states)>0 and 'success' in locals() and success:raise
            finally:
                try:env.close_env()
                except Exception:pass
        assert len(records)==50
        write(dest/'new_manifest.json',dict(task=task,episodes=records,complete=True,checkpoint=str(spec['checkpoint']),
                                           source_rate_hz=15,horizon=10,candidates=4,new_seed_range=[9200000,9201000]))
    finally:
        if client is not None and hasattr(client,'close'):client.close()
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)

def train(task):
    dest=EXP/task
    old,test,_=load_cohort(OUT/'matched_sft_data',task)
    reference=json.loads((OUT/'rank_distance_v1'/task/'report.json').read_text())
    ck=torch.load(OUT/'rank_distance_v1'/task/'final_full50/last.pt',map_location='cpu',weights_only=False)
    norm=ck['normalization'];delta=reference['delta']
    encoder=load_encoder(SPECS[task]['encoder'],'imagenet' if task in ('place_empty_cup','place_dual_shoes') else 'r3m',torch.device('cuda'))
    new=[]
    for r in json.loads((dest/'new_manifest.json').read_text())['episodes']:
        path=dest/r['file'];assert digest(path)==r['sha256']
        with np.load(path) as z:e={k:z[k] for k in ['state','expert','policy']};images=z['images']
        e['features']=encode_views(encoder,[[a[v] for a in images] for v in range(3)],torch.device('cuda'))
        e['episode']=str(r['seed']);new.append(e)
    del encoder
    train_rows=old+new;assert len(train_rows)==100 and not(set(e['episode'] for e in train_rows)&set(e['episode'] for e in test))
    base.seed_all(1042);model=ChunkCoverageCritic(norm,horizon=10,width=256).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);rng=np.random.default_rng(1042)
    tensors={k:torch.as_tensor(np.concatenate([e[k] for e in train_rows]),device='cuda') for k in ['state','expert','policy','features']}
    ds=np.concatenate([distance(e,norm['action_scale'].numpy()) for e in train_rows]);gap_all=torch.as_tensor(ds[:,J]-ds[:,I],device='cuda')
    sizes=np.array([len(e['state']) for e in train_rows]);offset=np.r_[0,np.cumsum(sizes)[:-1]];curve=[]
    for step in range(1,10001):
        group=rng.integers(100,size=256);idx=torch.as_tensor(offset[group]+rng.integers(sizes[group]),device='cuda')
        neg=torch.as_tensor(rng.integers(4,size=256),device='cuda');f,s=tensors['features'][idx],tensors['state'][idx]
        pos=model(f,s,tensors['expert'][idx]);scores=model(f.repeat_interleave(4,0),s.repeat_interleave(4,0),tensors['policy'][idx].reshape(-1,10,14)).reshape(-1,4)
        bce=.5*(F.softplus(-pos).mean()+F.softplus(scores[torch.arange(256,device='cuda'),neg]).mean())
        gap=gap_all[idx];mask=gap.abs()>delta;cnt=mask.sum(1)
        rank=((F.softplus(-gap.sign()*(scores[:,I]-scores[:,J]))*mask).sum(1)/cnt.clamp_min(1)).sum()/(cnt>0).sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True);(bce+rank).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%1000==0:
            row=dict(step=step,bce=bce.item(),rank=rank.item())
            for split,es in [('train100',train_rows),('test20',test)]:
                correct=total=0
                for e in es:
                    _,score=base.score_episode(model.eval(),e,'cuda')
                    d=distance(e,norm['action_scale'].numpy());g=d[:,J]-d[:,I]
                    m=((score[:,I]-score[:,J])*np.sign(g))[abs(g)>delta]
                    correct+=(m>0).sum()+.5*(m==0).sum();total+=len(m)
                row[split]=float(correct/total)
            model.train();curve.append(row);print(task,row,flush=True)
            write(dest/'curve.json',curve)
    final=dest/'final_full100';final.mkdir(exist_ok=True)
    torch.save(dict(model=model.state_dict(),normalization=norm,step=10000),final/'last.pt')
    write(dest/'report.json',dict(task=task,train_episodes=100,test_episodes=20,horizon=10,candidates=4,
        steps=10000,source={'horizon':10},checkpoint_relative='final_full100/last.pt',delta=delta,normalization='fixed original train50',curve=curve,
        original_report_sha256=digest(OUT/'rank_distance_v1'/task/'report.json'),
        checkpoint_sha256=digest(final/'last.pt'),new_manifest_sha256=digest(dest/'new_manifest.json')))

if __name__=='__main__':
    torch.set_num_threads(4)
    for task in ['place_dual_shoes','place_empty_cup']:
        collect(task);train(task)
