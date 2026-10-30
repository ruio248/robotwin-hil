"""Sequential paired SAPIEN branch execution with a mandatory replay gate."""
from __future__ import annotations
import contextlib
import copy
import json
import os
from pathlib import Path
import pickle
import random
import time
import numpy as np
import torch
from coverage_sampling.robotwin import SceneSnapshot, physical_signature, _Discard
from tail_value.prepare import FrozenEncoder
from .common import atomic_json, atomic_npz, sha256, weights
from .scorers import load_scorers


class Snapshot(SceneSnapshot):
    ENV_FIELDS=SceneSnapshot.ENV_FIELDS+('info','episode_metadata','stage_snapshots',
        'left_joint_path','right_joint_path','left_cnt','right_cnt','need_plan','stage_success_tag',
        'step_lim','save_freq','fix_gripper','file_path','_benchmark_physics_steps')
    ROBOT_FIELDS=SceneSnapshot.ROBOT_FIELDS+('left_joint_path','right_joint_path')
    def __init__(self,env):
        super().__init__(env)
        self.torch_rng=torch.get_rng_state()
        self.env_existing=set(vars(env));self.robot_existing=set(vars(env.robot))
        self.sleep_states=[]
        for actor in env.scene.get_all_actors():
            for component in getattr(actor,'components',()):
                if hasattr(component,'is_sleeping') and hasattr(component,'wake_up'):
                    value=component.is_sleeping
                    self.sleep_states.append((component,bool(value() if callable(value) else value)))
    def restore(self):
        # Delete branch-created Python metadata, not just overwrite old fields.
        for obj,old in ((self.env,self.env_existing),(self.env.robot,self.robot_existing)):
            for key in set(vars(obj))-old:delattr(obj,key)
        super().restore();torch.set_rng_state(self.torch_rng)
        for component,sleeping in self.sleep_states:
            if sleeping:component.put_to_sleep()
            else:component.wake_up()
    def persist(self,path):
        artifact={'physics':self.physics,'environment':self.env_state,'robot':self.robot_state,
                  'python_rng':self.python_rng,'numpy_rng':self.numpy_rng,'torch_rng':self.torch_rng,
                  'rigid_body_sleeping':[sleeping for _,sleeping in self.sleep_states],
                  'drives':[(qf,qacc,[(p,v) for _,p,v in joints]) for _,qf,qacc,joints in self.articulations]}
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:pickle.dump(artifact,f,protocol=5)
        os.replace(tmp,path)


class CountedScene:
    def __init__(self,scene,env):self.original=scene;self.env=env
    def __getattr__(self,key):return getattr(self.original,key)
    def step(self):
        self.original.step();self.env._benchmark_physics_steps+=1


def signature(env):
    parts=[physical_signature(env)]
    for art in env.scene.get_all_articulations():
        for j in art.get_active_joints():
            parts.extend([np.atleast_1d(j.get_drive_target()),np.atleast_1d(j.get_drive_velocity_target())])
    return np.concatenate(parts)


def assert_restored(env,snap,expected,tol):
    snap.restore();error=float(np.max(np.abs(signature(env)-expected)))
    if error>tol:raise RuntimeError(f'Restore signature error {error} > {tol}')
    return error


def setup(config,seed,output):
    from eval_policy_xpolicylab import class_decorator,load_task_args
    user={'task_name':'handover_to_tray','task_config':config['protocol']['task_config'],
          'policy_name':'Pi_05_RobotTwin','ckpt_setting':'9999'}
    args,_=load_task_args(user)
    args.update(eval_mode=True,eval_instruction='seen',render_freq=0,need_plan=True,
                save_data=False,save_video=False,save_path=str(output),save_freq=15)
    env=class_decorator('handover_to_tray')
    env.setup_demo(now_ep_num=0,seed=seed,is_test=False,**args)
    env.step_lim=900;env.set_instruction(config['protocol']['prompt'])
    env.eval_video_path=None;env._benchmark_physics_steps=0
    env.scene=CountedScene(env.scene,env)
    return env


def update(client,env,config):
    from eval_policy_xpolicylab import robotwin_obs_to_xpolicylab
    obs=env.get_obs()
    client.call(func_name='update_obs',obs=robotwin_obs_to_xpolicylab(obs,
        instruction=config['protocol']['prompt'],frequency=30,task_env=env))
    return obs


def infer(client,noise):
    a=np.asarray(client.call(func_name='benchmark_action',obs={'noise':noise}),np.float32)
    if a.shape!=(len(noise),10,14) or not np.isfinite(a).all():raise ValueError('Bad policy output')
    return a


def execute(env,chunk,frames=None):
    for action in chunk:
        if env.eval_success or env.take_action_cnt>=env.step_lim:break
        env.get_obs()
        env.take_action(action,action_type='qpos')
        if frames is not None:frames.append(env.get_obs()['observation']['head_camera']['rgb'].copy())


def status(env,started,physics_start):
    return {'success':bool(env.eval_success or env.check_success()),'policy_actions':int(env.take_action_cnt),
        'elapsed_seconds':time.monotonic()-started,'physics_steps':env._benchmark_physics_steps-physics_start,
        'planning_success':bool(env.plan_success),'dropped':bool(env.bar.get_pose().p[2]<.70),
        'pose_error':{k:float(v) for k,v in env.success_metrics().items() if isinstance(v,(float,int,np.number))}}


def expert(env):
    started=time.monotonic();physics=env._benchmark_physics_steps
    if env.eval_success:return {**status(env,started,physics),'reason':'already_terminal'}
    final=None
    for event in env.recovery_stage_iterator():
        if event['phase']=='done':final=event.get('result',event)
    if final is None:raise RuntimeError('Expert recovery missing terminal record')
    return {**status(env,started,physics),'recovery':final,'success':bool(final['success'])}


def replay_prefix(env,baseline,index):
    for chunk in baseline[:index]:execute(env,chunk)


def prepare_scene(config,client,seed,fraction,directory):
    from eval_policy_xpolicylab import safe_close_env
    directory.mkdir(parents=True,exist_ok=True)
    shape=client.call(func_name='benchmark_info')['noise_shape']
    noise_file=directory/'noise.npz'
    if not noise_file.exists():
        rng=np.random.default_rng(seed)
        atomic_npz(noise_file,{'baseline':rng.standard_normal((90,1,*shape),dtype=np.float32),
            'candidates':rng.standard_normal((8,*shape),dtype=np.float32),
            'continuation':rng.standard_normal((3,90,1,*shape),dtype=np.float32)})
    with np.load(noise_file,allow_pickle=False) as f:noises=dict(f)
    path=directory/'baseline.npz'
    if not path.exists():
        env=setup(config,seed,directory);chunks=[];states=[]
        try:
            client.call(func_name='reset')
            for noise in noises['baseline']:
                if env.eval_success or env.take_action_cnt>=900:break
                update(client,env,config);states.append(signature(env))
                chunk=infer(client,noise)[0];chunks.append(chunk);execute(env,chunk)
            if not chunks:raise RuntimeError('Empty baseline trajectory')
            atomic_npz(path,{'chunks':np.stack(chunks),'signatures':np.stack(states)})
            atomic_json(directory/'baseline.json',{'seed':seed,'success':bool(env.eval_success),
                'decisions':len(chunks),'actions':env.take_action_cnt,'physics_steps':env._benchmark_physics_steps})
        finally:safe_close_env(env)
    with np.load(path,allow_pickle=False) as f:base=dict(f)
    index=int(round(fraction*(len(base['chunks'])-1)))
    env=setup(config,seed,directory);replay_prefix(env,base['chunks'],index)
    error=float(np.max(abs(signature(env)-base['signatures'][index])))
    if error>config['protocol']['restore_atol']:
        safe_close_env(env);raise RuntimeError(f'Baseline prefix replay diverged seed={seed}, error={error}')
    client.call(func_name='reset');obs=update(client,env,config)
    pool_file=directory/'candidates.npz'
    if pool_file.exists():
        with np.load(pool_file,allow_pickle=False) as f:pool=f['chunks']
    else:
        pool=np.concatenate([infer(client,noises['candidates'][i:i+4]) for i in (0,4)])
        atomic_npz(pool_file,{'chunks':pool,'signature':signature(env)})
    atomic_json(directory/'state.json',{'seed':seed,'fraction_stratum':fraction,'baseline_decision':index,
        'baseline_decisions':len(base['chunks']),'policy_actions':env.take_action_cnt,
        'prefix_replay_max_error':error,'noise_sha256':sha256(noise_file),'candidate_sha256':sha256(pool_file)})
    return env,pool,noises,obs


def validity_gate(config,client,output):
    from eval_policy_xpolicylab import safe_close_env
    tol=config['protocol']['restore_atol'];results=[]
    for i,seed in enumerate(range(200000,200003)):
        directory=output/f'scene_{seed}'
        env,pool,noises,obs=prepare_scene(config,client,seed,config['protocol']['branch_fractions'][i],directory)
        try:
            snap=Snapshot(env);initial=signature(env);snap.persist(directory/'start_snapshot.pkl')
            update(client,env,config);one=infer(client,noises['candidates'][:1]);two=infer(client,noises['candidates'][:1])
            noise_error=float(np.max(abs(one-two)))
            if noise_error>tol:raise RuntimeError(f'Fixed-noise inference mismatch: {noise_error}')
            endpoints=[];success=[];errors=[]
            for candidate in (0,1,0,1):
                assert_restored(env,snap,initial,tol);execute(env,pool[candidate])
                current=signature(env);ok=bool(env.eval_success)
                if len(endpoints)<2:endpoints.append(current);success.append(ok)
                else:
                    e=float(np.max(abs(current-endpoints[candidate])));errors.append(e)
                    if e>tol or success[candidate]!=ok:raise RuntimeError(f'Chunk replay mismatch seed={seed} candidate={candidate} error={e}')
            recovery=expert(env)
            assert_restored(env,snap,initial,tol);execute(env,pool[0])
            e=float(np.max(abs(signature(env)-endpoints[0])))
            if e>tol or bool(env.eval_success)!=success[0]:raise RuntimeError(f'Expert contamination seed={seed}: {e}')
            results.append({'seed':seed,'fixed_noise_error':noise_error,'replay_errors':errors,
                'post_expert_replay_error':e,'expert_gate_result':recovery})
            print('GATE',seed,'passed',flush=True)
        finally:safe_close_env(env)
    atomic_json(output/'gate.json',{'passed':True,'states':results,'tolerance':tol,
        'implementation_sha256':sha256(Path(__file__)),
        'protocol_sha256':config['protocol_sha256']})


def audit_seeds(config,output):
    """Inspect historical collection/evaluation records, not arbitrary numerical model files."""
    low,high=config['protocol']['branch_seeds'];conflicts=[];scanned=[]
    for root in config['paths']['seed_audit_roots']:
        for pattern in ('**/episode.json','**/eval_records.jsonl','**/rollout_records.jsonl'):
            for file in Path(root).glob(pattern):
                if output in file.parents:continue
                scanned.append(str(file))
                text=file.read_text(errors='strict')
                records=[json.loads(line) for line in text.splitlines() if line.strip()] if file.suffix=='.jsonl' else [json.loads(text)]
                for record in records:
                    for r in (record,record.get('record',{}),record.get('metadata',{})):
                        seed=r.get('seed',r.get('episode_seed'))
                        if isinstance(seed,int) and low<=seed<=high:conflicts.append({'file':str(file),'seed':seed})
    atomic_json(output/'seed_audit.json',{'scanned':scanned,'conflicts':conflicts,'range':[low,high]})
    if conflicts:raise RuntimeError('New scene seed range conflicts with recorded historical data')


def run(config,output,resume):
    from eval_policy_xpolicylab import build_policy_client,safe_close_env
    torch.set_num_threads(4);output=Path(output);paths=config['paths']
    if paths['policy_url'].endswith(':18300'):raise ValueError('Never use production HIL service')
    audit_seeds(config,output)
    asset_path=output/'assets.json'
    policy_root=Path(paths['policy_checkpoint'])
    if not asset_path.exists():
        assets={'policy':{str(p.relative_to(policy_root)):sha256(p) for p in sorted(policy_root.rglob('*')) if p.is_file()},
                'encoder_sha256':sha256(Path(paths['encoder'])),
                'policy_config_sha256':sha256(Path(paths['policy_config']))}
        atomic_json(asset_path,assets)
    client=build_policy_client({'policy_server_url':paths['policy_url'],'xpolicylab_root':paths['xpolicylab'],
                                'evaluation_id':'coverage-benchmark','trial_id':'paired-branch'})
    try:
        info=client.call(func_name='benchmark_info')
        if info['api']!='benchmark_fixed_noise_v1':raise ValueError('Wrong policy service')
        atomic_json(output/'service.json',info)
        gate=output/'gate.json'
        if gate.exists():
            record=json.loads(gate.read_text())
            if not record['passed'] or record.get('implementation_sha256')!=sha256(Path(__file__)):
                raise ValueError('Gate implementation changed; rerun validation in a fresh output')
        else:
            with contextlib.redirect_stdout(_Discard()):validity_gate(config,client,output)
        print('ALL_REPLAY_GATES_PASSED',flush=True)
        if config.get('_gate_only'):return
        # Fit may still be in progress. Do not start formal execution without frozen calibration.
        fit=Path(paths['fit'])
        if not (fit/'manifest.json').exists():raise RuntimeError('REPLAY_GATE_PASSED_WAITING_FOR_FIT')
        manifest=json.loads((fit/'manifest.json').read_text())
        if not manifest['complete'] or manifest['protocol_sha256']!=config['protocol_sha256']:raise ValueError('Wrong fit')
        for model in manifest['models'].values():
            if sha256(fit/model['file'])!=model['sha256']:raise ValueError('Changed branch scorer')
        frozen_fit=sha256(fit/'manifest.json')
        scorers=load_scorers(fit);cal=json.loads((fit/'calibration.json').read_text())
        encoder=FrozenEncoder('cpu',Path(paths['encoder']))
        primary=config['protocol']['primary_methods'];completed={}
        for si,seed in enumerate(range(200000,200030)):
            directory=output/f'scene_{seed}'
            with contextlib.redirect_stdout(_Discard()):
                env,pool,noises,obs=prepare_scene(config,client,seed,[.2,.5,.8][si%3],directory)
            try:
                snap=Snapshot(env);initial=signature(env);snap.persist(directory/'start_snapshot.pkl')
                images={name:obs['observation'][camera]['rgb'] for name,camera in zip(
                    ('cam_high','cam_left_wrist','cam_right_wrist'),('head_camera','left_camera','right_camera'))}
                state=np.asarray(obs['joint_action']['vector'],np.float32);features=encoder(images)
                scores={n:m.score(features,state,pool) for n,m in scorers.items()}
                ws={n:weights(scores[n],cal[n]) for n in primary}
                schemes={f'candidate_{i}':pool[i] for i in range(8)}
                schemes.update({n:np.tensordot(ws[n],pool,axes=(0,0)) for n in primary});schemes['uniform']=pool.mean(0)
                atomic_npz(directory/'scores_weights.npz',{'features':features,'state':state,
                    **{'score_'+n:s for n,s in scores.items()},**{'weight_'+n:w for n,w in ws.items()},
                    **{'action_'+n:a for n,a in schemes.items()}})
                for name,chunk in schemes.items():
                    dest=directory/(name+'.json')
                    if dest.exists():
                        saved=json.loads(dest.read_text())
                        if saved.get('fit_manifest_sha256')!=frozen_fit or saved.get('candidate_sha256')!=sha256(directory/'candidates.npz'):
                            raise ValueError('Completed branch belongs to different models/candidates')
                        completed[f'{seed}/{name}']=sha256(dest);continue
                    with contextlib.redirect_stdout(_Discard()):
                        assert_restored(env,snap,initial,1e-4);execute(env,chunk)
                        end=Snapshot(env);end_signature=signature(env);end.persist(directory/(name+'_endpoint.pkl'))
                        within=bool(env.eval_success);continuations=[]
                        for rep in range(3):
                            assert_restored(env,end,end_signature,1e-4)
                            client.call(func_name='reset');started=time.monotonic();physics=env._benchmark_physics_steps
                            frames=[] if si<3 and rep==0 else None
                            for noise in noises['continuation'][rep]:
                                if env.eval_success or env.take_action_cnt>=900:break
                                update(client,env,config);execute(env,infer(client,noise)[0],frames)
                            continuation=status(env,started,physics);continuations.append(continuation)
                            if frames:
                                import imageio.v2 as imageio
                                imageio.mimwrite(directory/(name+'_rep0.mp4'),frames,fps=10,macro_block_size=1)
                        assert_restored(env,end,end_signature,1e-4);recovered=expert(env)
                    p=float(np.mean([r['success'] for r in continuations]));r=int(recovered['success'])
                    atomic_json(dest,{'seed':seed,'scheme':name,'chunk_success':within,'p_auto':p,'r_expert':r,
                        'U':(1-p)*r,'continuations':continuations,'expert':recovered,
                        'fit_manifest_sha256':frozen_fit,'candidate_sha256':sha256(directory/'candidates.npz'),
                        'endpoint_sha256':sha256(directory/(name+'_endpoint.pkl'))})
                    completed[f'{seed}/{name}']=sha256(dest)
                    atomic_json(output/'progress.json',{'completed_schemes':len(completed),'total_schemes':390,'last':f'{seed}/{name}'})
                    print('BRANCH',seed,name,'p_auto',p,'r_expert',r,flush=True)
            finally:safe_close_env(env)
        atomic_json(output/'manifest.json',{'complete':len(completed)==390,'completed':completed,
            'protocol_sha256':config['protocol_sha256'],'fit_manifest_sha256':sha256(fit/'manifest.json')})
    finally:client.close()
