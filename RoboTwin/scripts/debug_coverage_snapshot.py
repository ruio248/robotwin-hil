#!/usr/bin/env python3
"""Read the failed gate's immutable prefix/candidates and localize expert pollution.

This is a diagnostic, not a scored experiment. It cannot launch formal branches.
"""
import argparse
import hashlib
import pickle
from pathlib import Path
import contextlib
import time
import numpy as np
from coverage_benchmark.common import load_config,atomic_json,atomic_npz
from coverage_benchmark.branch import setup,execute,replay_prefix,Snapshot,signature,expert
from coverage_sampling.robotwin import _Discard
import coverage_benchmark.branch as branch_module


def fields(obj):
    result={}
    for key,value in vars(obj).items():
        if isinstance(value,(str,int,float,bool,type(None),list,dict,tuple,np.ndarray)):
            try:result[key]=hashlib.sha256(pickle.dumps(value)).hexdigest()
            except Exception:pass
    return result


def signature_labels(env):
    """Use exactly the ordering of branch.signature; identify error units."""
    labels=[]
    def add(name,value):
        labels.extend(f'{name}[{i}]' for i in range(np.asarray(value).size))
    for i,actor in enumerate(env.scene.get_all_actors()):
        name=f'actor:{i}:{actor.name}';pose=actor.get_pose()
        add(name+'.position_m',pose.p);add(name+'.quaternion',pose.q)
        for component in getattr(actor,'components',()):
            if hasattr(component,'linear_velocity'):
                add(name+'.linear_velocity_m_s',component.linear_velocity)
                add(name+'.angular_velocity_rad_s',component.angular_velocity)
    for i,art in enumerate(env.scene.get_all_articulations()):
        name=f'articulation:{i}';pose=art.get_root_pose()
        for field,value in [('position_m',pose.p),('quaternion',pose.q),('qpos',art.get_qpos()),('qvel',art.get_qvel())]:add(name+'.'+field,value)
        if hasattr(art,'get_root_linear_velocity'):
            add(name+'.linear_velocity',art.get_root_linear_velocity());add(name+'.angular_velocity',art.get_root_angular_velocity())
    for i,art in enumerate(env.scene.get_all_articulations()):
        for j,joint in enumerate(art.get_active_joints()):
            add(f'drive:{i}:{j}.position',joint.get_drive_target());add(f'drive:{i}:{j}.velocity',joint.get_drive_velocity_target())
    assert len(labels)==len(signature(env))
    return labels


def main():
    from eval_policy_xpolicylab import safe_close_env
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--tag',default=time.strftime('%Y%m%d_%H%M%S'))
    args=p.parse_args();config=load_config(args.config,'ubuntu');path=args.scene
    if not args.tag.replace('_','').replace('-','').isalnum():raise ValueError('Invalid diagnostic tag')
    report=path/f'snapshot_debug_{args.tag}.json';traces_file=path/f'snapshot_debug_{args.tag}_traces.npz'
    if report.exists() or traces_file.exists():raise FileExistsError('Preserve existing diagnostic evidence; choose a new tag')
    import json
    record=json.loads((path/'state.json').read_text())
    with np.load(path/'baseline.npz',allow_pickle=False) as d:base=d['chunks']
    with np.load(path/'candidates.npz',allow_pickle=False) as d:candidates=d['chunks']
    results=[]
    with contextlib.redirect_stdout(_Discard()):
        env=setup(config,record['seed'],path)
        try:
            replay_prefix(env,base,record['baseline_decision']);snap=Snapshot(env);initial=signature(env);labels=signature_labels(env)
            before={key:fields(obj) for key,obj in [('env',env),('robot',env.robot),('left_mplib',env.robot.left_mplib_planner),('right_mplib',env.robot.right_mplib_planner)]}
            def trial(label):
                snap.restore();restored=signature(env);traces=[];steps=[]
                for a in candidates[0]:
                    execute(env,a[None]);traces.append(signature(env));steps.append(env._benchmark_physics_steps)
                results.append({'label':label,'restore_error':float(max(abs(restored-initial))),'steps':steps})
                return np.stack(traces)
            first=trial('before_expert');second=trial('repeat_before_expert')
            recovery=expert(env)
            post_expert={key:fields(obj) for key,obj in [('env',env),('robot',env.robot),('left_mplib',env.robot.left_mplib_planner),('right_mplib',env.robot.right_mplib_planner)]}
            after=trial('after_expert')
            restored_fields={key:fields(obj) for key,obj in [('env',env),('robot',env.robot),('left_mplib',env.robot.left_mplib_planner),('right_mplib',env.robot.right_mplib_planner)]}
            atomic_npz(traces_file,{'first':first,'second':second,'after_expert':after})
            atomic_json(report,{'trials':results,'recovery':recovery,
                'branch_sha256':hashlib.sha256(Path(branch_module.__file__).read_bytes()).hexdigest(),
                'diagnostic_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'sleep_states':[sleeping for _,sleeping in snap.sleep_states],
                'signature_labels':labels,
                'max_error_fields':[labels[i] for i in np.argmax(abs(after-first),axis=1)],
                'repeat_errors':np.max(abs(second-first),axis=1).tolist(),
                'post_expert_errors':np.max(abs(after-first),axis=1).tolist(),
                'max_error_indices':np.argmax(abs(after-first),axis=1).tolist(),
                'changed_fields_after_expert':{k:[f for f in post_expert[k] if post_expert[k][f]!=before[k].get(f)] for k in before},
                'changed_fields_after_replay':{k:[f for f in restored_fields[k] if restored_fields[k][f]!=before[k].get(f)] for k in before}})
        finally:safe_close_env(env)
    print('DIAGNOSTIC_SAVED',report)


if __name__=='__main__':main()
