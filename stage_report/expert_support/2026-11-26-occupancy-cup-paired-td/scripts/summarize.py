"""Summarize paired online results with common time and termination conventions."""
import argparse
import json
import numpy as np
import torch

from run import ROOT, SPEC, sha, write
from score_outcomes import load_encoder
import train_occupancy
from train_occupancy import GAMMA, mlp, load_episodes, rewards

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('native','random','max','min'))
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    train_occupancy.DEVICE=args.device
    torch.set_num_threads(4)
    training=json.loads((ROOT/'training.json').read_text())
    assert sha(ROOT/'d.pt')==training['d_sha256']
    assert sha(SPEC['encoder'])==training['encoder_sha256']
    payload=torch.load(ROOT/'d.pt',weights_only=False)
    model=mlp(1550).to(args.device).eval()
    model.load_state_dict(payload['model'])
    encoder=load_encoder(SPEC['encoder'],'imagenet',args.device)
    runs={}
    for mode in ([args.mode] if args.mode else ('native','random','max','min')):
        rows=[]
        episodes=load_episodes(mode,encoder)
        assert len(episodes)==20 and all(e['scene'] not in sum(training['sft'],[]) for e in episodes)
        for ep,trace in zip(episodes,json.loads((ROOT/mode/'trace/manifest.json').read_text())['episodes']):
            with torch.no_grad():
                x=train_occupancy.state_input(train_occupancy.tensor(ep['features']),train_occupancy.tensor(ep['state']),payload['normalization'])
                raw=torch.nn.functional.softplus(-model(x).flatten()[1:]).cpu().numpy()
            r=np.minimum(raw, training.get('cap', 5.0))
            high=r>training['threshold']
            pose_delta=np.linalg.norm(np.diff(ep['poses'],axis=0),axis=-1)
            gripper_delta=abs(np.diff(ep['state'][:,[6,13]],axis=0))
            geom=[json.loads(line) for line in (ROOT/mode/'trace'/trace['directory']/'geometry.jsonl').read_text().splitlines()]
            assert [g['steps'] for g in geom]==ep['steps'].tolist()
            cup=np.array([g['cup'] for g in geom]); coaster=np.array([g['coaster'] for g in geom])
            horizontal=np.linalg.norm(cup[:,:2]-coaster[:,:2],axis=1)
            vertical=abs(cup[:,2]-coaster[:,2])
            lifted=bool(np.max(cup[:,2]-cup[0,2])>.04)
            near=bool(((horizontal<.035)&(vertical<.015)).any())
            rows.append(dict(scene=ep['scene'],success=ep['success'],steps=int(ep['steps'][-1]),queries=len(r),
                discounted_reward=float(r@GAMMA**np.arange(len(r))), reward_trace=r.tolist(), raw_reward_trace=raw.tolist(),
                reward_per_visited_state=float(r.mean()), low_coverage_fraction=float(high.mean()),
                # Treat time to first crossing as administratively censored at the task budget.
                first_low_coverage_step=int(ep['steps'][1:][high][0]) if high.any() else None,
                step_limit=ep['limit'], eef_idle_fraction=float(np.mean(np.max(pose_delta,axis=1)<.002)),
                gripper_large_change_fraction=float(np.mean(np.max(gripper_delta,axis=1)>.5)),
                lifted=lifted, reached_target_geometry=near, terminal_xy_error=float(horizontal[-1]),
                failure_mode=None if ep['success'] else 'no_lift' if not lifted else
                    'lifted_without_target_alignment' if not near else 'aligned_without_success'))
        result=dict(episodes=rows,successes=sum(e['success'] for e in rows),
            first_crossing_count=sum(e['first_low_coverage_step'] is not None for e in rows))
        for key in ('discounted_reward','reward_per_visited_state','low_coverage_fraction','steps',
                    'eef_idle_fraction','gripper_large_change_fraction','lifted','reached_target_geometry','terminal_xy_error'):
            result[key]=float(np.mean([r[key] for r in rows]))
        observed=[e['first_low_coverage_step'] for e in rows if e['first_low_coverage_step'] is not None]
        result['first_crossing_median_among_crossers']=float(np.median(observed)) if observed else None
        if mode!='native':
            decisions=[json.loads(line) for line in (ROOT/mode/'decisions.jsonl').read_text().splitlines()]
            assert len(decisions)==sum(e['queries'] for e in rows)
            assert all(len(d['scores'])==32 for d in decisions)
            if mode in ('max','min'):
                order=np.argmax if mode=='max' else np.argmin
                assert all(d['index']==int(order(d['scores'])) for d in decisions)
            result['candidate_diagnostics']={k:float(np.mean([r[k] for r in decisions])) for k in
                ('range','pair_gap','ess_over_k','joint_rms_mrad','gripper_rms')}
        runs[mode]=result
    if args.mode:
        write(ROOT/args.mode/'metrics.json',runs[args.mode])
        return
    assert all([e['scene'] for e in r['episodes']]==[e['scene'] for e in runs['native']['episodes']] for r in runs.values())
    # Compare equal-length prefixes within each scene the four methods share.
    common_lengths=[min(result['episodes'][i]['queries'] for result in runs.values()) for i in range(20)]
    for result in runs.values():
        for row,n in zip(result['episodes'],common_lengths):
            row['matched_prefix_mean_reward']=float(np.mean(row['reward_trace'][:n]))
        result['matched_prefix_mean_reward']=float(np.mean(
            [row['matched_prefix_mean_reward'] for row in result['episodes']]))
    rng=np.random.default_rng(42)
    samples=rng.integers(20,size=(20000,20))
    comparisons={}
    for mode, reference in [('max','random'),('min','random'),('max','min'),('random','native')]:
        comparisons[f'{mode}_minus_{reference}']={}
        for key in ('discounted_reward','reward_per_visited_state','low_coverage_fraction','matched_prefix_mean_reward','success'):
            diff=np.array([float(a[key])-float(b[key]) for a,b in zip(runs[mode]['episodes'],runs[reference]['episodes'])])
            comparisons[f'{mode}_minus_{reference}'][key]=dict(mean=float(diff.mean()),
                ci95=np.quantile(diff[samples].mean(1),[.025,.975]).tolist())
    write(ROOT/'online.json',dict(runs=runs,comparisons=comparisons,gamma=GAMMA,
        threshold=training['threshold'],termination='all rewards following the task terminal are zero',
        q_sha256=training['q_sha256'],d_sha256=training['d_sha256']))

if __name__=='__main__':
    main()
