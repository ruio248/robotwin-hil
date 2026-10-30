from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .common import atomic_json, paired_bootstrap, KS, sha256
from .offline import correlation


def validation_report(config,root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[json.loads(p.read_text()) for p in sorted((root/'scores').glob('validation_*.json'))]
    if len(files)!=30:raise ValueError('Need exactly 30 diagnostic episodes')
    methods=config['protocol']['primary_methods'];result={}
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for method,ax in zip(methods,axes.flat):
        e=np.array([r['expert_scores'][method] for f in files for r in f['rows']])
        p=np.array([r['scores'][method] for f in files for r in f['rows']])
        gap=e-p.mean(1);ranges=np.ptp(p,axis=1)
        stats=[r['metrics'][method]['4'] for f in files for r in f['rows']]
        result[method]={'observations':len(e),'episodes':30,
            'expert_mean':float(e.mean()),'policy_mean':float(p.mean()),'expert_minus_policy_mean':float(gap.mean()),
            'within_observation_range_quantiles':np.quantile(ranges,[0,.25,.5,.75,1]).tolist(),
            'normalized_range_median':float(np.median([m['b_range_normalized'] for m in stats])),
            'ess_fraction_median':float(np.median([m['ess_fraction'] for m in stats])),
            'tilted_joint_rms_median':float(np.median([m['tilted_joint_rms'] for m in stats]))}
        edges=np.histogram_bin_edges(np.concatenate([e,p.reshape(-1)]),bins=45)
        ax.hist(e,bins=edges,density=True,histtype='step',label='Expert chunks')
        ax.hist(p.reshape(-1),bins=edges,density=True,histtype='step',label='Policy chunks')
        ax.set(title=method,xlabel='Score (method-specific scale)',ylabel='Density');ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle('30 internal diagnostic episodes; score distributions are not coverage ground truth')
    for ext in ('png','pdf'):fig.savefig(output/f'internal_validation_distribution.{ext}',dpi=180)
    plt.close(fig)
    atomic_json(output/'internal_validation_metrics.json',result)
    return result


def plots_offline(config,root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest=json.loads((root/'manifest.json').read_text());files=[]
    if not manifest.get('complete'):raise ValueError('Full offline report requires all cohorts')
    validation_report(config,root,output)
    for name,digest in manifest['completed'].items():
        path=root/'scores'/(name+'.json')
        if sha256(path)!=digest:raise ValueError('Changed offline score result')
        files.append(json.loads(path.read_text()))
    pre=[f for f in files if f['metadata']['cohort']=='hil_pre_k32']
    methods=config['protocol']['primary_methods']+['observation','td_s42_1000','td_s42_20000']
    calibration=json.loads((Path(config['paths']['fit'])/'calibration.json').read_text())
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(output/(name+'.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
    # Every observation's max-min is calculated BEFORE episode/bin aggregation.
    curves={}
    metrics=['joint_rms_mean','joint_rms_max','joint_f_mean','joint_f_max','b_range',
             'b_range_normalized','tilted_joint_rms','ess_fraction','max_weight']
    for method in methods:
        curves[method]={}
        for metric in metrics:
            fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained');series={}
            for k in KS:
                bins=defaultdict(list)
                for ep in pre:
                    local=defaultdict(list)
                    for row in ep['rows']:
                        idx=min(19,int(row['progress']*20))
                        local[idx].append(row['metrics'][method][str(k)][metric])
                    for idx,values in local.items():bins[idx].append(float(np.mean(values)))
                points=[{'progress':(b+.5)*.05,'episodes':len(values),'mean':float(np.mean(values)),
                         'median':float(np.median(values)),'q25':float(np.quantile(values,.25)),
                         'q75':float(np.quantile(values,.75))} for b,values in sorted(bins.items())]
                series[str(k)]=points
                ax.plot([r['progress']*100 for r in points],[r['mean'] for r in points],label=f'K={k}')
            curves[method][metric]=series
            ax.set(xlabel='Policy progress before first takeover (%)',ylabel=metric,
                   title=f'{method}: full-chunk {metric}')
            ax.legend();ax.grid(alpha=.2);save(fig,method+'_'+metric)
    # Score stability is compared within each observation, never across states.
    stability={};relations={};diagnostics={}
    for method in methods:
        rs=[];mean_shift=[];gaps=[];ess=[]
        for ep in pre:
            for row in ep['rows']:
                m=row['metrics'][method]['32'];rs.append(m['action_gap_spearman']);mean_shift.append(m['tilted_joint_rms'])
                gaps.append(m['b_range_normalized']);ess.append(m['ess_fraction'])
        vals=[r for r in rs if r is not None]
        relations[method]={'spearman_quantiles':np.quantile(vals,[0,.25,.5,.75,1]).tolist() if vals else None,
            'defined_observations':len(vals),'degenerate_observations':len(rs)-len(vals),
            'normalized_gap_median':float(np.median(gaps)),'tilted_joint_rms_median':float(np.median(mean_shift)),
            'ess_fraction_median':float(np.median(ess)),
            'permuted_tilted_joint_rms_median':float(np.median([
                r['metrics'][method]['32']['permuted_tilted_joint_rms'] for ep in pre for r in ep['rows']])),
            'calibration':calibration[method]}
    for family in ('td_1000','td_10000','td_20000','disc'):
        names=[f'td_s{s}_{family.split("_")[1]}' if family.startswith('td') else f'disc_s{s}' for s in (42,43,44)]
        result=[]
        for ep in pre:
            for row in ep['rows']:
                for i,j in ((0,1),(0,2),(1,2)):
                    a,b=map(lambda n:np.asarray(row['scores'][n]),(names[i],names[j]))
                    result.append({'pair':[names[i],names[j]],'rho':correlation(a,b),
                                   'same_min':bool(np.argmin(a)==np.argmin(b)),
                                   'tied':bool(np.sum(a==a.min())>1 or np.sum(b==b.min())>1)})
        defined=[x['rho'] for x in result if x['rho'] is not None]
        stability[family]={'rows':result,'argmin_agreement':float(np.mean([x['same_min'] for x in result])),
            'median_rank_correlation':float(np.median(defined)) if defined else None}
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for method in methods:
        rows=[r['metrics'][method]['32'] for ep in pre for r in ep['rows']]
        valid=[r['action_gap_spearman'] for r in rows if r['action_gap_spearman'] is not None]
        if valid:axes[0].hist(valid,bins=np.linspace(-1,1,31),histtype='step',density=True,label=method)
        axes[1].scatter([r['b_range_normalized'] for r in rows],[r['tilted_joint_rms'] for r in rows],s=5,alpha=.3,label=method)
    axes[0].set(xlabel='Within-observation Spearman: action distance vs score gap',ylabel='Density')
    axes[1].set(xlabel='Normalized candidate score range',ylabel='Tilted - uniform chunk joint RMS (rad)',xscale='symlog')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'action_score_and_mean_shift')
    hil=[f for f in files if f['metadata']['cohort']=='hil_full']
    historical=[f for f in files if f['metadata']['cohort']=='historical_rollout']
    # A compact primary-method comparison; all methods use the same observations.
    fig,axes=plt.subplots(4,3,figsize=(13,13),layout='constrained')
    for row,method in zip(axes,config['protocol']['primary_methods']):
        for ax,metric,label in zip(row,['b_range_normalized','ess_fraction','tilted_joint_rms'],
                ['Score range / calibration SD','ESS / K','Weighted - uniform joint RMS (mrad)']):
            for k in KS:
                points=curves[method][metric][str(k)]
                scale=1000 if metric=='tilted_joint_rms' else 1
                ax.plot([p['progress']*100 for p in points],[p['mean']*scale for p in points],label=f'K={k}')
            ax.set(title=method,ylabel=label,xlabel='Pre-takeover policy progress (%)');ax.grid(alpha=.2)
            if metric=='ess_fraction':ax.set_ylim(0,1.05)
            if metric!='ess_fraction':ax.set_yscale('symlog',linthresh=.001)
            ax.legend(fontsize=7)
    fig.suptitle('50 saved HIL episodes: episode-balanced bin means\nTemperatures frozen; calibration ESS is NOT matched across all methods')
    save(fig,'primary_methods_offline_comparison')
    for method in methods:
        before=min(e['metadata']['takeover'] for e in hil)
        after=min(len(e['rows'])-e['metadata']['takeover'] for e in hil)
        x=np.arange(-before,after)
        matrix=np.array([[np.mean(ep['rows'][t+ep['metadata']['takeover']]['scores'][method]) for t in x] for ep in hil])
        fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
        ax.plot(x,matrix.mean(0),label='Mean across 50 saved HIL trajectories')
        ax.fill_between(x,*np.quantile(matrix,[.25,.75],axis=0),alpha=.2,label='Between-trajectory 25–75%')
        ax.axvline(0,color='gray',linestyle='--',label='First human-control saved frame')
        ax.set(title=method,xlabel='Saved frame relative to first takeover (not seconds)',ylabel='Mean score of 4 full policy suggestions')
        ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,method+'_hil_aligned')
        diagnostics[method]={'hil_frames':x.tolist(),'hil_mean':matrix.mean(0).tolist()}
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');groups={}
        prefix=min(len(ep['rows']) for ep in historical)
        for outcome,color in ((True,'#2166ac'),(False,'#d95f0e')):
            rows=[np.array([np.mean(r['scores'][method]) for r in ep['rows']]) for ep in historical if ep['metadata']['success']==outcome]
            progress=np.linspace(0,1,101);arr=np.stack([np.interp(progress,np.linspace(0,1,len(r)),r) for r in rows])
            pref=np.stack([r[:prefix] for r in rows]);groups[str(outcome)]={'n':len(rows),'prefix_mean':pref.mean(0).tolist()}
            for ax,xx,yy in ((axes[0],progress,arr),(axes[1],np.arange(prefix),pref)):
                ax.plot(xx,yy.mean(0),color=color,label=f'{"Success" if outcome else "Failure"} n={len(rows)}')
                ax.fill_between(xx,*np.quantile(yy,[.25,.75],axis=0),color=color,alpha=.18)
        axes[0].set(xlabel='Normalized saved-trajectory progress',ylabel='Mean candidate score',title=method)
        axes[1].set(xlabel='Common-prefix saved frame (not seconds)',title=f'All 100 trajectories; prefix={prefix}')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        save(fig,method+'_success_failure');diagnostics[method]['historical']=groups
    atomic_json(output/'offline_metrics.json',{'progress_curves':curves,'relations':relations,
        'stability':stability,'trajectory_diagnostics':diagnostics,
        'limitations':['Internal diagnostic and selected historical cohorts, not coverage ground truth',
            'Larger K naturally increases sample extrema; not evidence of a wider underlying policy',
            'Beta capped at 100; ESS concentration not fully matched between methods',
            'SFT metadata 15 Hz and online configuration 30 Hz do not define fixed chunk physical duration'],
        'counts':{'pre_hil_episodes':len(pre),'pre_hil_observations':sum(len(ep['rows']) for ep in pre),
            'full_hil_episodes':len(hil),'full_hil_observations':sum(len(ep['rows']) for ep in hil),
            'historical_episodes':len(historical),'historical_observations':sum(len(ep['rows']) for ep in historical)}})
    return relations


def analyze_branches(config,root,output):
    manifest=json.loads((root/'manifest.json').read_text())
    if not manifest['complete'] or len(manifest['completed'])!=390:raise ValueError('Need all 390 paired schemes')
    methods=config['protocol']['primary_methods'];data=defaultdict(dict);ranking=defaultdict(list)
    for key,digest in manifest['completed'].items():
        seed,name=key.split('/');file=root/f'scene_{seed}'/(name+'.json')
        if sha256(file)!=digest:raise ValueError('Branch result hash changed')
        r=json.loads(file.read_text());data[int(seed)][name]=r
    if len(data)!=30:raise ValueError('Need 30 independent scene seeds')
    comparisons={}
    for method in methods:
        comparisons[method]={metric:paired_bootstrap([d[method][metric]-d['uniform'][metric] for d in data.values()])
                             for metric in ('p_auto','r_expert','U')}
        comparisons[method]['by_stratum']={str(f):{metric:paired_bootstrap([
            d[method][metric]-d['uniform'][metric] for seed,d in data.items() if [.2,.5,.8][(seed-200000)%3]==f])
            for metric in ('p_auto','r_expert','U')} for f in (.2,.5,.8)}
    per_scene_rank=[]
    for seed,d in sorted(data.items()):
        with np.load(root/f'scene_{seed}'/'scores_weights.npz',allow_pickle=False) as archive:record=dict(archive)
        for name,scores in record.items():
            if not name.startswith('score_'):continue
            method=name.removeprefix('score_');rows=[d[f'candidate_{i}'] for i in range(8)]
            # Tied minima/maxima use their expected outcome, not arbitrary first-index selection.
            low=np.flatnonzero(np.isclose(scores,scores.min(),atol=1e-12,rtol=0))
            high=np.flatnonzero(np.isclose(scores,scores.max(),atol=1e-12,rtol=0))
            r={'seed':seed,'method':method}
            for metric in ('p_auto','r_expert','U'):
                y=np.array([v[metric] for v in rows]);random_value=float(y.mean())
                r[metric]={'low':float(y[low].mean()),'high':float(y[high].mean()),'uniform_candidate':random_value,
                           'low_minus_random':float(y[low].mean()-random_value),
                           'high_minus_random':float(y[high].mean()-random_value)}
            per_scene_rank.append(r)
    for method in sorted({r['method'] for r in per_scene_rank}):
        ranking[method]={metric:paired_bootstrap([r[metric]['low_minus_random'] for r in per_scene_rank if r['method']==method])
                         for metric in ('p_auto','r_expert','U')}
    result={'weighted_mean_vs_uniform_mean':comparisons,'low_candidate_vs_uniform_candidate':dict(ranking),
            'per_scene_rank':per_scene_rank,'scene_unit':30,'bootstrap_repeats':10000}
    atomic_json(output/'branch_metrics.json',result)
    return result


def run(config,output,resume):
    output=Path(output);root=output.parent;relations=None;branch=None
    if (root/'offline'/'manifest.json').exists():relations=plots_offline(config,root/'offline',output)
    if (root/'branch'/'manifest.json').exists():branch=analyze_branches(config,root/'branch',output)
    # Partial reports explicitly distinguish real results from unfinished comparisons.
    atomic_json(output/'manifest.json',{'offline_complete':relations is not None,'branch_complete':branch is not None,
        'complete':relations is not None and branch is not None,'protocol_sha256':config['protocol_sha256']})
