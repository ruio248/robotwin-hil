"""Summarize paired click BCE and distance-ranking experiment outputs."""
import json
import numpy as np
from run_online20 import OUT,write

from train_click_rank import experiment_task
task=experiment_task()
def load(path):
    return json.loads(path.read_text())
old=OUT/'matched_sft_models'/task
rank=OUT/'rank_distance_v1'/task
online=OUT/task/'online_k32_20'
new=online.with_name(online.name+'_rank_distance')
r=load(rank/'report.json')
pairs={k:sum(row[k] for row in r['paired_diagnostics']) for k in r['paired_diagnostics'][0] if k!='scene'}
baseline=load(online/'summary.json')['scene_results']
changed=load(new/'summary.json')['scene_results']
assert [(x['scene_seed'],x['baseline_success']) for x in baseline]==[(x['scene_seed'],x['baseline_success']) for x in changed]
def compare(before,after):
    diff=np.array(after,dtype=int)-np.array(before,dtype=int)
    return dict(before=sum(before),after=sum(after),drops=[s['scene_seed'] for s,d in zip(changed,diff) if d<0],
                gains=[s['scene_seed'] for s,d in zip(changed,diff) if d>0],
                change_ci=np.quantile(diff[np.random.default_rng(42).integers(len(diff),size=(10000,len(diff)))].mean(1),[.025,.975]).tolist())
after=[x['min_support_success'] for x in changed]
dec=[json.loads(line) for line in (new/'decisions.jsonl').read_text().splitlines()]
assert all(len(x['bce_scores'])==len(x['scores'])==32 and np.argmin(x['scores'])==x['selected'] for x in dec)
result=dict(delta=r['delta'],offline=pairs,online_queries=len(dec),
            online_switches=sum(int(np.argmin(x['bce_scores'])!=x['selected']) for x in dec),
            vs_bce=compare([x['min_support_success'] for x in baseline],after),
            vs_sft=compare([x['baseline_success'] for x in changed],after))
for name,path in [('bce',old),('rank',rank)]:
    test=load(path/'report.json')['test20']
    outcome=load(path/'sft_outcomes.json')
    result[name]={k:float(np.mean([x[k] for x in test])) for k in ['candidate_pair_gap_mean','expert_rank','ess_over_k_beta1']}
    result[name].update({k:outcome[k] for k in ['success_score_mean','failure_score_mean','success_auc']})
write(rank/'ablation_summary.json',result)
print(json.dumps(result,indent=2))
