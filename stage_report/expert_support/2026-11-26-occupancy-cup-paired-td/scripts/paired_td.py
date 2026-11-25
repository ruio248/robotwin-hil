"""Rescore genuine H10 transitions with the frozen paired occupancy D."""
import json
import shutil
import numpy as np
import torch
import train_occupancy as t
from run import ROOT, REPO, SPEC, write, sha
from score_outcomes import encode_views, load_encoder

old = REPO / 'outputs/occupancy_cup_20261125'
paired = REPO / 'outputs/occupancy_cup_paired_20261125'
assert ROOT not in (old, paired)
ROOT.mkdir(exist_ok=True)
for name in ('collect', 'native'):
    path = ROOT / name
    if not path.exists():
        path.symlink_to(old / name)

torch.set_num_threads(4)
t.base.seed_all(1042)
encoder = load_encoder(SPEC['encoder'], 'imagenet', t.DEVICE)
shutil.copyfile(paired/'C_paired/d.pt', ROOT/'d.pt')
checkpoint = torch.load(ROOT/'d.pt', weights_only=False)
d = t.mlp(1550).to(t.DEVICE).eval()
d.load_state_dict(checkpoint['model'])
norm = checkpoint['normalization']
split = json.loads((old/'training.json').read_text())['sft']
assert split == json.loads((paired/'C_paired/report.json').read_text())['split'][1]
audit = []
trace = ROOT/'collect/trace'
for ep in json.loads((trace/'manifest.json').read_text())['episodes']:
    seed = ep['scene_seed']
    with np.load(trace/ep['directory']/ep['queries'][0]['file']) as query, np.load(paired/f'sft_{seed}.npz') as match:
        a = np.stack([query['rgb_'+c] for c in t.CAMERAS])
        b = match['rgb'][0]
        x = np.concatenate([query['state_'+c] for c in t.STATE])
        y = match['state'][0]
        def logit(rgb, state):
            feat = encode_views(encoder, [[img] for img in rgb], t.DEVICE)
            with torch.no_grad():
                return float(d(t.state_input(t.tensor(feat),t.tensor(state[None]),norm)).item())
        audit.append(dict(seed=seed,rgb_max=int(np.abs(a.astype(int)-b).max()),state_max=float(abs(x-y).max()),query_logit=logit(a,x),paired_logit=logit(b,y)))
write(ROOT/'input_audit.json', audit)
assert all(a['rgb_max']==0 and a['state_max']==0 and abs(a['query_logit']-a['paired_logit'])<1e-5 for a in audit), audit
episodes = {ep['scene']:ep for ep in t.load_episodes('collect',encoder)}
groups = [[episodes[i] for i in ids] for ids in split]
profiles = []
for ep in episodes.values():
    with torch.no_grad():
        logits = d(t.state_input(t.tensor(ep['features']),t.tensor(ep['state']),norm)).flatten()[1:]
        raw = torch.nn.functional.softplus(-logits).cpu().numpy()
    profiles.append(dict(seed=ep['scene'],raw=raw.tolist(),clipped=raw.clip(max=5).tolist()))
write(ROOT/'query_rewards.json',profiles)
expert_rewards = []
for seed in split[0]:
    with np.load(paired/f'expert_{seed}.npz') as z:
        imgs=z['rgb'][:-1]
        feat=encode_views(encoder,[list(imgs[:,i]) for i in range(3)],t.DEVICE)
        expert_rewards.extend(t.rewards(d,dict(features=feat,state=z['state'][:-1]),norm))
train,val=[t.transition_batch(g,d,norm) for g in groups]
report=dict(sft=split,encoder_sha256=sha(SPEC['encoder']),d_sha256=sha(ROOT/'d.pt'),gamma=t.GAMMA,cap=t.CAP,
            q_train_transitions=len(train['reward']),q_validation_transitions=len(val['reward']),threshold=float(np.quantile(expert_rewards,.9)))
for key,action in [('q',True),('v',False)]:
    report[key+'_curve']=t.train_value(train,val,norm,action)
    report[key+'_sha256']=sha(ROOT/(key+'.pt'))
write(ROOT/'training.json',report)
