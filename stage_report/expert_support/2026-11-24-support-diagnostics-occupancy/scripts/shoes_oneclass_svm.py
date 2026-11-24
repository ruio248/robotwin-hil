"""Expert-only support estimation with episode-disjoint validation."""
import json
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from train import REPO,load_cohort,base,digest
from train_click_rank import distance,I,J

ROOT=REPO/'outputs/support_three_tasks_20261123'
DEST=ROOT/'shoes_oneclass_svm'
def main():
    rows,test,source=load_cohort(ROOT/'matched_sft_data','place_dual_shoes')
    entries=json.loads((ROOT/'matched_sft_data/place_dual_shoes/manifest.json').read_text())['episodes']
    for group,split in [(rows,'train'),(test,'test')]:
        for row,entry in zip(group,[e for e in entries if e['split']==split]):
            with np.load(ROOT/'shoes_dinov2_10k'/(entry['id']+'.npz')) as z:
                assert str(z['source'])==row['sha256']
                row['features']=z['features'].reshape(len(row['state']),-1)
    split=json.loads((ROOT/'shoes_dinov2_split40_100k/split.json').read_text())
    lookup={r['episode']:r for r in rows}
    train=[lookup[r['episode']] for r in split['train']]
    validation=[lookup[r['episode']] for r in split['validation']]
    norm={k:v.numpy() for k,v in base.normalization(train,list(range(40))).items()}
    gaps=np.concatenate([distance(r,norm['action_scale'])[:,J]-distance(r,norm['action_scale'])[:,I] for r in train])
    delta=float(np.quantile(abs(gaps),.25))
    reference=json.loads((ROOT/'shoes_dinov2_split40_100k/report.json').read_text())
    assert delta==reference['delta']
    pca=PCA(n_components=32,whiten=True,svd_solver='full')
    pca.fit(np.concatenate([r['features'] for r in train]))
    def inputs(row):
        n=len(row['state'])
        visual=pca.transform(row['features'])/np.sqrt(32)
        state=(row['state']-norm['state_mean'])/norm['state_scale']/np.sqrt(14)
        context=np.concatenate([visual,state],axis=1)
        actions=np.concatenate([row['expert'][:,None],row['policy']],axis=1)
        actions=(actions-norm['action_mean'])/norm['action_scale']
        return np.repeat(context,5,axis=0),actions.reshape(n*5,140)/np.sqrt(140)
    prepared={r['episode']:inputs(r) for r in train+validation+test}
    def design(row,weight):
        context,actions=prepared[row['episode']]
        return np.concatenate([context,weight*actions],axis=1)
    def evaluate(model,group,weight):
        records=[]
        for row in group:
            scores=model.decision_function(design(row,weight)).reshape(-1,5)
            expert,policy=scores[:,0],scores[:,1:]
            d=distance(row,norm['action_scale'])
            gaps=d[:,J]-d[:,I]
            mask=abs(gaps)>delta
            margins=((policy[:,I]-policy[:,J])*np.sign(gaps))[mask]
            records.append(dict(episode=row['episode'],anchors=len(expert),
                expert_wins=float(((expert[:,None]>policy)+.5*(expert[:,None]==policy)).sum()),
                expert_accepted=int((expert>=0).sum()),
                pairs=int(mask.sum()),correct=float((margins>0).sum()+.5*(margins==0).sum()),
                candidate_pair_gap=float(abs(policy[:,I]-policy[:,J]).mean())))
        n=sum(r['anchors'] for r in records)
        return dict(expert_rank=sum(r['expert_wins'] for r in records)/(4*n),
            expert_acceptance=sum(r['expert_accepted'] for r in records)/n,
            candidate_rank=sum(r['correct'] for r in records)/sum(r['pairs'] for r in records),
            candidate_pair_gap=sum(r['candidate_pair_gap']*r['anchors'] for r in records)/n,
            episodes=records)
    DEST.mkdir(exist_ok=True)
    grid=[]
    best=None
    for weight in (0.3,1.,3.,10.):
        x=np.concatenate([design(r,weight)[::5] for r in train])
        # Median expert-pair distance gives a training-only kernel scale.
        rng=np.random.default_rng(42)
        left=rng.integers(len(x),size=20000);right=rng.integers(len(x),size=20000)
        median=float(np.median(np.sum((x[left]-x[right])**2,axis=1)))
        for multiplier in (0.1,1.,10.):
            for nu in (0.01,0.05,0.1):
                gamma=multiplier/median
                model=OneClassSVM(kernel='rbf',gamma=gamma,nu=nu,tol=1e-5)
                model.fit(x)
                metrics=evaluate(model,validation,weight)
                record=dict(action_weight=weight,gamma_multiplier=multiplier,gamma=gamma,nu=nu,
                    validation={k:v for k,v in metrics.items() if k!='episodes'})
                grid.append(record)
                if best is None or metrics['expert_rank']>best['validation']['expert_rank']:
                    best=record
                    joblib.dump(dict(model=model,pca=pca,normalization=norm,config=record),DEST/'model.joblib')
                print(record,flush=True)
    artifact=joblib.load(DEST/'model.joblib')
    report=dict(source=source,split=split,encoder='frozen DINOv2 CLS',pca_components=32,
        selection='maximum validation expert rank, grid order breaks ties',
        delta=delta,best=best,grid=grid,checkpoint_sha256=digest(DEST/'model.joblib'))
    for name,group in [('train',train),('validation',validation),('test20',test)]:
        report[name]=evaluate(artifact['model'],group,best['action_weight'])
    base.atomic_json(DEST/'report.json',report)
    print('RESULT', {s:{k:v for k,v in report[s].items() if k!='episodes'} for s in ['train','validation','test20']},flush=True)
if __name__=='__main__':
    main()
