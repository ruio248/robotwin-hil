import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from coverage_benchmark.common import action_metrics, calibrate, weights, paired_bootstrap, validate_chunks, verify_fit
from coverage_benchmark.scorers import normalization, obs_features, act_features, NeuralScorer, SupportScorer
from tail_value.chunk import ChunkCoverageCritic
from tail_value.model import atomic_checkpoint
from tail_value.cache import atomic_npz, atomic_json, sha256


class TestBenchmark(unittest.TestCase):
    def test_frozen_fit_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            atomic_npz(root/'validation.npz',{'state':np.zeros((2,14))})
            for name in ('models','split','calibration'):atomic_json(root/(name+'.json'),{})
            manifest={'complete':True,'protocol_sha256':'test','models':{},
                **{name+'_sha256':sha256(root/(name+'.json')) for name in ('split','calibration')},
                'validation_sha256':sha256(root/'validation.npz')}
            atomic_json(root/'manifest.json',manifest)
            self.assertTrue(verify_fit(root,'test')['complete'])
            atomic_json(root/'calibration.json',{'changed':True})
            with self.assertRaisesRegex(ValueError,'calibration.json'):verify_fit(root,'test')
    def test_full_chunk(self):
        a=np.zeros((4,10,14)); a[1,-1,0]=1
        m=action_metrics(a)
        self.assertEqual(m['first_joint_rms_mean'],0)
        self.assertGreater(m['last_joint_rms_mean'],0)
        self.assertAlmostEqual(m['joint_f_max'],1)
        self.assertAlmostEqual(m['joint_rms_max'],1/np.sqrt(120))
        a[:]=0;a[1,:,6]=1
        self.assertEqual(action_metrics(a)['joint_f_max'],0)
        with self.assertRaises(ValueError):validate_chunks(a[:,0])
    def test_calibration_and_permutation(self):
        p=np.arange(40).reshape(10,4)*.2;e=np.arange(10)*.2
        c=calibrate(e,p)
        w=weights(p[0],c); order=[2,0,3,1]
        np.testing.assert_allclose(weights(p[0,order],c),w[order])
        self.assertAlmostEqual(c['achieved_ess_fraction'],.5,places=8)
        self.assertGreater(w[0],w[-1])
        np.testing.assert_allclose(weights(np.ones(4),calibrate(np.ones(3),np.ones((3,4)))),.25)
    def test_global_scale_balanced(self):
        c=calibrate([0,0],np.full((2,4),2.))
        self.assertEqual(c['mean'],1);self.assertEqual(c['scale'],1)
        self.assertTrue(c['degenerate'])
    def test_statistics_train_only(self):
        train={'state':np.ones((2,14)),'expert_chunks':np.ones((2,10,14))}
        n=normalization(train)
        self.assertTrue(np.all(n['action_scale']==.05))
        self.assertEqual(obs_features(np.ones((2,1536)),train['state'],n).shape,(2,1550))
        np.testing.assert_equal(act_features(train['expert_chunks'],n),0)
    def test_scene_bootstrap(self):
        r=paired_bootstrap(np.ones(30));self.assertEqual(r['scenes'],30);self.assertEqual(r['ci95'],[1.,1.])
    def test_neural_order_and_tail_action(self):
        torch.manual_seed(42)
        n={k:np.zeros(14) if 'mean' in k else np.ones(14) for k in ('state_mean','state_scale','action_mean','action_scale')}
        model=ChunkCoverageCritic(n,10,256)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt'
            atomic_checkpoint(path,{'representation':'absolute_joint_chunk','config':{'horizon':10,'width':256},'normalization':n,'model':model.state_dict()})
            m=NeuralScorer(path); a=np.zeros((4,10,14));a[1,-1,0]=1
            b=m.score(np.zeros(1536),np.zeros(14),a)
            self.assertNotEqual(b[0],b[1])
            np.testing.assert_allclose(m.score(np.zeros(1536),np.zeros(14),a[[3,1,0,2]]),b[[3,1,0,2]],atol=1e-7)
    def test_svm_export_parity(self):
        from sklearn.svm import OneClassSVM
        rng=np.random.default_rng(42)
        n={k:np.zeros(14) if 'mean' in k else np.ones(14) for k in ('state_mean','state_scale','action_mean','action_scale')}
        f=rng.normal(size=(8,1536));s=rng.normal(size=(8,14));a=rng.normal(size=(8,10,14))
        x=np.concatenate([obs_features(f,s,n),act_features(a,n)],axis=1)/np.sqrt(2)
        svm=OneClassSVM(nu=.05,gamma=.3).fit(x)
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'svm.npz'
            atomic_npz(file,{**n,'svm_gamma':np.array(.3),'svm_support':svm.support_vectors_,
                            'svm_dual':svm.dual_coef_,'svm_intercept':svm.intercept_})
            scorer=SupportScorer(file,'svm')
            for i in range(8):self.assertAlmostEqual(scorer.score(f[i],s[i],a[i:i+1])[0],svm.decision_function(x[i:i+1])[0],places=10)
    def test_episode_split_and_last_anchor(self):
        from coverage_benchmark.data import sft_data
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);entries=[]
            for i in range(47):
                a={'features':np.zeros((3,1536)),'state':np.full((3,14),i),
                   'expert_chunks':np.full((3,10,14),i),'policy_chunks':np.zeros((3,4,10,14)),
                   'frame_index':np.array([0,10,20]),'valid_transition':np.array([True,True,False])}
                file=root/f'{i:03d}.npz';atomic_npz(file,a)
                entries.append({'file':file.name,'sha256':sha256(file),'split':'train' if i<2 else 'val'})
            atomic_json(root/'manifest.json',{'schema_version':1,'complete':True,
                'config':{'representation':'absolute_joint_chunk','horizon':10,'num_candidates':4},'episodes':entries})
            data,split=sft_data(root,expected=False)
            self.assertEqual(len(data['state']),94)
            self.assertEqual(sum(data['split']=='train'),4)
            self.assertEqual(sum(data['split']=='cal'),30)
            self.assertFalse(set(split['calibration_episodes'])&set(split['diagnostic_episodes']))
            self.assertFalse(np.any(data['frame_index']==20))


if __name__=='__main__':unittest.main()
