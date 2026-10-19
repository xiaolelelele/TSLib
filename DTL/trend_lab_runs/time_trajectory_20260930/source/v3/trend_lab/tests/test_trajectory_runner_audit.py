"""Independent integration review; tiny deterministic fits are mocked.

These tests check orchestration, validation selection, persisted reload and
test access order. They do not claim real training efficacy or GPU parity.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from trend_lab.data import Split
from trend_lab import trajectory_run as runner


class TrajectoryRunnerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)

    def setUp(self):
        rng=np.random.default_rng(62)
        self.x=rng.normal(size=(4,4,1))
        self.b=rng.normal(size=(4,2,1))
        self.mu=rng.normal(size=(4,2,1))
        p=rng.uniform(size=(4,2,1,3));self.p=p/p.sum(-1,keepdims=True)
        self.prior=np.array([[.4,.2,.4]])
        self.meta={'lookback':4,'horizon':2,'columns':['value'],'mean':[0.],'std':[1.],
                   'train_min':[-4.],'train_max':[4.],
                   'spec':{'name':'audit_tiny','grid_mode':'continuous'},
                   'nested':{'optional':None,'bool':True,'list':[1,2.5,'text']}}
        self.cfg={'lambda_grid':[0,.1,1],'alpha_grid':[0,.1],'device':'cpu',
                  'reference_root':'/home/lgc/TSLib','batch':4,'lr':.001,'epochs':1,'patience':1}

    def test_lambda_integer_endpoint_keys_survive_test_regeneration(self):
        validation,_,_=runner.decisions(self.x,self.b,self.mu,self.p,self.prior,self.meta,self.cfg)
        families=['posterior','top1_gap','mean_is_base','training_prior','greedy']
        for endpoint in (0,1):
            choices={family:next(k for k in validation if k.startswith(family+'_') and float(k.rsplit('_',1)[1])==endpoint)
                     for family in families}
            test,_,_=runner.decisions(self.x,self.b,self.mu,self.p,self.prior,self.meta,self.cfg,validation=False,choices=choices)
            for key in choices.values():
                self.assertIn(key,test)
                np.testing.assert_allclose(test[key],validation[key],atol=0,rtol=0)
        fallback,_,_=runner.decisions(self.x,self.b,self.mu,self.p,self.prior,self.meta,self.cfg,
                                      validation=False,choices={family:None for family in families})
        self.assertEqual(set(fallback),{'mean_only_grid','mean_raw'})

    def test_selector_rejects_direction_only_or_mse_only_improvement(self):
        x=np.zeros((1,4,1));y=np.array([[[2.],[3.]]]);base=np.array([[[-1.],[1.]]])
        candidates={'joint':np.array([[[1.],[2.]]]),
                    'direction_bad_mse':np.array([[[10.],[20.]]]),
                    'same_direction':base.copy()}
        selected,diagnostic=runner.choose(x,y,candidates,base)
        self.assertEqual(selected,'joint')
        self.assertEqual([r['eligible'] for r in diagnostic['candidates']],[True,False,False])
        fallback,_=runner.choose(x,y,{'base':base},base)
        self.assertIsNone(fallback)

    def test_native_bounds_rounding_preserve_exact_first_zero(self):
        meta={'mean':[12.345],'std':[7.25],'spec':{'grid_mode':'native'}}
        zero=(-np.asarray(meta['mean'])/np.asarray(meta['std']))[None,None,:]
        anchor=np.repeat(zero,4,axis=1);truth=np.repeat(zero,2,axis=1)
        bounded=runner.bounded(np.full((1,2,1),-100.),meta)
        np.testing.assert_array_equal(bounded,truth)
        score=runner.score(anchor,truth,bounded)
        self.assertEqual(score['raw_accuracy'],1.)
        self.assertEqual(score['recall'][1],1.)
        np.testing.assert_allclose(runner.rounded(truth,meta),truth,atol=0,rtol=0)

    def test_mock_fit_end_to_end_lock_order_and_disk_reload(self):
        rng=np.random.default_rng(651)
        splits={}
        for i,name in enumerate(('train','earlystop','calibration','validation','test')):
            x=rng.normal(size=(4,4,1));y=rng.normal(size=(4,2,1))
            splits[name]=Split(x,y,np.arange(4)+i*10)
        tensors={k:(torch.tensor(v.x,dtype=torch.float32),torch.tensor(v.y,dtype=torch.float32)) for k,v in splits.items()}
        fits=[]
        def mocked_fit(name,model,loss,monitor,splits,cfg,seed):
            model.eval()
            value=loss(*splits['train']);point=monitor(*splits['earlystop'])
            self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(point))
            fits.append(name)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            return {'method':name,'best_epoch':0,'executed_epochs':0,'earlystop_loss':float(point),'total':sum(p.numel() for p in model.parameters())},state
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);shared_folder=root/'shared';case=root/'case'
            original_score=runner.score
            access=[]
            def watched_score(x,y,z):
                if y is splits['test'].y:
                    self.assertTrue((case/'validation_lock.json').exists())
                    access.append('test_score')
                return original_score(x,y,z)
            with patch.object(runner,'fit_model',mocked_fit),patch.object(runner,'score',watched_score):
                shared=runner.neural_shared(splits,tensors,self.meta,self.cfg,shared_folder,2024)
                self.assertFalse(any(k.startswith('test__') for k in shared))
                self.assertFalse(access)
                runner.run_case(splits,tensors,self.meta,self.cfg,case,2024,'dlinear',shared,shared_folder)
            self.assertEqual(len(fits),10)
            self.assertTrue(access)
            state=torch.load(case/'checkpoint.pt',weights_only=True,map_location='cpu')
            self.assertEqual(set(state['states']),{'mse_base','cosdir','tdalign','raw_head','mean_estimator','residual_capacity','residual_direction_capacity'})
            # Mock fitting never updates weights: these persisted states are
            # actual inputs presented to the trainer, checking common starts.
            for left,right in (('mse_base','cosdir'),('mse_base','tdalign'),
                               ('residual_capacity','residual_direction_capacity')):
                self.assertEqual(set(state['states'][left]),set(state['states'][right]))
                for key in state['states'][left]:
                    torch.testing.assert_close(state['states'][left][key],state['states'][right][key],atol=0,rtol=0)
            self.assertEqual(state['spec']['metadata']['nested'],self.meta['nested'])
            result=json.loads((case/'result.json').read_text())
            self.assertEqual(len(result['reload_checks']),7)
            self.assertTrue(all(r['max_abs_error']<=2e-5 for r in result['reload_checks']))
            shared_result=json.loads((shared_folder/'result.json').read_text())
            self.assertEqual(len(shared_result['reload_checks']),3)
            self.assertEqual(len(shared_result['test_reload_checks']),3)
            self.assertEqual(Path(shared_result['test_authorized_by_lock']),case/'validation_lock.json')


if __name__=='__main__':unittest.main()
