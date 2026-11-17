import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,unittest,tempfile
import numpy as np
import pandas as pd
import torch
OUT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(OUT))
from isolated_models import build_control,VARIANTS,load_control,checkpoint
from prepare_data import load_seed,assert_identity
from qgeognn_al.training.predictor import seed_everything,loader_pair,target_loss
from qgeognn_al.schemas.conditions import ConditionNormalization
from qgeognn_al.evaluation.point import point_metrics

class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.d,atom,angle,_=load_seed(157)
        cls.a,cls.b=next(iter(zip(*loader_pair(atom,angle,cls.d['check_indices'][:3],2048))))
        cls.meta=pd.read_csv(OUT/'data/seed_157/metadata.csv')
        cls.norm=ConditionNormalization(**cls.d['normalization'])
    def models(self):
        result={}
        for k in VARIANTS:
            seed_everything(self.d['initialization_seed']);result[k]=build_control(k,self.norm,30,150).eval()
        return result
    def test_zero_skip_equivalence_and_parameter_count(self):
        m=self.models()
        with torch.no_grad():p={k:v(self.a,self.b) for k,v in m.items()}
        self.assertTrue(torch.equal(p['A_original'],p['B_amount_skip']))
        self.assertTrue(torch.equal(p['A_original'],p['C_volume_skip']))
        self.assertEqual(sum(v.numel() for v in m['B_amount_skip'].parameters()),sum(v.numel() for v in m['C_volume_skip'].parameters()))
        self.assertEqual(sum(v.numel() for v in m['A_original'].parameters()),458952)
    def test_volume_units_and_identity(self):
        assert_identity(self.a,self.d['ordered_ids'],self.meta['V/ul'])
        m=self.models()['C_volume_skip']
        expected=(self.a.sample_volume_ul.reshape(-1)-30)/120
        self.assertTrue(torch.equal(m.skip_input(self.a),expected))
        bad=self.a.clone();bad.sample_id=list(reversed(bad.sample_id))
        with self.assertRaises(ValueError):assert_identity(bad,self.d['ordered_ids'],self.meta['V/ul'])
    def test_amount_uses_original_typed_path(self):
        m=self.models()['B_amount_skip']
        self.assertTrue(torch.equal(m.skip_input(self.a),m.condition_branch.typed_inputs(self.a)[:,2]))
    def test_skip_shared_across_endpoint_quantiles(self):
        for kind in ['B_amount_skip','C_volume_skip']:
            m=self.models()[kind]
            with torch.no_grad():
                m.head[0].weight.zero_();m.head[0].bias.fill_(5.);m.skip_weight.copy_(torch.tensor([2.,-3.]))
                pred=m(self.a,self.b);x=m.skip_input(self.a)
                expected=torch.relu(5+x[:,None]*torch.tensor([2.,2.,2.,-3.,-3.,-3.]))
            self.assertTrue(torch.equal(pred,expected))
            self.assertTrue(torch.equal(pred[:,0],pred[:,2]));self.assertTrue(torch.equal(pred[:,3],pred[:,5]))
            self.assertTrue((pred>=0).all())
    def test_holdout_disjoint_and_not_validation(self):
        self.assertFalse(set(self.d['train_ids'])&set(self.d['check_ids']))
        self.assertFalse(set(self.d['validation_ids'])&set(self.d['history_ids']))
    def test_metrics_use_frozen_combined_loss_direction(self):
        y=np.array([[1.,2.],[3.,6.]])
        p=np.repeat(y,3,axis=1);p[:,1]+=.5;p[:,4]+=1
        r=point_metrics(y,p,{'V1':1.,'V2':2.})
        self.assertAlmostEqual(r['combined_normalized_rmse'],.5)
    def test_old_checkpoint_not_accepted_by_control_loader(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.pt';torch.save({'model_variant':'qgeognn_v2'},p)
            with self.assertRaises(ValueError):load_control(p)
    def test_new_checkpoint_variant_binding_is_strict(self):
        m=self.models()['B_amount_skip']
        saved=checkpoint(m,'B_amount_skip',self.norm,self.d['preprocessing'],self.d['training_config'],self.d['volume_scale'],{})
        saved['model_variant']=VARIANTS['C_volume_skip']
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'wrong.pt';torch.save(saved,p)
            with self.assertRaises(ValueError):load_control(p)

if __name__=='__main__':unittest.main()
