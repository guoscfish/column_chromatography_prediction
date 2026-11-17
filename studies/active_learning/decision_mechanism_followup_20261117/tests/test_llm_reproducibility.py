"""Check transparent adjustment and frozen-packet comparison helpers."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('llm_rep',Path(__file__).resolve().parents[1]/'llm_reproducibility.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class ReproducibilityTests(unittest.TestCase):
    def test_strip_only_cw(self):
        packet={'packet_sha256':'a','candidates':{'columns':['id','cw_distance_percentile','q50'],'rows':[['x',.7,3]]},
                'observed':{'columns':['id'],'rows':[['h']]},'pending':{'columns':[],'rows':[]},'memory':[]}
        stripped=M.strip_cw_packet(packet)
        self.assertEqual(stripped['candidates'],{'columns':['id','q50'],'rows':[['x',3]]})
        self.assertEqual(packet['candidates']['rows'][0],['x',.7,3])

    def test_group_projection_orthogonal(self):
        rng=np.random.default_rng(17);groups=np.repeat(np.arange(5),20);x=rng.normal(size=(100,3))
        y=groups*.1+x@np.array([.2,-.4,.1])+rng.normal(size=100)*.01
        residual,r2,_=M.residualize(y,x,groups)
        self.assertGreater(r2,.99)
        for group in np.unique(groups):self.assertAlmostEqual(float(residual[groups==group].mean()),0,places=12)
        np.testing.assert_allclose(x.T@residual,0,atol=1e-10)

    def test_response_projection_known_scale(self):
        x=np.arange(100)[:,None]/100;y=.1+2*x[:,0]
        residual,r2,_=M.residualize(y,x)
        np.testing.assert_allclose(residual,0,atol=1e-12);self.assertAlmostEqual(r2,1)

    def test_matching_excludes_selected(self):
        score=np.array([100.,200.,1.,2.,3.,4.,5.,6.]);mask=np.array([1,1,0,0,0,0,0,0],bool)
        difference,n=M.matched_excess(score,mask,[0]*8,minimum=5)
        self.assertEqual(n,2);self.assertAlmostEqual(difference,150-3.5)

    def test_matching_missing_support(self):
        difference,n=M.matched_excess([1,2,3],[1,0,0],[0,1,1])
        self.assertEqual(n,0);self.assertTrue(np.isnan(difference))

    def test_single_factor_type(self):
        def row(pe,volume):return {'smiles':'CC','conditions':{'PE/EA':pe,'Density g/ml':1.,'V/ul':volume,'loading solvent':'PE','Volume of loading solvent/ul':200}}
        metadata={'h':row('10/1',50),'a':row('10/1',100),'b':row('20/1',100)}
        support,pairs=M.factor_allocation(['a','b'],['h'],metadata)
        self.assertEqual(support['V/ul'],.5);self.assertEqual(len(pairs['PE/EA']),1)
        self.assertEqual(len(pairs['V/ul']),0)


if __name__=='__main__':unittest.main()
