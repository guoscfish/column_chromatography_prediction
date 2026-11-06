import unittest
import numpy as np
from diagnostics import strict_pairs, fixed_candidate_positions, fusion_features


class DiagnosticsTests(unittest.TestCase):
    def test_exact_pairs_preserve_fraction_and_precision(self):
        def row(i,v,pe='2/1',density=1):
            return {'id':str(i),'smiles':'CCO','conditions':{'PE/EA':pe,'Density g/ml':density,
                'V/ul':v,'loading solvent':'EA','Volume of loading solvent/ul':200}}
        data=[row(0,'34.01360544'),row(1,'68.02721088','4/2'),row(2,'68.02721088','4/2',2)]
        pairs=dict(strict_pairs(data))
        self.assertEqual(pairs['sample_volume'].tolist(),[[0,1]])
        self.assertEqual(len(pairs['pe']),0)

    def test_candidate_order_independent_of_targets(self):
        ids=[f'x{i}' for i in range(50)]
        a=fixed_candidate_positions(ids,12,157)
        rev=list(reversed(ids))
        b=fixed_candidate_positions(rev,12,157)
        self.assertEqual([ids[i] for i in a],[rev[i] for i in b])

    def test_fusion_never_fits_evaluation_rows(self):
        rng=np.random.default_rng(1)
        g=rng.normal(size=(90,128));l=rng.normal(size=(90,150))
        a,meta=fusion_features(g,l,70)
        g[70:]+=1000;l[70:]-=300
        b,_=fusion_features(g,l,70)
        np.testing.assert_allclose(a[:70],b[:70],atol=1e-10)
        self.assertEqual(a.shape,(90,128))
        self.assertEqual(meta['projection_parameters'],8256)


if __name__=='__main__':unittest.main()
