"""Artifact-level tests of exposure guards, matched identity and curve support."""
from pathlib import Path
import json,unittest
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1]

class MatchedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=pd.read_csv(OUT/'matched_response_pairs.csv')
        cls.meta=pd.read_csv(OUT/'source_metadata.csv').set_index('sample_id')
        cls.roles=pd.read_csv(OUT/'formal_split_roles.csv')
        cls.quarantine=set(json.loads((OUT/'LABEL_BOUNDARY_INCIDENT.json').read_text())['quarantined_ids'])
    def test_no_quarantined_or_own_test_endpoint_is_scored(self):
        self.assertFalse((set(self.p.low_id)|set(self.p.high_id))&self.quarantine)
        for seed,g in self.p.groupby('formal_seed'):
            roles=self.roles[self.roles.row_seed==seed].set_index('sample_id').source_role
            self.assertTrue(roles.loc[list(set(g.low_id)|set(g.high_id))].isin(['train','validation']).all())
    def test_all_four_methods_share_identical_pair_and_truth(self):
        for _,g in self.p.groupby(['formal_seed','al_seed','pair_key','target']):
            self.assertEqual(set(g.method),{'Fully_trained','AL_L525','AL_L1005','AL_L1485'})
            self.assertEqual(len(g),4);self.assertEqual(g.delta_true.nunique(),1)
    def test_strict_single_factor_conditions(self):
        fields=['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']
        for row in self.p[['axis','low_id','high_id']].drop_duplicates().itertuples(index=False):
            a,b=self.meta.loc[row.low_id],self.meta.loc[row.high_id]
            self.assertEqual(a.canonical_smiles,b.canonical_smiles)
            changed=[k for k in fields if a[k]!=b[k]]
            self.assertEqual(changed,[row.axis])
    def test_curves_support_is_not_truth_error_scoring(self):
        p=pd.read_csv(OUT/'controlled_volume_curves.csv')
        self.assertEqual(set(p.volume_ul),{30,60,90,120,150})
        self.assertTrue((p.locally_supported==((p.volume_ul>=p.local_min_ul)&(p.volume_ul<=p.local_max_ul))).all())
        self.assertNotIn('delta_true',p.columns)
        pool=json.loads((OUT/'CURVE_POOL.json').read_text())['cases']
        self.assertEqual(len(pool),15)
        self.assertFalse({sid for c in pool for sid in c['observed_ids']}&self.quarantine)

if __name__=='__main__':unittest.main()
