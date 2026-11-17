import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_learnability import classify_pair,pair_numbers

def record(volume=50,pe='10/1',solvent='PE',loading=200):
    return dict(smiles='CCO',conditions={'PE/EA':pe,'Density g/ml':.9,'V/ul':volume,
        'loading solvent':solvent,'Volume of loading solvent/ul':loading})

class LearnabilityTests(unittest.TestCase):
    def test_same_molecule_single_axis_and_orientation(self):
        meta={'low':record(50),'high':record(150),'multi':record(150,pe='5/1')}
        match=classify_pair('high','low',meta)
        self.assertEqual(match[:3],(2,'low','high'))
        self.assertIsNone(classify_pair('low','multi',meta))
    def test_equivalent_ratio_and_EA_direction(self):
        meta={'a':record(pe='10/2'),'b':record(pe='5/1'),'c':record(pe='1/1')}
        self.assertIsNone(classify_pair('a','b',meta))
        self.assertEqual(classify_pair('c','a',meta)[:3],(0,'a','c'))
    def test_near_zero_observed_not_direction_evidence(self):
        x=pair_numbers([0,.01,1,-2],[100,.01,.01,1],threshold=.5,flat=.1)
        self.assertEqual(x['n_meaningful'],2)
        self.assertAlmostEqual(x['flat_fraction_meaningful'],.5)
        self.assertAlmostEqual(x['wrong_direction_nonflat_fraction_meaningful'],.5)
        self.assertAlmostEqual(x['direction_accuracy_meaningful'],.5)
        self.assertTrue(all(np.isfinite(v) for v in x.values()))
    def test_flat_baseline_excess_error(self):
        x=pair_numbers([2,-3],[0,0])
        self.assertAlmostEqual(x['excess_difference_error_over_flat'],0)
        self.assertAlmostEqual(x['flat_fraction_meaningful'],1)
    def test_compression_and_wrong_direction_are_distinct(self):
        flat=pair_numbers([2],[.01]);wrong=pair_numbers([2],[-1])
        self.assertEqual(flat['direction_accuracy_meaningful'],1)
        self.assertEqual(flat['direction_correct_nonflat_fraction_meaningful'],0)
        self.assertEqual(wrong['wrong_direction_nonflat_fraction_meaningful'],1)

if __name__=='__main__':unittest.main()
