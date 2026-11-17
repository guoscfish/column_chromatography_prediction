import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metrics import hamming,continuous_distance,rarity,coverage_gain,random_batches,percentile,aulc,align_gain,conditional_logit

class MetricTests(unittest.TestCase):
    def test_discrete_and_continuous(self):
        self.assertAlmostEqual(hamming([[0,1]],[[1,1]])[0,0],.5)
        self.assertAlmostEqual(continuous_distance([[0,1]],[[.1,1]],[1,2])[0,0],.05)
        self.assertGreater(continuous_distance([[0]],[[.9]],[1])[0,0],continuous_distance([[0]],[[.1]],[1])[0,0])
    def test_rarity_smoothing(self):
        r=rarity(np.array([[0],[1],[2]]),[0,1],[3])
        np.testing.assert_allclose(r,[-np.log(2/5),-np.log(2/5),-np.log(1/5)])
        gap=rarity(np.array([[0],[7]]),[0],[2])
        self.assertAlmostEqual(gap[1],-np.log(1/3))
        normalized=rarity(np.array([[0],[1],[2]]),[0,1],[3],normalized=True)
        np.testing.assert_allclose(normalized,r/np.log(5))
    def test_coverage_monotonicity(self):
        b=np.array([1.,2.]);d=np.array([[.5],[3.]])
        g=coverage_gain(b,d)
        self.assertGreaterEqual(g,0)
        self.assertGreaterEqual(coverage_gain(b,np.column_stack([d,[.1,.1]])),g)
    def test_random_baseline(self):
        x=random_batches([1,2,3,4],500,2,23)
        np.testing.assert_array_equal(x,random_batches([1,2,3,4],500,2,23))
        self.assertTrue(all(len(set(v))==2 for v in x))
        self.assertTrue(set(x.ravel())<={1,2,3,4})
        self.assertAlmostEqual(percentile(2,[1,2,3]),.5)
        self.assertLess(abs(x.mean()-2.5),.1)
    def test_horizon_and_nonmonotonic_loss(self):
        self.assertAlmostEqual(aulc([333,365,397],[1,.2,.5]),.475)
        self.assertAlmostEqual(align_gain({333:.2,365:.5},333,365),-.3)
        with self.assertRaises(ValueError):aulc([333,397],[1,.2])
        with self.assertRaises(ValueError):align_gain({333:1,397:.2},333,365)
    def test_logit_extreme_stability(self):
        result=conditional_logit([np.array([[1000.],[0.],[-1000.]])],1.)
        self.assertTrue(result.success)
        self.assertTrue(np.isfinite(result.fun))
        self.assertGreater(result.x[0],0)

if __name__=='__main__':unittest.main()
