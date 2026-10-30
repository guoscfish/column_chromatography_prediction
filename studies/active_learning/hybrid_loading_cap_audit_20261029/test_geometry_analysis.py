"""Analytic checks for frozen geometry, without any historical labels/models."""
import unittest
import numpy as np
import geometry_analysis as g


class GeometryTests(unittest.TestCase):
    def test_sixteen_and_thirty_two_are_supported(self):
        rng=np.random.default_rng(17)
        x=rng.normal(size=(96,12))
        for k in (16,32):
            m,p=g.geometry(x,8,np.arange(k))
            self.assertEqual(len(p['distance_percentile']),k)
            self.assertTrue(0<=m['coverage_distance_relative_drop']<=1)
            self.assertTrue(0<=m['remaining_U_coverage_distance_relative_drop']<=1)

    def test_analytic_coverage_and_mechanical_zeroing(self):
        # L={0}; U={2,4,8}; choose 4. Before={2,4,8}; after={2,0,4}.
        x=np.array([[0.],[2.],[4.],[8.]])
        m,_=g.geometry(x,1,[1])
        self.assertAlmostEqual(m['coverage_distance_relative_drop'],8/14)
        self.assertAlmostEqual(m['remaining_U_coverage_distance_relative_drop'],4/10)
        self.assertAlmostEqual(m['mechanical_selected_zeroing_drop_component'],4/14)
        self.assertAlmostEqual(m['nonselected_drop_component_full_U_denominator'],4/14)
        self.assertAlmostEqual(m['selected_zeroing_share_of_total_gain'],.5)

    def test_same_denominator_marginal_additivity(self):
        rng=np.random.default_rng(44);x=rng.normal(size=(130,10))
        a=np.arange(16);b=np.arange(16,32);union=np.r_[a,b]
        ma,_=g.geometry(x,10,a,common_excluded=union,other_positions=b)
        mb,_=g.geometry(x,10,b,common_excluded=union,other_positions=a)
        mall,_=g.geometry(x,10,union,common_excluded=union)
        for solo,margin,total in [
            ('coverage_distance_relative_drop','marginal_after_other_relative_to_original_coverage','coverage_distance_relative_drop'),
            ('common_remaining_U_coverage_distance_relative_drop','common_remaining_U_marginal_after_other_relative_to_original_coverage','common_remaining_U_coverage_distance_relative_drop')]:
            self.assertAlmostEqual(ma[solo]+mb[margin],mall[total])
            self.assertAlmostEqual(mb[solo]+ma[margin],mall[total])
            self.assertLessEqual(ma[margin],ma[solo]+1e-12)

    def test_rank_and_relative_coverage_scale_invariance(self):
        rng=np.random.default_rng(31);x=rng.normal(size=(100,6))
        a,p=g.geometry(x,10,np.arange(16))
        b,q=g.geometry(7*x+12,10,np.arange(16))
        np.testing.assert_allclose(p['distance_percentile'],q['distance_percentile'])
        for k in ['coverage_distance_relative_drop','remaining_U_coverage_distance_relative_drop','centered_batch_effective_rank']:
            self.assertAlmostEqual(a[k],b[k])

    def test_duplicate_selection_and_out_of_pool_fail(self):
        x=np.arange(20,dtype=float).reshape(-1,1)
        for p in ([1,1],[-1],[20]):
            with self.assertRaises(AssertionError):g.geometry(x,2,p)

    def test_metadata_rejects_wrong_checkpoint_or_order(self):
        md={'contract':{'checkpoint':'frozen','indices':[4,3,2,1]}}
        g.validate_bank_metadata(md,'frozen',[4,3,2,1])
        with self.assertRaises(AssertionError):g.validate_bank_metadata(md,'other',[4,3,2,1])
        with self.assertRaises(AssertionError):g.validate_bank_metadata(md,'frozen',[1,2,3,4])

    def test_average_rank_ties(self):
        x=np.array([[0.],[1.],[-1.],[2.]])
        _,p=g.geometry(x,1,[0,1])
        np.testing.assert_allclose(p['distance_percentile'],[.5,.5])


if __name__=='__main__':unittest.main()
