import unittest
import numpy as np

from scripts.transfer_aware_acquisition import (
    facility_location_select, percentile_rank, target_representativeness,
    transfer_aware_selections, transfer_prediction_shift,
)


class TransferAwareAcquisitionTests(unittest.TestCase):
    def test_transfer_shift_definition(self):
        source=np.array([[1.,10.],[5.,8.]])
        target=np.array([[3.,6.],[1.,16.]])
        expected=.5*(np.abs(target[:,0]-source[:,0])/2.+np.abs(target[:,1]-source[:,1])/4.)
        np.testing.assert_allclose(transfer_prediction_shift(source,target,{"V1":2.,"V2":4.}),expected)

    def test_percentile_rank_average_ties(self):
        np.testing.assert_allclose(percentile_rank(np.array([1.,2.,2.,4.])),[.25,.625,.625,1.])

    def test_representativeness_prefers_dense_region(self):
        dense=np.array([[0.,0.],[.1,0.],[0.,.1],[.1,.1]])
        z=np.vstack([dense,[[10.,10.],[20.,20.]]])
        representativeness,_=target_representativeness(z,k=2)
        self.assertGreater(representativeness[:4].mean(),representativeness[4:].mean())

    def test_facility_location_is_deterministic_and_covers_clusters(self):
        ids=["b","a","d","c"]
        z=np.array([[0.,0.],[0.,.1],[10.,10.],[10.,10.1]])
        first=facility_location_select(ids,z,2)
        self.assertEqual(first,facility_location_select(ids,z,2))
        self.assertEqual(len(first),2)
        self.assertTrue(any(x in first for x in ("a","b")))
        self.assertTrue(any(x in first for x in ("c","d")))

    def test_three_strategy_contract(self):
        ids=[f"s{i:02d}" for i in range(40)]
        shift=np.arange(40.); uncertainty=np.arange(40.)[::-1]
        z=np.column_stack([np.arange(40.),np.zeros(40)])
        selected,audit=transfer_aware_selections(ids,shift,uncertainty,z,10,.25)
        self.assertEqual(set(selected),{"transfer_shift","transfer_shift_uncertainty","transfer_shift_uncertainty_representative"})
        for values in selected.values(): self.assertEqual(len(values),len(set(values))); self.assertEqual(len(values),10)
        self.assertLessEqual(set(selected["transfer_shift_uncertainty_representative"]),set(audit["shortlist_ids"]))


if __name__ == "__main__": unittest.main()
