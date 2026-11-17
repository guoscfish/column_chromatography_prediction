"""v2 CRN regression and independent BALD/streaming oracle checks."""
from pathlib import Path
import sys

import numpy as np
import pytest

V2=Path(__file__).resolve().parents[2]/"studies/active_learning/epig_prediction_oriented_v2_20261117"
sys.path.insert(0,str(V2))
sys.path.insert(0,str(V2.parents[2]/"src"))
from acquisition_v2 import (coupled_reference_positions,GaussianStateWithBALD,
    IndependentStreamingCache,select_streaming_batch,batch_target_information)
from math_oracle import posterior,oracle_scores


def test_common_random_numbers_fix_distribution_salted_order():
    import hashlib
    ids=[f"r{i}" for i in range(70)]
    old=lambda dist: sorted(range(70),key=lambda i:hashlib.sha256(f"epig_v1:157:{dist}:{ids[i]}".encode()).digest())
    assert old("P1") != old("P2")  # reproduces original confounding even identical population
    full=coupled_reference_positions(157,np.arange(70),ids)
    shared=coupled_reference_positions(157,np.arange(10,70),ids)
    assert shared.tolist()==[i for i in full if i>=10]
    assert coupled_reference_positions(157,np.arange(70),ids,16).tolist()==full[:16].tolist()
    assert coupled_reference_positions(157,np.arange(69,-1,-1),ids).tolist()==full.tolist()


@pytest.mark.parametrize("q",[1,2,3])
def test_generic_bald_matches_dense_and_is_reference_independent(q):
    p=posterior(q=q)
    a=GaussianStateWithBALD(p,np.arange(2,11),[0,1])
    b=GaussianStateWithBALD(p,np.arange(2,11),[3,8,10])
    for selected in ([],[4],[4,6]):
        if selected:
            index=selected[-1]-2;a.condition(index);b.condition(index)
        expected=oracle_scores(p,np.arange(2,11),[0,1],selected,"bald")
        np.testing.assert_allclose(a.scores("bald"),expected,atol=1e-10)
        np.testing.assert_allclose(a.scores("bald"),b.scores("bald"),atol=1e-10)


@pytest.mark.parametrize("objective",["epig","ivr","bald"])
@pytest.mark.parametrize("q",[1,2,3])
def test_streaming_matches_dense_conditioning_and_batch_chain(tmp_path,objective,q):
    p=posterior(q=q,independent=True)
    cache=IndependentStreamingCache(p,np.arange(2,11),tmp_path,target_chunk=2,candidate_chunk=3)
    result=select_streaming_batch(cache,np.arange(11),objective=objective,batch_size=5)
    chosen=[]
    for trace in result["trace"]:
        expected=oracle_scores(p,np.arange(2,11),np.arange(11),chosen,objective)
        assert trace["candidate_index"]==int(np.argmax(expected))
        assert trace["conditional_score"]==pytest.approx(max(expected),abs=1e-9)
        chosen.append(trace["row_position"])
    assert cache.max_covariance_shape==[2,3]
    info=batch_target_information(p,np.arange(11),chosen,chunk=2)
    if objective=="epig":
        assert sum(t["conditional_score"] for t in result["trace"])==pytest.approx(info,abs=1e-10)


def test_dependent_outputs_rejected_and_invalid_candidates(tmp_path):
    with pytest.raises(ValueError,match="independent outputs"):
        IndependentStreamingCache(posterior(),np.arange(2,11),tmp_path)
    with pytest.raises(ValueError):
        IndependentStreamingCache(posterior(independent=True),[2,2],tmp_path)


def test_streaming_candidate_order_preserves_ids(tmp_path):
    p=posterior(independent=True)
    candidates=np.array([9,3,7,2,4,6,8,5,10])
    cache=IndependentStreamingCache(p,candidates,tmp_path,target_chunk=3,candidate_chunk=2)
    r=select_streaming_batch(cache,[0,1,4,10],batch_size=4)
    chosen=[]
    for trace in r["trace"]:
        expected=oracle_scores(p,candidates,[0,1,4,10],chosen,"epig")
        assert trace["row_position"]==candidates[np.argmax(expected)]
        chosen.append(trace["row_position"])
