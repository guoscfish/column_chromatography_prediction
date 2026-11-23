"""Regression tests for matching, physical conversion and information boundaries."""
import copy
import numpy as np
import pandas as pd
import pytest

from run import KEYS, design, fraction, join_predictions, label_frame, physical, stable, verified_feedback
from al_models import require_validation_ids


def synthetic():
    base = dict(zip(KEYS, ['CC','10-00-0','Silica 4g',.98,1.,.1,50.,200.,10.,'PE']))
    return pd.DataFrame([{**base,'sample_id': sid,'V/ul': volume,'V1': value,'V2': value+2,'C':value+1,'W':2.,**changes}
                         for sid, volume, value, changes in [('a',50,1,{}),('b',50,3,{}),('c',100,5,{}),
                                                              ('d',150,6,{}),('e',175,20,{'Purity':.97}),
                                                              ('f',200,30,{'CAS':'20-00-0'})]])


def test_exact_match_replicates_and_curvature():
    pairs, levels, curvature = design(synthetic(),{})
    assert len(pairs) == 2 and len(levels) == 3 and len(curvature) == 1
    assert pairs[0]['low_ids'] == 'a|b' and pairs[0]['high_ids'] == 'c'
    assert pairs[0]['d_V1'] == 3 and pairs[0]['delta_mass_mg'] == 50
    assert curvature[0]['chord_V1'] == 1


def test_physical_units():
    assert physical(847,10) == pytest.approx(7.058333333333334)
    assert fraction('40/1') == fraction('80/2') == 1/41
    np.testing.assert_allclose(physical([1200,2400],[10,12]),[10,24])


def test_mixed_prediction_state_and_role_are_rejected():
    frame = synthetic().assign(checkpoint=['x','x','y','x','x','x'])
    for target in ('V1','V2','C','W'):
        frame[f'p_{target}'] = frame[target]
    with pytest.raises(ValueError,match='mixed checkpoints'):
        design(frame,{})
    truth = frame.set_index('sample_id')[['V1','V2']]
    with pytest.raises(ValueError,match='single-state'):
        join_predictions(truth,frame.set_index('sample_id')[['V1','V2','checkpoint']])
    partition = pd.DataFrame({'sample_id':['a','b','c'],'role':['validation','u0','test']})
    assert list(require_validation_ids(partition, [], ['b'])) == ['a']
    with pytest.raises(ValueError,match='validation role'):
        require_validation_ids(partition,['a'],['b'])


def test_receipt_rejects_unrevealed_or_rebound_feedback():
    ids = [f'id{index:02}' for index in range(32)]
    metadata = dict(selected_ids=ids, unlabeled_ids=ids, labeled_ids=[], source_sha256='abc')
    records = [dict(candidate_id=name,pred_V1_ml=0.,pred_V2_ml=1.) for name in ids]
    frozen = dict(new_batch_labels_revealed=False,premeasurement_predictions=records)
    feedback = dict(batch_freeze_sha256='abc',records=copy.deepcopy(records))
    audit = dict(roles='u0',acquisitions_frozen=True,requested_rows=32,requested_ids_hash=stable(ids))
    receipt = dict(audit=audit,batch_freeze_sha256='abc',feedback_sha256=stable(feedback))
    assert list(verified_feedback(metadata,frozen,feedback,receipt).index) == ids
    feedback['records'][0]['true_V1_ml'] = 13
    with pytest.raises(ValueError,match='invalid accepted-label receipt'):
        verified_feedback(metadata,frozen,feedback,receipt)
    with pytest.raises(ValueError,match='invalid accepted-label receipt'):
        verified_feedback({**metadata,'unlabeled_ids':ids[1:]},frozen,feedback,receipt)


def test_label_projection_serializes_only_whitelisted_id(tmp_path, monkeypatch):
    import run

    local = tmp_path/'roles.csv'
    local.write_text('sample_id,V1\nvalidation-id,1\nfuture-id,9\ntest-id,12\n')
    monkeypatch.setattr(run,'track',lambda path:path)
    assert list(label_frame(local,pd.Series(['validation-id']),open)) == [{'sample_id':'validation-id','V1':'1'}]
