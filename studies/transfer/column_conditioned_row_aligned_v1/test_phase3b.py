"""Standard metric and score-boundary tests with arbitrary synthetic responses only."""
import inspect
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_phase3b as b


def test_standard_metric_and_tail_maxima():
    truth=np.array([[1.,3.],[3.,7.],[5.,11.]])
    pred=np.zeros((3,6));pred[:,[1,4]]=truth+np.array([[1.,2.],[-1.,-2.],[2.,3.]])
    scales={'V1':2.,'V2':4.}
    values,tails=b.metrics(truth,pred,scales,[3.,7.],[3.,7.],['a','b','c'])
    assert values['combined_normalized_rmse']==b.old.point_metrics(truth,pred,scales)['combined_normalized_rmse']
    assert values['combined_normalized_rmse']==pytest.approx(np.sqrt(.5*(2./4+(17./3)/16)))
    assert values['Center_rmse']==pytest.approx(np.sqrt((1.5**2+1.5**2+2.5**2)/3))
    assert values['Width_rmse']==pytest.approx(1.)
    assert tails[0]['tail_count']==2 and tails[0]['tail_sse']==5
    assert values['V1_max_abs_error']==2 and values['V1_max_error_sample_id']=='c'


def test_empty_tail_is_nan_not_zero():
    truth=np.array([[1.,3.],[2.,5.]])
    pred=np.ones((2,6))
    _,tails=b.metrics(truth,pred,{'V1':1.,'V2':2.},[99.,99.],[2.,3.],['a','b'])
    assert all(x['tail_count']==0 and np.isnan(x['tail_rmse']) for x in tails)


def test_pairing_preserves_gain_definition_and_scale_identity():
    rows=[]
    for seed,a,c in [(1,1.,.8),(2,4.,5.)]:
        for arm,value in [('A1',a),('A2',c)]:
            rows.append(dict(outer_seed=seed,column='25g',arm=arm,test_ids_sha256=str(seed),scale_V1=1.,scale_V2=2.,V1_r2=.2,V2_r2=.3,
                             **{name:value for name in b.LOSS_METRICS}))
    result=pd.DataFrame(rows);pair=b.pairing(result)
    assert pair.combined_normalized_rmse_relative_gain.mean()==pytest.approx(-.025)
    assert 1-result.loc[result.arm.eq('A2'),'combined_normalized_rmse'].mean()/result.loc[result.arm.eq('A1'),'combined_normalized_rmse'].mean()==pytest.approx(-.16)
    result.loc[0,'scale_V1']=3.
    with pytest.raises(RuntimeError):b.pairing(result)


def test_one_shot_cached_completion_precedes_label_reads():
    source=inspect.getsource(b.score_once)
    assert source.index('verify_completed();')<source.index('preflight()')<source.index('old._read_authorized_truth(')
    assert source.index("'PHASE3B_TRAIN_SCALE_AUDIT.json'")<source.index("# The single authorized outer response access boundary.")
    assert 'old.fit_model(' not in inspect.getsource(b) and 'fit_fixed(' not in inspect.getsource(b)
    assert '.score(' not in inspect.getsource(b)
