"""Fixed-budget equivalence and outer-label exclusion; observed outer responses never read."""
from copy import deepcopy
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_phase3a as s
import test_protocol as t


@pytest.mark.parametrize('arm',['A1','A2'])
def test_fixed_refit_updates_match_original_bitwise(tmp_path,monkeypatch,arm):
    torch.set_num_threads(1)
    data=t.synthetic_data.__wrapped__()
    train={c:list(range(8)) for c in data}
    for c in data:
        data[c]['positions']['gradient_train']=list(range(8))
        data[c]['positions']['heldout']=list(range(8,12))
    monkeypatch.setattr(s.old,'STUDY',tmp_path)
    (tmp_path/'protocol.json').write_text('{}')
    original,fit,_=s.old.fit_model(data,train,None,arm,17,tmp_path/'original',fixed_epochs=3)
    changed,new=s.fit_fixed(data,train,arm,17,tmp_path/'new',3,'synthetic')
    for name,value in original.state_dict().items():
        torch.testing.assert_close(value,changed.state_dict()[name],rtol=0,atol=0)
    a,b=pd.read_csv(tmp_path/'original/history.csv'),pd.read_csv(tmp_path/'new/history.csv')
    for name in a.columns:
        np.testing.assert_array_equal(a[name],b[name])
    assert new['optimizer_steps']==3==new['fixed_epochs']==fit['best_epoch']
    checkpoint=torch.load(tmp_path/'new/final.pt',weights_only=False)
    assert set(checkpoint['model_state_dict'])==set(changed.state_dict())
    assert any('running_mean' in name for name in checkpoint['model_state_dict'])


def test_loader_whitelist_and_train_only_requests(monkeypatch):
    read_csv=pd.read_csv
    calls=[]
    def checked(path,*args,**kwargs):
        path=Path(path)
        if path==s.old.SOURCE_DATA or path.name.startswith('filtered_features_'):
            assert kwargs.get('usecols')==s.old.FEATURES
            calls.append(path)
        return read_csv(path,*args,**kwargs)
    requests=[]
    def reader(path,ids):
        requests.append((Path(path),list(ids)))
        return np.column_stack([np.arange(len(ids))+1.,2*np.arange(len(ids))+3.])
    monkeypatch.setattr(pd,'read_csv',checked)
    data,access=s.load_data(s.SEEDS[0],truth_reader=reader)
    assert len(calls)==len(requests)==len(access)==3
    for c,(path,ids) in zip(s.old.TASKS,requests):
        assert ids==data[c]['roles']['gradient_train']
        for role,idx in data[c]['positions'].items():
            if role!='gradient_train':
                assert not set(ids)&set(data[c]['roles'][role])
                assert np.isnan(data[c]['truth'][idx]).all()
                assert all(torch.count_nonzero(data[c]['atoms'][i].y)==0 for i in idx)
    assert len(requests[0][1])==3330


def test_replay_loader_reads_no_labels():
    def forbidden(*args,**kwargs):
        raise AssertionError('feature-only replay requested labels')
    data,access=s.load_data(s.SEEDS[0],parse_training_labels=False,truth_reader=forbidden)
    assert access==[]
    for c in data:
        assert np.isnan(data[c]['truth']).all()
        assert all(torch.count_nonzero(atom.y)==0 for atom in data[c]['atoms'])


def test_authorize_full_rejects_partial_or_heldout_training():
    data=t.synthetic_data.__wrapped__()
    train={c:list(data[c]['positions']['gradient_train']) for c in data}
    with pytest.raises(RuntimeError,match='exactly equal'):
        s.authorize_full(data,{**train,'25g':train['25g'][:-1]})
    with pytest.raises(RuntimeError,match='exactly equal'):
        s.authorize_full(data,{**train,'4g':train['4g']+[8]})


def synthetic_results():
    return pd.DataFrame([{'outer_seed':1,'arm':'A1','column':c,'inner_fold':f,'best_epoch':ep}
                        for f,ep in enumerate([1,2,500,4,5],1) for c in s.TARGETS])


def test_joint_budget_median_and_invalid_cases():
    frame=synthetic_results()
    assert s.budget_from_results(frame,1,'A1')==([1,2,500,4,5],4)
    for bad in [frame.iloc[:-1],pd.concat([frame,frame.iloc[[0]]])]:
        with pytest.raises(RuntimeError):s.budget_from_results(bad,1,'A1')
    for value in [0,501,1.5,np.nan,3]:
        bad=frame.astype({'best_epoch':float});bad.loc[0,'best_epoch']=value
        with pytest.raises(RuntimeError):s.budget_from_results(bad,1,'A1')


def test_no_outer_score_or_metric_selector_in_phase3a():
    text=inspect.getsource(s)
    assert 'old.score(' not in text and 'old.metric_rows(' not in text
    fit=inspect.getsource(s.fit_fixed)
    assert 'old.predict(' not in fit and 'joint_validation_score' not in fit and 'patience' not in fit


def test_prediction_column_identity_and_finite_guards():
    data={'25g':{'roles':{'test':['a','b']}}}
    frame=pd.DataFrame([dict(column='25g',outer_seed=1,arm='A1',role='test',sample_id=sid,
                            **dict(zip(s.old.QUANTILES,[1.,2.,3.,4.,5.,6.]))) for sid in ['a','b']],columns=s.PRED_COLUMNS)
    assert s.check_prediction(frame,data,1,'A1','25g','test')['rows']==2
    for bad in [frame.iloc[::-1],frame[frame.columns[::-1]],frame.assign(V1_q50=np.nan)]:
        with pytest.raises(RuntimeError):s.check_prediction(bad,data,1,'A1','25g','test')
