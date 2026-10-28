"""Phase 2 permission boundary, unchanged fit updates, and paired summary checks."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_phase1 as p
import run_phase2 as r
import test_protocol as t


def test_only_fit_code_change_is_source_validation_label_diagnostics():
    original=ast.parse(inspect.getsource(p.fit_model))
    new=ast.parse(inspect.getsource(r.fit_model))
    original.body[0].body.pop(0); new.body[0].body.pop(0)
    loop=next(n for n in ast.walk(new) if isinstance(n,ast.For) and any(isinstance(x,ast.If) and ast.unparse(x.test)=="c == '4g'" for x in n.body))
    loop.body=[n for n in loop.body if not(isinstance(n,ast.If) and ast.unparse(n.test)=="c == '4g'")]
    assert ast.dump(original,include_attributes=False)==ast.dump(new,include_attributes=False)


@pytest.mark.parametrize('arm',['A1','A2'])
def test_source_validation_diagnostic_omission_preserves_weights_and_target_selection(tmp_path,monkeypatch,arm):
    torch.set_num_threads(1)
    data=t.synthetic_data.__wrapped__()
    new_data=deepcopy(data)
    new_data['4g']['truth'][8:]=np.nan
    for atom in new_data['4g']['atoms'][8:]:
        atom.y.zero_()
    train,valid={c:list(range(8)) for c in data},{c:list(range(8,12)) for c in data}
    monkeypatch.setattr(p,'STUDY',tmp_path);monkeypatch.setattr(r,'STUDY',tmp_path)
    monkeypatch.setattr(r.old,'TRAINING',{**r.old.TRAINING,'maximum_epochs':3})
    (tmp_path/'protocol.json').write_text('{}')
    old_run,new_run=tmp_path/'old',tmp_path/'new';old_run.mkdir();new_run.mkdir()
    original,old_fit,_=p.fit_model(data,train,valid,arm,17,old_run)
    changed,new_fit,_=r.fit_model(new_data,train,valid,arm,17,new_run)
    assert old_fit['best_epoch']==new_fit['best_epoch']
    for name,value in original.state_dict().items():
        torch.testing.assert_close(value,changed.state_dict()[name],rtol=0,atol=0)
    a,b=pd.read_csv(old_run/'history.csv'),pd.read_csv(new_run/'history.csv')
    for name in a.columns:
        if name.startswith('4g_') or name=='validation_loss_4g':
            continue
        np.testing.assert_array_equal(a[name],b[name])
    assert 'validation_loss_4g' not in b and '4g_V1_rmse' not in b


def test_phase2_loader_requests_source_train_only_and_whitelisted_features(monkeypatch):
    read_csv=pd.read_csv
    calls=[]
    def checked(path,*args,**kwargs):
        path=Path(path)
        if path==r.old.SOURCE_DATA or path.name.startswith('filtered_features_'):
            assert kwargs.get('usecols')==r.old.FEATURES
            calls.append(str(path))
        return read_csv(path,*args,**kwargs)
    requests=[]
    def reader(path,ids):
        requests.append((Path(path),list(ids)))
        return np.column_stack([np.arange(len(ids))+1.,2*np.arange(len(ids))+3.])
    monkeypatch.setattr(pd,'read_csv',checked)
    data,access=r.load_data(r.NEW_SEEDS[0],truth_reader=reader)
    source_ids=requests[0][1]
    assert set(source_ids)==set(data['4g']['roles']['gradient_train']) and len(source_ids)==3330
    assert not set(source_ids)&set(data['4g']['roles']['source_validation'])
    assert np.isnan(data['4g']['truth'][data['4g']['positions']['source_validation']]).all()
    assert all(torch.count_nonzero(data['4g']['atoms'][i].y)==0 for i in data['4g']['positions']['source_validation'])
    for c in r.TARGETS:
        ids=set(next(ids for path,ids in requests if path.name==f'filtered_canonical_{c}.csv'))
        assert ids==set(data[c]['roles']['gradient_train'])
        assert not ids&set(data[c]['roles']['validation']+data[c]['roles']['test'])
    assert len(calls)==3 and not any(item['outer_labels_parsed'] for item in access)


def test_phase2_rejects_pilot_fit_and_unauthorized_seed():
    with pytest.raises(ValueError):
        r.run_fold(p.PILOT_SEED,1)
    with pytest.raises(ValueError):
        r.load_data(1)
