from types import SimpleNamespace
import numpy as np
import pytest
from src.qgeognn_al.active_learning_v2 import cw_factorial_extension as extension


def test_extension_acquisition_freezes_predictions_and_refuses_state_drift(tmp_path,monkeypatch):
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixture-checkpoint')
    c=SimpleNamespace(seed=157,transforms={'cw':np.eye(2)},atom=[],angle=[],ids=lambda x:[f'row-{i}' for i in x])
    l,u=np.arange(2),np.arange(2,36)
    bank=SimpleNamespace(features=np.random.default_rng(8).normal(size=(36,8)),audit={'fixture':True})
    monkeypatch.setattr(extension,'load_predictor_checkpoint',lambda path:None)
    monkeypatch.setattr(extension,'extract_linear_output_gradient_sketches',lambda *args,**kwargs:bank)
    monkeypatch.setattr(extension,'predict_outputs',lambda model,a,b,idx:(np.ones((len(idx),6)),idx))
    d=tmp_path/'round_06';p={'limit':24}
    picks=extension.acquisition(c,'cw','ivr',l,u,checkpoint,d,p)
    assert len(picks)==len(set(picks))==32
    assert (d/'premeasurement.csv.gz').exists() and (d/'batch_freeze.json').exists()
    monkeypatch.setattr(extension,'extract_linear_output_gradient_sketches',lambda *args,**kwargs:pytest.fail('cache should reuse frozen batch'))
    np.testing.assert_array_equal(picks,extension.acquisition(c,'cw','ivr',l,u,checkpoint,d,p))
    with pytest.raises(AssertionError):extension.acquisition(c,'cw','ivr',l,u[::-1],checkpoint,d,p)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(AssertionError):extension.acquisition(c,'cw','ivr',l,u,checkpoint,d,p)
