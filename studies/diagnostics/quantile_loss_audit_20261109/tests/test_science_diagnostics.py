"""Diagnostic boundaries and known numerical examples; all inputs are synthetic."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from quantile_loss_audit_20261109.quantiles import quantile_summary,sorting_diagnostic,ranking_metrics,score_overlap
from quantile_loss_audit_20261109.incremental import fixed_group_folds,incremental_width_diagnostic,seed_bootstrap_interval
from quantile_loss_audit_20261109.loss_diagnostics import target_components,measure_gradients,state_hash


def test_raw_crossings_are_not_erased_by_sorting_and_equality_is_separate():
    q=np.array([[3.,1.,2.],[0.,1.,2.],[1.,1.,1.],[3.,2.,1.]])
    before=q.copy();r=quantile_summary(np.array([2.,1.,1.,2.]),q)
    assert r['any_crossing_rate']==.5
    assert r['negative_width_rate']==.5
    assert r['q10_eq_q50_rate']==.25
    assert r['q50_eq_q90_rate']==.25
    assert r['central80_coverage']==.5
    s=sorting_diagnostic(np.array([2.,1.,1.,2.]),q)
    assert s['q50_changed_rate']==.25
    np.testing.assert_array_equal(q,before)


def test_unlabeled_rows_produce_no_target_metric():
    r=quantile_summary(None,np.array([[0.,1.,2.],[3.,2.,1.]]))
    assert r['target_metrics_status']=='NOT_MEASURED'
    assert 'q50_mae' not in r
    with pytest.raises(ValueError):quantile_summary([1.],np.zeros((2,3)))


def test_perfect_error_ranking_has_zero_ause_and_stable_top20():
    e=np.arange(1.,11.)
    r=ranking_metrics(e,e)
    assert r['ause']==0
    assert r['top20_count']==2
    assert r['top20_error_enrichment']==pytest.approx(9.5/5.5)
    assert score_overlap(e,e)['top20_jaccard']==1.
    assert ranking_metrics(np.zeros(4),np.zeros(4))['top20_error_enrichment'] is None


def test_group_folds_and_width_incremental_planted_signal_are_held_out():
    rng=np.random.default_rng(9);x=rng.normal(size=(150,2));w=rng.random((150,1));y=3+4*w[:,0]+rng.normal(scale=.01,size=150)
    groups=[f'compound-{i//3}' for i in range(150)]
    folds=fixed_group_folds(groups)
    for i in range(0,150,3):assert len(set(folds[i:i+3]))==1
    result=incremental_width_diagnostic(x,w,y,groups)
    assert result['with_width_oof_rmse']<.1*result['baseline_oof_rmse']
    assert result['interpretation'].endswith('no causal or AULC claim')


def test_seed_interval_is_local_rng_and_rejects_nonfinite():
    np.random.seed(42);expected=np.random.random(3);np.random.seed(42)
    r=seed_bootstrap_interval([1.,2.,3.]);np.testing.assert_array_equal(np.random.random(3),expected)
    assert r['lower95']<=r['mean']<=r['upper95']
    with pytest.raises(ValueError):seed_bootstrap_interval([np.nan])


def test_loss_components_preserve_current_definition():
    y=torch.tensor([1.,3.,2.]);q=torch.tensor([[2.,0.,1.],[1.,2.,3.],[4.,1.,0.]])
    pieces=target_components(y,q)
    assert set(pieces)=={'q10_pinball','q50_mse','q90_pinball','cross_q10_q50','cross_q50_q90'}
    assert float(sum(pieces.values())) == pytest.approx(4.5666666667, abs=1e-6)


def test_gradient_measurement_has_no_optimizer_or_state_update():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=torch.nn.Linear(1,4);self.condition_branch=torch.nn.Linear(1,4);self.head=torch.nn.Linear(4,6)
        def forward(self,atom,angle):return self.head(self.backbone(atom.x)+self.condition_branch(angle.x))
    torch.manual_seed(7);model=Tiny().eval();before=state_hash(model)
    batches=[(SimpleNamespace(x=torch.tensor([[1.],[2.]])),SimpleNamespace(x=torch.tensor([[.5],[.7]])),torch.tensor([[1.,2.],[2.,4.]]))]
    r=measure_gradients(model,batches,row_count=2)
    assert r['state_before']==r['state_after']==before
    assert len(r['components'])==24 and len(r['competition'])==2
    assert all(p.grad is None for p in model.parameters())
    assert r['optimizer_steps']==0
    with pytest.raises(ValueError):measure_gradients(model.train(),batches,row_count=2)


def test_snapshot_role_gate_discards_poisoned_u_and_test_targets(tmp_path):
    import csv
    from quantile_loss_audit_20261109.prepare_inputs import permitted_rows,FEATURES
    rows=[]
    for identity,role,value in [('a','l0','1.5'),('b','validation','2.5'),('c','u0','MUST_NOT_PARSE'),('d','test','MUST_NOT_PARSE')]:
        row={key:'synthetic' for key in FEATURES};row.update(sample_id=identity,V1_ml=value,V2_ml=value);rows.append(row)
    path=tmp_path/'mixed.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FEATURES+['V1_ml','V2_ml']);writer.writeheader();writer.writerows(rows)
    selected=permitted_rows(path,{'a':'l0','b':'validation','c':'u0','d':'test'},{'a':0,'b':1,'c':2,'d':3})
    assert [row['sample_id'] for row in selected]==['a','b']
    assert [row['V1_ml'] for row in selected]==[1.5,2.5]


def test_scientific_model_namespace_cannot_reuse_mutable_main_modules(tmp_path,monkeypatch):
    from quantile_loss_audit_20261109.run_audit import source_namespace_isolated
    monkeypatch.setitem(sys.modules,'src.qgeognn_al.fake',SimpleNamespace(__file__=str(tmp_path/'live'/'fake.py')))
    with pytest.raises(ValueError,match='mutable/mixed'):
        source_namespace_isolated(tmp_path/'frozen')
