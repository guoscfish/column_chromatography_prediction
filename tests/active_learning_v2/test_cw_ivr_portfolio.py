import inspect
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.qgeognn_al.active_learning_v2 import cw_ivr_portfolio as selector
from src.qgeognn_al.active_learning_v2 import cw_ivr_portfolio_study as study
from src.qgeognn_al.active_learning_v2 import cw_ivr_portfolio_runner as runner
from src.qgeognn_al.active_learning_v2 import cw_ivr_portfolio_reporting as reporting
from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select
from src.qgeognn_al.active_learning_v2.ivr import conditional_batch_ivr
from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore
from src.qgeognn_al.training.predictor import atomic_json


@pytest.mark.parametrize('seed',[7,41,73])
def test_incremental_pure_selectors_match_historical_exactly(seed):
    x=np.random.default_rng(seed).normal(size=(100,12))
    cw=selector.CoverageState(x[30:],x[:30])
    ivr=selector.IVRState(x,np.arange(30),np.arange(30,100))
    a,b=[],[]
    for _ in range(32):
        p,d=cw.propose();a.append(p);cw.condition(p)
        p=ivr.propose();b.append(p);ivr.condition(p)
    assert a==lcmd_tp_select(x[30:],x[:30],32).selected_pool_positions.tolist()
    old=conditional_batch_ivr(x,np.arange(30),np.arange(30,100),32)
    assert b==old['selected_candidate_positions']
    np.testing.assert_allclose(ivr.scores()[1].mean(),old['trace'][-1]['integrated_variance_after'],atol=1e-14)


def test_shared_context_direct_coverage_and_covariance():
    x=np.random.default_rng(17).normal(size=(90,10))
    cw=selector.CoverageState(x[20:],x[:20])
    ivr=selector.IVRState(x,np.arange(20),np.arange(20,90))
    external=[6,17,43]
    for p in external:
        cw.condition(p);ivr.condition(p)
    expected=selector._squared_distances(x[20:],np.r_[x[:20],x[20+np.array(external)]])
    np.testing.assert_allclose(cw.minimum[cw.available],expected.min(1)[cw.available],atol=1e-13)
    z=ivr.x
    direct=np.linalg.inv(np.eye(10)+z[np.r_[np.arange(20),20+np.array(external)]].T@z[np.r_[np.arange(20),20+np.array(external)]])
    np.testing.assert_allclose(ivr.q,z@direct,atol=1e-13)
    assert cw.selected==ivr.selected==external


def test_synthetic_high_overlap_is_unique_balanced_deterministic():
    # Both selectors favor the same large-norm direction, many near duplicates.
    x=np.arange(1,101,dtype=float)[:,None]*np.array([[1.,2.,3.]])
    a=selector.select_portfolio(x,x,20)
    b=selector.select_portfolio(x,x,20)
    assert a==b
    assert len(a['selected_pool_positions'])==len(set(a['selected_pool_positions']))==32
    assert [r['expert'] for r in a['trace']]==['cw','ivr']*16
    assert [r['shared_context_count'] for r in a['trace']]==list(range(32))
    assert [r['coverage_center_count'] for r in a['trace']]==list(range(20,52))
    assert a['diagnostics']['independent_proposal_overlap_count']>0
    assert a['diagnostics']['cw_turns']==a['diagnostics']['ivr_turns']==16
    with pytest.raises(TypeError):
        selector.select_portfolio(x,x,20,labels=np.ones(100))


def test_selector_has_no_label_access_or_dynamic_quota():
    assert list(inspect.signature(selector.select_portfolio).parameters)==['cw_features','ivr_features','labeled_count']
    assert study.SEEDS==(157,6101)
    assert study.ACTIVE_LABEL_BUDGETS==tuple(range(333,654,32))
    assert study.TOLERANCE==.01


def test_actual_l333_lineage_and_read_only_provenance():
    audit=study.audit_reuse()
    assert len(audit['anchors'])==2
    assert len(audit['comparators'])==10
    assert all(a['checkpoint_reused'] and a['new_anchor_fits']==0 for a in audit['anchors'])
    assert all(a['initialization_hash'] for a in audit['anchors'])


def test_report_barrier_runs_before_any_label_store(monkeypatch,tmp_path):
    calls=[]
    def fail(*a): raise RuntimeError('incomplete global freeze')
    monkeypatch.setattr(runner,'finalize_pre_test',fail)
    monkeypatch.setattr(reporting,'RestrictedLabelStore',lambda *a:calls.append(a))
    with pytest.raises(RuntimeError,match='incomplete'):
        reporting.report(tmp_path)
    assert not calls


def test_seed6101_requires_seed157_engineering_gate(monkeypatch,tmp_path):
    atomic_json(tmp_path/'global_pre_test_freeze.json',{'status':'PENDING_TRAJECTORIES'})
    atomic_json(tmp_path/'engineering_smoke.json',{'status':'PASS','new_fits':0})
    monkeypatch.setattr(runner,'validate_prepared',lambda *a:None)
    def fail(*a): raise RuntimeError('157 missing')
    monkeypatch.setattr(runner,'verify_seed',fail)
    with pytest.raises(RuntimeError,match='157 missing'):
        runner.execute_seed(6101,tmp_path)


@pytest.fixture
def synthetic_runtime(tmp_path,monkeypatch):
    roles={'l0':np.arange(333),'u0':np.arange(333,700),'validation':np.arange(700,710),'test':np.arange(710,720)}
    partition=pd.DataFrame({'sample_id':[f'id{i}' for i in range(720)],'canonical_index':np.arange(720),
        'role':[k for k,v in roles.items() for _ in v]})
    source=tmp_path/'data.csv'
    pd.DataFrame({'sample_id':partition.sample_id,'V1_ml':[i if i<710 else 'TEST_FORBIDDEN' for i in range(720)],
        'V2_ml':[2*i if i<710 else 'TEST_FORBIDDEN' for i in range(720)]}).to_csv(source,index=False)
    calls=[]
    class Context:
        seed=157
        protocol_hash='frozen'
        def __init__(self):
            self.study=tmp_path
            self.runtime=tmp_path/'runtime/seed_157'
            self.roles=roles
            self.l0_truth=np.array([[i,2*i] for i in range(333)],dtype=np.float32)
        def ids(self,indices):return [f'id{i}' for i in indices]
        def new_store(self):
            store=RestrictedLabelStore(source,partition)
            original=store.reveal
            def reveal(ids,purpose):
                if purpose=='after_acquisition_fit':
                    r=len(store.acquisition_ids)//32-1
                    assert (self.runtime/study.METHOD/f'round_{r:02d}/round_freeze.json').exists()
                    batch=pd.read_csv(self.runtime/study.METHOD/f'round_{r:02d}/acquisition_artifacts/selected_next_batch.csv')
                    assert list(ids)==batch.sample_id.tolist()
                calls.append(('reveal',purpose,tuple(ids)))
                return original(ids,purpose)
            store.reveal=reveal
            return store
        def fit(self,method,r,l,truth,path):
            assert r>0
            path.mkdir(parents=True,exist_ok=True)
            if not (path/'best.pt').exists():
                calls.append(('fit',r))
                (path/'best.pt').write_text(f'model{r}')
                (path/'predictions.csv.gz').write_text(f'prediction{r}')
            return {'reuse_status':'new_fit','checkpoint_state_hash':f'model{r}','fit_contract_hash':f'fit{r}'}
    context=Context()
    anchor=tmp_path/'anchor'
    anchor.mkdir()
    (anchor/'best.pt').write_text('model0')
    (anchor/'predictions.csv.gz').write_text('prediction0')
    monkeypatch.setattr(runner,'_round0_evaluation',lambda c:(anchor/'best.pt',anchor/'predictions.csv.gz',
        {'reuse_status':'reused_historical_round0','checkpoint_state_hash':'model0','fit_contract_hash':'fit0'}))
    def acquire(c,m,r,l,u,model,directory):
        assert model.read_text()==f'model{r}'
        calls.append(('extract_current',r))
        a=directory/'acquisition_artifacts';a.mkdir(exist_ok=True)
        ids=c.ids(u[:32])
        runner._write_csv_once(a/'selected_next_batch.csv',pd.DataFrame({'sample_id':ids,'expert':['cw','ivr']*16}))
        return ids,{'source_round':r,'shared_context':True}
    monkeypatch.setattr(runner,'_acquire',acquire)
    return context,calls


def test_full_trajectory_freeze_before_reveal_no_anchor_fit_and_resume(synthetic_runtime):
    context,calls=synthetic_runtime
    first=runner.run_trajectory(context,study.METHOD)
    assert first['final_active_labels']==653
    assert [c[1] for c in calls if c[0]=='fit']==list(range(1,11))
    assert [c[1] for c in calls if c[0]=='extract_current']==list(range(10))
    assert all(c[1]!='final_test_evaluation' for c in calls if c[0]=='reveal')
    calls.clear()
    again=runner.run_trajectory(context,study.METHOD)
    assert first==again
    assert not [c for c in calls if c[0]=='fit']
    assert len(runner.verify_seed(157,context.runtime.parents[1]))==11
    (context.runtime/study.METHOD/'round_03/state.csv').write_text('tampered')
    with pytest.raises(RuntimeError,match='nested artifact'):
        runner.verify_seed(157,context.runtime.parents[1])


def test_interrupted_resume_keeps_frozen_selection(synthetic_runtime,monkeypatch):
    context,calls=synthetic_runtime
    fit=context.fit
    def interrupted(m,r,*args):
        if r==4: raise RuntimeError('power loss')
        return fit(m,r,*args)
    context.fit=interrupted
    with pytest.raises(RuntimeError,match='power loss'):
        runner.run_trajectory(context,study.METHOD)
    context.fit=fit
    result=runner.run_trajectory(context,study.METHOD)
    assert result['final_active_labels']==653
    assert [c[1] for c in calls if c[0]=='fit']==list(range(1,11))


def test_aulc_regret_negative_and_complete_grid():
    rows=[]
    for s in study.SEEDS:
        for m in (study.METHOD,*study.COMPARATORS):
            y=.49 if m==study.METHOD else .5 if m=='center_width_lcmd' else .52
            for n in study.ACTIVE_LABEL_BUDGETS:
                rows.append({'outer_seed':s,'method':m,'active_label_count':n,reporting.METRIC:y,
                             'V1_RMSE':y,'V2_RMSE':y,'V1_R2':1-y,'V2_R2':1-y})
    curves=pd.DataFrame(rows)
    summary,means,regret,rs=reporting.summarize(curves)
    assert regret.loc[regret.method.eq(study.METHOD),'regret'].iloc[0]==pytest.approx(-.01)
    assert reporting.decide(summary,means,rs)['decision']=='STRONG_PORTFOLIO_SIGNAL'
    with pytest.raises(ValueError,match='grid'):
        reporting.summarize(curves.iloc[1:])


def test_shared_gradient_pass_preserves_both_geometries(monkeypatch):
    import torch
    from torch_geometric.data import Data
    from src.qgeognn_al.active_learning_v2.gradient_features import (
        extract_linear_output_gradient_sketches_many,extract_linear_output_gradient_sketches,
        extract_q50_gradient_sketches)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(1,6)
        def forward(self,atom,angle):return self.layer(atom.x.float())
    torch.manual_seed(4)
    model=Toy()
    atom=[Data(x=torch.tensor([[v]]),y=torch.zeros(1,2),canonical_position=torch.tensor(i)) for i,v in enumerate([1.,2.,3.])]
    angle=[Data(x=torch.zeros(1,1)) for _ in atom]
    transform=np.array([[.2,.2],[-.3,.3]])
    calls=[]
    original=torch.autograd.grad
    def tracked(*a,**k):
        calls.append(1)
        return original(*a,**k)
    monkeypatch.setattr(torch.autograd,'grad',tracked)
    result=extract_linear_output_gradient_sketches_many(model,atom,angle,[0,1,2],{'cw':transform,'ivr':np.diag([.5,.25])},dimension=16,sketch_seed=71)
    assert len(calls)==6
    cw=extract_linear_output_gradient_sketches(model,atom,angle,[0,1,2],transform,dimension=16,sketch_seed=71)
    ivr=extract_q50_gradient_sketches(model,atom,angle,[0,1,2],(2.,4.),dimension=16,sketch_seed=71)
    np.testing.assert_allclose(result['cw'].features,cw.features,atol=1e-6)
    np.testing.assert_allclose(result['ivr'].features,ivr.features,atol=1e-6)
