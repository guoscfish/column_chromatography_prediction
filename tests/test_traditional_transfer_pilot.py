"""Readiness contracts for the matched validation-only pilot."""
from copy import deepcopy
import inspect
import json
import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from src.qgeognn_al.transfer import adaptation as a
from src.qgeognn_al.transfer.source_anchored import SourceAnchoredTransfer
from src.qgeognn_al.models import load_predictor_checkpoint
from scripts.studies import run_traditional_transfer_recipe_pilot as pilot

@pytest.fixture(scope='module')
def context():
    return pilot.prepare_context()

@pytest.fixture
def tiny():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=nn.Module()
            self.backbone.convs=nn.ModuleList([nn.Sequential(nn.Linear(2,2),nn.BatchNorm1d(2)) for _ in range(5)])
            self.condition_branch=nn.Linear(2,2)
            self.head=nn.Linear(2,6)
        def forward(self, atom, angle):
            return self.head(self.backbone.convs[4](atom.x)+self.condition_branch(atom.x))
    torch.manual_seed(17)
    model=Tiny()
    atoms=[Data(x=torch.tensor([[i/10, (i%3)/2]],dtype=torch.float32), y=torch.tensor([[i+1.,i*2+3.]])) for i in range(8)]
    angles=[Data(x=torch.zeros(1,1)) for _ in atoms]
    prep={'target_scales':a.fit_target_scales(atoms,range(6))}
    return model,atoms,angles,prep

def fit_tiny(tiny, **kwargs):
    model,atoms,angles,prep=tiny
    config=dict(maximum_epochs=3,patience=3,learning_rate=.05,bn_policy='source_stats',lbfgs_max_iter=3)
    config.update(kwargs)
    return a.train_target_adaptation(model,atoms,angles,range(6),[6,7],prep,mode='historical_shallow',config=config)

def test_scaled_loss_scale_one():
    y=torch.tensor([1.,5.]); p=torch.tensor([[3.,2.,1.],[2.,4.,6.]])
    assert torch.allclose(a.scaled_quantile_target_loss(y,p,1),a.quantile_target_loss(y,p))

def test_scaled_loss_invariance():
    y=torch.tensor([1.,5.]); p=torch.tensor([[3.,2.,1.],[2.,4.,6.]])
    assert torch.allclose(a.scaled_quantile_target_loss(y*17,p*17,3*17),a.scaled_quantile_target_loss(y,p,3))

def test_scale_train_only(tiny):
    _,atoms,_,prep=tiny
    atoms[6].y.fill_(float('nan')); atoms[7].y.fill_(float('nan'))
    assert a.fit_target_scales(atoms,range(6))==prep['target_scales']

def test_historical_exact_inventory():
    model=load_predictor_checkpoint(pilot.SOURCE)
    historical=SourceAnchoredTransfer(model,'shallow')
    a.configure_trainable(model,'historical_shallow')
    actual={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
    expected={n.replace('target_head.','head.'):p.numel() for n,p in historical.named_parameters() if p.requires_grad}
    assert actual==expected
    audit=pilot.pd.read_csv(pilot.ROOT/'studies/transfer/source_anchored_shared_transfer/training_audit.csv')
    assert set(audit.loc[audit.method.eq('standard_shallow_finetune'),'trainable_parameters'])=={sum(actual.values())}=={36387}
    assert not any('angle' in n or 'bond_float' in n or 'convs.3.' in n for n in actual)

def test_direct_stage_b_same_scope(tiny,monkeypatch):
    model,atoms,angles,prep=tiny
    a.configure_trainable(model,'historical_shallow')
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    original=a.train_target_adaptation
    scopes=[]
    def wrapper(*args,**kwargs):
        result=original(*args,**kwargs)
        scopes.append({n for n,p in args[0].named_parameters() if p.requires_grad})
        return result
    monkeypatch.setattr(a,'train_target_adaptation',wrapper)
    config=dict(maximum_epochs=2,patience=2,bn_policy='source_stats')
    a.train_staged_target_adaptation(model,atoms,angles,range(6),[6,7],prep,stage_b_mode='historical_shallow',stage_a_config=config,stage_b_config=config)
    assert scopes[1]==expected

def test_stage_b_inherits_best_not_final(tiny,monkeypatch):
    model,atoms,angles,prep=tiny
    original=a.train_target_adaptation
    scores=iter([1.,3.,4.,1.,2.,3.])
    monkeypatch.setattr(a,'point_metrics',lambda *args: {'combined_normalized_rmse':next(scores)})
    states=[]; predictions=[]; raw_epoch_states=[]
    original_predict=a.predict_point
    def capture_predict(*args,**kwargs):
        result=original_predict(*args,**kwargs)
        raw_epoch_states.append(deepcopy(args[0].state_dict()))
        return result
    monkeypatch.setattr(a,'predict_point',capture_predict)
    def wrapper(*args,**kwargs):
        if states:
            for n,v in args[0].state_dict().items(): assert torch.equal(v,states[0][n])
            np.testing.assert_allclose(original_predict(model,atoms,angles,[6,7])[1],predictions[0],rtol=1e-6,atol=1e-6)
        result=original(*args,**kwargs)
        if not states:
            assert result.best_epoch==1 and result.epochs_run==3
            states.append(deepcopy(model.state_dict()))
            assert all(torch.equal(v,raw_epoch_states[0][n]) for n,v in states[0].items())
            assert any(not torch.equal(v,raw_epoch_states[2][n]) for n,v in states[0].items())
            predictions.append(original_predict(model,atoms,angles,[6,7])[1])
        return result
    monkeypatch.setattr(a,'train_target_adaptation',wrapper)
    config=dict(maximum_epochs=3,patience=3,bn_policy='source_stats',learning_rate=.01)
    a.train_staged_target_adaptation(model,atoms,angles,range(6),[6,7],prep,stage_b_mode='historical_shallow',stage_a_config=config,stage_b_config=config)

def test_source_bn_stats_buffers_frozen(tiny):
    model=tiny[0]; before=a.snapshot_bn_buffers(model)
    affine=model.backbone.convs[4][1].weight.detach().clone()
    fit_tiny(tiny)
    assert a.bn_buffer_drift(model,before)==0
    assert all(torch.equal(before[n],v) for n,v in a.snapshot_bn_buffers(model).items())
    assert not torch.equal(affine,model.backbone.convs[4][1].weight)

def test_current_bn_not_forced_frozen(tiny):
    before=a.snapshot_bn_buffers(tiny[0]); fit_tiny(tiny,bn_policy='current')
    assert a.bn_buffer_drift(tiny[0],before)>0

def test_lbfgs_closure_parameter_update(tiny):
    before=deepcopy(tiny[0].state_dict()); fit=fit_tiny(tiny,optimizer='lbfgs')
    assert a.parameter_drift(tiny[0],before)>0
    assert all(np.isfinite(row['reported_step_loss']) and np.isfinite(row['post_step_train_loss']) for row in fit.history)
    assert np.isfinite(a.predict_point(tiny[0],tiny[1],tiny[2],[6,7])[1]).all()

def test_source_preprocessing_checkpoint_derived(context):
    source=torch.load(pilot.SOURCE,weights_only=False)['preprocessing']
    assert context[5]['scaler']==source['scaler']
    assert context[6]['source_preprocessing_hash']==pilot.digest(source)

def test_graph_full_population(context):
    audit=context[6]
    assert audit['expected_filtered_rows']==audit['actual_graph_rows']==408
    assert audit['missing_sample_ids']==audit['extra_sample_ids']==[]

def test_frozen_row_identity(context):
    frozen=pilot.pd.read_csv(pilot.FROZEN/'split_manifest.csv')
    seed=json.loads((pilot.FROZEN/'protocol.json').read_text())['outer_seeds'][0]
    expected=frozen.loc[frozen.column.eq('25g') & frozen.protocol.eq('row') & frozen.outer_seed.eq(seed),context[0].columns].reset_index(drop=True)
    pilot.pd.testing.assert_frame_equal(context[0],expected)
    assert seed==context[6]['target_outer_seed']!=pilot.SOURCE_CHECKPOINT_SEED

def test_roles_disjoint_and_complete(context):
    frame=context[0]; roles=[set(frame.loc[frame.role.eq(r),'sample_id']) for r in ['gradient_train','validation','test']]
    assert not (roles[0]&roles[1] or roles[0]&roles[2] or roles[1]&roles[2])
    assert len(set.union(*roles))==408
    assert context[6]['split_counts']=={'gradient_train':320,'validation':5,'test':83}

def test_test_truth_inaccessible_to_fit_api(context):
    assert 'test_indices' not in inspect.signature(a.train_target_adaptation).parameters
    audit=context[6]
    assert not set(audit['fit_sample_ids'])&set(audit['test_sample_ids'])
    assert len(context[1])==len(context[3])+len(context[4])==325
    assert not {'V1_ml','V2_ml','t1','t2','verified t1','verified t2'}&set(pilot.FEATURES)
    assert '--score-test' not in inspect.getsource(pilot.main)

def test_finite_real_source_output(context):
    model=load_predictor_checkpoint(pilot.SOURCE)
    assert np.isfinite(a.predict_point(model,context[1],context[2],context[4])[1]).all()

def test_quantile_crossing_contract():
    y=torch.tensor([2.]); ordered=torch.tensor([[1.,2.,3.]])
    crossed=torch.tensor([[3.,2.,1.]])
    assert a.quantile_target_loss(y,crossed)>a.quantile_target_loss(y,ordered)
    with pytest.raises(ValueError): a.quantile_target_loss(y,torch.ones(1,6))

def test_source_checkpoint_hash(context):
    frozen=json.loads((pilot.FROZEN/'protocol.json').read_text())
    assert context[6]['source_checkpoint_sha256']==frozen['paper_style_current_v2']['source_checkpoint_sha256']

def test_truth_reader_never_parses_test_cells(context, monkeypatch):
    read=pilot.pd.read_csv
    test_ids=set(context[6]['test_sample_ids'])
    identity=read(pilot.FROZEN/'filtered_canonical_25g.csv',usecols=['sample_id'])
    protected={i+1 for i,s in enumerate(identity.sample_id) if s in test_ids}
    reads=[]
    def guard(path,*args,**kwargs):
        if str(path)==str(pilot.FROZEN/'filtered_canonical_25g.csv'):
            if {'V1_ml','V2_ml'} & set(kwargs.get('usecols', [])):
                assert protected <= set(kwargs.get('skiprows', []))
                reads.append(True)
        return read(path,*args,**kwargs)
    monkeypatch.setattr(pilot.pd,'read_csv',guard)
    pilot.prepare_context()
    assert reads==[True]

def test_markdown_export_without_optional_dependencies():
    text=pilot.markdown_table(pilot.pd.DataFrame([{'method':'P0','score':1.0}]))
    assert '| method | score |' in text and '| P0 | 1.0 |' in text

def test_recovery_requires_all_seven_stage_results(tmp_path):
    pilot.write_json(tmp_path/'PILOT_PROTOCOL.json',{})
    pilot.pd.DataFrame([{'method':'P0','stage':'B'}]).to_csv(tmp_path/'pilot_validation_metrics.csv',index=False)
    pilot.pd.DataFrame([{'method':'P0'}]).to_csv(tmp_path/'pilot_training_history.csv',index=False)
    pilot.pd.DataFrame([{'method':'P0'}]).to_csv(tmp_path/'pilot_bn_drift.csv',index=False)
    with pytest.raises(AssertionError): pilot.finalize_existing(tmp_path)
