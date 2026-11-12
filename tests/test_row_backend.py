"""Row backend wiring and gradient visibility contracts with owned mock fits.

No optimizer, real checkpoint load, provider, candidate truth or science fit is
executed. Numerical/model equivalence is measured by the separate frozen suite.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.qgeognn_al.al.workflow import row_backend
from src.qgeognn_al.al.representations import Authorization, RepresentationConfig
from src.qgeognn_al.al.representations.latent import GraphRow, predictor_state_hash
from src.qgeognn_al.infra.contracts import BaselineProvenance, content_hash
from src.qgeognn_al.predictor.row import PreparedRowData
from src.qgeognn_al.predictor.conditions import ConditionNormalization


def tiny_model():
    model=torch.nn.Linear(1,1)
    with torch.no_grad():model.weight.fill_(1.);model.bias.zero_()
    return model


def prepared_data():
    # Canonical order deliberately differs from each round's L-then-U order.
    names=('u2','l0a','val','l0b','u1')
    rows={name:GraphRow(name,
        Data(x=torch.tensor([[i]],dtype=torch.long),edge_index=torch.empty((2,0),dtype=torch.long),
             edge_attr=torch.empty((0,1)),canonical_position=torch.tensor(i)),
        Data(edge_index=torch.empty((2,0),dtype=torch.long),edge_attr=torch.empty((0,1))))
        for i,name in enumerate(names)}
    norm=ConditionNormalization(0.,10.,0.,10.,'4g','source_train',4,'1'*64,'2'*64)
    return PreparedRowData(norm,{'target_scales':{'V1':1.,'V2':2.}},rows,
                           '3'*64,'4'*64,'5'*64,157,('l0a','l0b'),('val',))


def provenance(data,**changes):
    values=dict(code_commit='b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28',dataset_hash=data.dataset_hash,
        split_hash=data.split_hash,model_checkpoint_hash='6'*64,protocol_hash='7'*64,source_kind='synthetic')
    return BaselineProvenance(**{**values,**changes})


def make_backend(monkeypatch,tmp_path,*,kind='latent',prepared=None,baseline=None,l0_targets=None):
    import src.qgeognn_al.predictor as predictor
    monkeypatch.setattr(predictor,'build_predictor',lambda *a,**k:tiny_model())
    monkeypatch.setattr(row_backend,'initialize_row_model',lambda *a,**k:(tiny_model(),predictor_state_hash(tiny_model())))
    data=prepared if prepared is not None else prepared_data()
    config=RepresentationConfig(kind,128 if kind=='latent' else 512,
                                json.dumps({'sketch_seed':4_000_194,'alpha':0.8}))
    return row_backend.QGeoGNNRowBackend(prepared=data,representation=config,
        provenance=baseline or provenance(data),output_root=tmp_path/'owned_run',
        l0_targets=np.array([[1.,2.],[3.,6.]]) if l0_targets is None else l0_targets,
        allow_training=False)


def context():
    return dict(seed=157,labeled_ids=('l0a','l0b','u2'),
        labels={'u2':[5.,10.],'l0b':[3.,6.],'l0a':[1.,2.]},
        validation_ids=('val',),validation_labels={'val':[7.,14.]})


def fake_fitter(backend,events,*,write_files=False):
    def fit(**kwargs):
        events.append(kwargs)
        member=kwargs['contract']['member']
        checkpoint=Path(kwargs['runtime'])/'best.pt'
        if write_files:
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            checkpoint.write_bytes(('owned mock checkpoint '+str(member)).encode())
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        else:digest=content_hash({'owned_mock_checkpoint':member})
        return {'initialization_hash':content_hash({'legacy_init':member}),
            'public_initialization_hash':backend.initial_state_hash,
            'checkpoint_sha256':digest,'checkpoint_state_hash':content_hash({'legacy_fitted_state':member}),
            'public_checkpoint_state_hash':content_hash({'public_fitted_state':member})}
    return fit


def test_row_backend_preserves_canonical_indices_label_order_and_public_hashes(monkeypatch,tmp_path):
    backend=make_backend(monkeypatch,tmp_path);events=[]
    monkeypatch.setattr(row_backend,'fit_row_from_initialization',fake_fitter(backend,events))
    result=backend.fit(**context());call=events[0]
    assert call['train_indices']==[1,3,0] and call['validation_indices']==[2]
    assert call['prediction_indices']==[1,3,0,2]
    assert tuple(call['prediction_sample_ids'])==('l0a','l0b','u2','val')
    np.testing.assert_array_equal(call['train_truth'],[[1.,2.],[3.,6.],[5.,10.]])
    np.testing.assert_array_equal(call['validation_truth'],[[7.,14.]])
    assert all('y' not in graph for graph in call['atom_base']+call['angle'])
    assert [int(graph.canonical_position) for graph in call['atom_base']]==list(range(5))
    assert call['initialization_seed']==157+2_000_003
    assert call['training_config']['seed']==157+3_000_017
    assert call['training_config']['split_sha256']==backend.provenance.split_hash
    assert call['allow_training'] is False
    assert result['initial_state_hash']==backend.initial_state_hash
    assert result['checkpoint_state_hash']==content_hash({'public_fitted_state':0})
    assert result['labeled_ids_hash']==content_hash(list(context()['labeled_ids']))
    assert result['checkpoint_path']==str(call['runtime']/'best.pt')
    assert not (tmp_path/'owned_run').exists()


@pytest.mark.parametrize('drift',[{'seed':887},{'validation_ids':('u1',)},
    {'labels':{'l0a':[1.,2.]}},{'validation_labels':{'u1':[7.,14.]}}])
def test_row_backend_role_and_outer_seed_drift_rejected_before_fitter(monkeypatch,tmp_path,drift):
    backend=make_backend(monkeypatch,tmp_path)
    monkeypatch.setattr(row_backend,'fit_row_from_initialization',lambda **k:pytest.fail('fitter must not run'))
    with pytest.raises(ValueError):backend.fit(**{**context(),**drift})
    assert not backend.root.exists()


def test_row_backend_graph_mutation_rejected_before_fitter(monkeypatch,tmp_path):
    backend=make_backend(monkeypatch,tmp_path)
    backend.features['u1'].atom.x[0,0]+=1
    monkeypatch.setattr(row_backend,'fit_row_from_initialization',lambda **k:pytest.fail('fitter must not run'))
    with pytest.raises(ValueError,match='graph content changed'):backend.fit(**context())
    assert not backend.root.exists()


@pytest.mark.parametrize('field',['dataset_hash','split_hash'])
def test_row_backend_requires_prepared_and_provenance_identities_to_agree(monkeypatch,tmp_path,field):
    data=prepared_data();wrong=provenance(data,**{field:'f'*64})
    with pytest.raises(ValueError,match='identity|identit|dataset|split'):
        make_backend(monkeypatch,tmp_path,prepared=data,baseline=wrong)
    assert not (tmp_path/'owned_run').exists()


def test_row_backend_requires_original_l0_target_alignment(monkeypatch,tmp_path):
    with pytest.raises(ValueError,match='L0|l0|target|align'):
        make_backend(monkeypatch,tmp_path,l0_targets=np.array([[1.,2.]]))
    assert not (tmp_path/'owned_run').exists()


def test_row_k3_stage_reuses_member_zero_and_fits_only_other_member_profiles(monkeypatch,tmp_path):
    from src.qgeognn_al.al.representations import ensemble
    from src.qgeognn_al.predictor import inference
    backend=make_backend(monkeypatch,tmp_path);events=[]
    monkeypatch.setattr(row_backend,'fit_row_from_initialization',fake_fitter(backend,events,write_files=True))
    first=backend.fit(**context());collected=[]
    def collect(members,**callbacks):
        collected.extend(members)
        for member in members:member.resource.verify()
        result,ids=callbacks['predict'](tiny_model(),members[0].candidate_ids)
        assert tuple(ids)==('u1',) and result.shape==(1,6)
        return 'owned_mock_k3'
    monkeypatch.setattr(ensemble,'collect_checkpoint_ensemble',collect)
    def outputs(model,atom,angle,positions):
        assert all('y' not in graph for graph in atom+angle)
        assert positions==[0] and int(atom[0].canonical_position)==0
        return np.zeros((1,6))
    monkeypatch.setattr(inference,'predict_outputs',outputs)
    state={'labeled_ids':context()['labeled_ids'],'labels':context()['labels'],'candidate_ids':('u1',)}
    assert backend.ensemble(fit=first,state=state,validation_ids=('val',),validation_labels={'val':[7.,14.]})=='owned_mock_k3'
    assert [call['contract']['member'] for call in events]==[0,1,2]
    assert [call['initialization_seed'] for call in events]==[157+2_000_003+i*1_000_033 for i in range(3)]
    assert [call['training_config']['seed'] for call in events]==[157+3_000_017+i*1_000_033 for i in range(3)]
    assert all(call['train_indices']==[1,3,0] for call in events)
    assert all(call['allow_training'] is False for call in events)
    assert [member.member for member in collected]==[0,1,2]
    assert len({member.resource.sha256 for member in collected})==3
    assert len({member.shared_fit_config_hash for member in collected})==1
    assert all(member.train_ids==state['labeled_ids'] and member.candidate_ids==('u1',) for member in collected)


@pytest.mark.parametrize('kind',['center_width','gradient','fused'])
def test_backend_gradient_dispatch_preserves_cw_l0_and_raw_scale_visibility(monkeypatch,tmp_path,kind):
    from src.qgeognn_al.al.representations import gradient,latent,fused
    backend=make_backend(monkeypatch,tmp_path,kind=kind)
    monkeypatch.setattr(backend,'_model',lambda fit:tiny_model().eval())
    captured=[];sentinel=object();fused_sentinel=object();latent_sentinel=object()
    def extract(model,**kwargs):captured.append(kwargs);return sentinel
    monkeypatch.setattr(gradient,'extract_gradient',extract)
    monkeypatch.setattr(latent,'extract_latent',lambda model,**kwargs:latent_sentinel)
    def combine(first,second,alpha):
        assert first is sentinel and second is latent_sentinel and alpha==0.8
        return fused_sentinel
    monkeypatch.setattr(fused,'fuse_snapshots',combine)
    auth=Authorization(('l0a','l0b'),('u2','u1'))
    result=backend.represent(fit={'checkpoint_hash':'9'*64},authorization=auth)
    packet=captured[0]
    assert packet['sketch_seed']==4_000_194
    assert [item.sample_id for item in packet['rows']]==list(auth.train_ids+auth.candidate_ids)
    assert packet['provenance'].model_checkpoint_hash=='9'*64
    if kind=='center_width':
        assert packet['kind']=='center_width' and 'scales' not in packet
        assert packet['l0_ids']==backend.l0_ids
        np.testing.assert_array_equal(packet['l0_targets'],[[1.,2.],[3.,6.]])
    else:
        assert packet['kind']=='gradient' and packet['scales']==(1.,2.)
        assert 'l0_ids' not in packet and 'l0_targets' not in packet
    assert result is (fused_sentinel if kind=='fused' else sentinel)


def completed_runtime(backend, labeled_ids):
    directory=backend.root/'member_0'/('L'+str(len(labeled_ids))+'_'+content_hash(list(labeled_ids))[:16])
    directory.mkdir(parents=True,exist_ok=True)
    for name in ('best.pt','predictions.csv.gz','fit_audit.json'):
        (directory/name).write_bytes(('owned mock '+name).encode())
    return directory


def test_row_recovery_only_reuses_complete_current_l_and_forces_no_new_fit(monkeypatch,tmp_path):
    from src.qgeognn_al.al.workflow.state import AmbiguousFit
    backend=make_backend(monkeypatch,tmp_path);backend.allow_training=True
    inputs=context();directory=completed_runtime(backend,inputs['labeled_ids']);events=[]
    monkeypatch.setattr(row_backend,'fit_row_from_initialization',fake_fitter(backend,events))
    before={p.name:p.read_bytes() for p in directory.iterdir()}
    recovered=backend.recover_fit(**inputs)
    assert recovered['checkpoint_path']==str(directory/'best.pt')
    assert len(events)==1 and events[0]['allow_training'] is False
    assert before=={p.name:p.read_bytes() for p in directory.iterdir()}
    # Same set with a different acquisition order is a different fit identity.
    with pytest.raises(AmbiguousFit):
        backend.recover_fit(**{**inputs,'labeled_ids':tuple(reversed(inputs['labeled_ids']))})
    assert len(events)==1
    for missing in ('best.pt','predictions.csv.gz','fit_audit.json'):
        piece=directory/missing;data=piece.read_bytes();piece.unlink()
        with pytest.raises(AmbiguousFit):backend.recover_fit(**inputs)
        assert len(events)==1
        piece.write_bytes(data)


def test_workflow_recovers_backend_completion_once_after_interrupted_outer_receipt(monkeypatch,tmp_path):
    from src.qgeognn_al.al.workflow import RunSpec,RoundWorkflow,LabelBroker
    from src.qgeognn_al.al.workflow.state import write_envelope,read_envelope
    backend=make_backend(monkeypatch,tmp_path);events=[]
    monkeypatch.setattr(row_backend,'fit_row_from_initialization',fake_fitter(backend,events))
    monkeypatch.setattr(backend,'fit',lambda **kwargs:pytest.fail('started fit must never invoke a new fit'))
    spec=RunSpec(seed=157,acquisition_seed=157,acquisition='lcmd',representation='latent',
        batch_size=1,target_budget=3,initial_ids=('l0a','l0b'),candidate_ids=('u2','u1'),validation_ids=('val',),
        dataset_hash=backend.provenance.dataset_hash,split_hash=backend.provenance.split_hash,
        initial_state_hash=backend.initial_state_hash,protocol_hash=backend.provenance.protocol_hash,
        execution_mode='authorized_scientific_run')
    labels=LabelBroker(spec=spec,initial_labels={'l0a':[1.,2.],'l0b':[3.,6.]},
        validation_labels={'val':[7.,14.]},reveal=None)
    directory=tmp_path/'workflow';workflow=RoundWorkflow(directory,spec=spec,backend=backend,labels=labels)
    completed_runtime(backend,spec.initial_ids)
    write_envelope(directory/'fit_0000.started.json',{'scope':spec.scope,'labeled_ids_hash':content_hash(list(spec.initial_ids))})
    def interrupted_save():raise OSError('owned crash after durable outer fit receipt')
    monkeypatch.setattr(workflow,'_save',interrupted_save)
    with pytest.raises(OSError,match='durable outer'):workflow.advance()
    receipt=read_envelope(directory/'fit_0000.json')['payload']
    assert receipt['fit']['initial_state_hash']==spec.initial_state_hash
    assert receipt['fit_context_hash']==workflow._fit_context_hash()
    assert len(events)==1 and events[0]['allow_training'] is False
    # The durable outer receipt now takes precedence; no recovery or fit repeats.
    resumed=RoundWorkflow(directory,spec=spec,backend=backend,labels=labels)
    assert resumed.advance()=='validate_pending' and len(events)==1
