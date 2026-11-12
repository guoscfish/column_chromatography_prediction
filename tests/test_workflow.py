"""Lifecycle engineering fixtures; no optimizer, real reveal, provider or model fit."""
import dataclasses
import json
from pathlib import Path
import numpy as np
import pytest
from src.qgeognn_al.al.workflow import RunSpec, RoundWorkflow, LabelBroker, AmbiguousReveal, AmbiguousFit
from src.qgeognn_al.al.workflow.backends import SyntheticBackend, QGeoGNNBackend
from src.qgeognn_al.al.workflow.state import read_envelope, write_envelope
from src.qgeognn_al.infra.contracts import content_hash


def setup(root, acquisition='lcmd', **changes):
    options = dict(seed=157, acquisition_seed=157000471, acquisition=acquisition, representation='latent',
                   batch_size=2, target_budget=6, initial_ids=('L0','L1'), candidate_ids=('U0','U1','U2','U3'),
                   validation_ids=('V0','V1'), dataset_hash='1'*64, split_hash='2'*64,
                   initial_state_hash='3'*64, protocol_hash='4'*64)
    options.update(changes); spec = RunSpec(**options)
    features = {'L0':[0.,0.], 'L1':[10.,0.], 'U0':[1.,0.], 'U1':[3.,1.], 'U2':[7.,1.], 'U3':[9.,0.]}
    features = {key: value + [0.] * 126 for key, value in features.items()}
    backend = SyntheticBackend(features=features,
                               validation_predictions={'V0':[0.,1.,2.,1.,2.,3.], 'V1':[2.,3.,4.,3.,4.,5.]},
                               initial_state_hash=spec.initial_state_hash, dataset_hash=spec.dataset_hash,
                               split_hash=spec.split_hash, code_commit='a'*40, protocol_hash=spec.protocol_hash)
    calls = []
    def reveal(ids, frozen):
        calls.append((ids,frozen)); return {identity: [2.,4.] for identity in ids}
    broker = LabelBroker(spec=spec, initial_labels={'L0':[1.,2.],'L1':[3.,6.]},
                         validation_labels={'V0':[1.2,2.2],'V1':[3.2,4.2]}, reveal=reveal)
    workflow = RoundWorkflow(root, spec=spec, backend=backend, labels=broker)
    return workflow, spec, backend, broker, calls


def ready(workflow):
    assert workflow.advance()=='validate_pending'
    assert workflow.advance()=='feedback_pending'
    assert workflow.advance()=='ready'


@pytest.mark.parametrize('method',['random','lcmd','maxdet','ivr','hybrid'])
def test_full_multiround_fit_freeze_reveal_retrain_validation_feedback(tmp_path,method):
    w,s,b,l,calls=setup(tmp_path,method)
    result=w.run()
    assert result['phase']=='complete' and len(result['labeled_ids'])==6 and not result['candidate_ids']
    assert len(calls)==2
    assert len([c for c in b.calls if c[0]=='fit_fixture'])==3
    assert len(list(tmp_path.glob('feedback_*.json')))==3
    assert all(not read_envelope(p)['payload']['fit']['training_performed'] for p in tmp_path.glob('fit_????.json'))
    for p in tmp_path.glob('round_*.freeze.json'):
        v=read_envelope(p)['payload']; assert v['outer_seed']==s.seed and v['seed']==s.acquisition_seed
        assert set(v['selected_ids'])<=set(s.candidate_ids) and v['budget']==2
    resumed=RoundWorkflow(tmp_path,spec=s,backend=b,labels=l)
    assert resumed.run()==result and len(calls)==2


def test_cross_seed_or_budget_resume_is_rejected_before_callbacks(tmp_path):
    w,s,b,l,calls=setup(tmp_path)
    other=dataclasses.replace(s,seed=887)
    broker=LabelBroker(spec=other,initial_labels=l.initial,validation_labels=l.validation,reveal=l.callback)
    with pytest.raises(ValueError,match='another seed'):
        RoundWorkflow(tmp_path,spec=other,backend=b,labels=broker)
    assert calls==[]


def test_agent_foreign_duplicate_and_extra_actions_rejected_before_freeze(tmp_path):
    w,s,b,l,c=setup(tmp_path);ready(w)
    for plan in [{'selected_ids':['U0','U0']},{'selected_ids':['U0','V0']},{'selected_ids':['U0','U1'],'fit':True}]:
        with pytest.raises(ValueError):w.advance(agent_plan=plan)
    assert not list(tmp_path.glob('*.freeze.json')) and c==[]


def test_agent_plan_freezes_before_reveal_and_has_no_provider(tmp_path):
    w,s,b,l,c=setup(tmp_path);ready(w)
    assert w.advance(agent_plan={'selected_ids':['U0','U2']})=='frozen'
    assert not c
    packet=read_envelope(tmp_path/'round_0000.freeze.json')['payload']
    assert packet['audit']['provider_invoked'] is False
    w.advance();assert c[0][0]==('U0','U2')


def test_changed_freeze_even_with_recomputed_envelope_fails_closed(tmp_path):
    w,s,b,l,c=setup(tmp_path);ready(w);w.advance()
    path=tmp_path/'round_0000.freeze.json';v=read_envelope(path)['payload'];v['selected_ids']=['U0','U3'];write_envelope(path,v)
    with pytest.raises(ValueError,match='freeze changed'):w.advance()
    assert c==[]


def test_ambiguous_reveal_crash_does_not_repeat_label_provider(tmp_path):
    w,s,b,l,c=setup(tmp_path);ready(w);w.advance()
    def failure(ids,frozen):c.append('started');raise OSError('after external callback started')
    l.callback=failure
    with pytest.raises(OSError):w.advance()
    resumed=RoundWorkflow(tmp_path,spec=s,backend=b,labels=l)
    with pytest.raises(AmbiguousReveal):resumed.advance()
    assert c==['started']


def test_receipt_recovery_after_state_write_failure_reveals_once(tmp_path):
    w,s,b,l,c=setup(tmp_path);ready(w);w.advance()
    original=w._save
    def failure():raise OSError('simulate failed state commit after label receipt')
    w._save=failure
    with pytest.raises(OSError):w.advance()
    assert len(c)==1
    resumed=RoundWorkflow(tmp_path,spec=s,backend=b,labels=l)
    resumed.advance();assert resumed.state['round']==1 and len(c)==1


def test_ambiguous_fit_does_not_silently_retrain(tmp_path):
    w,s,b,l,c=setup(tmp_path)
    def failure(**kwargs):b.calls.append(('fit_started',None));raise OSError('unknown fit outcome')
    b.fit=failure
    with pytest.raises(OSError):w.advance()
    resumed=RoundWorkflow(tmp_path,spec=s,backend=b,labels=l)
    with pytest.raises(AmbiguousFit):resumed.advance()
    assert len(b.calls)==1


def test_wrong_label_receipt_cannot_mutate_training_set(tmp_path):
    w,s,b,l,c=setup(tmp_path);ready(w);w.advance()
    l.callback=lambda ids,frozen:{'V0':[0.,0.]}
    with pytest.raises(ValueError,match='exactly'):w.advance()
    assert w.state['labeled_ids']==['L0','L1']


def test_wrong_representation_fit_identity_fails_before_freeze(tmp_path):
    w,s,b,l,c=setup(tmp_path);ready(w)
    b.protocol_hash='f'*64
    with pytest.raises(ValueError,match='representation|recipe'):w.advance()
    assert not c


def test_output_symlink_and_runtime_roots_rejected(tmp_path):
    w,s,b,l,c=setup(tmp_path/'run')
    target=tmp_path/'active';target.mkdir()
    with pytest.raises(ValueError,match='protected'):
        RoundWorkflow(target/'bad',spec=s,backend=b,labels=l,forbidden_output_roots=[target])
    link=tmp_path/'link';link.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        RoundWorkflow(link/'bad',spec=s,backend=b,labels=l)


def test_real_training_backend_is_denied_before_any_training_import():
    backend=object.__new__(QGeoGNNBackend);backend.allow_training=False
    with pytest.raises(PermissionError,match='separate explicit authorization'):
        backend.fit(seed=157,labeled_ids=(),labels={},validation_ids=(),validation_labels={})


def test_bad_roles_or_labels_rejected(tmp_path):
    with pytest.raises(ValueError,match='disjoint'):setup(tmp_path,validation_ids=('L0','V1'))
    w,s,b,l,c=setup(tmp_path/'good')
    with pytest.raises(ValueError,match='finite'):
        LabelBroker(spec=s,initial_labels={'L0':[np.nan,0.],'L1':[1.,2.]},validation_labels=l.validation,reveal=None)


def test_rehashed_state_cannot_inject_unrevealed_U_into_training(tmp_path):
    w,s,b,l,c=setup(tmp_path)
    path=tmp_path/'state.json';state=read_envelope(path)['payload']
    state['labeled_ids'].append('U0');state['candidate_ids'].remove('U0');state['labels']['U0']=[123.,456.]
    write_envelope(path,state)
    with pytest.raises(ValueError,match='lineage'):
        RoundWorkflow(tmp_path,spec=s,backend=b,labels=l)
    assert not b.calls and not c


def test_rehashed_known_label_value_and_changed_backend_recipe_rejected(tmp_path):
    w,s,b,l,c=setup(tmp_path)
    w.state['labels']['L0']=[99.,100.]
    with pytest.raises(ValueError,match='lineage'):w.advance()
    w,s,b,l,c=setup(tmp_path/'recipe')
    b.predictions['V0'][1]=99.
    with pytest.raises(ValueError,match='recipe'):w.advance()
    assert not b.calls


def test_validation_labels_cannot_be_changed_after_run_creation(tmp_path):
    w,s,b,l,c=setup(tmp_path)
    l.validation['V0'][0]=999.
    with pytest.raises(ValueError,match='capability changed'):w.advance()


def test_fixed_trajectory_profile_uses_one_U0_permutation_across_rounds(tmp_path):
    w,s,b,l,c=setup(tmp_path,'random',random_profile='fixed_trajectory',target_budget=5)
    w.run()
    expected=np.random.default_rng(s.acquisition_seed).permutation(list(s.candidate_ids)).tolist()[:3]
    assert [identity for ids,_ in c for identity in ids]==expected
    assert [len(ids) for ids,_ in c]==[2,1]
