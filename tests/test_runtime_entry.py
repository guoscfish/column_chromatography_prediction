"""Unified commands, immutable replay and authorization/label boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from src.qgeognn_al.al.workflow.entry import configuration_plan, report_run
from src.qgeognn_al.al.workflow.state import AwaitingLabels, read_envelope
from src.qgeognn_al.infra.execution import ExecutionGrant
from src.qgeognn_al.infra.contracts import content_hash
from tests.test_workflow import setup, ready

ROOT=Path(__file__).resolve().parents[1]


def test_validate_and_dry_run_have_no_execution_side_effect(tmp_path):
    for command in ('validate','dry_run'):
        result=subprocess.run([sys.executable,'-m','scripts.run_al',command,'--config','configs/offline_synthetic.json',
                               '--output',str(tmp_path/'not_created')],cwd=ROOT,capture_output=True,text=True,check=True)
        value=json.loads(result.stdout)
        assert value['new_training'] is False and value['paid_requests']==0
        assert not (tmp_path/'not_created').exists()


def test_explicit_grant_binds_config_output_and_actions(tmp_path):
    config={'profile':'random32'};output=tmp_path/'run';file=tmp_path/'grant.json'
    with pytest.raises(PermissionError):ExecutionGrant.read(None,config=config,output=output)
    file.write_text(json.dumps({'schema_version':1,'protocol_hash':content_hash(config),
                               'output':str(output),'actions':['training']}))
    grant=ExecutionGrant.read(file,config=config,output=output)
    grant.require('training')
    with pytest.raises(PermissionError):grant.require('reveal')
    with pytest.raises(PermissionError):ExecutionGrant.read(file,config={'profile':'other'},output=output)
    assert not output.exists()


def test_real_run_without_grant_denies_before_loading_private_data(tmp_path):
    config={'schema_version':3,'execution_mode':'qgeognn_row','profile':'random32','outer_seed':157,'target_budget':365,
        'resources':{k:{'path':'missing','root':'missing','sha256':'0'*64} for k in ('source','partition','graph_cache','l0_labels','validation_labels')}}
    config['resources']['reveal_directory']='missing'
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    result=subprocess.run([sys.executable,'-m','scripts.run_al','authorized_real_run','--config',str(path),
                           '--output',str(tmp_path/'run')],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode!=0 and 'separate explicit authorization' in result.stderr
    assert not (tmp_path/'run').exists()


def test_label_availability_does_not_create_ambiguous_reveal_marker(tmp_path):
    workflow,spec,backend,labels,calls=setup(tmp_path/'run')
    ready(workflow);workflow.advance();labels.ready=lambda *args:False
    with pytest.raises(AwaitingLabels):workflow.advance()
    assert not calls and not list((tmp_path/'run').glob('*.reveal_started.json'))
    labels.ready=lambda *args:True
    workflow.advance()
    assert len(calls)==1 and workflow.state['round']==1


def test_selector_cannot_override_frozen_scientific_identity(tmp_path):
    workflow,spec,backend,labels,calls=setup(tmp_path/'run')
    class Selector:
        contract_hash='5'*64
        def select(self,**kwargs):
            return {'scope':'foreign','selected_ids':['U0','U1']}
    workflow.selector=Selector();workflow.state['selector_contract_hash']=workflow.selector.contract_hash
    ready(workflow)
    with pytest.raises(ValueError,match='reserved'):workflow.advance()
    assert not calls and not list((tmp_path/'run').glob('*.freeze.json'))


def test_report_matches_budget_grid_without_reading_test_data(tmp_path):
    workflow,*_=setup(tmp_path/'run');workflow.run()
    result=report_run(tmp_path/'run')
    assert [r['labels'] for r in result['validation_curve']]==[2,4,6]
    assert result['rounds']==2 and result['normalized_aulc']>=0
    before={p.name:p.read_bytes() for p in (tmp_path/'run').iterdir() if p.is_file()}
    report_run(tmp_path/'run')
    assert before=={p.name:p.read_bytes() for p in (tmp_path/'run').iterdir() if p.is_file()}


@pytest.mark.parametrize('kind',['symlink','hardlink'])
def test_atomic_artifact_writes_cannot_follow_foreign_temporary_links(tmp_path,kind):
    import os
    from src.qgeognn_al.al.workflow.state import write_envelope
    from src.qgeognn_al.predictor.training import atomic_json
    foreign=tmp_path/'protected';foreign.write_bytes(b'original identity')
    for function,name in ((write_envelope,'state.json'),(atomic_json,'preprocessing.json')):
        path=tmp_path/name;temporary=tmp_path/(name+'.tmp')
        if kind=='symlink':temporary.symlink_to(foreign)
        else:os.link(foreign,temporary)
        with pytest.raises(ValueError,match='linked'):function(path,{'value':1})
        assert foreign.read_bytes()==b'original identity' and not path.exists()
        temporary.unlink()


def test_owned_multiround_recording_uses_one_workflow_and_rejects_input_drift(tmp_path):
    import copy
    from src.qgeognn_al.al.workflow.replay import ReplayBackend
    from src.qgeognn_al.al.workflow.state import RoundWorkflow,LabelBroker
    from src.qgeognn_al.infra.contracts import BaselineProvenance
    import dataclasses
    source,spec,backend,labels,calls=setup(tmp_path/'source')
    final=source.run();recording={'initial_state_hash':spec.initial_state_hash,'rounds':[]}
    current_l=list(spec.initial_ids);current_u=list(spec.candidate_ids);truth=dict(labels.initial)
    for number in range(3):
        fit=read_envelope(tmp_path/'source'/f'fit_{number:04d}.json')['payload']['fit']
        feedback=read_envelope(tmp_path/'source'/f'feedback_{number:04d}.json')['payload']['validation']
        recording['rounds'].append({'labeled_ids':list(current_l),'candidate_ids':list(current_u),'fit':fit,
            'fit_input_sha256':content_hash({'seed':spec.seed,'labeled_ids':list(current_l),'labels':dict(truth),
                'validation_ids':list(spec.validation_ids),'validation_labels':labels.validation}),
            'validation_input_sha256':content_hash({'validation_ids':list(spec.validation_ids),'validation_labels':labels.validation}),
            'validation_metrics':feedback,'features':{'train':[backend.features[i].tolist() for i in current_l],
                'candidates':[backend.features[i].tolist() for i in current_u],
                'config':{'kind':'latent','dimension':128},
                'provenance':BaselineProvenance('a'*40,spec.dataset_hash,spec.split_hash,fit['checkpoint_hash'],spec.protocol_hash,'synthetic').to_dict()}})
        if number<2:
            chosen=list(calls[number][0]);current_l+=chosen;current_u=[i for i in current_u if i not in chosen]
            truth.update({i:[2.,4.] for i in chosen})
    replay_spec=dataclasses.replace(spec,execution_mode='frozen_replay')
    broker=LabelBroker(spec=replay_spec,initial_labels=labels.initial,validation_labels=labels.validation,
                      reveal=lambda ids,frozen:{i:[2.,4.] for i in ids})
    replay=RoundWorkflow(tmp_path/'replay',spec=replay_spec,backend=ReplayBackend(recording),labels=broker)
    result=replay.run()
    assert result['labeled_ids']==final['labeled_ids'] and result['round']==2
    assert [read_envelope(p)['payload']['selected_ids'] for p in sorted((tmp_path/'source').glob('*.freeze.json'))]==[read_envelope(p)['payload']['selected_ids'] for p in sorted((tmp_path/'replay').glob('*.freeze.json'))]
    before={p.name:p.read_bytes() for p in (tmp_path/'replay').iterdir()}
    assert replay.run()==result and before=={p.name:p.read_bytes() for p in (tmp_path/'replay').iterdir()}
    corrupted=copy.deepcopy(recording);corrupted['rounds'][0]['fit_input_sha256']='0'*64
    failed=RoundWorkflow(tmp_path/'drift',spec=replay_spec,backend=ReplayBackend(corrupted),labels=broker)
    with pytest.raises(ValueError,match='fit inputs'):failed.advance()
    assert not list((tmp_path/'drift').glob('*.freeze.json'))


def test_independent_clone_guard_also_protects_unregistered_git_worktrees(tmp_path):
    from src.qgeognn_al.infra.execution import runtime_path,snapshot_resource
    import hashlib
    foreign=tmp_path/'foreign';foreign.mkdir();(foreign/'.git').write_text('gitdir: elsewhere')
    file=foreign/'snapshot';file.write_text('owned boundary fixture');file.chmod(0o444)
    with pytest.raises(ValueError,match='protected'):runtime_path(foreign/'run')
    with pytest.raises(ValueError,match='independent snapshots'):
        snapshot_resource({'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'root':str(foreign)})
    assert not (foreign/'run').exists()


def test_real_source_identity_rejects_dirty_and_untracked_scientific_code(tmp_path):
    from src.qgeognn_al.infra.execution import scientific_source_identity
    repo=tmp_path/'project';repo.mkdir();(repo/'src').mkdir();file=repo/'src/model.py';file.write_text('loss = 1\n')
    def git(*args):return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
    git('init');git('config','user.name','Owned fixture');git('config','user.email','fixture@example.invalid')
    git('add','src/model.py');git('-c','commit.gpgsign=false','commit','-m','fixture')
    before=scientific_source_identity(repo)
    (repo/'notes.md').write_text('documentation changes are allowed')
    assert scientific_source_identity(repo)==before
    file.write_text('loss = 2\n')
    with pytest.raises(ValueError,match='dirty code'):scientific_source_identity(repo)
    file.write_text('loss = 1\n');(repo/'src/other.py').write_text('hidden = 1')
    with pytest.raises(ValueError,match='untracked'):scientific_source_identity(repo)


def test_retired_generic_fit_scaffold_cannot_start_alternative_training():
    from src.qgeognn_al.al.workflow.backends import QGeoGNNBackend
    backend=object.__new__(QGeoGNNBackend);backend.allow_training=True
    with pytest.raises(NotImplementedError,match='QGeoGNNRowBackend'):
        backend.fit(seed=157,labeled_ids=(),labels={},validation_ids=(),validation_labels={})
