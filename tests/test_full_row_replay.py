"""External full-row input/checkpoint regression; never falls back to a worktree.

Only derived seven-column X, frozen zero-y graphs and previously revealed L333
observations are accepted. The external manifest declares every readonly resource
and hash. No source target CSV, optimizer, provider or test-label API is used.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np
import pandas as pd
import pytest
import torch

from src.qgeognn_al.infra.contracts import content_hash
from src.qgeognn_al.infra.resources import SnapshotFile
from src.qgeognn_al.predictor.row import RowDataset, FEATURE_COLUMNS, CONFIRMATION_SEEDS
from src.qgeognn_al.predictor.row_fit import configure_row_runtime
from src.qgeognn_al.predictor.model import load_predictor_checkpoint
from src.qgeognn_al.predictor.inference import predict_outputs
from src.qgeognn_al.al.representations.latent import predictor_state_hash
from src.qgeognn_al.al.agents.validation import BASE, PRED, CW_FIELD, _only, validate_card
from src.qgeognn_al.al.agents.catalog import OBSERVED_FIELDS

pytestmark = pytest.mark.frozen_regression
ENVIRONMENT = 'QGEOGNN_FULL_ROW_REGRESSION_MANIFEST'
SOURCE_COMMIT = 'b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28'
SIX_COLUMNS = tuple(f'pred_{target}_{quantile}_ml' for target in ('V1','V2') for quantile in ('q10','q50','q90'))


def external_file(path):
    """Explicit external snapshots cannot borrow even readonly worktree files."""
    path=Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('full-row fixture needs an absolute path without symlinks')
    for parent in path.parents:
        if (parent/'.git').exists():
            raise ValueError('external regression cannot read from a Git worktree')
    metadata=path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink!=1 or metadata.st_mode & 0o222:
        raise ValueError('full-row fixture must be an independent readonly regular file')
    return path


def resource(record,manifest):
    if set(record)!={'path','root','sha256'}:
        raise ValueError('undeclared full-row resource capabilities')
    if str(Path(record['root']).resolve()) not in manifest['allowed_snapshot_roots']:
        raise ValueError('resource root is not explicitly authorized by the manifest')
    path=external_file(record['path'])
    return SnapshotFile(path,record['sha256'],Path(record['root'])).verify()


@pytest.fixture(scope='module')
def full_row_inputs():
    value=os.getenv(ENVIRONMENT)
    if not value:
        pytest.skip('BLOCKED_EXTERNAL_NOT_CONFIGURED: '+ENVIRONMENT+' requires authorized readonly snapshots')
    path=external_file(value)
    manifest=json.loads(path.read_text())
    if (manifest.get('schema_version')!=1 or manifest.get('source_kind')!='readonly_full_row_regression'
            or manifest.get('source_commit')!=SOURCE_COMMIT):
        raise ValueError('unsupported full-row source identity/schema')
    assert tuple(manifest['feature_columns'])==FEATURE_COLUMNS
    assert {case['seed'] for case in manifest['cases']}==set(CONFIRMATION_SEEDS)
    assert len(manifest['cases'])==5
    frame=pd.read_csv(resource(manifest['features'],manifest))
    assert set(frame.columns)==set(FEATURE_COLUMNS) and len(frame)==4163
    graph_cache=torch.load(resource(manifest['graph_cache'],manifest),map_location='cpu',weights_only=False)
    previous_threads=torch.get_num_threads()
    configure_row_runtime(cpu_threads=2)
    yield manifest,frame,graph_cache
    torch.set_num_threads(previous_threads)


def catalog_record(case,profile,manifest):
    bundle=case['catalogs'][profile]
    freeze_raw=resource(bundle['batch_freeze.json'],manifest).read_bytes()
    anchor=bundle['tracked_freeze_anchor']
    assert anchor['source_commit']==SOURCE_COMMIT
    assert hashlib.sha256(freeze_raw).hexdigest()==anchor['sha256']
    assert hashlib.sha1(b'blob '+str(len(freeze_raw)).encode()+b'\0'+freeze_raw).hexdigest()==anchor['git_blob']
    freeze=json.loads(freeze_raw)
    for name in ('catalog.json','contract.json'):
        assert freeze['artifact_hashes'][name]==bundle[name]['sha256']
    catalog=json.loads(resource(bundle['catalog.json'],manifest).read_text())
    contract=json.loads(resource(bundle['contract.json'],manifest).read_text())
    assert (contract['seed'],contract['round'])==(case['seed'],0)
    assert contract['catalog_sha256']==bundle['catalog.json']['sha256']
    assert set(catalog)=={'candidates','pending','observed','salt'}
    assert len(contract['L_t_ids'])==333 and len(contract['U_t_ids'])==2997
    assert set(contract['L_t_ids']).isdisjoint(contract['U_t_ids'])
    assert set(contract['L_t_ids'])=={card['id'] for card in catalog['observed']}
    assert set(contract['U_t_ids'])=={card['id'] for name in ('candidates','pending') for card in catalog[name]}
    for name in ('candidates','pending','observed'):
        for card in catalog[name]:
            observed=name=='observed'
            _only(card,BASE | (set(OBSERVED_FIELDS) if observed else PRED | {CW_FIELD}),name)
            validate_card(card,observed)
            if observed:
                assert card['record_kind']=='initial_L333'
    return catalog,contract


@pytest.mark.parametrize('seed',CONFIRMATION_SEEDS)
def test_full_row_preparation_and_all_candidate_six_outputs_match_frozen_catalogs(full_row_inputs,tmp_path,seed):
    manifest,frame,cache=full_row_inputs
    case=next(item for item in manifest['cases'] if item['seed']==seed)
    assert case['budget']==333
    partition_path=resource(case['partition'],manifest)
    partition_raw=partition_path.read_bytes()
    assert case['partition_git_anchor']['source_commit']==SOURCE_COMMIT
    assert hashlib.sha1(b'blob '+str(len(partition_raw)).encode()+b'\0'+partition_raw).hexdigest()==case['partition_git_anchor']['git_blob']
    partition=pd.read_csv(partition_path)
    dataset=RowDataset(frame,partition,cache,source_sha256=case['source_dataset_sha256'],
        split_sha256=case['partition']['sha256'],graph_cache_sha256=manifest['graph_cache']['sha256'])
    assert dataset.partition_hash==case['source_split_semantic_sha256']
    catalog,contract=catalog_record(case,'free32_cwaware',manifest)
    observed={card['id']:card for card in catalog['observed']}
    assert set(observed)==set(dataset.l0_ids)
    # This is the only target input: previously revealed initial observations.
    known_l0=np.asarray([[observed[i]['V1_ml'],observed[i]['V2_ml']] for i in dataset.l0_ids],dtype=np.float32)
    prepared=dataset.prepare(known_l0,tmp_path/'scaler.json')
    context=json.loads(resource(case['context'],manifest).read_text())
    graph_receipt=json.loads(resource(case['graphs_receipt'],manifest).read_text())
    assert content_hash(context['contract'])==context['contract_hash']==graph_receipt['contract_hash']
    assert graph_receipt['contract']==context['contract']
    assert graph_receipt['sha256']==case['graphs']['sha256']
    assert context['contract']['source_sha256']==case['source_dataset_sha256']
    assert context['contract']['source_graph_cache_sha256']==manifest['graph_cache']['sha256']
    assert context['contract']['partition_hash']==dataset.partition_hash
    assert asdict(prepared.normalization)==context['contract']['normalization']
    assert prepared.preprocessing==context['contract']['preprocessing']
    assert prepared.preprocessing['target_scale_fit_role']=='l0_labels_only'
    assert prepared.preprocessing['validation_rows_used_for_scaling']==0
    assert prepared.preprocessing['test_rows_used']==0
    original_atom,original_angle=torch.load(resource(case['graphs'],manifest),map_location='cpu',weights_only=False)
    assert len(original_atom)==len(original_angle)==len(prepared.rows)==4163
    assert list(prepared.rows)==frame.sample_id.astype(str).tolist()
    for position,item in enumerate(prepared.rows.values()):
        assert 'y' not in item.atom and 'y' not in item.angle
        assert 'y' in original_atom[position] and torch.count_nonzero(original_atom[position].y).item()==0
        for original,current,exclude in ((original_atom[position],item.atom,{'y'}),(original_angle[position],item.angle,set())):
            assert set(original.keys())-exclude==set(current.keys())
            for key in current.keys():
                if isinstance(original[key],torch.Tensor):
                    assert original[key].dtype==current[key].dtype
                    assert torch.equal(original[key],current[key]),(seed,position,key)
                else:
                    assert original[key]==current[key]
    checkpoint=resource(case['checkpoint'],manifest)
    assert contract['checkpoint_sha256']==case['checkpoint']['sha256']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['normalization']==asdict(prepared.normalization)
    assert payload['preprocessing']==prepared.preprocessing
    assert payload['training_config']['split_sha256']==dataset.partition_hash
    model=load_predictor_checkpoint(checkpoint)
    before=predictor_state_hash(model)
    positions={identity:n for n,identity in enumerate(prepared.rows)}
    candidate_ids=contract['U_t_ids']
    actual=predict_outputs(model,[item.atom for item in prepared.rows.values()],
        [item.angle for item in prepared.rows.values()],[positions[i] for i in candidate_ids],2048)
    assert actual.shape==(2997,6) and predictor_state_hash(model)==before
    for profile in ('free32_cwaware','free32_nocw'):
        original,other=catalog_record(case,profile,manifest)
        assert other['checkpoint_sha256']==case['checkpoint']['sha256']
        assert other['U_t_ids']==candidate_ids
        lookup={card['id']:card for name in ('candidates','pending') for card in original[name]}
        expected=np.asarray([[lookup[i][column] for column in SIX_COLUMNS] for i in candidate_ids],dtype=np.float32)
        np.testing.assert_array_equal(actual,expected)


# Tracked M bytes for already-revealed feedback; no targets are embedded here.
SCIENTIST_FEEDBACK_ANCHORS = {0: {'feedback.json': {'sha256': '12fe3d7fa3f58d16e1adacca6650581e03a124457c47e8b49e762e1ff1cfc30b',
                       'git_blob': '07c4d60cad444ed73943901a6c97536463ab41b7'},
     'label_access_receipt.json': {'sha256': '3f07efe3d8adb111c4579fef73e1609002b56e1276e4f96dcc0936cc3e5acc55',
                                   'git_blob': '0a5ce9342945548a420026d4d2ae3b20d7ffe3ac'}},
 1: {'feedback.json': {'sha256': '69b29f4dcebcbafee2b5346f4de60ba6aa78dd6c90f3250c3d4b7bd04f1354c1',
                       'git_blob': 'b8c2d1408dcb82b4f7ba000c0062162cbdd42682'},
     'label_access_receipt.json': {'sha256': '0ac94103db6d66b2042cb4f621d2b5255878a660746f28d1760a479e47d1a8b7',
                                   'git_blob': 'c814a720df4253296ae54798954bb85eb4c8d723'}}}


def test_recorded_scientist_freeze_reveal_feedback_lineage_has_no_new_truth_or_fit():
    """Reuse the private-agent manifest and tracked feedback sidecars, never a truth store."""
    from src.qgeognn_al.al.agents.recorded import _private_case
    value=os.getenv('QGEOGNN_PRIVATE_AGENT_REPLAY_MANIFEST')
    if not value:
        pytest.skip('BLOCKED_EXTERNAL_NOT_CONFIGURED: private Scientist6101 round0/1 manifest required')
    chosen=None
    for entry in value.split(os.pathsep):
        path=external_file(entry);manifest=json.loads(path.read_text())
        if manifest.get('schema_version')!=2 or manifest.get('source_commit')!=SOURCE_COMMIT:
            raise ValueError('unsupported private recorded source identity')
        cases=sorted((c for c in manifest['cases'] if c['profile']=='cw16_llm16_scientist_v2'
                      and c['seed']==6101 and c['round'] in (0,1)),key=lambda c:c['round'])
        if len(cases)==2:
            chosen=path,cases
            break
    if chosen is None:
        pytest.skip('BLOCKED_EXTERNAL_NOT_CONFIGURED: both Scientist6101 rounds and feedback sidecars required')
    path,cases=chosen
    data={c['round']:_private_case(c,path.parent) for c in cases}
    def original(round_index,name):return json.loads(data[round_index]['files'][name]['text'])
    start=original(0,'contract.json')
    labeled=list(start['L_t_ids']);unlabeled=list(start['U_t_ids'])
    known={c['id']:[c['V1_ml'],c['V2_ml']] for c in original(0,'catalog.json')['observed']}
    assert len(known)==len(labeled)==333 and set(known)==set(labeled)
    for round_index in (0,1):
        contract=original(round_index,'contract.json');catalog=original(round_index,'catalog.json')
        frozen=original(round_index,'batch_freeze.json')
        assert contract['L_t_ids']==labeled and contract['U_t_ids']==unlabeled
        assert {c['id']:[c['V1_ml'],c['V2_ml']] for c in catalog['observed']}==known
        records={}
        for name,anchor in SCIENTIST_FEEDBACK_ANCHORS[round_index].items():
            sidecar=path.parent/'revealed_feedback'/f'round_{round_index:02d}'/name
            if not sidecar.exists():
                pytest.skip('BLOCKED_EXTERNAL_NOT_CONFIGURED: original tracked feedback sidecars required')
            safe=external_file(sidecar)
            raw=SnapshotFile(safe,anchor['sha256'],path.parent).verify().read_bytes()
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==anchor['git_blob']
            records[name]=json.loads(raw)
        feedback,receipt=records['feedback.json'],records['label_access_receipt.json']
        freeze_hash=data[round_index]['files']['batch_freeze.json']['sha256']
        assert feedback['batch_freeze_sha256']==receipt['batch_freeze_sha256']==freeze_hash
        assert receipt['feedback_sha256']==content_hash(feedback)
        assert receipt['audit']['purpose']=='after_acquisition_fit' and receipt['audit']['acquisitions_frozen'] is True
        assert receipt['audit']['requested_ids_hash']==content_hash(sorted(frozen['batch_ids']))
        assert (feedback['seed'],feedback['method'],feedback['round'])==(6101,'cw16_llm16_scientist_v2',round_index)
        assert len(frozen['batch_ids'])==len(set(frozen['batch_ids']))==32
        assert set(frozen['batch_ids'])<=set(unlabeled)
        assert [row['candidate_id'] for row in feedback['records']]==frozen['batch_ids']
        for row,prediction in zip(feedback['records'],frozen['premeasurement_predictions']):
            for target in ('V1','V2'):
                assert row[f'pred_{target}_ml']==prediction[f'pred_{target}_ml']
                assert row[f'error_{target}_ml']==prediction[f'pred_{target}_ml']-row[f'true_{target}_ml']
            known[row['candidate_id']]=[row['true_V1_ml'],row['true_V2_ml']]
        labeled+=frozen['batch_ids']
        unlabeled=[i for i in unlabeled if i not in set(frozen['batch_ids'])]
        assert len(labeled)==len(known)==333+32*(round_index+1)
    assert len(labeled)==397 and set(labeled).isdisjoint(unlabeled)
