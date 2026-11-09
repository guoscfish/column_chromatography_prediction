"""Strict real frozen-checkpoint/cache regression, only authorized L0/validation."""
import ast
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader
from src.qgeognn_al.predictor import load_predictor_checkpoint
from src.qgeognn_al.predictor.data import build_model_data
from src.qgeognn_al.infra.resources import SnapshotFile
import legacy_reference

ROOT=Path(__file__).resolve().parents[2]
RECORDS=[]


@pytest.fixture(scope='module')
def snapshots():
    value=os.getenv('QGEOGNN_REGRESSION_MANIFEST')
    if not value:pytest.skip('BLOCKED: independent authorized real checkpoint/cache fixtures not configured')
    path=Path(value);manifest=json.loads(path.read_text())
    source=SnapshotFile(Path(manifest['snapshot_manifest']),manifest['snapshot_manifest_sha256'],path.parent).verify()
    metadata=json.loads(source.read_text())
    assert metadata['label_access']['u0_values_converted']==0 and metadata['label_access']['test_values_converted']==0
    yield path.parent,manifest,metadata
    destination=os.getenv('QGEOGNN_MODEL_EQUIVALENCE_OUTPUT')
    if destination:Path(destination).write_text(json.dumps({'status':'NEW_OFFLINE_MEASURED','frozen_code_commit':manifest['source_code_commit'],'training_performed':False,'test_or_u_truth_read':False,'records':RECORDS},indent=2)+'\n')


@pytest.mark.parametrize('seed',[157,887,2357,6101,12203])
def test_real_checkpoint_six_outputs_hidden_and_rebuilt_graph_cache(seed,snapshots):
    root,manifest,metadata=snapshots
    entry=next(x for x in manifest['fixtures'] if x['seed']==seed)
    for key in ['checkpoint','graphs','expected']:
        SnapshotFile(Path(entry[key]),entry[key+'_sha256'],root).verify()
    model=load_predictor_checkpoint(Path(entry['checkpoint']));model.eval()
    checkpoint=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
    assert sum(p.numel() for p in model.parameters())==458952
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in checkpoint['model_state_dict'].items())
    fixture=torch.load(entry['graphs'],map_location='cpu',weights_only=False)
    assert set(fixture['roles'])<={'l0','validation'}
    atom,angle=fixture['atom'],fixture['angle'];expected=np.load(entry['expected'])
    a,b=next(zip(DataLoader(atom,batch_size=len(atom),shuffle=False),DataLoader(angle,batch_size=len(angle),shuffle=False)))
    with torch.no_grad():output=model(a,b).numpy();hidden=model.extract_representation(a,b).numpy()
    np.testing.assert_allclose(output,expected['outputs'],**entry['tolerance'])
    np.testing.assert_allclose(hidden,expected['hidden'],**entry['tolerance'])
    config=next(x for x in metadata['seeds'] if x['seed']==seed)
    SnapshotFile(Path(config['authorized_rows']),config['authorized_rows_sha256'],root).verify()
    frame=pd.read_csv(config['authorized_rows'])
    assert set(frame['role'])<={'l0','validation'}
    cache_record=next(x for x in metadata['source_files'] if x.get('source','').endswith('graph_cache_4g.pt'))
    path=SnapshotFile(Path(cache_record['snapshot']),cache_record['sha256'],root).verify()
    cache=torch.load(path,map_location='cpu',weights_only=False)
    rebuilt_a,rebuilt_b=build_model_data(frame,cache,pd.DataFrame(),checkpoint['preprocessing']['scaler'])
    positions={str(value):i for i,value in enumerate(frame['sample_id'])}
    for old_a,old_b,identity in zip(atom,angle,fixture['sample_ids']):
        new_a,new_b=rebuilt_a[positions[identity]],rebuilt_b[positions[identity]]
        for name in ['x','edge_index','edge_attr','y','canonical_position']:
            assert torch.equal(old_a[name],new_a[name]),(seed,identity,name)
        for name in ['edge_index','edge_attr']:
            assert torch.equal(old_b[name],new_b[name]),(seed,identity,name)
    RECORDS.append({'seed':seed,'budget':333,'allowed_roles':['l0','validation'],'rows':len(atom),
                    'output_max_abs_difference':float(np.max(np.abs(output-expected['outputs']))),
                    'hidden_max_abs_difference':float(np.max(np.abs(hidden-expected['hidden']))),
                    'state_values_exact':True,'raw_graphs_rebuilt_bitwise_equal':True,'checkpoint_sha256':entry['checkpoint_sha256']})


@pytest.mark.parametrize('old,new,names',[
 ('src/qgeognn_al/training/predictor.py','src/qgeognn_al/predictor/training.py',['target_loss','train_source','fit_source_preprocessing']),
 ('src/qgeognn_al/data.py','src/qgeognn_al/predictor/data.py',['build_model_data','minmax_transform','fit_standardizer']),
 ('src/qgeognn_al/models/qgeognn_v2.py','src/qgeognn_al/predictor/model.py',['GeometryBackbone','QGeoGNNV2','build_predictor','predictor_checkpoint','load_predictor_checkpoint'])])
def test_moved_scientific_bodies_are_ast_identical_to_frozen_commit(old,new,names):
    before=ast.parse(legacy_reference.source(old));after=ast.parse((ROOT/new).read_text())
    a={n.name:n for n in before.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    b={n.name:n for n in after.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name in names:assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False)
