from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
from src.qgeognn_al.al.acquisitions import select
from src.qgeognn_al.al.acquisitions.lcmd import lcmd_tp_select
from src.qgeognn_al.al.acquisitions.maxdet import conditional_gradient_maxdet
from src.qgeognn_al.al.acquisitions.ivr import conditional_batch_ivr
from src.qgeognn_al.al.acquisitions.hybrid import hybrid_select
from src.qgeognn_al.al.acquisitions.random import random_control_positions, random_control_seed
from src.qgeognn_al.al.representations import Authorization, FeatureSnapshot, RepresentationConfig
from src.qgeognn_al.infra.contracts import BaselineProvenance, canonical_bytes
from tests.support import historical_reference as legacy
from tests.support.fixtures import ALGORITHM_RECORDS, inputs, snapshot, record


def _fixture_file(value):
    """Missing opt-in private inputs are BLOCKED; invalid inputs still fail closed."""
    path = Path(value)
    assert not path.is_symlink(), 'frozen regression fixture must be a regular owned file'
    if not path.is_file():
        pytest.skip('BLOCKED: authorized frozen regression fixture is unavailable: ' + str(path))
    return path


@pytest.mark.frozen_regression
@pytest.mark.parametrize('seed', [157, 887, 2357, 6101, 12203])
def test_five_frozen_trained_checkpoints_raw_cw_and_latent_on_label_free_u_features(seed):
    """B migration only: 8 known L0 + 8 U inputs; no validation/test targets.

    This is an exact numerical regression of fixed source/model inputs, not a
    replay of an entire historical AL round. True K3 ensemble is not available;
    the Hybrid signal here explicitly repeats one model to exercise tied ranks.
    """
    import csv
    import torch
    import pandas as pd
    from src.qgeognn_al.predictor.model import load_predictor_checkpoint
    from src.qgeognn_al.predictor.data import build_model_data
    from src.qgeognn_al.predictor.training import loader_pair
    from src.qgeognn_al.al.representations.gradient import extract_q50_gradient_sketches, extract_linear_output_gradient_sketches
    from src.qgeognn_al.al.representations.center_width import center_width_transform
    from src.qgeognn_al.al.representations.latent import GraphRow, extract_latent, predictor_state_hash
    path = os.getenv('QGEOGNN_READONLY_REGRESSION_MANIFEST')
    if not path: pytest.skip('BLOCKED: explicit authorized readonly real-checkpoint snapshot manifest required')
    manifest = json.loads(_fixture_file(path).read_text())
    item = next(row for row in manifest['seeds'] if row['seed'] == seed)
    feature_file = Path(item['label_free_u0_features']); authorized_file = Path(item['authorized_rows'])
    checkpoint_file = Path(item['checkpoint']['snapshot'])
    for file, digest in [(feature_file, item['label_free_u0_features_sha256']),
                         (authorized_file, item['authorized_rows_sha256']),
                         (checkpoint_file, item['checkpoint']['sha256'])]:
        _fixture_file(file)
        assert file.stat().st_mode & 0o222 == 0
        assert hashlib.sha256(file.read_bytes()).hexdigest() == digest
    feature_columns = ['sample_id', 'canonical_smiles', 'Density g/ml', 'V/ul', 'loading solvent',
                       'Volume of loading solvent/ul', 'PE/EA', 'canonical_index', 'role']
    known = pd.read_csv(authorized_file, usecols=feature_columns)
    known = known.loc[known.role.eq('l0')].sort_values('canonical_index')
    pool = pd.read_csv(feature_file)
    assert not {'V1_ml', 'V2_ml', 'V1_true', 'V2_true'} & set(pool.columns)
    assert set(pool.role) == {'u0'}
    chosen = pd.concat([known.iloc[:8], pool.iloc[:8]], ignore_index=True)
    chosen['V1_ml'] = 0.; chosen['V2_ml'] = 0.  # Synthetic graph sentinels, never candidate truth.
    initial_targets = []
    with authorized_file.open() as stream:
        for row in csv.DictReader(stream):
            if row['role'] == 'l0':
                initial_targets.append([float(row['V1_ml']), float(row['V2_ml'])])
    assert len(initial_targets) == 333
    cached = next(row for row in manifest['source_files'] if row.get('snapshot', '').endswith('/graph_cache_4g.pt'))
    graph_file = Path(cached['snapshot'])
    _fixture_file(graph_file)
    assert hashlib.sha256(graph_file.read_bytes()).hexdigest() == cached['sha256']
    cache = torch.load(graph_file, map_location='cpu', weights_only=False)
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    atom, angle = build_model_data(chosen, cache, None, checkpoint['preprocessing']['scaler'])
    for graph in atom:
        del graph.y
    torch.set_num_threads(1)
    model = load_predictor_checkpoint(checkpoint_file); state_before = predictor_state_hash(model)
    indices = list(range(16)); scales = tuple(checkpoint['preprocessing']['target_scales'][target] for target in ['V1', 'V2'])
    old = legacy.gradient()
    sketch_seed = seed + 4_000_037
    old_raw = old.extract_q50_gradient_sketches(model, atom, angle, indices, scales, dimension=512, sketch_seed=sketch_seed)
    new_raw = extract_q50_gradient_sketches(model, atom, angle, indices, scales, dimension=512, sketch_seed=sketch_seed)
    np.testing.assert_array_equal(new_raw.features, old_raw.features)
    assert new_raw.parameter_audit == old_raw.parameter_audit
    transform, _ = center_width_transform(np.array(initial_targets))
    old_cw = old.extract_linear_output_gradient_sketches(model, atom, angle, indices, transform, dimension=512, sketch_seed=sketch_seed)
    new_cw = extract_linear_output_gradient_sketches(model, atom, angle, indices, transform, dimension=512, sketch_seed=sketch_seed)
    np.testing.assert_array_equal(new_cw.features, old_cw.features)
    ids = tuple(chosen.sample_id.astype(str)); auth = Authorization(ids[:8], ids[8:])
    canonical_sha = next(row['sha256'] for row in manifest['source_files'] if row['source'].endswith('canonical_4g.csv'))
    provenance = BaselineProvenance(manifest['scientific_base_commit'], canonical_sha, item['partition_sha256'],
                                    item['checkpoint']['sha256'], item['partition_sha256'], 'authorized_snapshot')
    rows = [GraphRow(identity, a, b) for identity, a, b in zip(ids, atom, angle)]
    latent = extract_latent(model, expected_state_hash=state_before, authorization=auth, rows=rows,
                            provenance=provenance, batch_size=16)
    frozen_latent = legacy.load('src/qgeognn_al/active_learning_v2/coverage.py', ['extract_representations'],
                                injected={'torch': torch, 'loader_pair': old.loader_pair})
    # loader_pair is supplied to the frozen gradient module; it is not a migrated dependency.
    before_h = frozen_latent.extract_representations(model, atom, angle, indices, batch_size=16)
    np.testing.assert_array_equal(np.vstack([latent.train, latent.candidates]), before_h)
    for name, values in [('raw', new_raw.features), ('CW', new_cw.features)]:
        features = FeatureSnapshot(authorization=auth, train_ids=auth.train_ids, candidate_ids=auth.candidate_ids,
                                   train_features=values[:8], candidate_features=values[8:],
                                   config=RepresentationConfig('gradient' if name == 'raw' else 'center_width', 512), predictor=provenance)
        for method in ['lcmd', 'maxdet', 'ivr']:
            before_x = old_raw.features if name == 'raw' else old_cw.features
            if method == 'lcmd':
                previous = legacy.lcmd().lcmd_tp_select(before_x[8:], before_x[:8], 3)
                positions = previous.selected_pool_positions.tolist(); trace = list(previous.trace)
            elif method == 'maxdet':
                previous = legacy.maxdet().conditional_gradient_maxdet(before_x, np.arange(8), np.arange(8, 16), 3)
                positions = previous.selected_candidate_positions.tolist(); trace = list(previous.trace)
            else:
                previous = legacy.ivr().conditional_batch_ivr(before_x, np.arange(8), np.arange(8, 16), 3)
                positions = previous['selected_candidate_positions']; trace = previous['trace']
            result = select(features, acquisition=method, budget=3, seed=seed)
            assert result.selected_ids == tuple(auth.candidate_ids[position] for position in positions)
            assert [{k:v for k,v in row.items() if k != 'sample_id'} for row in result.trace] == trace
    a, b = next(zip(*loader_pair(atom, angle, indices, batch_size=16)))
    with torch.no_grad(): prediction = model(a, b).cpu().numpy()[8:]
    ensemble = np.stack([prediction]*3)
    previous = legacy.hybrid().v2_hybrid_select(before_h, ensemble, 8, scales, 3)
    result = select(latent, acquisition='hybrid', budget=3, seed=seed,
                    numerical_signals={'candidate_ids': auth.candidate_ids, 'ensemble_predictions': ensemble, 'target_scales': scales})
    assert result.selected_ids == tuple(auth.candidate_ids[position] for position in previous['selected_positions'])
    assert predictor_state_hash(model) == state_before
    assert all(parameter.grad is None for parameter in model.parameters())
    record('frozen_trained_representation', f'seed={seed}/L333/subsample_8L0_8U',
           [old_raw.audit['sketch_mapping_sha256']], [new_raw.audit['sketch_mapping_sha256']],
           {'parameter_count': sum(p.numel() for p in model.parameters()), 'raw_512_exact': True,
            'CW_512_exact': True, 'latent_128_exact': True, 'U_truth_converted': 0,
            'checkpoint_sha256': item['checkpoint']['sha256'], 'hybrid_ensemble': 'synthetic repeats of one frozen model; not a true K3 regression'})



@pytest.mark.frozen_regression
@pytest.mark.parametrize('round_index', [0, 6, 20])
def test_full_frozen_universe_replays_actual_lcmd_and_other_numerical_kernels(round_index):
    """Full 3330-row frozen numeric inputs; historical LCMD IDs exactly replay.

    Historical CSV scores use prespecified rtol=1e-12/atol=1e-8 to accommodate
    serialization/platform BLAS. Old/new kernels in the same process must be
    exactly equal. Counterfactual MaxDet/IVR comparisons are not AL experiments.
    """
    import csv
    path = os.getenv('QGEOGNN_FROZEN_ACQUISITION_MANIFEST')
    if not path: pytest.skip('BLOCKED: explicit readonly frozen acquisition feature manifest required')
    manifest = json.loads(_fixture_file(path).read_text()); item = next(row for row in manifest['records'] if row['round'] == round_index)
    root = Path(item['directory'])
    for relative, meta in item['files'].items():
        _fixture_file(root/relative)
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == meta['sha256']
        assert (root/relative).stat().st_mode & 0o222 == 0
    contract = json.loads((root/'acquisition_artifacts/current_gradient_features.npz.contract.json').read_text())['contract']
    assert contract['test_truth_access_count'] == 0 and contract['dimension'] == 512
    with np.load(root/'acquisition_artifacts/current_gradient_features.npz', allow_pickle=False) as archive:
        features = archive['features']; positions = archive['canonical_indices']
    labeled_positions = np.load(root/'labeled_indices.npy', allow_pickle=False)
    pool_positions = np.load(root/'unlabeled_indices.npy', allow_pickle=False)
    l_ids = tuple(np.load(root/'labeled_ids.npy', allow_pickle=False).tolist())
    u_ids = tuple(np.load(root/'unlabeled_ids.npy', allow_pickle=False).tolist())
    np.testing.assert_array_equal(positions, np.r_[labeled_positions, pool_positions])
    assert positions.tolist() == contract['ordered_indices']
    assert hashlib.sha256(canonical_bytes(sorted(l_ids))).hexdigest() == contract['L_t_ids_hash']
    assert hashlib.sha256(canonical_bytes(sorted(u_ids))).hexdigest() == contract['U_t_ids_hash']
    assert len(features) == 3330 and len(l_ids) == item['budget']
    count = len(l_ids); auth = Authorization(l_ids, u_ids)
    snapshot_x = FeatureSnapshot(authorization=auth, train_ids=l_ids, candidate_ids=u_ids,
                                  train_features=features[:count], candidate_features=features[count:],
                                  config=RepresentationConfig('gradient', 512),
                                  predictor=BaselineProvenance(legacy.FROZEN_COMMIT, '2'*64, '3'*64,
                                                              contract['evaluation_checkpoint_sha256'], contract['evaluation_checkpoint_state_hash']))
    old_lcmd = legacy.lcmd().lcmd_tp_select(features[count:], features[:count], 32)
    result = select(snapshot_x, acquisition='lcmd', budget=32, seed=157)
    selected = tuple(u_ids[position] for position in old_lcmd.selected_pool_positions)
    assert result.selected_ids == selected
    with (root/'acquisition_artifacts/selected_next_batch.csv').open() as stream:
        historical = list(csv.DictReader(stream))
    assert [int(row['selection_order']) for row in historical] == list(range(32))
    assert tuple(row['sample_id'] for row in historical) == selected
    assert [int(row['previous_U_t_position']) for row in historical] == old_lcmd.selected_pool_positions.tolist()
    with (root/'acquisition_artifacts/lcmd_trace.csv').open() as stream:
        historical_trace = list(csv.DictReader(stream))
    for before, frozen_row in zip(old_lcmd.trace, historical_trace):
        for key, value in before.items():
            np.testing.assert_allclose(value, float(frozen_row[key]), rtol=1e-12, atol=1e-8)
    assert [{k:v for k,v in row.items() if k != 'sample_id'} for row in result.trace] == list(old_lcmd.trace)
    for method in ['maxdet', 'ivr']:
        # The original IVR outer-training universe is the fixed initial L0+U0 order.
        first = next(row for row in manifest['records'] if row['round'] == 0)
        first_root = Path(first['directory'])
        _fixture_file(first_root/'labeled_ids.npy')
        _fixture_file(first_root/'unlabeled_ids.npy')
        reference = tuple(np.load(first_root/'labeled_ids.npy', allow_pickle=False).tolist()+
                          np.load(first_root/'unlabeled_ids.npy', allow_pickle=False).tolist())
        if method == 'maxdet':
            old_result = legacy.maxdet().conditional_gradient_maxdet(features, np.arange(count), np.arange(count, len(features)), 32)
            expected_positions = old_result.selected_candidate_positions.tolist(); old_trace = list(old_result.trace)
            new_result = select(snapshot_x, acquisition=method, budget=32, seed=157)
        else:
            lookup = {identity:i for i, identity in enumerate(l_ids+u_ids)}
            fixed_lookup = {identity:i for i, identity in enumerate(reference)}
            fixed = features[[lookup[identity] for identity in reference]]
            old_result = legacy.ivr().conditional_batch_ivr(fixed,
                          np.array([fixed_lookup[identity] for identity in l_ids]),
                          np.array([fixed_lookup[identity] for identity in u_ids]), 32)
            expected_positions = old_result['selected_candidate_positions']; old_trace = old_result['trace']
            new_result = select(snapshot_x, acquisition=method, budget=32, seed=157,
                                 numerical_signals={'reference_ids': reference})
        assert new_result.selected_ids == tuple(u_ids[position] for position in expected_positions)
        assert [{k:v for k,v in row.items() if k != 'sample_id'} for row in new_result.trace] == old_trace
    record('full_frozen_LCMD', f'seed=157/L{item["budget"]}->L{item["budget"]+32}',
           list(selected), list(result.selected_ids), result.trace)
    ALGORITHM_RECORDS[-1].update(input_features_sha256=item['files']['acquisition_artifacts/current_gradient_features.npz']['sha256'],
                       evaluation_checkpoint_sha256=contract['evaluation_checkpoint_sha256'],
                       source_round=round_index, representation_refresh='current_fit',
                       centers_L_t_ids_hash=contract['L_t_ids_hash'],
                       universe_rows=3330, historical_LCMD_selection_replayed=True,
                       MaxDet_IVR='counterfactual numerical equivalence only; no new actual selection/freeze/reveal',
                       historical_trace_tolerance={'rtol':1e-12, 'atol':1e-8})



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
from tests.support import historical_reference as legacy_reference

ROOT=Path(__file__).resolve().parents[1]
RECORDS=[]



@pytest.fixture(scope='module')
def snapshots():
    value=os.getenv('QGEOGNN_REGRESSION_MANIFEST')
    if not value:pytest.skip('BLOCKED: independent authorized real checkpoint/cache fixtures not configured')
    path=_fixture_file(value);manifest=json.loads(path.read_text())
    _fixture_file(manifest['snapshot_manifest'])
    source=SnapshotFile(Path(manifest['snapshot_manifest']),manifest['snapshot_manifest_sha256'],path.parent).verify()
    metadata=json.loads(source.read_text())
    assert metadata['label_access']['u0_values_converted']==0 and metadata['label_access']['test_values_converted']==0
    yield path.parent,manifest,metadata
    destination=os.getenv('QGEOGNN_MODEL_EQUIVALENCE_OUTPUT')
    if destination:Path(destination).write_text(json.dumps({'status':'NEW_OFFLINE_MEASURED','frozen_code_commit':manifest['source_code_commit'],'training_performed':False,'test_or_u_truth_read':False,'records':RECORDS},indent=2)+'\n')


@pytest.mark.frozen_regression
@pytest.mark.parametrize('seed',[157,887,2357,6101,12203])
def test_real_checkpoint_six_outputs_hidden_and_rebuilt_graph_cache(seed,snapshots):
    root,manifest,metadata=snapshots
    entry=next(x for x in manifest['fixtures'] if x['seed']==seed)
    for key in ['checkpoint','graphs','expected']:
        _fixture_file(entry[key])
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
    _fixture_file(config['authorized_rows'])
    SnapshotFile(Path(config['authorized_rows']),config['authorized_rows_sha256'],root).verify()
    frame=pd.read_csv(config['authorized_rows'])
    assert set(frame['role'])<={'l0','validation'}
    cache_record=next(x for x in metadata['source_files'] if x.get('source','').endswith('graph_cache_4g.pt'))
    _fixture_file(cache_record['snapshot'])
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
