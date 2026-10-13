"""Boundary and pairing tests; model equivalence uses synthetic labels only."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('row_aligned', HERE / 'run_phase1.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fold_ids(ids, cols, folds):
    return [(set(np.asarray(ids)[t]), set(np.asarray(ids)[v])) for t, v in folds]


def test_complete_deterministic_row_folds_and_permutation_invariance():
    ids = [f'{c}_{i}' for c in r.TARGETS for i in range(13)]
    cols = [c for c in r.TARGETS for _ in range(13)]
    first = r.row_target_folds(ids, cols)
    assert fold_ids(ids, cols, first) == fold_ids(ids, cols, r.row_target_folds(ids, cols))
    permutation = np.random.default_rng(3).permutation(len(ids))
    pi, pc = np.asarray(ids)[permutation], np.asarray(cols)[permutation]
    assert fold_ids(ids, cols, first) == fold_ids(pi, pc, r.row_target_folds(pi, pc))
    assert fold_ids(ids, cols, first) != fold_ids(ids, cols, r.row_target_folds(ids, cols, seed=43))
    assert sorted(np.concatenate([v for _, v in first])) == list(range(len(ids)))
    for train, valid in first:
        assert not set(np.asarray(ids)[train]) & set(np.asarray(ids)[valid])
        for c in r.TARGETS:
            assert sum(np.asarray(cols)[train] == c) and sum(np.asarray(cols)[valid] == c)
        assert set(train) | set(valid) == set(range(len(ids)))


@pytest.mark.parametrize('ids,cols', [([], []), (['a'], ['25g']),
    (['a']*10, ['25g']*5+['40g']*5), ([str(i) for i in range(8)], ['25g']*4+['40g']*4)])
def test_invalid_or_shared_id_folds_fail_closed(ids, cols):
    with pytest.raises(ValueError):
        r.row_target_folds(ids, cols)


def test_row_allows_same_molecule_but_not_same_identity_across_sides():
    data = {}
    for c in r.old.TASKS:
        data[c] = {'frame': pd.DataFrame({'sample_id': [c+'_a', c+'_b'], 'canonical_smiles': ['C', 'C']}),
                   'positions': {'gradient_train': [0, 1] if c != '4g' else [0], 'source_validation': [1]}}
    train, valid = {c: [0] for c in data}, {c: [1] for c in data}
    r.authorize_fit(data, train, valid)
    data['40g']['frame'].loc[1, 'sample_id'] = '25g_a'
    with pytest.raises(RuntimeError, match='same sample_id'):
        r.authorize_fit(data, train, valid)


def test_outer_identity_cannot_enter_train_or_validation():
    data = {c: {'frame': pd.DataFrame({'sample_id': [c+'_a', c+'_b', c+'_outer']}),
                'positions': {'gradient_train': [0, 1] if c != '4g' else [0], 'source_validation': [1]}}
            for c in r.old.TASKS}
    for side in ['train', 'valid']:
        train, valid = {c: [0] for c in data}, {c: [1] for c in data}
        (train if side == 'train' else valid)['25g'] = [2]
        with pytest.raises(RuntimeError, match='unauthorized'):
            r.authorize_fit(data, train, valid)


def test_authorized_reader_skips_forbidden_label_cells(tmp_path):
    path = tmp_path / 'truth.csv'
    path.write_text('sample_id,V1_ml,V2_ml,t1,t2\ntrain,1,2,999,999\ntest,FORBIDDEN,FORBIDDEN,999,999\n')
    actual = r.old._read_authorized_truth(path, ['train'])
    np.testing.assert_array_equal(actual, [[1, 2]])


@pytest.mark.parametrize('column', ['25g', '40g'])
def test_loader_explicit_feature_whitelist_and_no_outer_graphs(monkeypatch, column):
    torch.set_num_threads(1)
    read_csv = pd.read_csv
    feature_path = r.old.PARENT/f'filtered_features_{column}.csv'
    feature_calls, label_requests = [], []
    def checked_csv(path, *args, **kwargs):
        if Path(path) == feature_path:
            assert kwargs.get('usecols') == r.old.FEATURES
            feature_calls.append(True)
        return read_csv(path, *args, **kwargs)
    def synthetic_truth(path, ids):
        label_requests.append(set(ids))
        return np.column_stack([np.arange(len(ids))+1., np.arange(len(ids))*2.+3.])
    monkeypatch.setattr(pd, 'read_csv', checked_csv)
    monkeypatch.setattr(r.old, '_read_authorized_truth', synthetic_truth)
    data = r.authorized_data(column, {})
    assert feature_calls == [True]
    expected = set(data['roles']['gradient_train'])
    assert label_requests == [expected]
    assert set(data['frame'].sample_id) == expected
    assert not set(data['frame'].sample_id) & set(data['roles']['validation']+data['roles']['test'])
    assert len(data['atoms']) == len(data['angles']) == len(expected)


def test_real_frozen_identity_schedule_and_pairing():
    schedule = r.row_schedule()
    assert set(schedule.outer_seed) == set(r.old.SEEDS)
    for seed in r.old.SEEDS:
        subset = schedule.loc[schedule.outer_seed.eq(seed)]
        gt = subset.loc[subset.role.eq('gradient_train')]
        folds = r.row_target_folds(gt.sample_id, gt.column)
        for train, valid in folds:
            assert not set(gt.iloc[train].sample_id) & set(subset.loc[subset.role.ne('gradient_train')].sample_id)
            assert not set(gt.iloc[valid].sample_id) & set(subset.loc[subset.role.ne('gradient_train')].sample_id)
    for fold in range(1, 6):
        assert r.initialization_audit(r.PILOT_SEED+fold*1009)['heads_exact']


@pytest.fixture(scope='module')
def synthetic_data():
    torch.set_num_threads(1)
    preprocessing = torch.load(r.old.source_path(), map_location='cpu', weights_only=False)['preprocessing']
    cache = torch.load(r.old.SOURCE_GRAPH_CACHE, map_location='cpu', weights_only=False)
    frame = pd.read_csv(r.old.SOURCE_DATA, usecols=r.old.FEATURES).iloc[:12].reset_index(drop=True)
    # Arbitrary synthetic endpoints. No observed endpoint/time cells are read.
    frame['V1_ml'] = np.arange(12, dtype=float)+1
    frame['V2_ml'] = np.arange(12, dtype=float)*2+3
    atoms, angles = r.build_model_data(frame, cache, None, preprocessing['scaler'])
    r.old.cache_frozen_prefix(r.load_predictor_checkpoint(r.old.source_path()), atoms, angles)
    data = {}
    for c in r.old.TASKS:
        cf = frame.copy(); cf.sample_id = c + '_' + cf.sample_id
        data[c] = {'frame': cf, 'atoms': atoms, 'angles': angles,
            'truth': cf[['V1_ml', 'V2_ml']].to_numpy(),
            'positions': {'gradient_train': list(range(8 if c == '4g' else 12)), 'source_validation': list(range(8, 12))}}
    return data


@pytest.mark.parametrize('arm', ['A1', 'A2'])
def test_original_fit_recipe_checkpoint_equivalence_synthetic_only(synthetic_data, tmp_path, monkeypatch, arm):
    data = synthetic_data
    train, valid = {c: list(range(8)) for c in data}, {c: list(range(8, 12)) for c in data}
    monkeypatch.setattr(r, 'STUDY', tmp_path)
    monkeypatch.setattr(r.old, 'STUDY', tmp_path)
    (tmp_path/'protocol.json').write_text('{}')
    monkeypatch.setattr(r.old, 'TRAINING', {**r.old.TRAINING, 'maximum_epochs': 1})
    monkeypatch.setattr(r.old, 'authorize_fit', r.authorize_fit)
    new_run, old_run = tmp_path/'new', tmp_path/'old'
    new_run.mkdir(); old_run.mkdir()
    new_model, fitted, _ = r.fit_model(data, train, valid, arm, 17, new_run)
    old_model, _, _ = r.old.fit_model(data, train, valid, arm, 17, old_run)
    assert fitted['optimizer_steps'] == fitted['optimizer_steps_at_best'] == 1
    for key, value in new_model.state_dict().items():
        torch.testing.assert_close(value, old_model.state_dict()[key], rtol=0, atol=0)
    new, old = pd.read_csv(new_run/'history.csv'), pd.read_csv(old_run/'history.csv')
    for name in old.columns:
        np.testing.assert_array_equal(new[name], old[name])


def test_graph_input_digest_ignores_labels_and_identity_but_includes_task(synthetic_data):
    atom, angle = synthetic_data['4g']['atoms'][0], synthetic_data['4g']['angles'][0]
    changed = atom.clone(); changed.y.fill_(999); changed.canonical_position.fill_(999)
    assert r.input_hash(atom, angle) == r.input_hash(changed, angle)
    changed.edge_attr[0, -1] += 1
    assert r.input_hash(atom, angle) != r.input_hash(changed, angle)
    assert r.input_hash(atom, angle, task='25g') != r.input_hash(atom, angle, task='40g')
