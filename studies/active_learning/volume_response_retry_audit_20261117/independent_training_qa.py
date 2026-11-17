#!/usr/bin/env python3
"""Independently verify frozen controls using identities and covariates only, no fits."""
from pathlib import Path
import argparse
import hashlib
import json
import lzma
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'studies/active_learning/nocw_free_selection_diagnostics_20261101'))
from condition_metrics import _canonical_conditions


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    sources = {}
    def js(path):
        path = Path(path)
        sources[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())
    def frame(path):
        path = Path(path)
        sources[str(path.relative_to(ROOT))] = digest(path)
        return pd.read_csv(path, dtype={'sample_id': str})
    gate = js(HERE / 'DATA_FEASIBILITY.json')
    for name, expected in gate['training_code_sha256'].items():
        assert digest(HERE / name) == expected
        sources[str((HERE / name).relative_to(ROOT))] = expected
    old = ROOT / 'studies/active_learning/decision_mechanism_followup_20261117'
    plan = frame(old / 'model_control_holdouts.csv')
    path = ROOT / 'studies/active_learning/llm_decision_behavior_audit_20261116/states.json.xz'
    sources[str(path.relative_to(ROOT))] = digest(path)
    snapshot = json.loads(lzma.decompress(path.read_bytes()))
    records, overlaps = [], []
    for seed in (157, 887, 2357, 6101, 12203):
        identity = js(HERE / f'data/seed_{seed}/identity.json')
        metadata = frame(HERE / f'data/seed_{seed}/metadata.csv').set_index('sample_id')
        partition = frame(ROOT / f'studies/active_learning/qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv')
        context = js(ROOT / f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/context.json')['contract']
        state = next(b for b in snapshot['states'] if b['method'] == 'new_free_cw_sol61_xhigh' and b['seed'] == seed and b['L_before'] == 1005)
        assert identity['history_ids'] == state['labeled_ids']
        history, train, check, validation = map(set, (identity['history_ids'], identity['train_ids'], identity['check_ids'], identity['validation_ids']))
        assert len(history) == 1005 and train | check == history and not train & check
        assert not history & validation and len(validation) == 416
        assert identity['validation_ids'] == partition.loc[partition.role.eq('validation'), 'sample_id'].tolist()
        assert identity['preprocessing'] == context['preprocessing']
        assert identity['normalization'] == context['normalization']
        config = identity['training_config']
        assert {k: config[k] for k in ('model_variant', 'optimizer', 'learning_rate', 'weight_decay', 'batch_size', 'maximum_epochs', 'patience', 'shuffle', 'loss_weights', 'checkpoint_selection', 'test_during_training')} == {
            'model_variant': 'qgeognn_v2', 'optimizer': 'Adam', 'learning_rate': .001, 'weight_decay': 0.,
            'batch_size': 2048, 'maximum_epochs': 1000, 'patience': 100, 'shuffle': 'deterministic_each_epoch',
            'loss_weights': {'V1': 1., 'V2': 1.}, 'checkpoint_selection': 'validation_combined_normalized_rmse', 'test_during_training': False}
        assert identity['initialization_seed'] == seed + 2000003 and config['seed'] == seed + 3000017
        meta = {sid: {'smiles': metadata.loc[sid, 'canonical_smiles'], 'conditions': {
            k: metadata.loc[sid, k] for k in ('PE/EA', 'Density g/ml', 'V/ul', 'loading solvent', 'Volume of loading solvent/ul')}} for sid in identity['ordered_ids']}
        def exact(sid):
            return (meta[sid]['smiles'],) + _canonical_conditions(meta[sid])
        def series(sid):
            c = _canonical_conditions(meta[sid])
            return (meta[sid]['smiles'],) + c[:2] + c[3:]
        check_series = {series(sid) for sid in check}
        check_exact = {exact(sid) for sid in check}
        initial = set(partition.loc[partition.role.eq('l0'), 'sample_id'])
        assert not check_series & {series(sid) for sid in train}
        assert not check_exact & {exact(sid) for sid in train}
        assert not check_series & {series(sid) for sid in initial}
        assert {sid for sid in history if series(sid) in check_series} == check
        planned = plan[(plan.seed == seed) & (plan.budget == 1005)]
        assert len(planned) == len(check_series) == 10
        planned_ids = {sid for text in planned.ids for sid in text.split('|')}
        assert planned_ids == check
        series_key = {}
        for row in planned.itertuples():
            for sid in row.ids.split('|'):
                series_key[series(sid)] = row.series_key
        for check_id in sorted(check):
            for validation_id in sorted(validation):
                if series(check_id) == series(validation_id):
                    overlaps.append({'seed': seed, 'series_key': series_key[series(check_id)],
                        'check_id': check_id, 'validation_id': validation_id,
                        'exact_recorded_conditions': exact(check_id) == exact(validation_id),
                        'check_volume_ul': float(metadata.loc[check_id, 'V/ul']),
                        'validation_volume_ul': float(metadata.loc[validation_id, 'V/ul'])})
        associated = {series(sid) for sid in validation if series(sid) in check_series}
        ids = identity['ordered_ids']
        for role in ('train', 'check', 'validation'):
            assert [ids[i] for i in identity[role + '_indices']] == identity[role + '_ids']
        records.append({'seed': seed, 'history_n': len(history), 'training_n': len(train), 'check_n': len(check),
            'validation_n': len(validation), 'check_series_n': len(check_series),
            'training_same_series_overlap_n': 0, 'training_exact_condition_overlap_n': 0,
            'initial_same_series_overlap_n': 0, 'full_history_series_isolation': True,
            'validation_exact_condition_rows': sum(exact(sid) in check_exact for sid in validation),
            'validation_same_series_rows': sum(series(sid) in check_series for sid in validation),
            'check_series_associated_with_validation': len(associated),
            'check_series_without_validation_association': len(check_series - associated),
            'fixed_target_scales_match_historical': True, 'original_training_config_matches': True})
    assert len(gate['planned_fits']) == gate['maximum_new_fits'] == 15
    return {'status': 'PASS_WITH_FIXED_VALIDATION_ASSOCIATIONS', 'per_seed': records,
        'new_fits_by_this_qa': 0, 'label_values_read_by_this_qa': 0,
        'frozen_training_sources_unchanged': True, 'planned_primary_local_metric':
        'Per seed mean over volume pairs for each endpoint, then equal V1/V2 mean, then equal seed mean; series-equal is descriptive sensitivity only',
        'limitation': 'Original validation stays fixed. Two seeds contain one validation row with the exact recorded conditions of a held check row; held IDs never enter training or stopping directly.',
        'source_sha256': sources}, overlaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    report, overlap = validate()
    if args.write:
        (HERE / 'training_independent_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        pd.DataFrame(report['per_seed']).to_csv(HERE / 'training_independent_qa.csv', index=False)
        pd.DataFrame(overlap).to_csv(HERE / 'check_series_validation_overlap.csv', index=False)
    print(json.dumps({'status': report['status'], 'seeds': len(report['per_seed']), 'overlap_metadata_rows': len(overlap)}))


if __name__ == '__main__':
    main()
