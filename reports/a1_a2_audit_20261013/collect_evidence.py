"""Reporting-only audit: saved metrics, identities, histories, and weights only.

Never calls training/scoring/prepare actions or reads canonical label tables.
Run from repository root with the existing fish Python environment.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MT = ROOT / 'studies/transfer/column_conditioned_multitask'
PC = ROOT / 'studies/transfer/column_conditioned_pcgrad'
PARENT = ROOT / 'studies/transfer/filtered_full_data_benchmark'
QUAL = ROOT / 'studies/predictor/final_4g_qualification'
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)

def main():
    # Only these result-study directories are hashed, never raw canonical truth.
    watched = {str(p.relative_to(ROOT)): sha(p) for d in (MT, PC) for p in d.rglob('*') if p.is_file()}
    (OUT / 'frozen_hashes_before.json').write_text(json.dumps(watched, indent=2) + '\n')
    summary = {'repository_head': '42ab126', 'raw_truth_read': False, 'retraining': False}
    source_path = ROOT / json.loads((QUAL / 'decision.json').read_text())['source_checkpoint_for_transfer']
    source = torch.load(source_path, map_location='cpu', weights_only=False)['model_state_dict']
    summary['source_sha256'] = sha(source_path)
    provenance = []
    for study in (MT, PC):
        record = json.loads((study / 'EXECUTION_PROVENANCE.json').read_text())
        for rel, expected in record['files'].items():
            actual = sha(ROOT / rel) if (ROOT / rel).exists() else None
            provenance.append(dict(study=study.name, file=rel, expected=expected, actual=actual, match=actual == expected))
    save('provenance.csv', provenance)
    schedule = pd.read_csv(MT / 'split_manifest.csv')
    source_split = pd.read_csv(QUAL / 'splits/row_seed_42.csv')
    summary['source_role_counts'] = source_split.split.value_counts().to_dict()
    source_train = set(source_split.loc[source_split.split.eq('train'), 'canonical_smiles'])
    count_rows, overlap_rows = [], []
    account = pd.read_csv(PARENT / 'label_accounting.csv')
    for (protocol, seed, col), frame in schedule.groupby(['protocol', 'outer_seed', 'column']):
        assert frame.sample_id.is_unique
        roles = frame.role.value_counts().to_dict()
        a = account.loc[account.protocol.eq(protocol) & account.seed.eq(seed) & account.column.eq(col)].iloc[0]
        for role in ('gradient_train', 'validation', 'test'):
            assert roles[role] == a[role + '_rows']
        row = dict(protocol=protocol, seed=int(seed), column=col, total=len(frame), **roles)
        row['parent_pool_to_train'] = int((frame.parent_role.eq('pool') & frame.role.eq('gradient_train')).sum())
        count_rows.append(row)
        donor = schedule.loc[schedule.protocol.eq(protocol) & schedule.outer_seed.eq(seed) & schedule.column.ne(col) & schedule.role.eq('gradient_train')]
        own = set(frame.loc[frame.role.eq('gradient_train'), 'canonical_smiles'])
        for role in ('validation', 'test'):
            hold = frame.loc[frame.role.eq(role)]
            for donor_name, groups in [('same_column', own), ('other_target', set(donor.canonical_smiles)), ('4g_source_train', source_train)]:
                overlap_rows.append(dict(protocol=protocol, seed=int(seed), column=col, role=role, donor=donor_name,
                                         heldout_rows=len(hold), heldout_molecules=hold.canonical_smiles.nunique(),
                                         overlapping_molecules=len(set(hold.canonical_smiles) & groups),
                                         overlapping_rows=int(hold.canonical_smiles.isin(groups).sum())))
    save('outer_counts.csv', count_rows)
    save('outer_identity_overlap.csv', overlap_rows)
    # These files have unsafe redundant time columns: ALWAYS select safe names.
    from scripts.studies.run_column_conditioned_multitask import FEATURES
    feature_rows, collision_rows = [], []
    for col in ('25g', '40g'):
        feature_path = PARENT / f'filtered_features_{col}.csv'
        with feature_path.open() as handle:
            header = handle.readline().strip().split(',')
        summary.setdefault('redundant_outcome_column_headers', {})[col] = [key for key in header if key in ('t1', 't2', 'verified t1', 'verified t2')]
        features = pd.read_csv(feature_path, usecols=FEATURES + ['Flow mL/min'])
        key_columns = [key for key in FEATURES if key != 'sample_id']
        features['input_key'] = list(map(tuple, features[key_columns].to_numpy()))
        for (protocol, seed), subset in schedule.loc[schedule.column.eq(col)].groupby(['protocol', 'outer_seed']):
            joined = subset[['sample_id', 'role']].merge(features, on='sample_id', validate='one_to_one')
            train_keys = set(joined.loc[joined.role.eq('gradient_train'), 'input_key'])
            for role, frame in joined.groupby('role'):
                ratio = frame['PE/EA'].str.split('/', expand=True).astype(float)
                ea = ratio[1] / ratio.sum(axis=1)
                loading = frame['Density g/ml'] * frame['V/ul']
                feature_rows.append(dict(protocol=protocol, seed=int(seed), column=col, role=role, n=len(frame),
                                         unique_molecules=frame.canonical_smiles.nunique(), ea_mean=ea.mean(), ea_sd=ea.std(ddof=0),
                                         loading_mean=loading.mean(), loading_sd=loading.std(ddof=0),
                                         flow_values=','.join(map(str, sorted(frame['Flow mL/min'].unique())))))
                if role != 'gradient_train':
                    collision_rows.append(dict(protocol=protocol, seed=int(seed), column=col, role=role,
                                               same_model_input_rows_in_train=int(frame.input_key.isin(train_keys).sum())))
        summary.setdefault('identical_model_input_counts', {})[col] = dict(total_rows=len(features), unique_input_rows=len(set(features.input_key)))
    save('feature_distributions.csv', feature_rows)
    save('same_input_overlap.csv', collision_rows)
    epoch_rows, checks, scale_rows = [], [], []
    for protocol, prefix in [('compound', ''), ('row', 'ROW_')]:
        results = pd.read_csv(MT / (prefix + 'INNER_RESULTS.csv'))
        identities = pd.read_csv(MT / (prefix + 'INNER_IDENTITIES.csv')).drop_duplicates()
        for (seed, fold), ids in identities.groupby(['outer_seed', 'inner_fold']):
            targets = ids.loc[ids.column.isin(['25g', '40g'])]
            assert not set(targets.loc[targets.role.eq('inner_train'), 'canonical_smiles']) & set(targets.loc[targets.role.eq('inner_validation'), 'canonical_smiles'])
            for col, sub in ids.groupby('column'):
                allowed = set(source_split.loc[source_split.split.eq('train'), 'sample_id']) if col == '4g' else set(schedule.loc[schedule.protocol.eq(protocol) & schedule.outer_seed.eq(seed) & schedule.column.eq(col) & schedule.role.eq('gradient_train'), 'sample_id'])
                assert set(sub.loc[sub.role.eq('inner_train'), 'sample_id']) <= allowed
                if col != '4g':
                    assert set(sub.sample_id) == allowed
                else:
                    assert set(sub.loc[sub.role.eq('inner_validation'), 'sample_id']) == set(source_split.loc[source_split.split.eq('validation'), 'sample_id'])
        for (seed, arm), sub in results.groupby(['outer_seed', 'arm']):
            a = sub.loc[sub.column.eq('25g')].sort_values('inner_fold')
            b = sub.loc[sub.column.eq('40g')].sort_values('inner_fold')
            assert a.best_epoch.tolist() == b.best_epoch.tolist()
            median = int(np.floor(a.best_epoch.median() + .5))
            final = MT / f'runtime/final/{protocol}/seed_{seed}/{arm}'
            fit = json.loads((final / 'fit.json').read_text())
            hist = pd.read_csv(final / 'history.csv')
            assert fit['epochs_run'] == fit['best_epoch'] == median == len(hist)
            assert hist.epoch.tolist() == list(range(1, median + 1))
            assert 'joint_validation_score' not in hist
            largest = schedule.loc[schedule.protocol.eq(protocol) & schedule.outer_seed.eq(seed) & schedule.role.eq('gradient_train')].groupby('column').size().max()
            assert fit['per_task_batch_size'] == min(682, largest)
            ck = torch.load(final / 'best.pt', map_location='cpu', weights_only=False)
            assert ck['best_epoch'] == median
            expected_steps = median * fit['steps_per_epoch']
            late_bn = {k: int(v) for k, v in ck['model_state_dict'].items() if k.endswith('num_batches_tracked') and k.startswith(('backbone.convs.3.', 'backbone.convs.4.', 'backbone.convs_bond_angle.3.'))}
            assert all(v - int(source[k]) == expected_steps for k, v in late_bn.items())
            epoch_rows.append(dict(protocol=protocol, seed=int(seed), arm=arm, fold_best_epochs=','.join(map(str, a.best_epoch)),
                                   fold_epochs_run=','.join(map(str, a.epochs_run)), median_epoch=median,
                                   final_epochs=fit['epochs_run'], steps_per_epoch=fit['steps_per_epoch'], total_optimizer_steps=expected_steps,
                                   inner_batch_sizes=','.join(map(str, a.per_task_batch_size)), final_batch_size=fit['per_task_batch_size'],
                                   inner_25g_train=','.join(map(str, a.inner_train_count)), inner_40g_train=','.join(map(str, b.inner_train_count))))
            for fold in range(1, 6):
                run = MT / f'runtime/inner/{protocol}/seed_{seed}/fold_{fold}/{arm}'
                history = pd.read_csv(run / 'history.csv')
                selected = sub.loc[sub.inner_fold.eq(fold)].iloc[0]
                best = int(history.loc[history.joint_validation_score.idxmin(), 'epoch'])
                assert best == int(selected.best_epoch)
                assert len(history) == int(selected.epochs_run)
                assert history.epoch.tolist() == list(range(1, len(history) + 1))
                for col in ('25g', '40g'):
                    current = sub.loc[sub.inner_fold.eq(fold) & sub.column.eq(col)].iloc[0]
                    for metric in ('V1_rmse', 'V2_rmse', 'combined_normalized_rmse'):
                        assert abs(current[metric] - history.loc[history.epoch.eq(best), col + '_' + metric].iloc[0]) < 1e-10
                    scale_rows.append(dict(protocol=protocol, seed=int(seed), fold=fold, arm=arm, column=col,
                                           scale_V1=current.scale_V1, scale_V2=current.scale_V2))
        # Saved targets have identical A1/A2 identities/scales/batches.
        keyed = results.set_index(['outer_seed', 'inner_fold', 'column', 'arm'])
        for seed, fold, col in keyed.index.droplevel('arm').unique():
            for field in ('inner_train_ids_sha256', 'inner_validation_ids_sha256', 'inner_train_count', 'inner_validation_count', 'scale_V1', 'scale_V2', 'per_task_batch_size', 'steps_per_epoch'):
                assert keyed.loc[(seed, fold, col, 'A1'), field] == keyed.loc[(seed, fold, col, 'A2'), field]
    save('epoch_bridge.csv', epoch_rows)
    save('inner_scales.csv', scale_rows)
    loss_rows = []
    for protocol in ('compound', 'row'):
        for seed in schedule.outer_seed.unique():
            row = dict(protocol=protocol, seed=int(seed))
            for arm in ('A1', 'A2'):
                history = pd.read_csv(MT / f'runtime/final/{protocol}/seed_{seed}/{arm}/history.csv')
                for col in ('4g', '25g', '40g'):
                    row[arm + '_' + col + '_last_train_loss'] = history['train_loss_' + col].iloc[-1]
            loss_rows.append(row)
    save('final_train_loss.csv', loss_rows)
    # Hash completion chains; check all frozen prefix weights and buffers without truth.
    trainable_prefixes = ('backbone.convs.3.', 'backbone.convs.4.', 'backbone.convs_bond_angle.3.',
                         'backbone.convs_bond_float.3.', 'backbone.convs_bond_embeding.3.', 'backbone.convs_angle_float.3.')
    frozen_keys = [k for k in source if k.startswith('backbone.') and not k.startswith(trainable_prefixes)]
    for study in (MT, PC):
        for complete in study.glob('runtime/**/complete.json'):
            rec = json.loads(complete.read_text())
            mismatches = [name for name, expected in rec['files'].items() if sha(complete.parent / name) != expected]
            assert not mismatches, (complete, mismatches)
            ck = torch.load(complete.parent / 'best.pt', map_location='cpu', weights_only=False)['model_state_dict']
            assert all(torch.equal(ck[k], source[k]) for k in frozen_keys)
            checks.append(dict(study=study.name, run=str(complete.parent.relative_to(study)), completion_hash_match=True,
                               frozen_tensors_equal=len(frozen_keys)))
    save('checkpoint_checks.csv', checks)
    summary['completion_checks'] = len(checks)
    summary['frozen_tensors_per_fit'] = len(frozen_keys)
    # Recompute paired changes and non-tail SSE using frozen aggregate metrics only.
    paired, tail_rows = [], []
    for protocol, prefix in [('compound', ''), ('row', 'ROW_')]:
        inner = pd.read_csv(MT / (prefix + 'INNER_RESULTS.csv'))
        outer = pd.read_csv(MT / (prefix + 'OUTER_RESULTS.csv'))
        tails = pd.read_csv(MT / (prefix + 'OUTER_TAIL_METRICS.csv'))
        for (seed, col), frame in outer.groupby(['outer_seed', 'column']):
            x = frame.set_index('method')
            row = dict(protocol=protocol, seed=int(seed), column=col)
            means = inner.loc[inner.outer_seed.eq(seed) & inner.column.eq(col)].groupby('arm').mean(numeric_only=True)
            row['inner_mean_nrmse_gain_pct'] = 100 * (1 - means.loc['A2', 'combined_normalized_rmse'] / means.loc['A1', 'combined_normalized_rmse'])
            for metric in ('combined_normalized_rmse', 'V1_rmse', 'V2_rmse', 'V1_mae', 'V2_mae', 'Center_rmse', 'Width_rmse'):
                row[metric + '_A1'], row[metric + '_A2'] = x.loc['A1', metric], x.loc['A2', metric]
                row[metric + '_delta'] = x.loc['A2', metric] - x.loc['A1', metric]
                row[metric + '_deterioration_pct'] = 100 * (x.loc['A2', metric] / x.loc['A1', metric] - 1)
            paired.append(row)
            n = int(schedule.loc[schedule.protocol.eq(protocol) & schedule.outer_seed.eq(seed) & schedule.column.eq(col) & schedule.role.eq('test')].shape[0])
            for endpoint in ('V1', 'V2'):
                t = tails.loc[tails.outer_seed.eq(seed) & tails.column.eq(col) & tails.endpoint.eq(endpoint)].set_index('method')
                assert t.loc['A1', 'tail_count'] == t.loc['A2', 'tail_count']
                assert t.loc['A1', 'threshold'] == t.loc['A2', 'threshold']
                record = dict(protocol=protocol, seed=int(seed), column=col, endpoint=endpoint, threshold=t.loc['A1', 'threshold'], n=n,
                              tail_count=int(t.loc['A1', 'tail_count']), A1_tail_rmse=t.loc['A1', 'tail_rmse'], A2_tail_rmse=t.loc['A2', 'tail_rmse'],
                              tail_change_pct=100 * (t.loc['A2', 'tail_rmse'] / t.loc['A1', 'tail_rmse'] - 1),
                              delta_tail_sse=t.loc['A2', 'tail_sse'] - t.loc['A1', 'tail_sse'],
                              delta_total_sse=t.loc['A2', 'total_sse'] - t.loc['A1', 'total_sse'])
                for arm in ('A1', 'A2'):
                    record[arm + '_non_tail_rmse'] = float(np.sqrt((t.loc[arm, 'total_sse'] - t.loc[arm, 'tail_sse']) / (n - t.loc[arm, 'tail_count'])))
                    record[arm + '_tail_sse_fraction'] = t.loc[arm, 'tail_sse_fraction']
                record['delta_non_tail_sse'] = record['delta_total_sse'] - record['delta_tail_sse']
                tail_rows.append(record)
    save('paired_errors.csv', paired)
    save('tail_decomposition.csv', tail_rows)
    # Algebra from saved aggregate metrics, not a request for held-out labels.
    distributions = []
    for protocol, prefix in [('compound', ''), ('row', 'ROW_')]:
        inner = pd.read_csv(MT / (prefix + 'INNER_RESULTS.csv'))
        outer = pd.read_csv(MT / (prefix + 'OUTER_RESULTS.csv'))
        for (seed, col), group in outer.groupby(['outer_seed', 'column']):
            design = np.column_stack([group.V1_rmse.to_numpy() ** 2, group.V2_rmse.to_numpy() ** 2])
            coefficients = np.linalg.lstsq(design, 2 * group.combined_normalized_rmse.to_numpy() ** 2, rcond=None)[0]
            for arm in ('A1', 'A2'):
                a = group.loc[group.method.eq(arm)].iloc[0]
                i = inner.loc[inner.outer_seed.eq(seed) & inner.column.eq(col) & inner.arm.eq(arm)]
                distributions.append(dict(protocol=protocol, seed=int(seed), column=col, arm=arm,
                    outer_scale_V1=1 / np.sqrt(coefficients[0]), outer_scale_V2=1 / np.sqrt(coefficients[1]),
                    inner_train_scale_V1_min=i.scale_V1.min(), inner_train_scale_V1_max=i.scale_V1.max(),
                    inner_train_scale_V2_min=i.scale_V2.min(), inner_train_scale_V2_max=i.scale_V2.max(),
                    outer_test_sd_V1=a.V1_rmse / np.sqrt(1 - a.V1_r2), outer_test_sd_V2=a.V2_rmse / np.sqrt(1 - a.V2_r2),
                    inner_val_sd_V1_mean=(i.V1_rmse / np.sqrt(1 - i.V1_r2)).mean(),
                    inner_val_sd_V2_mean=(i.V2_rmse / np.sqrt(1 - i.V2_r2)).mean(),
                    abs_center_mean_error=np.sqrt(max(0, a.Center_rmse ** 2 - a.Center_error_variance)),
                    abs_width_mean_error=np.sqrt(max(0, a.Width_rmse ** 2 - a.Width_error_variance))))
    save('aggregate_distribution_diagnostics.csv', distributions)
    shifts = []
    for seed in schedule.outer_seed.unique():
        a = pd.read_csv(MT / f'runtime/final/row/seed_{seed}/A1/predictions_blind.csv')
        b = pd.read_csv(MT / f'runtime/final/row/seed_{seed}/A2/predictions_blind.csv')
        a = a.loc[a.column.eq('40g') & a.role.eq('test')].set_index('sample_id')
        b = b.loc[b.column.eq('40g') & b.role.eq('test')].set_index('sample_id').loc[a.index]
        delta = b[['V1_q50', 'V2_q50']] - a[['V1_q50', 'V2_q50']]
        shifts.append(dict(seed=int(seed), mean_delta_V1=delta.V1_q50.mean(), mean_delta_V2=delta.V2_q50.mean(),
                           mean_delta_center=delta.mean(axis=1).mean(), mean_delta_width=(delta.V2_q50 - delta.V1_q50).mean(),
                           delta_endpoint_correlation=delta.corr().iloc[0, 1]))
    save('prediction_only_shifts.csv', shifts)
    from scripts.studies.run_column_conditioned_pcgrad import mechanism_summary
    raw, projected = pd.read_csv(PC / 'GRADIENT_RAW.csv'), pd.read_csv(PC / 'GRADIENT_PROJECTED.csv')
    for key in ('protocol', 'outer_seed', 'inner_fold', 'epoch', 'step', 'permutation_seed'):
        assert raw[key].equals(projected[key])
    summary['pcgrad_mechanism_recomputed'] = mechanism_summary(raw, projected)
    for manifest in ('COMPOUND_SCORE_MANIFEST.json', 'ROW_SCORE_MANIFEST.json', 'PREDICTION_FREEZE_MANIFEST.json', 'ROW_PREDICTION_FREEZE_MANIFEST.json'):
        rec = json.loads((MT / manifest).read_text())
        assert all(sha(MT / rel) == expected for rel, expected in rec['files'].items())
    after = {str(p.relative_to(ROOT)): sha(p) for d in (MT, PC) for p in d.rglob('*') if p.is_file()}
    assert watched == after
    summary['frozen_files_unchanged'] = len(after)
    summary['status'] = 'PASS'
    (OUT / 'evidence_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
