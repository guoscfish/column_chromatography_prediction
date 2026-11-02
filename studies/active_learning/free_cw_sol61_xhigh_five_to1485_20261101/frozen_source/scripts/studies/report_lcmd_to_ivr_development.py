#!/usr/bin/env python3
"""Frozen two-seed exploratory comparison; never a five-seed confirmation."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.qgeognn_al.active_learning_v2 import lcmd_to_ivr_runner as runner
from src.qgeognn_al.active_learning_v2 import lcmd_to_ivr_study as study
from src.qgeognn_al.active_learning_v2.benchmark_reporting import metric_row
from src.qgeognn_al.active_learning_v2.efficiency_reporting import crossings
from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore
from src.qgeognn_al.active_learning_v2.sequential_protocol import label_budget

SEEDS = (157, 6101)
METHODS = ("lcmd", "kernel_ivr", study.METHOD)
METRIC = "combined_normalized_RMSE"


def summarize(curves, full):
    expected = {(s, m, b) for s in SEEDS for m in METHODS for b in study.ACTIVE_LABEL_BUDGETS}
    actual = list(zip(curves.outer_seed, curves.method, curves.active_label_count))
    if len(actual) != len(expected) or set(actual) != expected or set(full.outer_seed) != set(SEEDS):
        raise RuntimeError("complete, unique two-seed matched grid required")
    rows, values = [], []
    for scope, seed, frame, reference in [
        *(('seed', seed, curves.loc[curves.outer_seed.eq(seed)],
           full.loc[full.outer_seed.eq(seed), METRIC].iloc[0]) for seed in SEEDS),
        ('development_mean', None, curves.groupby(['method', 'active_label_count'], as_index=False)
         [[METRIC, 'V1_RMSE', 'V2_RMSE', 'V1_R2', 'V2_R2']].mean(), full[METRIC].mean()),
    ]:
        random = frame.loc[frame.method.eq("lcmd") & frame.active_label_count.eq(333), METRIC]
        e0 = float(random.iloc[0])
        targets = {"T_R30": float(frame.loc[frame.method.eq('lcmd') & frame.active_label_count.eq(1005), METRIC].iloc[0]),
                   **{f"T{p}": reference + (1-p/100)*(e0-reference) for p in (80, 90, 95)}}
        for method in METHODS:
            arm = frame.loc[frame.method.eq(method)].sort_values('active_label_count')
            x, y = arm.active_label_count.to_numpy(), arm[METRIC].to_numpy()
            early = arm.loc[arm.active_label_count.between(333, 653)]
            late = arm.loc[arm.active_label_count.between(653, 1005)]
            row = {'scope': scope, 'outer_seed': seed, 'method': method,
                   'full_AULC': float(np.trapezoid(y, x) / 672),
                   'early_AULC_333_653': float(np.trapezoid(early[METRIC], early.active_label_count) / 320),
                   'late_AULC_653_1005': float(np.trapezoid(late[METRIC], late.active_label_count) / 352),
                   'NRMSE_at_1005': float(y[-1]),
                   'V1_RMSE_at_1005': float(arm.V1_RMSE.iloc[-1]),
                   'V2_RMSE_at_1005': float(arm.V2_RMSE.iloc[-1]),
                   'V1_R2_at_1005': float(arm.V1_R2.iloc[-1]),
                   'V2_R2_at_1005': float(arm.V2_R2.iloc[-1]),
                   'matched_full_data_NRMSE': float(reference)}
            for name, threshold in targets.items():
                crossed = crossings(x, y, threshold)
                values.append({'scope': scope, 'outer_seed': seed, 'method': method,
                               'target': name, 'threshold': threshold, **crossed})
            rows.append(row)
    summary = pd.DataFrame(rows)
    pairs = []
    for seed in SEEDS:
        frame = summary.loc[summary.scope.eq('seed') & summary.outer_seed.eq(seed)].set_index('method')
        for comparator in METHODS[:-1]:
            pairs.append({'outer_seed': seed, 'comparator': comparator,
                          **{f'delta_{key}': float(frame.loc[study.METHOD, key] - frame.loc[comparator, key])
                             for key in ('full_AULC', 'late_AULC_653_1005', 'NRMSE_at_1005',
                                         'V1_RMSE_at_1005', 'V2_RMSE_at_1005')}})
    return summary, pd.DataFrame(values), pd.DataFrame(pairs)


def development_report():
    study.validate_seal()
    for seed in SEEDS:
        runner.verify_trajectory(seed)
    old = pd.read_csv(study.BASELINE / 'results/learning_curve_metrics.csv')
    pure = pd.read_csv(study.IVR / 'results/learning_curve_metrics.csv')
    prefix = old.loc[old.method.eq('lcmd') & old.outer_seed.isin(SEEDS)
                     & old.active_label_count.lt(653)].copy()
    prefix['method'] = study.METHOD
    rows, access = [], []
    for seed in SEEDS:
        partition = study._partition(seed)
        test_ids = partition.loc[partition.role.eq('test'), 'sample_id'].astype(str).tolist()
        runtime = study.STUDY / f'runtime/seed_{seed}'
        lineage = study.read_json(study.STUDY / f'lineage/seed_{seed}.json')
        selected = lineage['labeled_ids'][333:] + [sample_id
                    for r in range(study.switch_round(), len(study.ACTIVE_LABEL_BUDGETS)-1)
                    for sample_id in study.read_json(runtime / f'round_{r:02d}/selection.json')['selected_ids']]
        store = RestrictedLabelStore(study.SOURCE_DATA, partition)
        store.freeze_acquisitions(selected)
        store.freeze_predictions()
        truth = store.reveal(test_ids, 'final_test_evaluation')
        access.extend({'outer_seed': seed, **item} for item in store.audit)
        for r in range(study.switch_round(), len(study.ACTIVE_LABEL_BUDGETS)):
            record = runner.verify_record(runtime / f'round_{r:02d}/freeze.json')
            prediction = pd.read_csv(study.ROOT / record['model_directory'] / 'predictions.csv.gz')
            if prediction.sample_id.astype(str).tolist() != test_ids:
                raise RuntimeError('test prediction IDs/order differ from frozen split')
            rows.append({'outer_seed': seed, 'method': study.METHOD, 'round': r,
                         **label_budget(study.ACTIVE_LABEL_BUDGETS[r]),
                         **metric_row(truth, prediction.drop(columns='sample_id').to_numpy(float),
                                      lineage['target_scales'])})
    curves = pd.concat([old.loc[old.outer_seed.isin(SEEDS) & old.method.eq('lcmd')],
                        pure.loc[pure.outer_seed.isin(SEEDS) & pure.method.eq('kernel_ivr')],
                        prefix, pd.DataFrame(rows)], ignore_index=True)
    for seed in SEEDS:
        a = curves.loc[curves.outer_seed.eq(seed) & curves.method.eq('lcmd')
                       & curves.active_label_count.le(653)].sort_values('active_label_count')
        b = curves.loc[curves.outer_seed.eq(seed) & curves.method.eq(study.METHOD)
                       & curves.active_label_count.le(653)].sort_values('active_label_count')
        if not np.array_equal(a[METRIC].to_numpy(), b[METRIC].to_numpy()):
            raise RuntimeError('switch prefix differs from historical LCMD')
    full = pd.read_csv(study.BASELINE / 'results/full_data_reference.csv')
    summary, targets, paired = summarize(curves, full.loc[full.outer_seed.isin(SEEDS)])
    destination = study.STUDY / 'development'
    destination.mkdir(exist_ok=True)
    for name, frame in (('learning_curves', curves), ('summary', summary),
                        ('targets', targets), ('paired', paired),
                        ('test_label_access_audit', pd.DataFrame(access))):
        frame.to_csv(destination / f'{name}.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for method, color in zip(METHODS, ('#268378', '#8055a0', '#d23d48')):
            frame = curves.loc[curves.outer_seed.eq(seed) & curves.method.eq(method)].sort_values('active_label_count')
            ax.plot(frame.active_label_count, frame[METRIC], label=method, color=color, marker='.', ms=3)
        ax.axvline(653, ls=':', color='black')
        ax.set(xlabel='Active labels', title=f'Seed {seed}')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Combined normalized RMSE')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination / 'learning_curves.png', dpi=160)
    plt.close(fig)
    record = {'status': 'TWO_SEED_DEVELOPMENT_REVEALED', 'seeds': list(SEEDS),
              'original_seal_sha256': study.sha256_file(study.STUDY / 'seal.json'),
              'amendment_sha256': study.sha256_file(study.STUDY / 'STAGED_EXECUTION_AMENDMENT.md'),
              'analysis_code_sha256': study.sha256_file(Path(__file__)),
              'trajectory_freezes': study.hashes(study.STUDY / f'runtime/seed_{s}/trajectory_freeze.json' for s in SEEDS),
              'results': study.hashes(destination.glob('*.csv')),
              'test_truth_access_count': len(SEEDS), 'independent_confirmation': False}
    study._write_json_once(destination / 'manifest.json', record)
    print(json.dumps({'status': record['status'], 'summary': summary.to_dict('records'),
                      'paired': paired.to_dict('records')}, indent=2))


if __name__ == '__main__':
    development_report()
