#!/usr/bin/env python3
"""Summarize already-authorized matched predictions without inference or labels."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
INPUTS = {}
METRICS = ['observed_abs_mean', 'predicted_abs_mean', 'difference_mae', 'amplitude_error_mean',
           'direction_accuracy', 'correct_nonflat_fraction', 'wrong_nonflat_fraction', 'flat_fraction']
KEYS = ['formal_seed', 'al_seed', 'axis', 'exposure', 'target', 'actual_threshold_ml']
METHODS = ['AL_L525', 'AL_L1005', 'AL_L1485', 'Fully_trained']
COMPARISONS = [('Full_vs_AL525', 'Fully_trained', 'AL_L525'),
               ('Full_vs_AL1005', 'Fully_trained', 'AL_L1005'),
               ('Full_vs_AL1485', 'Fully_trained', 'AL_L1485'),
               ('AL1005_vs_AL525', 'AL_L1005', 'AL_L525'),
               ('AL1485_vs_AL1005', 'AL_L1485', 'AL_L1005')]


def read(path):
    raw = path.read_bytes()
    assert raw == path.read_bytes(), ('unstable derived input', str(path))
    INPUTS[str(path.relative_to(ROOT))] = {'path': str(path.relative_to(ROOT)),
                                         'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return raw


def numbers(actual, predicted, threshold):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    meaningful = np.abs(actual) > threshold
    flat = np.abs(predicted) < .1
    correct = np.sign(actual) == np.sign(predicted)
    result = {'n_pairs': len(actual), 'n_meaningful': int(meaningful.sum()),
              'observed_abs_mean': float(np.abs(actual).mean()), 'predicted_abs_mean': float(np.abs(predicted).mean()),
              'difference_mae': float(np.abs(actual - predicted).mean()),
              'amplitude_error_mean': float(np.abs(np.abs(actual) - np.abs(predicted)).mean()),
              'actual_threshold_ml': threshold, 'predicted_flat_threshold_ml': .1}
    for name, values in [('direction_accuracy', correct), ('correct_nonflat_fraction', correct & ~flat),
                         ('wrong_nonflat_fraction', (actual * predicted < 0) & ~flat), ('flat_fraction', flat)]:
        result[name] = float(values[meaningful].mean()) if meaningful.any() else np.nan
    return result


def verify_permissions(metadata):
    permitted = pd.read_csv(io.BytesIO(read(OUT / 'al_common_pair_exposure.csv')))
    incident = json.loads(read(OUT / 'LABEL_BOUNDARY_INCIDENT.json'))
    quarantine = set(incident['quarantined_ids'])
    unique = metadata[['formal_seed', 'al_seed', 'axis', 'exposure', 'pair_key', 'low_id', 'high_id']].drop_duplicates()
    expected = permitted.rename(columns={'formal_row_seed': 'formal_seed', 'formal_exposure_class': 'exposure'})
    checked = unique.merge(expected[['formal_seed', 'al_seed', 'axis', 'exposure', 'pair_key', 'low_id', 'high_id', 'allow_new_scoring']],
                           on=['formal_seed', 'al_seed', 'axis', 'exposure', 'pair_key', 'low_id', 'high_id'],
                           how='left', validate='one_to_one')
    assert len(checked) == len(unique) and checked.allow_new_scoring.notna().all()
    assert checked.allow_new_scoring.all()
    assert not set(checked.low_id) & quarantine and not set(checked.high_id) & quarantine
    assert set(checked.exposure) <= {'both_source_train', 'mixed_source_train_validation', 'both_source_validation'}
    return len(unique)


def validate_metrics(pairs, metrics):
    maximum_error = 0.
    for key, group in pairs.groupby(['formal_seed', 'al_seed', 'axis', 'exposure', 'method', 'target']):
        selected = metrics
        for name, value in zip(['formal_seed', 'al_seed', 'axis', 'exposure', 'method', 'target'], key):
            selected = selected[selected[name] == value]
        assert set(selected.actual_threshold_ml) == {.1, .5, 1.} and len(selected) == 3
        for row in selected.to_dict('records'):
            expected = numbers(group.delta_true, group.delta_pred, row['actual_threshold_ml'])
            for name, value in expected.items():
                if pd.isna(value):
                    assert pd.isna(row[name])
                else:
                    maximum_error = max(maximum_error, abs(value - row[name]))
    assert maximum_error < 1e-10
    return maximum_error


def hierarchical_methods(metrics, pairs):
    formal_rows, final_rows = [], []
    keys = ['formal_seed', 'axis', 'exposure', 'method', 'target', 'actual_threshold_ml']
    for key, group in metrics.groupby(keys):
        row = dict(zip(keys, key))
        row.update(aggregation_level='within_formal_equal_AL_campaigns', n_formal_models=1,
                   n_al_campaigns=len(group), included_al_seeds='|'.join(map(str, sorted(group.al_seed))),
                   comparison_pair_rows=int(group.n_pairs.sum()))
        source = pairs[(pairs.formal_seed == row['formal_seed']) & (pairs.axis == row['axis']) &
                       (pairs.exposure == row['exposure']) & (pairs.method == row['method']) & (pairs.target == row['target'])]
        row.update(unique_pair_ids=source.pair_key.nunique(), unique_molecules=source.molecule.nunique(),
                   unique_series=source.series_key.nunique())
        for metric in METRICS:
            row[metric] = group[metric].mean()
            row['nonmissing_AL_campaigns_' + metric] = int(group[metric].notna().sum())
        formal_rows.append(row)
    formal = pd.DataFrame(formal_rows)
    keys = ['axis', 'exposure', 'method', 'target', 'actual_threshold_ml']
    for key, group in formal.groupby(keys):
        row = dict(zip(keys, key))
        row.update(aggregation_level='equal_formal_after_equal_AL_campaigns', formal_seed='ALL',
                   n_formal_models=len(group), n_al_campaigns=int(group.n_al_campaigns.sum()),
                   included_al_seeds='model-specific; see within-formal rows',
                   comparison_pair_rows=int(group.comparison_pair_rows.sum()))
        source = pairs[(pairs.axis == row['axis']) & (pairs.exposure == row['exposure']) &
                       (pairs.method == row['method']) & (pairs.target == row['target'])]
        row.update(unique_pair_ids=source.pair_key.nunique(), unique_molecules=source.molecule.nunique(), unique_series=source.series_key.nunique())
        for metric in METRICS:
            row[metric] = group[metric].mean()
            row['nonmissing_formal_models_' + metric] = int(group[metric].notna().sum())
        final_rows.append(row)
    return pd.concat([formal, pd.DataFrame(final_rows)], ignore_index=True)


def paired_improvements(metrics):
    rows = []
    for key, group in metrics.groupby(KEYS):
        indexed = group.set_index('method')
        assert set(indexed.index) == set(METHODS)
        for comparison, first, second in COMPARISONS:
            a, b = indexed.loc[first], indexed.loc[second]
            assert a.n_pairs == b.n_pairs and a.n_meaningful == b.n_meaningful
            assert abs(a.observed_abs_mean - b.observed_abs_mean) < 1e-12
            row = dict(zip(KEYS, key))
            row.update(aggregation_level='formal_AL_campaign', comparison=comparison, first_method=first,
                       second_method=second, n_pairs=int(a.n_pairs), n_meaningful=int(a.n_meaningful),
                       first_mae=a.difference_mae, second_mae=b.difference_mae,
                       mae_improvement=b.difference_mae - a.difference_mae,
                       mae_relative_improvement=1 - a.difference_mae / b.difference_mae if b.difference_mae > 1e-12 else np.nan,
                       first_predicted_abs=a.predicted_abs_mean, second_predicted_abs=b.predicted_abs_mean,
                       predicted_amplitude_change=a.predicted_abs_mean - b.predicted_abs_mean,
                       observed_abs=a.observed_abs_mean, first_direction=a.direction_accuracy, second_direction=b.direction_accuracy,
                       direction_change=a.direction_accuracy - b.direction_accuracy,
                       first_flat=a.flat_fraction, second_flat=b.flat_fraction, flat_reduction=b.flat_fraction - a.flat_fraction)
            rows.append(row)
    individual = pd.DataFrame(rows)
    measures = ['first_mae', 'second_mae', 'mae_improvement', 'first_predicted_abs', 'second_predicted_abs',
                'predicted_amplitude_change', 'observed_abs', 'first_direction', 'second_direction',
                'direction_change', 'first_flat', 'second_flat', 'flat_reduction']
    formal_rows, all_rows = [], []
    keys = ['formal_seed', 'axis', 'exposure', 'target', 'actual_threshold_ml', 'comparison', 'first_method', 'second_method']
    for key, group in individual.groupby(keys):
        row = dict(zip(keys, key))
        row.update(aggregation_level='within_formal_equal_AL_campaigns', al_seed='ALL', n_al_campaigns=len(group),
                   n_improved_AL_campaigns=int((group.mae_improvement > 1e-12).sum()),
                   n_worse_AL_campaigns=int((group.mae_improvement < -1e-12).sum()))
        row.update(group[measures].mean().to_dict())
        row['mean_campaign_relative_improvement'] = group.mae_relative_improvement.mean()
        row['mae_relative_improvement'] = 1 - row['first_mae'] / row['second_mae'] if row['second_mae'] > 1e-12 else np.nan
        formal_rows.append(row)
    formal = pd.DataFrame(formal_rows)
    keys = ['axis', 'exposure', 'target', 'actual_threshold_ml', 'comparison', 'first_method', 'second_method']
    for key, group in formal.groupby(keys):
        row = dict(zip(keys, key))
        row.update(aggregation_level='equal_formal_after_equal_AL_campaigns', formal_seed='ALL', al_seed='ALL',
                   n_formal_models=len(group), n_improved_formal_models=int((group.mae_improvement > 1e-12).sum()),
                   n_worse_formal_models=int((group.mae_improvement < -1e-12).sum()))
        row.update(group[measures].mean().to_dict())
        row['mean_campaign_relative_improvement'] = group.mean_campaign_relative_improvement.mean()
        row['mae_relative_improvement'] = 1 - row['first_mae'] / row['second_mae'] if row['second_mae'] > 1e-12 else np.nan
        all_rows.append(row)
    return pd.concat([individual, formal, pd.DataFrame(all_rows)], ignore_index=True)


def molecule_heterogeneity(pairs):
    rows = []
    keys = ['formal_seed', 'al_seed', 'axis', 'exposure', 'target', 'molecule']
    for key, group in pairs.groupby(keys):
        indexed = {method: values for method, values in group.groupby('method')}
        full, al = indexed['Fully_trained'], indexed['AL_L1005']
        a, b = numbers(full.delta_true, full.delta_pred, .5), numbers(al.delta_true, al.delta_pred, .5)
        rows.append(dict(zip(keys, key)) | {'comparison': 'Full_vs_AL1005', 'n_pairs': a['n_pairs'],
            'n_series': full.series_key.nunique(), 'report_eligible_n_pairs_at_least_two': a['n_pairs'] >= 2,
            'full_mae': a['difference_mae'], 'al1005_mae': b['difference_mae'],
            'mae_improvement': b['difference_mae'] - a['difference_mae'],
            'full_predicted_abs': a['predicted_abs_mean'], 'al1005_predicted_abs': b['predicted_abs_mean'],
            'observed_abs': a['observed_abs_mean'], 'full_direction': a['direction_accuracy'],
            'al1005_direction': b['direction_accuracy'], 'full_flat': a['flat_fraction'], 'al1005_flat': b['flat_fraction']})
    return pd.DataFrame(rows)


def denominators(pairs):
    unique = pairs[['formal_seed', 'al_seed', 'axis', 'exposure', 'pair_key', 'series_key', 'molecule']].drop_duplicates()
    rows = []
    for keys in [['axis'], ['axis', 'exposure'], ['formal_seed', 'axis', 'exposure']]:
        for key, group in unique.groupby(keys):
            if not isinstance(key, tuple):
                key = (key,)
            rows.append(dict(zip(keys, key)) | {'count_level': '|'.join(keys), 'comparison_pair_rows': len(group),
                'unique_physical_pair_ids': group.pair_key.nunique(), 'unique_molecules': group.molecule.nunique(),
                'unique_series': group.series_key.nunique(), 'formal_models': group.formal_seed.nunique(),
                'formal_AL_campaign_cells': len(group[['formal_seed', 'al_seed']].drop_duplicates())})
    result = pd.DataFrame(rows)
    result['exposure'] = result.exposure.fillna('all_exposures')
    return result


def markdown(frame):
    frame = frame.copy()
    for key in frame:
        if pd.api.types.is_numeric_dtype(frame[key]):
            frame[key] = frame[key].map(lambda value: f'{value:.5g}' if not pd.isna(value) else 'NA')
    return '| ' + ' | '.join(frame.columns) + ' |\n|' + '|'.join(['---'] * len(frame.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in frame.to_numpy())


def write_report(pairs, aggregate, paired, molecules, counts):
    final = aggregate[(aggregate.aggregation_level == 'equal_formal_after_equal_AL_campaigns') &
                      (aggregate.actual_threshold_ml == .5)]
    primary = final[final.axis == 'V/ul'][['exposure', 'method', 'target', 'n_formal_models', 'comparison_pair_rows',
        'unique_pair_ids', 'observed_abs_mean', 'predicted_abs_mean', 'difference_mae', 'direction_accuracy', 'flat_fraction']]
    comparison = paired[(paired.aggregation_level == 'equal_formal_after_equal_AL_campaigns') &
                        (paired.actual_threshold_ml == .5) & paired.comparison.isin(['Full_vs_AL525', 'Full_vs_AL1005', 'Full_vs_AL1485'])]
    delta_cols = ['exposure', 'comparison', 'target', 'mae_improvement', 'mae_relative_improvement',
                  'predicted_amplitude_change', 'direction_change', 'flat_reduction', 'n_improved_formal_models', 'n_formal_models']
    pe = final[final.axis == 'PE/EA'][['exposure', 'method', 'target', 'observed_abs_mean', 'predicted_abs_mean', 'difference_mae', 'direction_accuracy', 'flat_fraction']]
    within = aggregate[(aggregate.aggregation_level == 'within_formal_equal_AL_campaigns') &
                       (aggregate.actual_threshold_ml == .5) & (aggregate.axis == 'V/ul') &
                       aggregate.method.isin(['AL_L525', 'AL_L1005', 'AL_L1485', 'Fully_trained'])]
    within = within[['formal_seed', 'exposure', 'method', 'target', 'n_al_campaigns', 'difference_mae', 'predicted_abs_mean', 'direction_accuracy', 'flat_fraction']]
    eligible = molecules[(molecules.axis == 'V/ul') & molecules.report_eligible_n_pairs_at_least_two]
    heterogeneity = eligible.groupby(['formal_seed', 'exposure', 'target']).agg(
        molecule_campaign_cells=('mae_improvement', 'size'), distinct_molecules=('molecule', 'nunique'),
        improved_cells=('mae_improvement', lambda x: (x > 1e-12).sum()),
        worse_cells=('mae_improvement', lambda x: (x < -1e-12).sum()),
        median_mae_improvement=('mae_improvement', 'median')).reset_index()
    volume_rows = counts[(counts.axis == 'V/ul') & counts.formal_seed.notna()]
    source_train = final[(final.axis == 'V/ul') & (final.exposure == 'both_source_train')].set_index(['method', 'target'])
    def number(method, target, metric):
        return float(source_train.loc[(method, target), metric])
    full1005 = comparison[(comparison.axis == 'V/ul') & (comparison.exposure == 'both_source_train') &
                          (comparison.comparison == 'Full_vs_AL1005')].set_index('target')
    text = f'''# 同一实验对上的AL与充分训练模型比较

**同pair主结果：** 在正式模型见过两端的体积对上，Fully相对AL L1005的差分MAE，V1仅改善{float(full1005.loc['V1','mae_improvement']):.5f}mL，V2反而恶化{-float(full1005.loc['V2','mae_improvement']):.5f}mL。三级等权汇总的Fully预测幅度V1/V2仅为真实绝对变化均值的{100*number('Fully_trained','V1','predicted_abs_mean')/number('Fully_trained','V1','observed_abs_mean'):.2f}%/{100*number('Fully_trained','V2','predicted_abs_mean')/number('Fully_trained','V2','observed_abs_mean'):.2f}%，均不足10%。因此，当前充分训练模型没有在这些已见pair上普遍解决体积幅度压缩；还不能唯一归因于网络、loss、覆盖、优化或噪声。

本报告只读取已通过source角色与3ID隔离门控的派生评分表；没有新训练、推理、API或新的原始目标store访问，真值差分全部来自根分析已授权的派生表。比较对象为AL五seed的L525/L1005及预先纳入的L1485扩展，与正式Row seed42/525/1101三模型。**正式seed525不是AL预算L525。** 每个比较行使用同一对ID与同一个合法delta_true，四种方法的实际变化值逐项一致。

## 分母与曝光必须先说明

原始评分表{len(pairs)}行，包含4方法×2端点；去掉方法与端点复制后为{len(pairs[['formal_seed','al_seed','axis','pair_key']].drop_duplicates())}个“正式模型×AL campaign×pair”比较记录。跨模型/AL历史去重后仅{len(pairs[['axis','pair_key']].drop_duplicates())}个物理pair ID，仍可共享分子与系列。不能将cross 3×5的15个组合当成15个配对随机seed或据此做强显著性结论。

{markdown(counts[counts.formal_seed.isna()][['axis','exposure','comparison_pair_rows','unique_physical_pair_ids','unique_molecules','unique_series','formal_models','formal_AL_campaign_cells']])}

{markdown(volume_rows[['formal_seed','exposure','comparison_pair_rows','unique_physical_pair_ids','unique_molecules','formal_AL_campaign_cells']])}

两端source-validation的体积对仅6条跨模型/AL比较记录（正式seed42/525/1101分别3/2/1条），样本不足，不能单凭该小子集确认泛化恢复或失败。原source-validation也参与历史checkpoint选择，本来就不是全新的确认集。source-train两端表示充分训练模型见过该pair；mixed只见过其中一端；曝光类别随正式模型变化，同一物理pair可能在不同模型下落入不同类别，类别数不能简单相加充当独立实验数。涉及当前正式模型source-test或隔离ID的记录全部不进入本表。

## 聚合顺序与主结果

**幅度变化：** 训练内同pair真实V1/V2绝对变化均值为{number('Fully_trained','V1','observed_abs_mean'):.4f}/{number('Fully_trained','V2','observed_abs_mean'):.4f}mL；AL L525预测{number('AL_L525','V1','predicted_abs_mean'):.4f}/{number('AL_L525','V2','predicted_abs_mean'):.4f}，L1005预测{number('AL_L1005','V1','predicted_abs_mean'):.4f}/{number('AL_L1005','V2','predicted_abs_mean'):.4f}，L1485预测{number('AL_L1485','V1','predicted_abs_mean'):.4f}/{number('AL_L1485','V2','predicted_abs_mean'):.4f}，Fully预测{number('Fully_trained','V1','predicted_abs_mean'):.4f}/{number('Fully_trained','V2','predicted_abs_mean'):.4f}mL。相对很早期有所恢复，不等于随训练规模普遍达到充分响应。

先在正式模型×AL campaign×曝光×端点的完全相同pair集合上计算指标，再在每个正式模型内对可评价AL campaigns等权，最后对正式模型等权。主阈值为真实绝对变化>0.5mL，flat为预测绝对变化<0.1mL；0.1/1.0mL敏感性完整保存。方向/flat没有有意义真实变化时保持NA，计数列标注有效campaign/模型数，不用0填充。

以下为体积主表，物理单位mL；差分MAE越低越好，响应幅度大不等于预测正确。

{markdown(primary)}

方向准确率包含预测幅度极小但符号相同的情况，可以与高flat比例同时出现；不能由较高方向准确率认定幅度已学会。`correct_nonflat_fraction`及有效分母在CSV保留，避免把近零预测的正确符号当成明显条件响应。

正mae_improvement表示Fully相对对应AL的差分MAE更低；正flat_reduction表示平坦比例降低。比值使用上述分层均值之比，另保留逐campaign相对变化，不用近零真实变化作分母。

{markdown(comparison[comparison.axis == 'V/ul'][delta_cols])}

## 正式模型逐一查看

{markdown(within)}

不同正式模型有各自scaler、训练/validation划分和随机训练轨迹。它们与AL五seed不是相同随机划分的重复对照；本结果能描述预测差异，不能把差异唯一归因于训练行数。AL三个budget的共同pair均在各自已见H中，正式模型的训练曝光则必须按上表分开解释。L1485仅是已有checkpoint的同pair扩展，不是新AL运行，也不与L525/L1005使用不同样本凑比较。

## PE/EA作为因素参照

{markdown(pe)}

{markdown(comparison[comparison.axis == 'PE/EA'][delta_cols])}

PE/EA与体积的真实物理变化范围不同，不以两者幅度高低评价因素的重要性；它只作为同一模型是否存在选择性响应不足的参照。

## 分子异质性与限制

以下只用于描述Full相对AL1005的异质性，要求每个分子/campaign cell至少2个pair，避免仅一个pair撑起分布。所有分子仍完整保存在CSV。表中cell可共享分子、pair与模型，正负计数不是独立科研重复。

{markdown(heterogeneity)}

不能只选择改善最大的分子或正式seed宣布问题解决。所有逐pair、逐campaign和分子cell的正反结果均保留；恢复或平坦曲线的展示由根分析的预定metadata曲线池处理，本脚本不事后更换样本。训练内响应改善不等于source-validation泛化改善，整体Row NRMSE也不能替代这些局部差分指标。

## 文件与复现

`matched_exposure_equal_models.csv`包含两级分层均值与有效分母；`matched_paired_improvements.csv`同时保存campaign、正式模型及正式模型等权三层配对变化；`matched_denominators.csv`区分比较记录和物理pair；`matched_molecule_heterogeneity.csv`保留所有分子cell。输入SHA及权限检查见`matched_summary_inputs.json`、`matched_summary_verification.json`。

入口：`python -B studies/predictor/full_training_volume_response_audit_20261117/matched_summary.py`。这里的“配对”仅指同一模型比较中的同pair，不是随机化干预或随机seed因果配对。没有打开原始AL真值列、正式目标文件或test预测/真值。
'''
    (OUT / 'MATCHED_AL_COMPARISON.md').write_text(text)


def main():
    read(Path(__file__))
    source = OUT / 'matched_response_pairs.csv'
    raw = read(source)
    metadata = pd.read_csv(io.BytesIO(raw), usecols=['formal_seed', 'al_seed', 'axis', 'exposure', 'pair_key', 'low_id', 'high_id'])
    comparison_count = verify_permissions(metadata)
    pairs = pd.read_csv(io.BytesIO(raw))
    assert np.isfinite(pairs[['delta_true', 'delta_pred']].to_numpy()).all()
    ids = ['formal_seed', 'al_seed', 'axis', 'exposure', 'pair_key', 'target']
    assert not pairs.duplicated(ids + ['method']).any()
    assert pairs.groupby(ids).method.apply(lambda x: set(x) == set(METHODS)).all()
    assert pairs.groupby(ids).delta_true.apply(lambda x: x.max() - x.min() < 1e-12).all()
    metrics = pd.read_csv(io.BytesIO(read(OUT / 'matched_seed_metrics.csv')))
    maximum_error = validate_metrics(pairs, metrics)
    aggregate = hierarchical_methods(metrics, pairs)
    paired = paired_improvements(metrics)
    molecules = molecule_heterogeneity(pairs)
    counts = denominators(pairs)
    aggregate.to_csv(OUT / 'matched_exposure_equal_models.csv', index=False)
    paired.to_csv(OUT / 'matched_paired_improvements.csv', index=False)
    molecules.to_csv(OUT / 'matched_molecule_heterogeneity.csv', index=False)
    counts.to_csv(OUT / 'matched_denominators.csv', index=False)
    write_report(pairs, aggregate, paired, molecules, counts)
    verification = {'status': 'PASS_AUTHORIZED_MATCHED_DERIVED_SUMMARY', 'input_score_rows': len(pairs),
        'authorized_comparison_pairs': comparison_count,
        'unique_physical_pair_ids': len(pairs[['axis', 'pair_key']].drop_duplicates()),
        'metric_recalculation_max_error': maximum_error, 'source_test_or_quarantine_scored': 0,
        'new_target_reads': 0, 'new_model_inference': 0, 'new_fits': 0, 'new_API_calls': 0,
        'weighting': 'pair within formal/AL/exposure/endpoint; AL campaigns equal within formal; formal models equal',
        'independence': 'crossed predictions, not 15 independent randomized seed pairs'}
    (OUT / 'matched_summary_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    (OUT / 'matched_summary_inputs.json').write_text(json.dumps({'sources': list(INPUTS.values())}, indent=2) + '\n')
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
