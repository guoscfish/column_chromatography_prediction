#!/usr/bin/env python3
"""Independently recompute frozen prediction scores; no model imports or fitting."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from io import BytesIO, StringIO
from itertools import combinations
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SEEDS = [157, 887, 2357, 6101, 12203]
KINDS = ['A_original', 'B_amount_skip', 'C_volume_skip']
PRED_COLUMNS = ['V1_q10', 'V1_q50', 'V1_q90', 'V2_q10', 'V2_q50', 'V2_q90']
SOURCES: dict[str, dict] = {}
CHECKS: list[dict] = []


def read(path: Path) -> bytes:
    data = path.read_bytes()
    assert data == path.read_bytes(), f'Input changed during QA: {path}'
    key = str(path.relative_to(ROOT))
    entry = dict(path=key, sha256=sha256(data).hexdigest(), bytes=len(data))
    assert key not in SOURCES or SOURCES[key] == entry
    SOURCES[key] = entry
    return data


def csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(StringIO(read(path).decode()), **kwargs)


def js(path: Path) -> dict:
    return json.loads(read(path))


def ratio(value: str) -> Fraction:
    text = str(value).strip().replace(':', '/')
    if '/' in text:
        numerator, denominator = text.split('/')
        return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
    return Fraction(Decimal(text))


def coordinates(row: dict) -> tuple:
    return (row['canonical_smiles'], ratio(row['PE/EA']), Decimal(row['Density g/ml']),
            Decimal(row['V/ul']), row['loading solvent'], Decimal(row['Volume of loading solvent/ul']))


def series_key(coordinate: tuple) -> tuple:
    return coordinate[:3] + coordinate[4:]


def response_stats(actual, predicted, threshold=.5) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    meaningful = np.abs(actual) > threshold
    nonflat = np.abs(predicted) >= .1
    correct = np.sign(actual) == np.sign(predicted)
    return dict(n=len(actual), n_meaningful=int(meaningful.sum()),
                difference_error_mean=float(np.mean(np.abs(predicted - actual))),
                observed_abs_mean=float(np.mean(np.abs(actual))),
                predicted_abs_mean=float(np.mean(np.abs(predicted))),
                amplitude_error_mean_ml=float(np.mean(np.abs(np.abs(predicted) - np.abs(actual)))),
                direction_accuracy_meaningful=float(correct[meaningful].mean()),
                flat_fraction_meaningful=float((~nonflat)[meaningful].mean()),
                observation_threshold_ml=threshold,
                prediction_flat_threshold_ml=.1)


def row_stats(truth: np.ndarray, prediction: np.ndarray, scales: dict) -> dict:
    result = {}
    for endpoint in range(2):
        error = prediction[:, 3*endpoint+1] - truth[:, endpoint]
        result[f'V{endpoint+1}_rmse'] = float(np.sqrt(np.mean(error**2)))
        result[f'V{endpoint+1}_mae'] = float(np.mean(np.abs(error)))
        denominator = float(np.sum((truth[:, endpoint] - np.mean(truth[:, endpoint]))**2))
        result[f'V{endpoint+1}_r2'] = 1 - float(np.sum(error**2)) / denominator
    result['combined_normalized_rmse'] = math.sqrt(sum(
        (result[f'V{i}_rmse']/scales[f'V{i}'])**2 for i in [1, 2]) / 2)
    return result


def compare(name: str, actual: pd.DataFrame, released: pd.DataFrame,
            keys: list[str], columns: list[str], tolerance=2e-12) -> None:
    assert not actual.duplicated(keys).any(), name
    assert not released.duplicated(keys).any(), name
    a, b = actual.set_index(keys).sort_index(), released.set_index(keys).sort_index()
    assert a.index.equals(b.index), (name, 'keys disagree')
    largest = 0.
    for column in columns:
        x, y = a[column], b[column]
        if pd.api.types.is_bool_dtype(x):
            assert np.array_equal(x.to_numpy(), y.to_numpy()), (name, column)
        else:
            difference = np.abs(x.to_numpy(float) - y.to_numpy(float))
            assert np.isfinite(difference).all(), (name, column)
            largest = max(largest, float(difference.max(initial=0)))
            assert float(difference.max(initial=0)) <= tolerance, (name, column, difference.max())
    CHECKS.append(dict(table=name, rows=len(a), compared_columns=columns,
                       maximum_absolute_difference=largest, tolerance=tolerance, status='PASS'))


def main() -> None:
    gate = js(OUT/'DATA_FEASIBILITY.json')
    for filename, expected in gate['training_code_sha256'].items():
        assert sha256(read(OUT/filename)).hexdigest() == expected
    assert len(list((OUT/'fits').glob('seed_*/*/started.json'))) == 15
    assert len(list((OUT/'fits').glob('seed_*/*/fit_audit.json'))) == 15
    assert not list((OUT/'fits').glob('seed_*/*/FAILED.json'))
    rows, endpoints, primary, robustness, independent_pairs, metadata_audit = [], [], [], [], [], []
    maximum_wide_delta_difference = 0.
    prediction_files = 0
    exact_prediction_order_files = 0
    for seed in SEEDS:
        identity = js(OUT/f'data/seed_{seed}/identity.json')
        label_path = OUT/f'data/seed_{seed}/authorized_labels.npz'
        label_bytes = read(label_path)
        assert sha256(label_bytes).hexdigest() == identity['authorized_labels_sha256']
        metadata_path = OUT/f'data/seed_{seed}/metadata.csv'
        metadata = csv(metadata_path, dtype=str, keep_default_na=False)
        assert SOURCES[str(metadata_path.relative_to(ROOT))]['sha256'] == identity['metadata_sha256']
        assert metadata.sample_id.tolist() == identity['ordered_ids']
        assert not metadata.sample_id.duplicated().any()
        coordinates_by_id = {r['sample_id']: coordinates(r) for r in metadata.to_dict('records')}
        grouped = defaultdict(list)
        for sid in identity['check_ids']:
            grouped[series_key(coordinates_by_id[sid])].append(sid)
        val_series = {series_key(coordinates_by_id[sid]) for sid in identity['validation_ids']}
        train_series = {series_key(coordinates_by_id[sid]) for sid in identity['train_ids']}
        assert not set(grouped) & train_series
        assert len(grouped) == 10
        related = set(grouped) & val_series
        exact_validation = {coordinates_by_id[sid] for sid in identity['validation_ids']}
        pair_ids = []
        for a, b in combinations(identity['check_ids'], 2):
            ca, cb = coordinates_by_id[a], coordinates_by_id[b]
            if series_key(ca) != series_key(cb) or ca[3] == cb[3]:
                continue
            low, high = (a, b) if ca[3] < cb[3] else (b, a)
            pair_ids.append((low, high, series_key(ca)))
        assert pair_ids
        metadata_audit.append(dict(seed=seed, check_rows=len(identity['check_ids']),
                                  validation_rows=len(identity['validation_ids']),
                                  volume_series=len(grouped), volume_pairs_per_endpoint=len(pair_ids),
                                  training_related_series=0, validation_related_series=len(related),
                                  validation_exact_condition_check_rows=sum(
                                      coordinates_by_id[sid] in exact_validation for sid in identity['check_ids']),
                                  validation_unrelated_series=10-len(related),
                                  validation_unrelated_pairs_per_endpoint=sum(k not in related for _, _, k in pair_ids)))
        with np.load(BytesIO(label_bytes), allow_pickle=False) as archive:
            # Deliberately access only the two authorized evaluation projections.
            truths = {scope: archive[f'{scope}_y'].copy() for scope in ['check', 'validation']}
        for scope, truth in truths.items():
            assert truth.shape == (len(identity[f'{scope}_ids']), 2)
            assert truth.dtype == np.float32 and np.isfinite(truth).all()
        check_positions = {sid: i for i, sid in enumerate(identity['check_ids'])}
        for kind in KINDS:
            base = OUT/f'fits/seed_{seed}/{kind}'
            audit = js(base/'fit_audit.json')
            assert audit['status'] == 'COMPLETE' and audit['seed'] == seed and audit['kind'] == kind
            assert audit['test_truth_used'] == audit['unobserved_targets_used'] == 0
            assert audit['holdout_used_for_early_stopping'] is False
            for scope in ['check', 'validation']:
                ids = identity[f'{scope}_ids']
                table = csv(base/f'{scope}_predictions.csv', dtype={'sample_id': str})
                prediction_files += 1
                exact_prediction_order_files += int(table.sample_id.tolist() == ids)
                assert table.columns.tolist() == ['sample_id'] + PRED_COLUMNS
                assert not table.sample_id.duplicated().any() and set(table.sample_id) == set(ids)
                prediction = table.set_index('sample_id').loc[ids, PRED_COLUMNS].to_numpy(float)
                assert np.isfinite(prediction).all()
                rm = row_stats(truths[scope], prediction, identity['preprocessing']['target_scales'])
                rows.append(dict(seed=seed, kind=kind, scope=scope, n=len(ids), **rm))
                if scope == 'validation':
                    assert abs(rm['combined_normalized_rmse']-audit['best_validation_nrmse']) < 1e-7
                    validation_row = rm['combined_normalized_rmse']
                    continue
                seed_endpoints, series_scores, unrelated_scores, wide_endpoint_scores = [], [], [], []
                for endpoint in range(2):
                    all_actual, all_predicted, wide_actual = [], [], []
                    by_series = defaultdict(lambda: ([], []))
                    unrelated_actual, unrelated_predicted = [], []
                    for low, high, key in pair_ids:
                        i, j = check_positions[low], check_positions[high]
                        # Preserve the released float32-label arithmetic, and audit
                        # the independent float64 subtraction as a precision check.
                        actual = float(truths['check'][j, endpoint]-truths['check'][i, endpoint])
                        wide = float(truths['check'][j, endpoint])-float(truths['check'][i, endpoint])
                        predicted = float(prediction[j, 3*endpoint+1]-prediction[i, 3*endpoint+1])
                        maximum_wide_delta_difference = max(maximum_wide_delta_difference, abs(actual-wide))
                        all_actual.append(actual); all_predicted.append(predicted); wide_actual.append(wide)
                        by_series[key][0].append(actual); by_series[key][1].append(predicted)
                        if key not in related:
                            unrelated_actual.append(actual); unrelated_predicted.append(predicted)
                        independent_pairs.append(dict(seed=seed, kind=kind, target=f'V{endpoint+1}',
                            low_id=low, high_id=high, actual_delta_ml=actual, predicted_delta_ml=predicted,
                            amplitude_error_ml=abs(abs(predicted)-abs(actual)), validation_related_series=key in related))
                    for threshold in [.1, .5, 1.]:
                        endpoints.append(dict(seed=seed, kind=kind, scope='check', axis='V/ul',
                            target=f'V{endpoint+1}', n_series=len(by_series),
                            **response_stats(all_actual, all_predicted, threshold)))
                    seed_endpoints.append(response_stats(all_actual, all_predicted))
                    series_scores.append(float(np.mean([
                        np.mean(np.abs(np.asarray(p)-np.asarray(a))) for a, p in by_series.values()])))
                    unrelated_scores.append(response_stats(unrelated_actual, unrelated_predicted)['difference_error_mean'])
                    wide_endpoint_scores.append(response_stats(wide_actual, all_predicted)['difference_error_mean'])
                local = float(np.mean([r['difference_error_mean'] for r in seed_endpoints]))
                primary.append(dict(seed=seed, kind=kind, difference_error_mean=local,
                                    combined_normalized_rmse=np.nan,
                                    float64_label_subtraction_mae=float(np.mean(wide_endpoint_scores))))
                robustness.append(dict(seed=seed, kind=kind, difference_error_mean_primary=local,
                    difference_error_mean_series_equal=float(np.mean(series_scores)),
                    difference_error_mean_validation_unrelated=float(np.mean(unrelated_scores))))
    raw_rows = pd.DataFrame(rows)
    primary = pd.DataFrame(primary).drop(columns='combined_normalized_rmse').merge(
        raw_rows.query("scope=='validation'")[['seed','kind','combined_normalized_rmse']], on=['seed','kind'], validate='one_to_one')
    endpoint_table, robust_table = pd.DataFrame(endpoints), pd.DataFrame(robustness)
    pair_table = pd.DataFrame(independent_pairs)
    numeric_rows = ['n','V1_rmse','V2_rmse','V1_mae','V2_mae','combined_normalized_rmse']
    # R2's original denominator is float32 arithmetic; it is outside the gates.
    compare('row_metrics.csv', raw_rows, csv(OUT/'row_metrics.csv'), ['seed','kind','scope'], numeric_rows)
    response_columns = ['n','n_meaningful','n_series','difference_error_mean','observed_abs_mean',
                        'predicted_abs_mean','amplitude_error_mean_ml','direction_accuracy_meaningful',
                        'flat_fraction_meaningful','prediction_flat_threshold_ml']
    released_local = csv(OUT/'local_response_metrics.csv').query("scope=='check' and axis=='V/ul'")
    compare('local_response_metrics.csv/check/V', endpoint_table, released_local,
            ['seed','kind','target','observation_threshold_ml'], response_columns)
    compare('control_endpoint_summary.csv', endpoint_table.query('observation_threshold_ml==.5'),
            csv(OUT/'control_endpoint_summary.csv'), ['seed','kind','target'],
            [c for c in response_columns if c not in ['n_meaningful','prediction_flat_threshold_ml']])
    released_pairs = csv(OUT/'control_response_pairs.csv').query("scope=='check' and axis=='V/ul'")
    compare('control_response_pairs.csv/check/V', pair_table, released_pairs,
            ['seed','kind','target','low_id','high_id'],
            ['actual_delta_ml','predicted_delta_ml','amplitude_error_ml','validation_related_series'])
    compare('control_seed_summary.csv', primary, csv(OUT/'control_seed_summary.csv'),
            ['seed','kind'], ['difference_error_mean','combined_normalized_rmse'])
    compare('control_robustness.csv', robust_table, csv(OUT/'control_robustness.csv'),
            ['seed','kind'], [c for c in robust_table.columns if c not in ['seed','kind']])
    equal_seed = primary.groupby('kind')[['difference_error_mean','combined_normalized_rmse']].mean().reset_index()
    compare('control_equal_seed_summary.csv', equal_seed, csv(OUT/'control_equal_seed_summary.csv'),
            ['kind'], ['difference_error_mean','combined_normalized_rmse'])
    paired, gates = [], []
    indexed = primary.set_index(['seed','kind'])
    equal = equal_seed.set_index('kind')
    for comparison, first, second in [('B-A',KINDS[1],KINDS[0]), ('C-A',KINDS[2],KINDS[0]), ('C-B',KINDS[2],KINDS[1])]:
        local_deltas, row_deltas, local_relative, row_relative = [], [], [], []
        for seed in SEEDS:
            a, b = indexed.loc[(seed,first)], indexed.loc[(seed,second)]
            local_deltas.append(a.difference_error_mean-b.difference_error_mean)
            row_deltas.append(a.combined_normalized_rmse-b.combined_normalized_rmse)
            local_relative.append(a.difference_error_mean/b.difference_error_mean-1)
            row_relative.append(a.combined_normalized_rmse/b.combined_normalized_rmse-1)
            paired.append(dict(seed=seed,comparison=comparison,row_nrmse_first=a.combined_normalized_rmse,
                row_nrmse_second=b.combined_normalized_rmse,row_nrmse_delta=row_deltas[-1],row_relative_delta=row_relative[-1],
                local_difference_mae_first=a.difference_error_mean,local_difference_mae_second=b.difference_error_mean,
                local_difference_mae_delta=local_deltas[-1],local_relative_delta=local_relative[-1]))
        local_improvement = 1-equal.loc[first,'difference_error_mean']/equal.loc[second,'difference_error_mean']
        row_change = equal.loc[first,'combined_normalized_rmse']/equal.loc[second,'combined_normalized_rmse']-1
        positives, nonworse = int((np.array(local_deltas)<0).sum()), int((np.array(row_deltas)<=0).sum())
        gates.append(dict(comparison=comparison,local_improvement_ratio_equal_seed_means=local_improvement,
            local_positive_seeds=positives,mean_paired_seed_relative_local_improvement=float(-np.mean(local_relative)),
            row_relative_change_equal_seed_means=row_change,row_nonworse_seeds=nonworse,
            row_worse_over_one_percent_seeds=int((np.array(row_relative)>.01).sum()),
            local_twenty_percent=bool(local_improvement>=.2),local_four_seeds=positives>=4,
            row_mean_not_worse_over_one_percent=bool(row_change<=.01),
            mechanism_reference_pass=bool(local_improvement>=.2 and positives>=4 and row_change<=.01),
            row_relevance_reference_pass=bool(row_change<=-.01 and nonworse>=4)))
    paired, gates = pd.DataFrame(paired), pd.DataFrame(gates)
    compare('paired_model_differences.csv', paired, csv(OUT/'paired_model_differences.csv'),
            ['seed','comparison'], list(paired.columns[2:]))
    compare('mechanism_gates.csv', gates, csv(OUT/'mechanism_gates.csv'), ['comparison'], list(gates.columns[1:]))
    assert prediction_files == exact_prediction_order_files == 30
    expected_pair_events = sum(r['volume_pairs_per_endpoint'] for r in metadata_audit) * 2 * len(KINDS)
    assert len(primary) == 15 and len(pair_table) == expected_pair_events
    assert maximum_wide_delta_difference <= 8e-6
    supplement = js(OUT/'ANALYSIS_SUPPLEMENT.json')
    read(OUT/'evaluate_controls.py'); read(OUT/'report_controls.py')
    read(OUT/'check_series_validation_overlap.csv')
    # Confirm the read set did not change while the analysis was running.
    for entry in list(SOURCES.values()):
        assert sha256((ROOT/entry['path']).read_bytes()).hexdigest() == entry['sha256']
    result = dict(status='PASS', new_fits=0, model_inference_calls=0, api_calls=0,
        prohibited_target_sources_read=[], checkpoint_files_read=[], authorized_label_arrays_accessed=['check_y','validation_y'],
        primary_definition='Within each seed, mean absolute signed-difference error over every distinct-volume pair, equally average V1 and V2, then equally average five seeds. No meaningful-response filtering for MAE.',
        row_definition='Original 416-row validation; Q50 endpoint RMSE normalized by frozen L333 target scales, then root mean square of the two normalized endpoint RMSE values; equal five-seed averaging.',
        independent_implementation='No imports from evaluate_controls, report_controls, model_learnability, point_metrics, torch or graph/model modules. Metadata series and formulas implemented here.',
        prediction_files=prediction_files, exact_prediction_order_files=exact_prediction_order_files,
        id_alignment='Unique exact ID set checked for every file; reindex by authoritative identity before scoring.',
        checks=CHECKS, metadata_denominators=metadata_audit,
        primary_seed_results=primary.to_dict('records'), equal_seed_results=equal_seed.to_dict('records'),
        endpoint_results=endpoint_table.query('observation_threshold_ml==.5').to_dict('records'),
        robustness_seed_results=robust_table.to_dict('records'),
        robustness_equal_seed_results=robust_table.groupby('kind').mean(numeric_only=True).drop(columns='seed').reset_index().to_dict('records'),
        paired_results=paired.to_dict('records'), gate_results=gates.to_dict('records'),
        float_precision=dict(released_label_dtype='float32',pair_truth_subtraction='float32, inherited from projection; reproduced exactly',
            maximum_float64_pair_delta_difference_ml=maximum_wide_delta_difference,
            maximum_primary_mae_difference_under_float64_subtraction_ml=float(np.max(np.abs(primary.difference_error_mean-primary.float64_label_subtraction_mae)))),
        sensitivity_declaration=supplement, input_sources=sorted(SOURCES.values(),key=lambda r:r['path']),
        qa_script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'RESULTS_INDEPENDENT_QA.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def markdown(frame: pd.DataFrame) -> str:
        columns=list(frame.columns)
        def fmt(v): return f'{v:.9f}' if isinstance(v,(float,np.floating)) else str(v)
        return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))
    text = '''# 15次固定拟合结果的独立QA

结论：**PASS**。直接读取15个拟合各自的check与原validation预测CSV，共30个文件；只使用本轮授权投影中的`check_y`与`validation_y`。每文件校验ID唯一性、ID集合完整性，再按identity的ID索引对齐，未用预测行号猜测标签。30份文件的原始顺序也全部一致。没有训练、模型推理、API请求、checkpoint读取或test/未揭示池标签读取。

## 独立重算口径与分母

本QA没有导入原评估、汇总、pair分类或point_metrics函数，也没有导入torch。独立解析SMILES、PE/EA分数、密度、上样溶剂与其体积，以同分子且全部非样品体积条件相同建立完整系列。每系列中样品体积不同的所有ID pair纳入；重复条件记录仍保留为独立ID，体积相同的两条记录不作为体积变化pair。H1005训练与检查系列重叠仍为0。

主指标：每seed内先对全部体积pair计算`abs((pred_high−pred_low)−(truth_high−truth_low))`，每端点求均值，再V1/V2等权，最后五seed等权。0.1/0.5/1mL阈值仅改变方向/平坦率的有意义真实变化集合，不过滤主MAE。平坦预测为绝对差分严格小于0.1mL；有意义真实变化为严格大于阈值。不得把pair数量作为科研重复数。

Row指标使用原416行validation的q50，每端点RMSE除以固定L333目标尺度，再求两端点归一化RMSE的平方均值开方；五seed等权，未以检查集Row成绩替换。

'''+markdown(pd.DataFrame(metadata_audit))+'''\n
## 逐seed与五seed等权

'''+markdown(primary[['seed','kind','difference_error_mean','combined_normalized_rmse']])+'''\n
'''+markdown(equal_seed)+'''\n
逐endpoint及逐pair对应的核对覆盖全部{expected_pair_events}条体积pair×端点×模型事件。逐seed主MAE、Row RMSE/MAE/NRMSE、各阈值方向/平坦率、幅度、pair身份与validation关联标记均与发布表一致；每项最大绝对误差与输入SHA见JSON。这是预测评分事件数量，不能当作独立实验重复数。

## 配对与参考门槛

'''+markdown(gates[['comparison','local_improvement_ratio_equal_seed_means','local_positive_seeds','row_relative_change_equal_seed_means','row_nonworse_seeds','row_worse_over_one_percent_seeds','mechanism_reference_pass','row_relevance_reference_pass']])+'''\n
15条逐seed模型配对差与全部参考门槛逐项一致。主要相对改善使用“五seed等权均值之比”；另存的“逐seed相对改善再平均”不是门槛计算依据。20%局部改善、至少4/5seed局部改善、平均Row恶化不超过1%三个条件联合判定机制参考门槛；Row相关性另需平均改善至少1%且至少4/5seed不变差。原计划没有完全明确相对改善聚合公式和Row1%应对均值还是每seed应用，不能宣称这些具体公式全部预注册；当前报告分列两种相对口径及逐seed恶化，结论不随这些解释改变。三组对比均未通过这两个参考门槛；没有用稳健性结果替换主口径。

## 两项描述性敏感性

series等权先在每个系列内平均pair误差，再在seed内系列等权、端点等权，最后五seed等权。validation关联剔除则独立从metadata定位任何有同系列原validation记录的检查系列，将其整体剔除，再保留原pair权重及同一冻结预测。剔除系列数为2/3/2/0/2，剩余8/7/8/10/8；2357及12203各有一个检查ID与validation的完整条件相同，已整体排除其系列。这些关联不含检查ID直接出现在训练中。

'''+markdown(robust_table.groupby('kind').mean(numeric_only=True).drop(columns='seed').reset_index())+'''\n
两项敏感性分别匹配`control_robustness.csv`的15条seed记录；声明发生在完成1次拟合时，明确标注描述性，不改变主门槛或checkpoint，也不补训练。原validation关联意味着检查集不是完全独立确认集，此限制不能因训练系列重叠为0而省略。

## 精度与复现

授权投影标签为float32，发布评估的真实pair差先以float32相减；QA逐项重现该精度，同时独立用float64标签相减检查舍入影响。JSON列出最大差异及主MAE影响，不改变任何结果或门槛。Q50 Row误差用CSV浮点预测直接计算。

运行`results_independent_qa.py`即可只用已冻结数值重建本报告与JSON。全部读取文件的SHA均记录，运行结束再次核对无变化；训练冻结代码SHA仍与DATA_FEASIBILITY一致。没有改动任何原始预测、标签、评估代码、训练文件或既有报告。
'''
    text = text.replace('{expected_pair_events}', str(expected_pair_events))
    (OUT/'RESULTS_INDEPENDENT_QA.md').write_text(text)
    print(json.dumps(dict(status='PASS', source_files=len(SOURCES), comparisons=len(CHECKS), prediction_files=prediction_files,
                         pair_endpoint_model_events=len(pair_table), maximum_float64_pair_delta_difference_ml=maximum_wide_delta_difference,
                         equal_seed_results=equal_seed.to_dict('records')),indent=2))


if __name__ == '__main__':
    main()
