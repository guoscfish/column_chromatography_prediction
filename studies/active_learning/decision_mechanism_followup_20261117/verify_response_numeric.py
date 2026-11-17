#!/usr/bin/env python3
"""Independently recompute response-summary arithmetic from accepted-pair tables."""
from pathlib import Path
import hashlib
import json
import pandas as pd

OUT = Path(__file__).resolve().parent


def main():
    pair_path = OUT / 'condition_response_pairs.csv'
    pairs = pd.read_csv(pair_path)
    errors, checked = [], 0
    for filename, keys in [
        ('response_seed_summary.csv', ['method', 'source_half', 'seed', 'axis', 'target']),
        ('response_phase_summary.csv', ['method', 'source_half', 'seed', 'phase', 'axis', 'target'])]:
        summary = pd.read_csv(OUT / filename)
        lookup = {key: group for key, group in pairs.groupby(keys, dropna=False)}
        for row in summary.to_dict('records'):
            group = lookup[tuple(row[key] for key in keys)]
            actual = [float(value) for value in group.actual_delta_ml]
            predicted = [float(value) for value in group.predicted_delta_ml]
            threshold, flat = row['observation_threshold_ml'], row['prediction_flat_threshold_ml']
            meaningful = [i for i, value in enumerate(actual) if abs(value) > threshold]
            nonflat = [i for i in meaningful if abs(predicted[i]) >= flat]

            def sign(value):
                return (value > 0) - (value < 0)

            result = {'n': len(actual), 'n_meaningful': len(meaningful),
                      'n_unique_pairs': group.pair_key.nunique(), 'n_molecules': group.molecule_key.nunique(),
                      'n_batches': group['round'].nunique(),
                      'observed_abs_mean': sum(map(abs, actual)) / len(actual),
                      'predicted_abs_mean': sum(map(abs, predicted)) / len(predicted),
                      'difference_error_mean': sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual),
                      'flat_prediction_fraction': sum(abs(value) < flat for value in predicted) / len(predicted),
                      'excess_difference_error_over_flat': sum(abs(a - p) - abs(a) for a, p in zip(actual, predicted)) / len(actual)}
            if meaningful:
                result.update(
                    direction_accuracy_meaningful=sum(sign(actual[i]) == sign(predicted[i]) for i in meaningful) / len(meaningful),
                    direction_correct_nonflat_fraction_meaningful=sum(sign(actual[i]) == sign(predicted[i]) and abs(predicted[i]) >= flat for i in meaningful) / len(meaningful),
                    wrong_direction_nonflat_fraction_meaningful=sum(actual[i] * predicted[i] < 0 and abs(predicted[i]) >= flat for i in meaningful) / len(meaningful),
                    flat_fraction_meaningful=sum(abs(predicted[i]) < flat for i in meaningful) / len(meaningful))
            if nonflat:
                result['direction_accuracy_given_nonflat_meaningful'] = sum(sign(actual[i]) == sign(predicted[i]) for i in nonflat) / len(nonflat)
            errors += [abs(float(row[key]) - value) for key, value in result.items()]
            checked += 1
    assert max(errors) < 1e-10
    new = pairs[(pairs.method == 'new_free_cw_sol61_xhigh') & (pairs.source_half == 'all32') & (pairs.axis == 'V/ul')]
    assert len(new) == 330 and len(new[['seed', 'pair_key']].drop_duplicates()) == 165
    reused = {}
    seed_equal = {}
    for endpoint, group in new.groupby('target'):
        reused[endpoint] = {'observed_abs_mean': sum(abs(value) for value in group.actual_delta_ml) / len(group),
                            'predicted_abs_mean': sum(abs(value) for value in group.predicted_delta_ml) / len(group)}
        by_seed = [q for _, q in group.groupby('seed')]
        seed_equal[endpoint] = {'observed_abs_mean': sum(sum(abs(value) for value in q.actual_delta_ml) / len(q) for q in by_seed) / len(by_seed),
                                'predicted_abs_mean': sum(sum(abs(value) for value in q.predicted_delta_ml) / len(q) for q in by_seed) / len(by_seed)}
    result = {'status': 'PASS_INDEPENDENT_RESPONSE_SUMMARY_ARITHMETIC',
              'summary_rows_checked': checked, 'maximum_absolute_error': max(errors),
              'reused_165_pairs': reused,
              'reused_165_pairs_seed_equal': seed_equal,
              'pair_table_sha256': hashlib.sha256(pair_path.read_bytes()).hexdigest(),
              'scope': 'Arithmetic of already accepted same-batch responses; no counterfactual truth lookup'}
    (OUT / 'response_independent_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'MODEL_NUMERIC_REVIEW.md').write_text(f'''# 模型条件响应的独立数值复核

状态：**通过**。未调用被测`pair_numbers`，以标量求和、符号分类和严格阈值比较独立核对逐seed及逐阶段汇总，共{checked}行。核对n、真实变化超过阈值的数量、唯一实验对/分子/批数、实际与预测绝对变化均值、差分误差、相对平坦预测的误差差值、flat比例和三种方向分类；最大绝对误差{max(errors):.3g}。

New CW的样品体积子集是330个端点事件，即165个同批实验对。先seed内按pair均值、再seed等权汇总时，独立复得旧报告V1实际/预测绝对变化为{seed_equal['V1']['observed_abs_mean']:.6f}/{seed_equal['V1']['predicted_abs_mean']:.6f}mL，V2为{seed_equal['V2']['observed_abs_mean']:.6f}/{seed_equal['V2']['predicted_abs_mean']:.6f}mL。若165对直接pair加权，V1为{reused['V1']['observed_abs_mean']:.6f}/{reused['V1']['predicted_abs_mean']:.6f}mL，V2为{reused['V2']['observed_abs_mean']:.6f}/{reused['V2']['predicted_abs_mean']:.6f}mL。两者口径不同，不应将旧seed等权数字直接称作165对加权均值。

复核了数据流：主模块先匹配batch freeze、label-access receipt、已接受ID及原始预测，再投影该实际批已揭示反馈。Counterfactual模块不读取该反馈，也没有向未选候选查询标签。这里的独立计算证明汇总算术一致，不额外证明实验条件真值正确或模型瓶颈的唯一原因。

`all32`与Hybrid来源子批存在包含关系，不能把不同source_half行简单相加作为独立实验重复。方向准确率区分所有有意义真实变化、非flat预测条件下的正确率，以及flat/错误方向占比；没有用近零真值作分母。

可执行复核入口：`python -B studies/active_learning/decision_mechanism_followup_20261117/verify_response_numeric.py`。绑定的配对表SHA256为`{result['pair_table_sha256']}`。本审阅不修改模型条件响应数据、训练配置或旧封存结果。
''')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
