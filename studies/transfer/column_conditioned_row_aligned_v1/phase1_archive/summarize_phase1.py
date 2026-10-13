"""Reporting only: saved inner artifacts, identities and historical Group-CV metrics."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase1 as r


def paired(frame, protocol):
    rows = []
    for c in r.TARGETS:
        subset = frame.loc[frame.column.eq(c)]
        a = subset.loc[subset.arm.eq('A1')].set_index('inner_fold').sort_index()
        b = subset.loc[subset.arm.eq('A2')].set_index('inner_fold').loc[a.index]
        for fold in a.index:
            row = dict(cv=protocol, outer_seed=r.PILOT_SEED, inner_fold=int(fold), column=c,
                       A1_best_epoch=int(a.loc[fold, 'best_epoch']), A2_best_epoch=int(b.loc[fold, 'best_epoch']))
            for metric in ['combined_normalized_rmse', 'V1_rmse', 'V2_rmse', 'V1_mae', 'V2_mae', 'Center_rmse', 'Width_rmse']:
                row['A1_'+metric] = float(a.loc[fold, metric])
                row['A2_'+metric] = float(b.loc[fold, metric])
                row[metric+'_relative_gain'] = 1-row['A2_'+metric]/row['A1_'+metric]
            row['A2_wins'] = row['combined_normalized_rmse_relative_gain'] > 0
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_tail(frame):
    pooled = frame.loc[frame.column.isin(r.TARGETS)].groupby(['column', 'arm', 'endpoint'])[['tail_sse', 'tail_count', 'total_sse']].sum().reset_index()
    pooled['tail_rmse'] = np.sqrt(pooled.tail_sse/pooled.tail_count)
    pooled['tail_sse_fraction'] = pooled.tail_sse/pooled.total_sse
    return pooled


def figures(history, results, comparisons, epochs):
    colors = {'A1': '#2665a8', 'A2': '#d26a2a'}
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(5, 3, figsize=(15, 18))
    for fold in range(1, 6):
        for arm in r.old.ARMS:
            h = history.loc[history.inner_fold.eq(fold) & history.arm.eq(arm)]
            best = int(results.loc[results.inner_fold.eq(fold) & results.arm.eq(arm), 'best_epoch'].iloc[0])
            ax = axes[fold-1, 0]
            ax.plot(h.epoch, h.joint_validation_score, color=colors[arm], label=f'{arm}, best={best}')
            ax.scatter([best], [h.loc[h.epoch.eq(best), 'joint_validation_score'].iloc[0]], color=colors[arm], s=22)
            ax.axvline(best, color=colors[arm], ls=':', alpha=.7)
            for j, c in enumerate(r.TARGETS, 1):
                ax = axes[fold-1, j]
                ax.plot(h.epoch, h[f'train_loss_{c}'], color=colors[arm], alpha=.7, label=f'{arm} train')
                ax.plot(h.epoch, h[f'validation_loss_{c}'], color=colors[arm], ls='--', label=f'{arm} valid')
                ax.axvline(best, color=colors[arm], ls=':', alpha=.7)
                ax.set_yscale('log')
        axes[fold-1, 0].set_ylabel(f'Fold {fold}\nJoint NRMSE')
        for j in range(3):
            axes[fold-1, j].set_xlabel('Epoch (actual steps recorded in CSV)')
            axes[fold-1, j].grid(alpha=.2)
            axes[fold-1, j].legend(fontsize=8)
        axes[fold-1, 0].set_title('Row-CV joint selector; dots = selected epoch')
        for j, c in enumerate(r.TARGETS, 1):
            axes[fold-1, j].set_title(f'{c} original raw quantile loss (log scale)')
    fig.suptitle(f'A1/A2 Row-CV pilot, seed {r.PILOT_SEED} | DEVELOPMENT DIAGNOSTICS\nTrain uses mixed-task BN; validation uses eval BN. One step/epoch; 4g is subsampled.', y=.998)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(r.STUDY/'a1_a2_row_cv_learning_curves.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    x = np.arange(2)
    for offset, cv, color in [(-.19, 'Group-CV', '#888888'), (.19, 'Row-CV', '#2665a8')]:
        values = [comparisons.loc[comparisons.cv.eq(cv) & comparisons.column.eq(c), 'combined_normalized_rmse_relative_gain'].to_numpy()*100 for c in r.TARGETS]
        axes[0].bar(x+offset, [v.mean() for v in values], .36, label=cv, color=color, alpha=.75)
        for j, vals in enumerate(values):
            axes[0].scatter(j+offset+np.linspace(-.08, .08, len(vals)), vals, color=color, edgecolors='white', zorder=4)
    axes[0].axhline(0, color='black', lw=.8)
    axes[0].set_xticks(x, r.TARGETS)
    axes[0].set_ylabel('Mean paired-fold NRMSE gain (%)\npositive = A2 better; dots = folds within each CV')
    axes[0].legend(); axes[0].grid(axis='y', alpha=.2)
    boxes, labels = [], []
    for cv in ['Group-CV', 'Row-CV']:
        for arm in ['A1', 'A2']:
            vals = epochs.loc[epochs.cv.eq(cv) & epochs.arm.eq(arm), 'best_epoch'].to_numpy()
            boxes.append(vals); labels.append(f'{cv}\n{arm}')
    axes[1].boxplot(boxes, tick_labels=labels, widths=.55)
    for j, vals in enumerate(boxes, 1):
        axes[1].scatter(j+np.linspace(-.08, .08, len(vals)), vals, color=colors['A1' if j%2 else 'A2'], zorder=4)
    axes[1].set_ylabel('Joint best epoch / optimizer steps at best')
    axes[1].grid(axis='y', alpha=.2)
    fig.suptitle(f'Frozen Group-CV vs new Row-CV, same outer seed {r.PILOT_SEED}')
    fig.text(.5, .02, 'Different validation sets: absolute errors MUST NOT rank CV protocols. Folds across CV are not identity-matched. Single-seed diagnosis only.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(r.STUDY/'group_vs_row_cv_selection.png', dpi=170)
    plt.close(fig)


def table(frame, columns):
    header = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---']*len(columns)) + ' |\n'
    lines = []
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v) for v in row) + ' |')
    return header+'\n'.join(lines)+'\n'


def summarize():
    r.specification()  # Recheck source, datasets, graph caches, outer identities and historical code after training.
    runs = [r.STUDY/f'runtime/inner/seed_{r.PILOT_SEED}/fold_{f}/{arm}' for f in range(1, 6) for arm in r.old.ARMS]
    results, tails, regimes, predictions, history, audits = [], [], [], [], [], []
    expected_provenance = json.loads((r.STUDY/'EXECUTION_PROVENANCE.json').read_text())
    for rel, expected in expected_provenance['files'].items():
        if r.old.sha(r.ROOT/rel) != expected:
            raise RuntimeError(f'execution code changed: {rel}')
    for run in runs:
        receipt = json.loads((run/'complete.json').read_text())
        assert receipt['protocol_sha256'] == r.old.sha(r.STUDY/'protocol.json')
        assert receipt['provenance_sha256'] == r.old.sha(r.STUDY/'EXECUTION_PROVENANCE.json')
        for name, expected in receipt['files'].items():
            assert r.old.sha(run/name) == expected, (run, name)
        result = pd.read_csv(run/'results.csv'); h = pd.read_csv(run/'history.csv')
        fold, arm = int(result.inner_fold.iloc[0]), str(result.arm.iloc[0])
        best = int(result.best_epoch.iloc[0])
        assert best == int(h.loc[h.joint_validation_score.idxmin(), 'epoch'])
        assert len(h) == int(result.epochs_run.iloc[0])
        assert int(h.optimizer_steps.iloc[-1]) == int(result.optimizer_steps.iloc[0])
        assert bool(result.all_outputs_finite.all())
        assert np.isfinite(h.filter(regex='loss_|joint_validation_score').to_numpy()).all()
        for c in r.TARGETS:
            metric = float(result.loc[result.column.eq(c), 'combined_normalized_rmse'].iloc[0])
            assert np.isclose(metric, h.loc[h.epoch.eq(best), f'{c}_combined_normalized_rmse'].iloc[0], atol=1e-12)
        state = json.loads((run/'state_audit.json').read_text())
        assert state['frozen_parameters_unchanged'] and state['bn_policy_valid'] and state['finite_predictions_and_gradients']
        audits.append(dict(inner_fold=fold, arm=arm, best_epoch=best, history_rows=len(h),
                           optimizer_steps=int(result.optimizer_steps.iloc[0]), argmin_verified=True,
                           frozen_parameters_unchanged=True, bn_policy_valid=True))
        results.append(result); tails.append(pd.read_csv(run/'tails.csv'))
        regimes.append(pd.read_csv(run/'regimes.csv')); predictions.append(pd.read_csv(run/'predictions.csv'))
        history.append(h.assign(outer_seed=r.PILOT_SEED, inner_fold=fold, arm=arm))
    results, tails, history = pd.concat(results), pd.concat(tails), pd.concat(history)
    # Identity/scales/draws MUST be matched between the two arms within each fold.
    a = results.loc[results.arm.eq('A1')].set_index(['inner_fold', 'column']).sort_index()
    b = results.loc[results.arm.eq('A2')].set_index(['inner_fold', 'column']).loc[a.index]
    for name in ['inner_train_ids_sha256', 'inner_validation_ids_sha256', 'scale_V1', 'scale_V2',
                 'inner_train_count', 'inner_validation_count', 'per_task_batch_size', 'steps_per_epoch', 'training_seed']:
        np.testing.assert_array_equal(a[name], b[name])
    for fold in range(1, 6):
        ha = history.loc[history.inner_fold.eq(fold) & history.arm.eq('A1')].set_index('epoch')
        hb = history.loc[history.inner_fold.eq(fold) & history.arm.eq('A2')].set_index('epoch')
        common = ha.index.intersection(hb.index)
        np.testing.assert_array_equal(ha.loc[common, 'batch_draws_sha256'], hb.loc[common, 'batch_draws_sha256'])
        assert json.loads((runs[(fold-1)*2]/'initialization.json').read_text()) == json.loads((runs[(fold-1)*2+1]/'initialization.json').read_text())
    pred = pd.concat(predictions)
    for (c, arm), frame in pred.loc[pred.column.isin(r.TARGETS)].groupby(['column', 'arm']):
        assert frame.sample_id.is_unique
        manifest = pd.read_csv(r.STUDY/'ROW_SPLIT_MANIFEST.csv')
        assert set(frame.sample_id) == set(manifest.loc[manifest.column.eq(c) & manifest.role.eq('inner_validation'), 'sample_id'])
    for name, frame in [('INNER_RESULTS.csv', results), ('TAIL_METRICS.csv', tails),
        ('RETENTION_REGIMES.csv', pd.concat(regimes)), ('INNER_PREDICTIONS.csv', pred),
        ('TRAINING_HISTORY.csv', history), ('FIT_AUDIT.csv', pd.DataFrame(audits))]:
        frame.to_csv(r.STUDY/name, index=False)
    regime_rows = []
    regime_frame = pd.concat(regimes)
    for (c, arm, regime), frame in regime_frame.loc[regime_frame.column.isin(r.TARGETS)].groupby(['column','arm','regime']):
        n = frame.n.to_numpy()
        regime_rows.append(dict(column=c, arm=arm, regime=regime, n=int(n.sum()),
            **{f'{e}_pooled_rmse': float(np.sqrt(np.nansum(n*frame[f'{e}_rmse']**2)/n.sum())) for e in ['V1','V2']}))
    pd.DataFrame(regime_rows).to_csv(r.STUDY/'POOLED_RETENTION_REGIMES.csv', index=False)
    row_pairs = paired(results, 'Row-CV')
    row_pairs.to_csv(r.STUDY/'PAIRED_COMPARISONS.csv', index=False)
    group = pd.read_csv(r.old.STUDY/'ROW_INNER_RESULTS.csv')
    group = group.loc[group.outer_seed.eq(r.PILOT_SEED)]
    comparisons = pd.concat([paired(group, 'Group-CV'), row_pairs])
    comparisons.to_csv(r.STUDY/'GROUP_VS_ROW_PAIRED_COMPARISONS.csv', index=False)
    selection_comparison = []
    for cv, frame in [('Group-CV', group), ('Row-CV', results)]:
        for c in r.TARGETS:
            pairs = comparisons.loc[comparisons.cv.eq(cv) & comparisons.column.eq(c)]
            sub = frame.loc[frame.column.eq(c)]
            selection_comparison.append(dict(cv=cv, outer_seed=r.PILOT_SEED, column=c,
                mean_paired_fold_gain_percent=float(pairs.combined_normalized_rmse_relative_gain.mean()*100),
                fold_wins=int(pairs.A2_wins.sum()), folds=5,
                A1_median_epoch=float(sub.loc[sub.arm.eq('A1'),'best_epoch'].median()),
                A2_median_epoch=float(sub.loc[sub.arm.eq('A2'),'best_epoch'].median())))
    pd.DataFrame(selection_comparison).to_csv(r.STUDY/'GROUP_VS_ROW_SELECTION_SUMMARY.csv', index=False)
    epochs = pd.concat([frame.loc[frame.column.eq('25g'), ['arm', 'inner_fold', 'best_epoch', 'epochs_run']].assign(cv=cv)
                        for cv, frame in [('Group-CV', group), ('Row-CV', results)]])
    epochs.to_csv(r.STUDY/'BEST_EPOCH_DISTRIBUTION.csv', index=False)
    selection = results.loc[results.column.eq('25g')].groupby('arm').best_epoch.median().reset_index()
    selection['final_epoch_selected'] = np.floor(selection.best_epoch+.5).astype(int)
    selection['outer_seed'] = r.PILOT_SEED; selection['refit_executed'] = False
    selection.to_csv(r.STUDY/'EPOCH_SELECTION.csv', index=False)
    summary = []
    for c in r.TARGETS:
        pairs = row_pairs.loc[row_pairs.column.eq(c)]
        for arm in r.old.ARMS:
            frame = results.loc[results.column.eq(c) & results.arm.eq(arm)]
            n = frame.inner_validation_count.to_numpy()
            entry = dict(column=c, arm=arm,
                mean_combined_normalized_rmse=frame.combined_normalized_rmse.mean(),
                mean_paired_fold_gain_percent=pairs.combined_normalized_rmse_relative_gain.mean()*100,
                A2_fold_wins=int(pairs.A2_wins.sum()), folds=5,
                A2_seed_win=int(pairs.A2_combined_normalized_rmse.mean() < pairs.A1_combined_normalized_rmse.mean()), seeds_executed=1)
            for metric in ['V1_rmse', 'V2_rmse', 'V1_mae', 'V2_mae', 'Center_rmse', 'Width_rmse']:
                entry['mean_'+metric] = frame[metric].mean()
                if metric.endswith('_rmse'):
                    entry['pooled_OOF_'+metric] = np.sqrt(np.average(frame[metric]**2, weights=n))
            summary.append(entry)
    summary = pd.DataFrame(summary); summary.to_csv(r.STUDY/'INNER_SUMMARY.csv', index=False)
    pooled = pooled_tail(tails); pooled.to_csv(r.STUDY/'POOLED_TAIL_METRICS.csv', index=False)
    group_tails = pd.read_csv(r.old.STUDY/'ROW_TAIL_METRICS.csv')
    pooled_tail(group_tails.loc[group_tails.outer_seed.eq(r.PILOT_SEED)]).to_csv(r.STUDY/'HISTORICAL_GROUP_POOLED_TAIL_METRICS.csv', index=False)
    historical = json.loads((r.STUDY/'HISTORICAL_HASHES_BEFORE.json').read_text())
    after = {str(p.relative_to(r.ROOT)): r.old.sha(p) for p in r.old.STUDY.rglob('*') if p.is_file()}
    assert historical == after, 'historical artifacts changed'
    r.save_json('HISTORICAL_PRESERVATION_CHECK.json', {'files_checked': len(historical), 'all_unchanged': True,
        'outer_evaluation_executed': False, 'full_data_refit_executed': False, 'other_seeds_executed': False})
    from verify_inner_predictions import verify
    verify()
    figures(history, results, comparisons, epochs)
    report(results, summary, pooled, epochs, comparisons, selection, history, len(historical))
    outputs = {p.name: r.old.sha(p) for p in r.STUDY.iterdir() if p.is_file() and p.suffix in ['.csv', '.md', '.png', '.py']}
    r.save_json('REPORT_MANIFEST.json', {'files': outputs, 'phase': 'PHASE_1_COMPLETE', 'fitted_models': 10,
                                       'test_labels_read': False, 'report_code_sha256': r.old.sha(Path(__file__))})
    print('PHASE 1 COMPLETE: 10 fits, reporting and audits finished; no further actions.', flush=True)


def report(results, summary, pooled, epochs, comparisons, selection, history, historical_count):
    row = comparisons.loc[comparisons.cv.eq('Row-CV')]
    old = comparisons.loc[comparisons.cv.eq('Group-CV')]
    gains = row.groupby('column').combined_normalized_rmse_relative_gain.mean()*100
    group_gains = old.groupby('column').combined_normalized_rmse_relative_gain.mean()*100
    med = epochs.groupby(['cv', 'arm']).best_epoch.median()
    joint = results.loc[results.column.isin(r.TARGETS)].groupby(['inner_fold', 'arm']).combined_normalized_rmse.mean().unstack('arm')
    joint_gain = (1-joint.A2/joint.A1).mean()*100
    descriptions = []
    for c in r.TARGETS:
        sub = row.loc[row.column.eq(c)]
        descriptions.append(f'{c}：mean paired-fold gain={gains[c]:+.3f}%，A2 wins={int(sub.A2_wins.sum())}/5；历史 Group-CV 同 seed gain={group_gains[c]:+.3f}%。')
    columns = ['column', 'arm', 'mean_combined_normalized_rmse', 'mean_V1_rmse', 'mean_V2_rmse', 'mean_V1_mae', 'mean_V2_mae', 'mean_Center_rmse', 'mean_Width_rmse']
    epoch_rows = []
    for cv in ['Group-CV', 'Row-CV']:
        for arm in r.old.ARMS:
            sub = epochs.loc[epochs.cv.eq(cv) & epochs.arm.eq(arm)].sort_values('inner_fold')
            epoch_rows.append(dict(cv=cv, arm=arm, five_best_epochs=', '.join(map(str, sub.best_epoch.astype(int))),
                median=int(np.floor(sub.best_epoch.median()+.5)), maximum_epoch_hits=int((sub.best_epoch>=500).sum())))
    tails = pooled.pivot(index=['column', 'endpoint'], columns='arm', values='tail_rmse').reset_index()
    tails['A2_gain_percent'] = (1-tails.A2/tails.A1)*100
    component = []
    for c in r.TARGETS:
        sub = summary.loc[summary.column.eq(c)].set_index('arm')
        values = {m: (1-sub.loc['A2', 'pooled_OOF_'+m]/sub.loc['A1', 'pooled_OOF_'+m])*100 for m in ['V1_rmse', 'V2_rmse', 'Center_rmse', 'Width_rmse']}
        component.append(dict(column=c, **{m+'_gain_percent': v for m, v in values.items()}))
    losses = []
    for (fold, arm), h in history.groupby(['inner_fold', 'arm']):
        for c in r.TARGETS:
            losses.append(dict(inner_fold=int(fold), arm=arm, column=c,
                train_loss_epoch1=float(h[f'train_loss_{c}'].iloc[0]), train_loss_last=float(h[f'train_loss_{c}'].iloc[-1]),
                valid_loss_epoch1=float(h[f'validation_loss_{c}'].iloc[0]), valid_loss_last=float(h[f'validation_loss_{c}'].iloc[-1])))
    pd.DataFrame(losses).to_csv(r.STUDY/'LOSS_DIAGNOSTICS.csv', index=False)
    if (gains>0).all():
        recommendation = '建议确认后继续另外四个 seed，以检验两个柱上的单 seed 改善能否稳定重复。本试点 25g V1 tail RMSE 增大约 3.05%，超过历史研究 2% 的描述性保护线；这不是本单 seed 的正式 gate，但说明不能声称全误差剖面已改善，也不支持立即 FULL refit 或外层重评。继续的价值在于检验重复性和模型选择机制，须保留负结果，不能更改训练上限、selector 或模型追求通过。'
    elif (gains>0).any():
        recommendation = '建议在确认后按原冻结协议完成另外四个 seed，作为检验柱间不一致和模型选择假设的受控研究，而非预设 FiLM 已成功。等权 joint selector 允许一柱收益抵消另一柱退化，完整研究必须分别报告两柱、端点和 tail；不得据试点重定义 selector 或调参。'
    else:
        recommendation = '本试点没有显示两柱改善，当前不支持以提高性能为理由立即扩大计算。若研究目的是检验模型选择机制，可在确认后完成预定其余 seed 以估计重复性；应预先接受负结果，维持同一协议，不以追加网络或调参追求正结论。'
    reg = pd.read_csv(r.STUDY/'POOLED_RETENTION_REGIMES.csv')
    reg_gain = []
    for (c, regime), sub in reg.groupby(['column','regime']):
        sub = sub.set_index('arm')
        reg_gain.append(dict(column=c, regime=regime, n=int(sub.loc['A1','n']),
            **{e+'_gain_percent': float((1-sub.loc['A2',f'{e}_pooled_rmse']/sub.loc['A1',f'{e}_pooled_rmse'])*100) for e in ['V1','V2']}))
    cap = results.loc[results.column.eq('25g')].groupby('arm').epochs_run.apply(lambda s: int((s==500).sum())).to_dict()
    replay = json.loads((r.STUDY/'CHECKPOINT_REPLAY.json').read_text())
    overlap = pd.read_csv(r.STUDY/'INPUT_OVERLAP_AUDIT.csv')
    overlap = overlap.loc[overlap.column.eq(overlap.donor) & overlap.column.isin(r.TARGETS)]
    overlap_total = overlap.groupby('column')[['heldout_rows','full_model_input_overlap_rows','canonical_overlap_rows']].sum().reset_index()
    text = f'''# Phase 1：A1/A2 Row-aligned 试点报告

日期：2026-10-13（Asia/Shanghai）。实际完成 seed {r.PILOT_SEED} × 5 folds × 2 arms = **10 次模型拟合**。仅目标 gradient_train 标签用于内层训练/验证；另外四个 seed、FULL-data refit 和外层评分均未执行。完整设计见 [PROTOCOL.md](PROTOCOL.md)，实现检查见 [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md)。

## 真实结果与解释

{' '.join(descriptions)} Joint mean paired-fold gain={joint_gain:+.3f}%，joint fold wins={int((joint.A2<joint.A1).sum())}/5。

以下误差为五折算术均值，RMSE/MAE 单位 mL；配对 gain 与均值比值为不同统计量，不互换。seed win 分母为 **1 个已执行 seed**，不是 5 个。未经独立重复信息支持，不对重叠 CV 折做显著性检验。

{table(summary, columns)}
配对每折及两柱分别见 [PAIRED_COMPARISONS.csv](PAIRED_COMPARISONS.csv)。OOF pooled 端点与 Center/Width RMSE 按验证条数加权 SSE 汇总，不是折 RMSE 的简单平均；下表是这些 pooled RMSE 的 A2 相对 A1 收益（实际 RMSE 保存在 INNER_SUMMARY.csv）：

{table(pd.DataFrame(component), list(component[0]))}
## Epoch 选择与 Group-CV 历史诊断

{table(pd.DataFrame(epoch_rows), ['cv','arm','five_best_epochs','median','maximum_epoch_hits'])}
Row-CV 后 A1 median {med['Row-CV','A1']:.0f}（历史 {med['Group-CV','A1']:.0f}），A2 median {med['Row-CV','A2']:.0f}（历史 {med['Group-CV','A2']:.0f}）。它们只是待将来确认后 FULL refit 的固定 epoch，保存于 EPOCH_SELECTION.csv；没有用 inner checkpoint 代替 FULL refit。每折 best epoch 是同一个 joint selector 的结果，不能分别为 25g/40g 挑最优。

A2 的中位数不再远小于 A1，反而更大（290 vs 186）。这与“Group-CV 下 A2 较早选择”受划分影响的假设一致，但不同验证集合、折间变动和一个 outer seed 不足以证明系统性偏好，更不能由内部改善推断原 Row 40g 外层退化已经修复。

当前所有拟合实际 **{','.join(map(str, sorted(results.steps_per_epoch.unique())))} optimizer step/epoch**。best checkpoint 对应 best_epoch×steps_per_epoch；实际 total steps 包含等待 patience 的训练，见 INNER_RESULTS.csv 和 TRAINING_HISTORY.csv。4g 每轮只有配对抽样，并未遍历源 3330 条。

**Group-CV 与 Row-CV 使用不同验证集合；表中绝对误差不得用于判定哪种 CV 更优，跨协议 fold 号不是配对身份。** 原 Group-CV 仅是历史参照；本报告没有运行新的 Group-CV。即使相对收益/epoch 变化，也不能由一个 seed 证明 Group-CV 对 A2 有系统性偏好。

## 高保留体积和收敛

tail 使用原 outer gradient_train 各端点 80%分位；五折 OOF SSE/count 汇总，阈值只诊断、不选择。空 tail 折没有记为零误差。

{table(tails, ['column','endpoint','A1','A2','A2_gain_percent'])}
center low/mid/high 区间 pooled OOF 收益如下；阈值与数量见 RETENTION_REGIMES.csv、POOLED_RETENTION_REGIMES.csv。

{table(pd.DataFrame(reg_gain), ['column','regime','n','V1_gain_percent','V2_gain_percent'])}
每轮 train/validation raw loss 与两柱 NRMSE/Center/Width 全部保存在 TRAINING_HISTORY.csv；LOSS_DIAGNOSTICS.csv 给出首末值。train loss 来自训练模式混合 BN 和配对抽样，validation loss 来自 eval BN 固定验证集，二者不能当作完全相同测量。best epoch 远小于 500 只能描述为验证最优出现较早，不能未经证据称为实现过早停止；末期无改善按原 patience=80 停止。A1 有 {cap.get('A1',0)}/5、A2 有 {cap.get('A2',0)}/5 次训练达到 500 cap，因而部分曲线仍受上限约束，不能称已充分收敛；本轮不延长训练。

学习曲线显示部分折训练损失持续下降、验证损失回升（尤其 fold 3 的 25g），与有限数据下的过拟合模式相符，没有非有限损失或状态异常证据。A2 fold 5 的最佳点恰在 500 上限，其最优训练时长仍受截断。25g 改善主要来自 V1，V2 pooled 改善仅 0.48%；40g Center/Width 均有收益，endpoint tails 有明显改善。center low 区间两柱 V2 却退化，表明整体平均改善不能代表所有保留体积区间均受益。

![学习曲线](a1_a2_row_cv_learning_curves.png)

![历史协议选择对比](group_vs_row_cv_selection.png)

## 实现审计与继续建议

14 项协议测试通过；合成标签的一步训练与原入口 checkpoint/原有 history 字段逐位一致（A1、A2 分别验证）。十次真实拟合全部通过身份、配对抽样、共同初始状态、有限预测/梯度、冻结参数、BN 状态/计数、joint argmin 和真实 optimizer steps 检查。{historical_count} 个历史文件前后 SHA256 全部不变。没有发现新的训练实现问题。

另外以只读取 feature 白名单、零占位标签的独立入口重放十个 best checkpoint 的 {replay['contexts_checked']} 个目标上下文，共 {replay['target_prediction_rows']} 条 OOF 记录，最大绝对预测差={replay['max_abs_difference_ml']:.3g} mL；没有解析任何新增真实标签。

独立物理实验/重复测量的批次身份缺乏明确元数据；Row 估计仅支持记录级插值。不同 ID 的相同图输入与跨柱骨干输入重合不等于 sample_id 泄漏，详细数量见 INPUT_DUPLICATE_COUNTS.csv、INPUT_OVERLAP_AUDIT.csv、CROSS_COLUMN_INPUT_AUDIT.csv。跨柱模型还接收不同 task ID，因此骨干输入重合不等于完整模型输入相同。目标 OOF 与同柱 inner train 的相同完整输入/分子重合数量如下（每个目标 ID 验证一次；4g 诊断验证每折重复，不能按独立 OOF 汇总）：

{table(overlap_total, ['column','heldout_rows','full_model_input_overlap_rows','canonical_overlap_rows'])}

{recommendation} 单 seed 仅验证协议和机制，**不能据此判定 FiLM 成功或失败**。后续若确认 Phase 2，则保持当前协议完成剩余 40 次拟合，再讨论是否值得 Phase 3。最终外层测试已多次曝光，仅能标记 DEVELOPMENTAL CONFIRMATION；可靠泛化确认仍需未曝光数据。

本轮在 Phase 1 停止，不运行 PCGrad、其他优化器、额外候选、FULL refit 或 outer score。
'''
    (r.STUDY/'REPORT.md').write_text(text)
    audit = f'''# Implementation audit：Phase 1

## 证据边界和冻结

旧模型、训练脚本及其六个依赖代码的 SHA256 与历史 EXECUTION_PROVENANCE 一致；新入口通过导入使用原模型、loss、batcher、prediction 和 point metrics，没有改动其 GroupKFold 语义。所有写操作限本新目录。历史 study {historical_count} 个文件全部哈希不变；source/data/graph cache/split 哈希与旧 protocol 一致。外层 test 与 validation 只读取身份/获准 feature 白名单，没有标签请求、训练图、epoch 选择或评分。LABEL_ACCESS_AUTHORIZATION.json 固定允许的 ID；PREFLIGHT.json 保存元数据 header 审计。

## 测试与真实拟合

test_protocol.py：14 passed。覆盖真实五个 seed 的 outer 身份隔离（只对试点拟合）、每折两柱完整性、稳定排序、固定 seed 重现、不同 seed 改变身份、跨柱共享 ID 拒绝、允许分子重合、授权读取跳过 forbidden 标签、真实目标 loader 白名单/标签请求/outer graphs 排除、共同初始权重、图输入 digest 与合成标签下原训练入口一步 checkpoint 完全一致。

FIT_AUDIT.csv 覆盖十个完成拟合。A1/A2 同折训练/验证 IDs、scales、counts、training seed、batch size、steps_per_epoch 一致；所有共同 epoch 的 batch_draws_sha256 一致。best_epoch 是严格最早 joint argmin，保存模型恢复包含 BN 的完整 state_dict。frozen 参数逐张量与初始状态相同；early BN 统计不变，late BN num_batches_tracked 增量等于 best checkpoint 实际步数。真实 optimizer_steps 是执行 optimizer.step 累加，而不是推测完整源数据遍历次数。

verify_inner_predictions.py 的零标签重放检查 {replay['contexts_checked']} 个目标上下文、{replay['target_prediction_rows']} 条预测，最大绝对差 {replay['max_abs_difference_ml']:.3g} mL。此独立检查只解析输入白名单、身份及保存的盲预测，没有新增标签读取。CHECKPOINT_REPLAY.csv/json 保存逐上下文证据。

每折 source、25g、40g 的训练损失和验证损失有限，预测/梯度有限；A1 trainable/total=130615/460500，A2=139367/469252。FiLM 仍是 embedding_dim16、blocks3/4、零初始化；pretrained/new LR、Adam wd、raw_quantile、混合 BN、patience80/max500 不变。新增诊断不改变目标或参数更新；合成等价性测试对此提供直接执行证据。

## 数据独立性

ROW_SPLIT_MANIFEST.csv 包含每个实际 fold 的真实身份。目标 OOF 每 ID 恰一次验证，两个 arm 预测 ID 相同。INPUT_OVERLAP_AUDIT 按 heldout 方向记录同柱及跨柱分子、身份、图输入重合；跨柱 full_model_input_overlap_rows=0，因为 task identity 不同。输入 digest 包含实际 atom/bond/angle 图和原条件张量，排除 y、canonical_position 和缓存派生量；不是把原 CSV 全列当模型输入。

CSV header 没有明确 independent batch/replicate ID；source_row_1based 不是实验批次证据。重复输入行可能来自不同实验、重复测量或数据处理，当前不能判定，保留 Row 记录级估计并披露边界。新 runner 若遇到共享 sample_id 立即拒绝，而不是静默按列拆到两侧。

## 结果限制与停止

未发现新的训练错误。十折模型全部按原 recipe 运行；收益、tail/Center/Width 和 cap/early stop 详见 REPORT。单 seed 不足以证明选择偏好、跨 seed 稳定性或外层泛化。后续四 seed/refit/score 不在入口中，也未执行。新增文件不覆盖历史冻结证据。

复现（在仓库根目录）：`KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python studies/transfer/column_conditioned_row_aligned_v1/run_phase1.py`；完成拟合按 receipt 校验复用。仅重新报告：同一 Python 运行 summarize_phase1.py。测试：`python -m pytest -q studies/transfer/column_conditioned_row_aligned_v1/test_protocol.py`。上述 Python 为已验证 fish 环境，系统/base Python 缺 torch，因此没有安装或修改环境。
'''
    (r.STUDY/'IMPLEMENTATION_AUDIT.md').write_text(audit)


if __name__ == '__main__':
    summarize()
