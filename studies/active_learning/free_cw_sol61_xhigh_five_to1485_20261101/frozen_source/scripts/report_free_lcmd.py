"""Read-only cross-revision validation summary; curves stop at frozen fits.

Run without arguments for a progress snapshot, or --require-complete once all
five trajectories have 37 frozen fits. This does not call a model or reveal labels.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'reports/free_llm_five_seeds_to1485_20261024/figures'
OUT = DEFAULT_OUT
REPORT_DIR = ROOT / 'reports/free_lcmd_five_seeds_to1485_20261026'
SEEDS = (157, 887, 2357, 6101, 12203)
FREE = 'free_llm32_scientist_v2'
TRADITIONAL = ('random', 'lcmd', 'hybrid')
BASE = ROOT / 'studies/active_learning/qgeognn_v2_row_sequential_b32'
LCMD_EXTENSION = BASE.parent / 'qgeognn_v2_row_lcmd_to1485_20261026'
FREE_AMENDMENT_CUTS = {887: 33, 12203: 24}
REVISIONS = ROOT / 'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions'
OLD_STAGES = (
    (21, 'sol_xhigh_hybrid_free_to1005_20261013'),
    (26, 'sol_xhigh_all_strategies_to1165_20261014'),
    (30, 'sol_xhigh_all_strategies_to1293_20261015'),
    (34, 'sol_xhigh_all_strategies_to1421_20261017'),
    (36, 'sol_xhigh_all_strategies_to1549_20261018'),
)
NEW_STAGES = (
    (21, 'sol61_xhigh_free_five_seeds_to1005_20261023'),
    (36, 'sol61_xhigh_free_five_seeds_to1485_20261024'),
)
SOURCES = {}


def sha(path):
    path = Path(path)
    with path.open('rb') as handle:
        value = hashlib.file_digest(handle, 'sha256').hexdigest()
    SOURCES[str(path.relative_to(ROOT))] = value
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def owner(seed, r):
    if seed in FREE_AMENDMENT_CUTS and r > FREE_AMENDMENT_CUTS[seed]:
        return REVISIONS / 'sol61_xhigh_free_to1485_extra3_20261026'
    return REVISIONS / next(name for end, name in
                            (OLD_STAGES if seed in (157, 6101) else NEW_STAGES) if r <= end)


def checked_fit(path, budget, initial):
    fit = read(path)
    assert fit['train_rows'] == budget, path
    assert fit['initialization_hash'] == initial['initialization_hash'], path
    assert fit['validation_ids_hash'] == initial['validation_ids_hash'], path
    assert fit['test_labels_used_for_fit_or_checkpoint_selection'] == 0, path
    assert sha(path.parent / 'best.pt') == fit['checkpoint_sha256'], path
    assert sha(path.parent / 'predictions.csv.gz') == fit['prediction_sha256'], path
    return {'labels': budget, 'nrmse': fit['best_validation_combined_normalized_rmse']}


def collect(require_complete):
    curves, verification = {}, {}
    for seed in SEEDS:
        initial = read(BASE / f'runtime/seed_{seed}/lcmd/round_00/model/fit_audit.json')
        methods = {}
        for method in TRADITIONAL:
            methods[method] = [checked_fit(BASE / f'runtime/seed_{seed}/{method}/round_{r:02d}/model/fit_audit.json',
                                          333 + 32*r, initial) for r in range(22)]
        lcmd_ids = []
        for r in range(21):
            path = BASE / f'runtime/seed_{seed}/lcmd/round_{r:02d}/acquisition_artifacts/contract.json'
            acq = read(path)
            for name, digest in acq['files'].items():
                assert sha(path.parent.parent / name) == digest
            ids_path = path.parent / 'selected_next_batch.csv'
            import csv
            with ids_path.open() as handle:
                chosen = [row['sample_id'] for row in csv.DictReader(handle)]
            assert stable_hash(chosen) == acq['selected_ids_ordered_hash']
            assert len(chosen) == len(set(chosen)) == 32
            lcmd_ids.extend(chosen)
        for r in range(22, 37):
            directory = LCMD_EXTENSION / f'runtime/seed_{seed}/lcmd/round_{r:02d}'
            path = directory / 'fit_freeze.json'
            if not path.exists():
                assert not require_complete, path
                break
            frozen = read(path)
            assert frozen['test_truth_access_count'] == 0
            for name, digest in frozen['files'].items():
                assert sha(directory / name) == digest
            previous = LCMD_EXTENSION / f'runtime/seed_{seed}/lcmd/round_{r-1:02d}'
            acquisition = read(previous / 'acquisition_artifacts/contract.json')
            for name, digest in acquisition['files'].items():
                assert sha(previous / name) == digest
            receipt_path = previous / 'label_access_receipt.json'
            receipt = read(receipt_path)
            assert sha(receipt_path) == frozen['label_receipt_sha256']
            assert receipt['acquisition_sha256'] == sha(previous / 'acquisition_artifacts/contract.json')
            chosen = receipt['selected_ids']
            assert len(chosen) == len(set(chosen)) == 32
            assert stable_hash(chosen) == acquisition['selected_ids_ordered_hash']
            assert receipt['audit']['acquisitions_frozen'] and receipt['audit']['purpose'] == 'after_acquisition_fit'
            assert receipt['audit']['requested_ids_hash'] == stable_hash(sorted(chosen))
            lcmd_ids.extend(chosen)
            methods['lcmd'].append(checked_fit(directory / 'model/fit_audit.json',333+32*r,initial))
        assert len(lcmd_ids) == len(set(lcmd_ids)) == 32*(len(methods['lcmd'])-1)
        rows, ids, ownership = [], [], []
        for r in range(37):
            stage = owner(seed, r)
            path = stage / f'runtime/seed_{seed}/{FREE}/round_{r:02d}/prediction_freeze.json'
            if not path.exists():
                assert not require_complete, path
                break
            frozen = read(path)
            assert frozen['test_truth_access_count'] == 0, path
            for kind in ('checkpoint', 'prediction'):
                assert sha(frozen[kind + '_path']) == frozen[kind + '_sha256'], path
            fit_path = Path(frozen['checkpoint_path']).parent / 'fit_audit.json'
            fit = read(fit_path)
            assert fit['train_ids_hash'] == frozen['L_t_ids_hash'], path
            rows.append(checked_fit(fit_path, 333 + 32*r, initial))
            ownership.append({'round': r, 'labels': 333+32*r, 'revision': stage.name,
                              'freeze': str(path.relative_to(ROOT))})
            if r == 0:
                continue
            # Selection r-1 belongs to the revision that produced fit r.
            selection = stage / f'selections/seed_{seed}/{FREE}/round_{r-1:02d}'
            batch = read(selection / 'batch_freeze.json')
            assert len(batch['batch_ids']) == 32 and not batch['pending_ids'], selection
            for name, digest in batch['artifact_hashes'].items():
                assert sha(selection / name) == digest, selection / name
            feedback = read(selection / 'feedback.json')
            receipt = read(selection / 'label_access_receipt.json')
            batch_hash = sha(selection / 'batch_freeze.json')
            assert feedback['batch_freeze_sha256'] == receipt['batch_freeze_sha256'] == batch_hash
            assert stable_hash(feedback) == receipt['feedback_sha256'], selection
            assert (feedback['seed'], feedback['method'], feedback['round']) == (seed, FREE, r-1)
            assert [x['candidate_id'] for x in feedback['records']] == batch['batch_ids']
            assert receipt['audit']['purpose'] == 'after_acquisition_fit'
            assert receipt['audit']['acquisitions_frozen'] is True
            assert receipt['audit']['requested_ids_hash'] == stable_hash(sorted(batch['batch_ids'])), selection
            for row, pred in zip(feedback['records'], batch['premeasurement_predictions']):
                assert row['candidate_id'] == pred['candidate_id']
                for target in ('V1', 'V2'):
                    assert row[f'pred_{target}_ml'] == pred[f'pred_{target}_ml']
                    assert row[f'error_{target}_ml'] == row[f'pred_{target}_ml'] - row[f'true_{target}_ml']
            ids.extend(batch['batch_ids'])
        assert rows and len(ids) == len(set(ids)) == 32*(len(rows)-1), seed
        if require_complete:
            assert len(rows) == 37 and rows[-1]['labels'] == 1485, seed
        methods[FREE] = rows
        curves[str(seed)] = methods
        verification[str(seed)] = {'verified_fits': len(rows), 'verified_batches': len(ids)//32,
                                    'unique_acquired_ids': len(ids), 'final_labels': rows[-1]['labels'],
                                    'fit_ownership': ownership,
                                    'initialization_hash': initial['initialization_hash'],
                                    'validation_ids_hash': initial['validation_ids_hash'],
                                    'lcmd_verified_fits': len(methods['lcmd']),
                                    'lcmd_verified_batches': len(lcmd_ids)//32,
                                    'lcmd_final_labels': methods['lcmd'][-1]['labels'],
                                    'test_truth_access_count': 0}
    return curves, verification


def summarize(curves, verification, np):
    free_common = min(curves[str(seed)][FREE][-1]['labels'] for seed in SEEDS)
    lcmd_common = min(curves[str(seed)]['lcmd'][-1]['labels'] for seed in SEEDS)
    summary = {'snapshot_time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'status': 'COMPLETE' if free_common == lcmd_common == 1485 else 'IN_PROGRESS',
               'free_common_budget': free_common, 'traditional_common_budget': 1005,
               'lcmd_common_budget': lcmd_common, 'free_lcmd_matched_budget': min(free_common,lcmd_common),
               'target_labels': 1485, 'seed_count': 5, 'test_evaluation_performed': False,
               'model_note': 'Seeds 157/6101 retain historical Astra/high prefixes; '
                             '887/2357/12203 use gpt-6.1-sol/xhigh throughout. '
                             'Post-primary amendment: +3 correction attempts only at seed887 r33 / seed12203 r24.',
               'latest_free': {str(seed): curves[str(seed)][FREE][-1] for seed in SEEDS},
               'latest_lcmd': {str(seed): curves[str(seed)]['lcmd'][-1] for seed in SEEDS},
               'shared_333_to_1005': {}, 'verification': verification}
    for method in (*TRADITIONAL, FREE):
        values = np.array([[x['nrmse'] for x in curves[str(seed)][method][:22]] for seed in SEEDS])
        assert values.shape == (5, 22)
        integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        summary['shared_333_to_1005'][method] = {
            'mean_validation_nrmse': float(values[:, -1].mean()),
            'sd_across_seeds': float(values[:, -1].std(ddof=1)),
            'mean_normalized_aulc': float((integrate(values, np.arange(333, 1006, 32), axis=1)/672).mean())}
    values = np.array([[x['nrmse'] for x in curves[str(seed)][FREE] if x['labels'] <= free_common]
                       for seed in SEEDS])
    summary['free_at_common_budget'] = {'labels': free_common,
        'mean_validation_nrmse': float(values[:, -1].mean()),
        'sd_across_seeds': float(values[:, -1].std(ddof=1))}
    matched = summary['free_lcmd_matched_budget']
    summary['free_vs_lcmd_at_matched_budget'] = {'labels':matched}
    for method in (FREE,'lcmd'):
        vals=np.array([next(row['nrmse'] for row in curves[str(seed)][method] if row['labels']==matched) for seed in SEEDS])
        summary['free_vs_lcmd_at_matched_budget'][method]={'mean':float(vals.mean()),'sd':float(vals.std(ddof=1))}
    return summary


def plot(curves, summary, np, plt):
    colors = {'random': '#8C99A7', 'lcmd': '#148B88', 'hybrid': '#D89025', FREE: '#A63B83'}
    names = {'random': 'Random32', 'lcmd': 'LCMD32', 'hybrid': 'Traditional Hybrid32', FREE: 'Free LLM32'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(17.4, 10.2))
    for ax, seed in zip(axes.flat, SEEDS):
        for method in (*TRADITIONAL, FREE):
            rows = curves[str(seed)][method]
            ax.plot([x['labels'] for x in rows], [x['nrmse'] for x in rows], color=colors[method],
                    lw=2.3 if method == FREE else 1.5, label=names[method])
        end = curves[str(seed)][FREE][-1]
        lcmd_end=curves[str(seed)]['lcmd'][-1]
        ax.set_title(f'Seed {seed} | Free {end["labels"]} / LCMD {lcmd_end["labels"]}', loc='left')
        ax.set_xlim(325, 1510)
        ax.set_ylim(.15, 1.15)
        ax.set_xticks([333, 621, 1005, 1229, 1485])
        ax.axvline(1005, color='#687387', ls=':', alpha=.4)
        ax.set_xlabel('Active training labels')
        ax.set_ylabel('Validation NRMSE')
        ax.grid(axis='y', alpha=.15)
    ax = axes.flat[-1]
    for method in (*TRADITIONAL, FREE):
        common=min(len(curves[str(seed)][method]) for seed in SEEDS)
        data = np.array([[x['nrmse'] for x in curves[str(seed)][method][:common]] for seed in SEEDS])
        ax.plot(np.arange(333,333+32*common,32), data.mean(axis=0), color=colors[method], label=names[method])
    ax.set_title('Five-seed means | each stops at verified common budget', loc='left',fontsize=10)
    ax.set_xlim(325, 1510)
    ax.set_xticks([333, 717, 1005, 1229, 1485])
    ax.set_xlabel('Active training labels')
    ax.set_ylabel('Mean validation NRMSE')
    ax.grid(axis='y', alpha=.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .95), ncol=4, frameon=False)
    fig.suptitle('Free LLM and LCMD32 | five-seed continuation to 1485', fontsize=19, y=.99)
    fig.text(.055, .025, 'Lower is better. Lines stop at verified fits; Random/Hybrid stop at 1005. LCMD includes verified continuation fits.\n'
             'Seeds 157/6101 retain historical Astra/high prefixes; 887/2357/12203 use Sol6.1/xhigh throughout.\n'
             'New amendment: +3 corrections at seed887 r33 / seed12203 r24; historical authorized prefix corrections retained (see report).\n'
             f'Snapshot: {summary["snapshot_time"]}. Status: {summary["status"]}. Validation only; no test evaluation.',
             fontsize=9.5, color='#5D6878', linespacing=1.5)
    fig.subplots_adjust(left=.055, right=.98, top=.865, bottom=.16, wspace=.29, hspace=.4)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT/f'current_five_seed_comparison.{ext}', dpi=170, facecolor='white')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.5,6.6))
    n=(summary['free_lcmd_matched_budget']-333)//32+1
    x=np.arange(333,333+32*n,32)
    for method in (FREE,'lcmd'):
        data=np.array([[row['nrmse'] for row in curves[str(seed)][method][:n]] for seed in SEEDS])
        mean=data.mean(axis=0);sd=data.std(axis=0,ddof=1)
        ax.plot(x,mean,color=colors[method],lw=2.5,label=f'{names[method]} (5 seeds)')
        ax.fill_between(x,mean-sd,mean+sd,color=colors[method],alpha=.13)
    ax.set(xlabel='Active training labels (matched completed budgets only)',ylabel='Validation NRMSE',
        title=f'Free LLM32 vs LCMD32 | matched five-seed mean ± SD to {x[-1]}')
    ax.grid(axis='y',alpha=.15);ax.legend(frameon=False)
    fig.text(.08,.025,'Validation only; lower is better. No extrapolation.\n'
        'Free retains historical model prefixes, authorized prefix corrections and the new correction amendment; see report.',fontsize=9,color='#5D6878')
    fig.tight_layout(rect=[0,.10,1,1])
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'free_vs_lcmd_five_seed_mean.{ext}',dpi=170,facecolor='white')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    for seed, color in zip(SEEDS, ('#5276B7', '#D67835', '#20958C', '#8C6CB0', '#B74571')):
        rows = curves[str(seed)][FREE]
        ax.plot([x['labels'] for x in rows], [x['nrmse'] for x in rows], lw=2, color=color,
                label=f'Seed {seed}: {rows[-1]["labels"]} labels, {rows[-1]["nrmse"]:.3f}')
    ax.axvline(summary['free_common_budget'], color='#687387', ls=':', alpha=.5)
    ax.set(xlim=(325, 1510), ylim=(.15, 1.15), xlabel='Active training labels',
           ylabel='Validation NRMSE (lower is better)', title='Free LLM32 | five frozen seed trajectories')
    ax.set_xticks([333, 525, 717, 1005, 1229, 1485])
    ax.grid(axis='y', alpha=.15)
    ax.legend(frameon=False)
    fig.text(.08, .03, f'Status: {summary["status"]}; common Free budget: {summary["free_common_budget"]}. '
             'Lines stop at verified completed fits.\nSeeds 157/6101 retain Astra/high prefixes; new three seeds use Sol6.1/xhigh.\n'
             'Authorized prefix corrections and the seed157 r5 ID transcription repair are retained; see report.',
             fontsize=9.5, color='#5D6878')
    fig.tight_layout(rect=[0, .10, 1, 1])
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT/f'free_llm_five_seed_curves.{ext}', dpi=170, facecolor='white')
    plt.close(fig)


def load_snapshot(directory, require_complete, np):
    """Redraw retained numeric evidence without requiring local training payloads.

    This validates the compact table's structure, not the unavailable runtime hashes.
    Source hashes retain the provenance of the original online collection.
    """
    curves = json.loads((directory/'validation_curves.json').read_text())
    saved = json.loads((directory/'current_summary.json').read_text())
    for seed in SEEDS:
        for method in (*TRADITIONAL, FREE):
            rows = curves[str(seed)][method]
            expected_length = 22 if method in ('random', 'hybrid') else None
            if not 22 <= len(rows) <= 37 or (expected_length and len(rows) != expected_length):
                raise ValueError(f'Invalid curve length: {seed}/{method}')
            for r, row in enumerate(rows):
                if row['labels'] != 333+32*r or not np.isfinite(row['nrmse']) or row['nrmse'] < 0:
                    raise ValueError(f'Invalid frozen curve point: {seed}/{method}/{r}')
    if saved.get('test_evaluation_performed') is not False:
        raise ValueError('Expected a validation-only snapshot')
    summary = summarize(curves, saved['verification'], np)
    summary['snapshot_time'] = saved['snapshot_time']
    if summary != saved:
        raise ValueError('Summary disagrees with retained curves')
    if require_complete and summary['status'] != 'COMPLETE':
        raise ValueError('All ten trajectories must reach 1485')
    sources = json.loads((directory/'source_hashes.json').read_text())
    return curves, summary, sources


def report(summary, destination, figures, curves=None):
    """One current report, with values derived from the same retained curves."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_figures = Path(os.path.relpath(figures, destination.parent)).as_posix()
    complete = sum(point['labels'] == 1485 for family in ('latest_free', 'latest_lcmd')
                   for point in summary[family].values())
    remaining = sum((1485-point['labels'])//32 for family in ('latest_free', 'latest_lcmd')
                    for point in summary[family].values())
    matched = summary['free_vs_lcmd_at_matched_budget']
    heading = '最终结果' if summary['status'] == 'COMPLETE' else '当前结果'
    lines = [f'# Free LLM32 与 LCMD32：五 seed {heading}', '',
             f"采集时间：{summary['snapshot_time']}。状态：{summary['status']}。",
             f'目标为每条1485训练标签（初始333 + 36批×32，37次拟合）；已完成{complete}/10条，还剩{remaining}次拟合。', '',
             '| Seed | Free标签 | Free validation NRMSE | LCMD标签 | LCMD validation NRMSE |',
             '| --- | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        free, lcmd = (summary[key][str(seed)] for key in ('latest_free', 'latest_lcmd'))
        lines.append(f"| {seed} | {free['labels']} | {free['nrmse']:.6f} | {lcmd['labels']} | {lcmd['nrmse']:.6f} |")
    difference = matched['lcmd']['mean'] - matched[FREE]['mean']
    relative = difference / matched['lcmd']['mean']
    direction = '低' if difference >= 0 else '高'
    area_note = ''
    if curves is not None:
        areas = {}
        for method in (FREE, 'lcmd'):
            values = []
            for seed in SEEDS:
                rows = [row for row in curves[str(seed)][method] if row['labels'] <= matched['labels']]
                area = sum((a['nrmse'] + b['nrmse']) / 2 * (b['labels'] - a['labels'])
                           for a, b in zip(rows, rows[1:])) / (matched['labels'] - 333)
                values.append(area)
            areas[method] = sum(values) / len(values)
        area_note = (f"全程333→{matched['labels']}标签的平均归一化曲线面积："
                     f"Free {areas[FREE]:.6f}，LCMD {areas['lcmd']:.6f}（越低越好）。"
                     '端点误差与整个采样过程的效率应分别比较。')
    lines += ['', f"五seed共同预算{matched['labels']}：Free平均NRMSE {matched[FREE]['mean']:.6f} ± "
                  f"{matched[FREE]['sd']:.6f}，LCMD {matched['lcmd']['mean']:.6f} ± {matched['lcmd']['sd']:.6f}"
                  '（均值 ± seed间样本标准差）。'
                  f'Free比LCMD{direction}{abs(difference):.6f}，相对LCMD{direction}{abs(relative):.2%}。', '',
              area_note, '',
              f'![五seed实际冻结曲线]({relative_figures}/current_five_seed_comparison.png)', '',
              '## 实现与证据范围', '',
              'LCMD续跑使用冻结的原梯度特征实现，512D CountSketch、q50输出梯度和corrected LCMD-TP；'
              '沿用原初始化、validation划分、batch32及训练规则。特征源码副本与原git8081e86一致，'
              '不采用当前源码后来新增的变换。见[扩展manifest](lcmd/extension_manifest.json)及[冻结源码](frozen_sources/gradient_features.py)。', '',
              'Free的157/6101保留历史Astra/high前缀，887/2357/12203使用gpt-6.1-sol/xhigh。'
              '887 r33与12203 r24有显式授权的额外3次逻辑纠正；原失败回执保留，其他科学规则不变。'
              '见[amendment](free/continuation_amendment.json)。', '',
              '历史前缀还保留157 r5、6101 r0/r19各一次已授权的额外纠正，以及157 r5的一次已授权ID抄写修正；'
              '这些例外均有原始回执，本次没有新增调用或修正。'
              '见[历史授权记录](../llm_eight_to1549_20261018/historical_protocol_deviations.json)。', '',
              '只画完成且冻结的拟合点。Random/Hybrid五seed只到1005，不外推。'
              '这些是validation开发轨迹，没有新增test评价，validation指标不进入selector，标签在批次冻结后揭示。', '',
              '## 全训练集参考与平台期', '',
              '固定五seed、同初始化与416条validation下，3330标签全训练集参考平均NRMSE为0.325433；'
              'LCMD1485为0.347582，相差0.022149（相对全量约6.81%）。'
              'LCMD从1357到1485只改善0.003371；后段收益放缓，但这不等于已达到全数据性能。'
              '参考值来自既有冻结fit audit，没有重新训练或读取test真值。'
              '见[逐seed比较](analysis/full_reference_validation_comparison_20261028.json)。', '',
              '## 重绘与复现', '',
              '从仓库根目录运行：', '', '```bash',
              'python scripts/report_free_lcmd.py',
              'python scripts/report_free_lcmd.py --from-snapshot reports/free_llm_five_seeds_to1485_20261024/figures',
              '```', '',
              '第一条读取本地runtime并核对冻结拟合、预测和批次哈希；第二条仅从Git保留的数值快照重绘。'
              '新克隆不包含检查点、逐fit预测和训练缓存，因此离线重绘不等于重新完成runtime完整审计。' +
              ('本轮已使用`--require-complete`完成全部十条37拟合/36批次核验；'
               '逐条最终审计与图表核验见[最终核验记录](final_audit.json)。'
               if summary['status'] == 'COMPLETE' else
               '全部十条完成后再用`--require-complete`生成最终报告并进行完整37拟合/36批次审计。'), '',
              '历史监控入口`reports/free_llm_five_seeds_to1485_20261024/plot_frozen_results.py`保留为兼容转发，'
              '绘图实现统一在`scripts/report_free_lcmd.py`。原阶段协议与失败回执保持原路径。', '']
    destination.write_text('\n'.join(lines))


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--from-snapshot', type=Path, help='Redraw retained JSON; no runtime reads')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--report', type=Path, help='Report destination; custom outputs default to OUTPUT/REPORT.md')
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(OUT/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    if args.from_snapshot:
        curves, summary, sources = load_snapshot(args.from_snapshot, args.require_complete, np)
    else:
        SOURCES.clear()
        curves, verification = collect(args.require_complete)
        summary = summarize(curves, verification, np)
        sources = SOURCES
    plot(curves, summary, np, plt)
    for name, value in [('validation_curves.json', curves), ('current_summary.json', summary),
                        ('source_hashes.json', sources)]:
        (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    report_path = args.report or ((REPORT_DIR if OUT == DEFAULT_OUT else OUT)/'REPORT.md')
    report(summary, report_path, OUT, curves)
    print(json.dumps({k: summary[k] for k in ('status', 'free_common_budget', 'lcmd_common_budget', 'latest_free')}))


if __name__ == '__main__':
    main()
