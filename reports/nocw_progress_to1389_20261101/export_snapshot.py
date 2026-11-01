"""Publish compact provenance for the latest audited NoCW validation curves.

Reads saved metadata only. Does not evaluate labels, train, launch workers,
send requests, or copy full request/response bodies into the report.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REV = ROOT / 'studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions'
SOURCE = REV / 'five_seed_nocw_to1005_20261029'
EXT = REV / 'five_seed_nocw_to1389_20261101'
SEEDS = (157, 887, 2357, 6101, 12203)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


summary = read(OUT / 'summary.json')
monitor = read(EXT / 'progress/monitor_state.json')
assert monitor['checked_utc'] == summary['audited_at_utc']
assert monitor['supervisor_alive'] and not monitor['active_issues']
assert monitor['paid_calls_during_audit'] == monitor['test_truth_reads'] == 0
policies = read(EXT / 'extension_to1389_amendment.json')
policy_paths = [SOURCE / name for name in policies['legacy_amendment_hashes']]
for path in policy_paths:
    assert sha(path) == policies['legacy_amendment_hashes'][path.name]
assert sha(EXT / 'extension_to1389_amendment.json') == monitor['amendment_sha256']
policy_paths.append(EXT / 'extension_to1389_amendment.json')

fits, batches, milestones = [], [], []
for seed in SEEDS:
    state = monitor['seeds'][str(seed)]
    assert state['process_alive'] and state['lock_held']
    budget = summary['status'][str(seed)]['completed_labels']
    assert state['budget'] == budget
    run = EXT if state['mode'] == 'EXTENSION' else SOURCE
    for r in range((budget-333)//32 + 1):
        frozen_path = run / f'runtime/seed_{seed}/llm32_nocw/round_{r:02d}/prediction_freeze.json'
        frozen = read(frozen_path)
        model = Path(frozen['checkpoint_path']).parent
        audit_path = model / 'fit_audit.json'
        audit = read(audit_path)
        assert audit['train_rows'] == 333 + 32*r
        assert audit['test_labels_used_for_fit_or_checkpoint_selection'] == frozen['test_truth_access_count'] == 0
        assert audit['checkpoint_sha256'] == frozen['checkpoint_sha256']
        assert audit['prediction_sha256'] == frozen['prediction_sha256']
        fits.append({'seed':seed, 'round':r, 'budget':333+32*r,
                     'prediction_freeze_path':str(frozen_path.relative_to(ROOT)),
                     'prediction_freeze_sha256':sha(frozen_path),
                     'fit_audit_path':str(audit_path.relative_to(ROOT)),
                     'fit_audit_sha256':sha(audit_path), 'fit_audit':audit})
        if r:
            batch_path = run / f'selections/seed_{seed}/llm32_nocw/round_{r-1:02d}/batch_freeze.json'
            batch = read(batch_path)
            assert len(batch['batch_ids']) == len(set(batch['batch_ids'])) == 32
            batches.append({'seed':seed, 'round':r-1, 'resulting_budget':333+32*r,
                            'batch_freeze_path':str(batch_path.relative_to(ROOT)),
                            'batch_freeze_sha256':sha(batch_path), 'batch_ids':batch['batch_ids'],
                            'artifact_hashes':batch['artifact_hashes']})
    path = SOURCE / f'progress/milestone_seed_{seed}_L1005.json'
    if path.exists():
        milestones.append({'seed':seed, 'path':str(path.relative_to(ROOT)), 'sha256':sha(path)})

# SOURCE owns all original physical attempts; EXT contributes only rounds >=21.
# This avoids counting imported prefix requests as new work.
attempts = []
for run in (SOURCE, EXT):
    for seed in SEEDS:
        for path in sorted((run/f'selections/seed_{seed}/llm32_nocw').glob('round_*/attempt_*.json')):
            match = re.fullmatch(r'attempt_(\d+)_(\d+)\.json', path.name)
            rnd = int(path.parent.name.split('_')[-1])
            if not match or (run == EXT and rnd < 21):
                continue
            before = sha(path)
            attempt = read(path)
            assert sha(path) == before, 'attempt changed during snapshot'
            assert isinstance(attempt.get('success'), bool), path
            attempts.append({'path':str(path.relative_to(ROOT)), 'sha256':before,
                             'seed':seed, 'round':rnd, 'turn':int(match[1]), 'attempt_index':int(match[2]),
                             'request_sha256':attempt.get('request_sha256'), 'success':attempt['success'],
                             'error_type':attempt.get('error_type'), 'http_status':attempt.get('http_status')})

write('fit_audits.json', {'fits':fits, 'test_truth_reads':0})
write('accepted_batch_index.json', {'batches':batches, 'batch_count':len(batches),
                                  'copied_prefix_batches_counted_once':True})
write('physical_attempt_index.json', {'attempts':attempts,
                                     'count':len(attempts),
                                     'imported_extension_prefix_attempts_excluded':True,
                                     'active_unfinished_attempts_excluded':True})
write('progress_snapshot.json', {'snapshot_time':summary['snapshot_time'],
                                'checked_utc':monitor['checked_utc'], 'target_budget':1389,
                                'supervisor_pid':monitor['supervisor_pid'], 'supervisor_alive':True,
                                'workers':monitor['seeds'], 'source_L1005_milestones':milestones,
                                'completed_batches':len(batches), 'completed_prediction_freezes':len(fits),
                                'new_model_calls_by_publication':0, 'new_fits_by_publication':0,
                                'test_truth_reads_by_publication':0})
write('amendment_index.json', {'original_protocol_sha256':policies['original_protocol_sha256'],
                             'amendments':[{'path':str(path.relative_to(ROOT)), 'sha256':sha(path)} for path in policy_paths]})

rows = '\n'.join(f'| {seed} | L{summary["status"][str(seed)]["completed_labels"]} | '
                 f'{summary["status"][str(seed)]["completed_batches"]}/33 | '
                 f'{summary["status"][str(seed)]["validation_nrmse"]:.6f} |' for seed in SEEDS)
means = summary['matched_means']
report = f'''# NoCW 五 seed 最新验证结果（目标 L1389）

快照时间：{summary['snapshot_time']}（北京时间）。运行核验时间：{monitor['checked_utc']}。

五个 worker 均经实际进程与持锁检查，继续运行到 L1389。该快照中五个 worker 均在接口退避重试；重试不计入有效采集进展。L1389 尚未完成，本目录为阶段结果。

| Seed | 已完成拟合到 | 已完成批次 | 最新验证 NRMSE |
|---|---:|---:|---:|
{rows}

指标为每轮已保存最佳 checkpoint 对应的 `validation_selection_score`，越低越好。曲线只使用存在 prediction freeze 且通过 checkpoint、prediction、训练 ID、初始化及验证集 ID 哈希核对的已完成拟合；共 {len(batches)} 个有效采集批次、{len(fits)} 个预测冻结（含五个初始模型）。

![最新方法对比](nocw_latest_comparison.png)

![NoCW 五 seed 曲线](nocw_five_seed_curves.png)

所有方法的五 seed 均值只比较共同完成的 L{summary['common_five_seed_budget']}：NoCW {means['nocw']:.6f}，Free LLM {means['free_llm32_scientist_v2']:.6f}，Hybrid {means['hybrid']:.6f}，LCMD {means['lcmd']:.6f}，Random {means['random']:.6f}。没有对尚未完成的轮次外推，也没有将不同 seed 数目的后段均值拼接。现有 Random/Hybrid 参照最多至 L1005；LCMD/Free 参照按对应 seed 当前完成预算截取。

此处为历史对照的描述性比较：Free LLM 的 seed 157/6101 保留 Astra/high 旧前缀；NoCW 全程使用 gpt-6.1-sol/xhigh。不能据此单独作严格配对的因果结论。

## 审计与复核

- 新旧入口 validate、原阶段审计和延长段审计均通过；原协议与全部 12 项修订 SHA 保持一致。
- `fit_audits.json` 保存拟合元数据，`accepted_batch_index.json` 保存已完成批次 ID 与冻结哈希；导入的旧前缀仅计一次。
- `physical_attempt_index.json` 记录原阶段物理 attempt 和延长段 round >=21 的终结记录哈希，支持三位及以上 attempt 编号，不重复统计复制的旧请求。
- `source_audit.json` 保存曲线源文件哈希；`amendment_index.json` 保存原协议与修订索引；`progress_snapshot.json` 保存本次实际进程核验与旧 L1005 里程碑哈希。
- 本次发布不调用模型、不重新训练、不读取测试真值，不将图表反馈用于采集。
- 完整输入、原始响应和本地模型运行目录继续按仓库运行数据规则保留本机；本次提交发布聚合曲线、元数据、冻结哈希、入口代码与修订，不修改任何原始 lineage。

本机已有全部冻结运行数据时，先运行新旧入口 validate 和监控审计，再依次执行本目录 `plot_latest.py`、`export_snapshot.py` 更新阶段快照。全部五 seed 达到 L1389 后另由新入口 report 生成最终报告；本快照不替代最终报告。
'''
(OUT/'REPORT.md').write_text(report)
manifest = {str(path.relative_to(OUT)):sha(path) for path in sorted(OUT.rglob('*'))
            if path.is_file() and path.name != 'snapshot_manifest.json' and '__pycache__' not in path.parts}
write('snapshot_manifest.json', {'snapshot_time':summary['snapshot_time'], 'files':manifest,
                                'source_protocol_sha256':policies['original_protocol_sha256'],
                                'extension_amendment_sha256':monitor['amendment_sha256']})
print(json.dumps({'status':'PASS', 'batches':len(batches), 'prediction_freezes':len(fits),
                  'physical_attempts':len(attempts), 'report_files':len(manifest)+1,
                  'model_calls':0, 'new_fits':0, 'test_truth_reads':0},ensure_ascii=False))
