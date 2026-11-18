#!/usr/bin/env python3
"""Hash-locked, read-only loading of the three qualified native V2 Row models."""
from __future__ import annotations

import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[_name] = '1'
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

from hashlib import sha256
from io import StringIO
from pathlib import Path
import gc
import json
import sys

# Import the scientific array libraries before torch on this qualified runtime.
import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.models.qgeognn_v2 import (
    load_predictor_checkpoint, validate_predictor_checkpoint,
)

FORMAL = ROOT/'studies/predictor/final_4g_qualification'
ENGINEERING = ROOT/'studies/predictor/final_v2_engineering'
EXPECTED_SHA256 = {
    42: 'fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0',
    525: 'd86d314c55a4d6068a58d965755b2b4e5a664f5e18d877016f2da5af85acae07',
    1101: 'df37cdff353536e825ae269ca03418024913adc39d6061f143c8c5d63aca5901',
}
SOURCES: dict[str, dict] = {}


def _read(path: Path) -> bytes:
    content = path.read_bytes()
    record = dict(path=str(path.relative_to(ROOT)), sha256=sha256(content).hexdigest(), bytes=len(content))
    prior = SOURCES.get(record['path'])
    if prior is not None and prior != record:
        raise ValueError(f'Audit source changed: {path}')
    SOURCES[record['path']] = record
    return content


def _json(path: Path) -> dict:
    return json.loads(_read(path))


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _stable_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_engineering_evidence(evidence: dict) -> None:
    """Validate the existing official R2-pruned→V2 equivalence evidence only."""
    _require(evidence['status'] == 'PASS', 'Official V2 engineering gate failed')
    _require(evidence['fixture_rows'] == 4163, 'Official engineering fixture changed')
    _require(evidence['max_abs_difference'] == evidence['metric_max_abs_difference'] == 0,
             'Official measured R2→V2 equality no longer exact')
    _require(evidence['direct_construction'] is True and evidence['legacy_model_constructed_by_active_api'] is False,
             'Official native V2 construction contract changed')
    conversion = evidence['conversion']
    _require(conversion['status'] == 'PASS' and conversion['retained_values_bitwise_equal'] is True,
             'Official historical state conversion did not preserve retained weights')
    _require(conversion['missing_keys'] == conversion['unexpected_keys'] == [],
             'Official historical state conversion keys do not match')


def load_formal(seed: int):
    """Return ``(eval_model, identity)`` after official schema and provenance checks.

    This function reads weights, split metadata and recorded configuration only.
    It never loads graph caches, source target CSVs, numerical prediction caches,
    or labels. No inference or conversion is performed. Unknown formats fail.
    """
    _require(type(seed) is int and seed in EXPECTED_SHA256, 'Only formal Row model seeds 42/525/1101 are authorized')
    protocol = _json(FORMAL/'protocol.json')
    gate_path = ENGINEERING/'equivalence_audit.json'
    evidence = _json(gate_path)
    _require(sha256(_read(gate_path)).hexdigest() == protocol['engineering_gate_sha256'],
             'Qualification protocol does not bind the current engineering evidence')
    check_engineering_evidence(evidence)
    split_manifest = _json(FORMAL/'splits/split_manifest.json')
    _require(split_manifest['regenerated'] is False, 'Formal splits were regenerated')
    old_manifest_path = ROOT/'studies/predictor/clean_4g_baseline_qualification/splits/split_manifest.json'
    _require(sha256(_read(old_manifest_path)).hexdigest() == split_manifest['source_manifest_sha256'],
             'Historical qualification split manifest identity mismatch')
    split_entry = next(r for r in split_manifest['splits'] if r['split_mode'] == 'row' and r['seed'] == seed)
    split_path = FORMAL/f'splits/row_seed_{seed}.csv'
    split_content = _read(split_path)
    _require(sha256(split_content).hexdigest() == split_entry['sha256'], 'Formal split SHA mismatch')
    _require(_read(ROOT/split_entry['path']) == split_content, 'Formal split differs from historical frozen original')
    split = pd.read_csv(StringIO(split_content.decode()), dtype={'sample_id': str})
    _require(not split.sample_id.duplicated().any(), 'Duplicate sample IDs in source split')
    counts = split['split'].value_counts().to_dict()
    _require(counts == split_entry['row_counts'] == {'train': 3330, 'validation': 416, 'test': 417},
             'Formal Row role counts changed')
    _require(len(split) == 4163 and split.seed.eq(seed).all() and split.split_mode.eq('row').all(),
             'Formal source split seed/mode/domain mismatch')
    fit_hash = _stable_hash(sorted(split.loc[split['split'].eq('train'), 'sample_id']))
    result = FORMAL/f'results/row/seed_{seed}'
    summary, config = _json(result/'run_summary.json'), _json(result/'config.json')
    recorded_norm = _json(result/'normalization.json')
    _require(summary['status'] == 'COMPLETE' and summary['mode'] == 'row' and summary['seed'] == seed,
             'Formal run is not the completed requested Row model')
    checkpoint_path = ROOT/summary['checkpoint_path']
    _require(checkpoint_path == FORMAL/f'runtime/row/seed_{seed}/best.pt', 'Formal checkpoint path mismatch')
    digest = sha256(_read(checkpoint_path)).hexdigest()
    _require(digest == summary['checkpoint_sha256'] == EXPECTED_SHA256[seed], 'Formal checkpoint SHA mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Unknown/R2 formats are not silently reinterpreted. These three archives are
    # native V2; the official active API performs all schema/input validations.
    validate_predictor_checkpoint(checkpoint)
    _require(checkpoint['training_config'] == config, 'Checkpoint training configuration differs from frozen result')
    config_without_hash = {k: v for k, v in config.items() if k != 'config_hash'}
    _require(_stable_hash(config_without_hash) == config['config_hash'] == summary['config_hash'],
             'Frozen training config hash mismatch')
    _require(config['seed'] == seed and config['mode'] == 'row' and config['split_sha256'] == split_entry['sha256'],
             'Training configuration split/seed/mode mismatch')
    preproc, norm = checkpoint['preprocessing'], checkpoint['normalization']
    _require(preproc['fit_role'] == norm['fit_role'] == 'source_train' and
             preproc['fit_rows'] == norm['fit_row_count'] == 3330 and
             preproc['fit_ids_hash'] == norm['fit_ids_hash'] == fit_hash,
             'Preprocessing/condition normalization source-train identity mismatch')
    _require(all(preproc[key] == 0 for key in ['validation_rows_used','test_rows_used','target_rows_used']),
             'Preprocessing consumed an unauthorized role')
    _require(recorded_norm == dict(preproc, condition=norm), 'Frozen preprocessing record differs from checkpoint')
    scaler_path = FORMAL/f'runtime/row/seed_{seed}/scaler.json'
    scaler_content = _read(scaler_path)
    _require(json.loads(scaler_content) == preproc['scaler'], 'Runtime scaler differs from checkpoint')
    _require(sha256(scaler_content).hexdigest() == norm['eluent_scaler_sha256'], 'Scaler byte identity mismatch')
    _require(set(preproc['target_scales']) == {'V1','V2'} and
             all(np.isfinite(x) and x > 0 for x in preproc['target_scales'].values()), 'Invalid recorded target scales')
    _require(checkpoint['provenance']['epoch'] == summary['best_epoch'] and
             checkpoint['provenance']['split_sha256'] == split_entry['sha256'], 'Best checkpoint epoch/split provenance mismatch')
    model = load_predictor_checkpoint(checkpoint_path, device='cpu')
    _require(sum(p.numel() for p in model.parameters()) == checkpoint['parameter_count'] == summary['parameter_count'] == 458952,
             'Formal model parameter count mismatch')
    _require(all(torch.equal(value, model.state_dict()[key]) for key, value in checkpoint['model_state_dict'].items()),
             'Official loader altered a state tensor')
    _require(not model.training and sha256(_read(checkpoint_path)).hexdigest() == digest,
             'Formal checkpoint changed during loading or model is not in eval mode')
    identity = dict(seed=seed, model_role='formal Row seed (not AL budget)',
        checkpoint_path=str(checkpoint_path.relative_to(ROOT)), checkpoint_sha256=digest,
        checkpoint_schema_version=checkpoint['schema_version'], model_variant=checkpoint['model_variant'],
        split_path=str(split_path.relative_to(ROOT)), split_sha256=split_entry['sha256'],
        role_counts=counts, domain_rows=4163, training_rows=3330,
        fit_ids_hash=fit_hash, normalization=norm, preprocessing=preproc,
        scaler_path=str(scaler_path.relative_to(ROOT)), scaler_sha256=sha256(scaler_content).hexdigest(),
        training_config=config, provenance=checkpoint['provenance'], parameter_count=458952,
        best_epoch=summary['best_epoch'], epochs_run=summary['epochs_run'],
        output_order=checkpoint['output_order'], output_unit='mL; target scales normalize evaluation only',
        input_schema_hash=checkpoint['input_schema_hash'],
        official_loader='src.qgeognn_al.models.qgeognn_v2.load_predictor_checkpoint',
        native_v2=True, conversion_performed=False, loader_state_bitwise_equal=True,
        engineering_evidence_path=str(gate_path.relative_to(ROOT)),
        engineering_evidence_sha256=protocol['engineering_gate_sha256'],
        source_data_path='experiments/e0_4g_baseline/canonical_4g.csv',
        source_data_sha256_recorded_only=protocol['source_sha256'],
        molecular_graph_cache_path='experiments/e0_4g_baseline/graph_cache_4g.pt',
        molecular_graph_cache_sha256_recorded_only=protocol['graph_cache_sha256'])
    return model, identity


def write_audit() -> None:
    SOURCES.clear()
    for path in [FORMAL/'README.md', FORMAL/'FINAL_4G_QUALIFICATION_REPORT.md', ENGINEERING/'README.md',
                 ROOT/'scripts/studies/run_final_4g_qualification.py', ROOT/'scripts/studies/run_final_v2_engineering.py',
                 ROOT/'src/qgeognn_al/historical/conversion.py', ROOT/'src/qgeognn_al/models/qgeognn_v2.py',
                 ROOT/'src/qgeognn_al/schemas/conditions.py', ROOT/'src/qgeognn_al/training/predictor.py']:
        _read(path)
    identities = []
    for seed in EXPECTED_SHA256:
        model, identity = load_formal(seed)
        identities.append(identity)
        del model
        gc.collect()
    for entry in SOURCES.values():
        _require(sha256((ROOT/entry['path']).read_bytes()).hexdigest() == entry['sha256'], 'Audit input changed at final check')
    manifest = dict(status='PASS', execution_scope='checkpoint_audit.py automated run; preliminary shell incident separately disclosed below',
        training_runs=0, api_calls=0, inference_calls=0,
        target_csv_files_read=[], label_arrays_read=[], numerical_prediction_caches_read=[],
        preliminary_shell_incident=dict(
            file='studies/predictor/final_4g_qualification/splits/qualification_dataset_manifest.csv',
            operation='An initial sed command mistakenly listed this target-bearing file after row_seed_42.csv; sed cumulative line numbers meant only the first four split-metadata lines were output.',
            target_values_returned=0, target_ids_returned=0,
            disclosed_to_parent=True, target_file_not_used_by_loader=True),
        downstream_scoring_must_exclude_global_quarantine=True,
        original_training_identity_unchanged_by_quarantine=True,
        official_engineering_evidence='Previously measured full 4163-row R2-pruned/V2 exact output equivalence; not re-evaluated here.',
        qualification_models_are_native_v2=True, conversion_performed=False,
        identities=identities, sources=sorted(SOURCES.values(),key=lambda r:r['path']),
        audit_script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'checkpoint_audit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# 正式充分训练Row checkpoint身份核查', '',
           '结论：三份正式权重均存在，身份与原资格验证记录一致，官方严格加载通过。每个模型真正训练3330行，原validation416行、test417行；完整4163行数据域不能称为全部训练数据。模型seed525与主动学习预算L525没有对应关系。', '',
           '| Row模型seed | 训练/validation/test行数 | best epoch / 已运行epochs | 参数 | checkpoint SHA256 |',
           '| --- | --- | --- | --- | --- |']
    for i in identities:
        lines.append(f"| {i['seed']} | 3330 / 416 / 417 | {i['best_epoch']} / {i['epochs_run']} | 458952 | {i['checkpoint_sha256']} |")
    lines += ['', '## 权重、split与预处理身份', '',
        '三份权重均位于`studies/predictor/final_4g_qualification/runtime/row/seed_{seed}/best.pt`，不是仓库根目录runtime，也不是早期E0模型。以对应run_summary记录定位，再同时验证固定SHA和原资格验证run_summary的SHA。split从final资格验证的splits目录读取，字节与clean资格验证原始split相同；逐ID唯一、seed/mode与3330/416/417行数均核验。source-train ID排序哈希同时匹配checkpoint的preprocessing与condition normalization。', '',
        '| seed | split SHA256 | V1目标尺度mL | V2目标尺度mL |', '| --- | --- | --- | --- |']
    for i in identities:
        scales=i['preprocessing']['target_scales']
        lines.append(f"| {i['seed']} | {i['split_sha256']} | {scales['V1']:.12f} | {scales['V2']:.12f} |")
    lines += ['', '条件量及分子/eluent scaler均来自source-train：fit_rows=3330，validation/test/target_rows_used均0。checkpoint的scaler、runtime/scaler.json与results/normalization.json逐值相同；scaler原文件SHA同时匹配condition normalization。三个模型的分子/eluent scaler恰好相同，不据此强行替换其各自身份。目标尺度来自冻结记录，未重新读取或计算标签。q50输出已为mL，目标尺度仅用于归一化评价，不能重复反归一化。', '',
        '## 官方加载及R2→V2等价边界', '',
        '实际二进制均为原生`schema_version=1`、`model_variant=qgeognn_v2`，状态键是backbone/head/condition_branch；这些正式模型直接构建有效V2网络，不是旧R2归档，因此本轮无转换。只读`load_formal(seed)`调用官方`load_predictor_checkpoint`，保留schema、input schema hash、归一化、6输出顺序、458952参数与strict state_dict校验，再逐tensor确认加载前后bitwise相同，并确认eval模式。未知seed、身份不符或旧格式直接阻断，不临时改键绕过契约。', '',
        '历史官方`final_v2_engineering/equivalence_audit.json`已证明R2_PRUNED经`historical.conversion.convert_pruned_to_standalone`迁移后，在固定4163行工程fixture上全部六输出最大差0、指标最大差0、保留参数bitwise相同、无missing/unexpected keys。final资格验证protocol用SHA绑定此gate。本轮验证gate状态与SHA，不重跑它，因为其历史执行会访问test标签。该证据属于模型工程语义等价，不表示本轮三个Row模型都由同一个R2 checkpoint转换而来。', '',
        '## 推理接口与权限', '',
        '在本目录Python路径下导入`from checkpoint_audit import load_formal`，调用`model, identity = load_formal(42)`。identity包含preprocessing/scaler、normalization、目标尺度、output_order、split路径与SHA、训练配置及权重SHA；使用每个模型自身输入契约进行图构建。接口只构建和加载CPU eval模型，不执行forward，不读取图缓存、target CSV、标签或数值预测缓存。训练3330行角色来自冻结split元数据。源canonical与分子图缓存SHA仅转录原protocol，未在此子审计直接读取其内容。', '',
        '## 执行边界与下游隔离', '',
        '本loader程序及14项契约测试不读取任何target CSV、标签或正式predictions.gz。初始人工式shell检查曾误把qualification_dataset_manifest.csv列为sed的第二个文件参数；sed行号跨文件累计，工具实际只返回第一个row split的4行元数据，没有返回目标值或该manifest中的ID，但进程确打开并顺读了目标承载文件。该范围失误已向主审计披露，不能把整个工具历史描述为从未打开过目标文件；它未进入loader、标签或评分输入。', '',
        '主审计另有事故预览ID的全局quarantine；后续所有新评分须先排除该集合，独立结果QA也以此为硬门。排除评分行不改变原模型3330行训练事实、任何原始source split或权重。本报告仅核验原模型身份，不把排除后的评分子集重新称为其训练清单。', '',
        '复现：`PYTHONDONTWRITEBYTECODE=1 /Users/fish/miniforge3/envs/chromatography/bin/python -B studies/predictor/full_training_volume_response_audit_20261117/checkpoint_audit.py`。新manifest记录loader实际读取的配置、split、权重及源码SHA，并在结束时再次核对未变化。自动审计与测试均0新训练、0模型forward、0LLM/API、0标签数组访问；不更改旧checkpoint或报告。']
    (OUT/'CHECKPOINT_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS', seeds=list(EXPECTED_SHA256), source_files=len(SOURCES), native_v2=True,
                         training_rows=[i['training_rows'] for i in identities], conversion_performed=False)))


if __name__ == '__main__':
    write_audit()
