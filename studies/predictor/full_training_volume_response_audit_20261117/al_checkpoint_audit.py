#!/usr/bin/env python3
"""Audit AL checkpoint identity and label-free common-pair exposure permissions."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
AL = ROOT / 'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
PREVIOUS = ROOT / 'studies/active_learning/volume_response_retry_audit_20261117'
STATES = ROOT / 'studies/active_learning/llm_decision_behavior_audit_20261116/states.json.xz'
SEEDS = [157, 887, 2357, 6101, 12203]
BUDGETS = [525, 1005, 1485]
FORMAL_SEEDS = [42, 525, 1101]
INPUTS = {}
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'studies/active_learning/nocw_free_selection_diagnostics_20261101'))
from condition_metrics import _canonical_conditions


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stable(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def read(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    assert raw == path.read_bytes(), ('unstable input', str(path))
    digest = sha(raw)
    if expected:
        assert digest == expected, ('source SHA mismatch', str(path))
    INPUTS[str(path.relative_to(ROOT))] = {'path': str(path.relative_to(ROOT)), 'sha256': digest, 'bytes': len(raw)}
    return raw


def js(path, expected=None):
    return json.loads(read(path, expected))


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def history_for(snapshot, seed, budget):
    states = [b for b in snapshot['states'] if b['method'] == 'new_free_cw_sol61_xhigh' and b['seed'] == seed]
    exact = [b for b in states if b['L_before'] == budget]
    if exact:
        assert len(exact) == 1
        state = exact[0]
        contract = js(ROOT / state['contract_source'])
        freeze = js(ROOT / state['source'], state['source_sha256'])
        assert INPUTS[state['contract_source']]['sha256'] == freeze['artifact_hashes']['contract.json']
        assert contract['L_t_ids'] == state['labeled_ids']
        history = list(contract['L_t_ids'])
        source = state['contract_source']
    elif budget == 1485:
        final = next(b for b in states if b['L_after'] == budget)
        contract = js(ROOT / final['contract_source'])
        freeze = js(ROOT / final['source'], final['source_sha256'])
        assert INPUTS[final['contract_source']]['sha256'] == freeze['artifact_hashes']['contract.json']
        assert freeze['batch_ids'] == final['selected_ids']
        assert not set(contract['L_t_ids']) & set(freeze['batch_ids'])
        history = list(contract['L_t_ids']) + list(freeze['batch_ids'])
        source = final['source']
    else:
        raise ValueError('No exact historical budget')
    assert len(history) == len(set(history)) == budget
    return history, source


def checkpoint_path(seed, budget):
    if seed not in SEEDS or budget not in BUDGETS:
        raise ValueError('Unknown AL seed or label budget')
    return AL / f'runtime/seed_{seed}/free_cw_sol61_xhigh_rerun32/round_{(budget - 333) // 32:02d}/model/best.pt'


def audit_checkpoints(snapshot):
    import torch
    from qgeognn_al.models import load_predictor_checkpoint, validate_predictor_checkpoint
    for path in [ROOT / 'src/qgeognn_al/models/qgeognn_v2.py', ROOT / 'src/qgeognn_al/schemas/conditions.py']:
        read(path)
    rows, identities = [], []
    for seed in SEEDS:
        context_path = ROOT / f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/context.json'
        context = js(context_path)['contract']
        target_scales = None
        for budget in BUDGETS:
            path = checkpoint_path(seed, budget)
            row = {'al_seed': seed, 'label_budget': budget, 'fit_round': (budget - 333) // 32,
                   'checkpoint_path': str(path.relative_to(ROOT)), 'exists': path.exists(), 'strict_load_pass': False}
            if not path.exists():
                rows.append(row)
                continue
            history, history_source = history_for(snapshot, seed, budget)
            receipt_path = path.parent.parent / 'prediction_freeze.json'
            receipt = js(receipt_path)
            raw = read(path, receipt['checkpoint_sha256'])
            fit_path = path.with_name('fit_audit.json')
            fit = js(fit_path)
            assert receipt['test_truth_access_count'] == 0
            assert fit['test_labels_used_for_fit_or_checkpoint_selection'] == 0
            assert fit['checkpoint_sha256'] == receipt['checkpoint_sha256']
            assert fit['train_rows'] == budget
            assert fit['train_ids_hash'] == receipt['L_t_ids_hash'] == stable(sorted(history))
            saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
            validate_predictor_checkpoint(saved)
            assert saved['preprocessing']['scaler'] == context['preprocessing']['scaler']
            assert saved['normalization'] == context['normalization']
            if target_scales is None:
                target_scales = saved['preprocessing']['target_scales']
            assert saved['preprocessing']['target_scales'] == target_scales
            model = load_predictor_checkpoint(path, device='cpu')
            assert not model.training
            assert sum(parameter.numel() for parameter in model.parameters()) == saved['parameter_count']
            identity = {'model_kind': 'AL_NewCW', 'al_seed': seed, 'label_budget': budget,
                        'fit_round': row['fit_round'], 'train_rows': budget,
                        'checkpoint_path': row['checkpoint_path'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
                        'checkpoint_schema_version': saved['schema_version'], 'model_variant': saved['model_variant'],
                        'input_schema_hash': saved['input_schema_hash'], 'output_order': saved['output_order'],
                        'normalization': saved['normalization'], 'preprocessing': saved['preprocessing'],
                        'training_config': saved['training_config'], 'parameter_count': saved['parameter_count'],
                        'source_history_ids': history, 'source_history_ids_hash': stable(sorted(history)),
                        'source_history_identity_path': history_source,
                        'fit_audit_path': str(fit_path.relative_to(ROOT)), 'prediction_receipt_path': str(receipt_path.relative_to(ROOT)),
                        'strict_cpu_eval_load': True, 'new_fit': False, 'label_values_loaded': False}
            identities.append(jsonable(identity))
            row.update(checkpoint_sha256=receipt['checkpoint_sha256'], strict_load_pass=True,
                       model_variant=saved['model_variant'], checkpoint_schema_version=saved['schema_version'],
                       input_schema_hash=saved['input_schema_hash'], parameter_count=saved['parameter_count'],
                       train_rows=budget, H_ids_hash=identity['source_history_ids_hash'],
                       scaler_sha256=stable(jsonable(saved['preprocessing']['scaler'])),
                       normalization_sha256=stable(jsonable(saved['normalization'])),
                       target_scale_V1=target_scales['V1'], target_scale_V2=target_scales['V2'],
                       official_loader='qgeognn_al.models.load_predictor_checkpoint',
                       forward_executed=False, labels_loaded=False)
            rows.append(row)
            del model, saved
    pd.DataFrame(rows).to_csv(OUT / 'al_checkpoint_inventory.csv', index=False)
    (OUT / 'al_checkpoint_identities.json').write_text(json.dumps(identities, ensure_ascii=False, indent=2) + '\n')
    return pd.DataFrame(rows), identities


def load_al(seed, budget):
    """Return a strictly verified CPU model and its own scaler; never labels."""
    from qgeognn_al.models import load_predictor_checkpoint
    expected_path = checkpoint_path(seed, budget)
    identities = js(OUT / 'al_checkpoint_identities.json')
    identity = next(x for x in identities if x['al_seed'] == seed and x['label_budget'] == budget)
    path = ROOT / identity['checkpoint_path']
    assert path == expected_path
    read(path, identity['checkpoint_sha256'])
    model = load_predictor_checkpoint(path, device='cpu')
    assert not model.training
    return model, identity


def pair_key(axis, low, high):
    return stable([axis, *sorted([low, high])])


def audit_pair_identities(snapshot, identities):
    fields = ['seed', 'budget', 'axis', 'molecule', 'low_id', 'high_id', 'low_level', 'high_level', 'checkpoint_sha256']
    path = PREVIOUS / 'frozen_response_pairs.csv'
    # Project identifiers and covariates only; never parse actual_delta_ml.
    pairs = pd.read_csv(path, usecols=fields).drop_duplicates()
    read(path)
    assert set(pairs.budget) == {525, 1005}
    meta = snapshot['metadata']
    identities = {(x['al_seed'], x['label_budget']): x for x in identities}
    rows = []
    for row in pairs.to_dict('records'):
        seed, budget = int(row['seed']), int(row['budget'])
        identity = identities[(seed, budget)]
        h = set(identity['source_history_ids'])
        low, high = row['low_id'], row['high_id']
        assert low in h and high in h
        assert row['checkpoint_sha256'] == identity['checkpoint_sha256']
        a, b = _canonical_conditions(meta[low]), _canonical_conditions(meta[high])
        assert a is not None and b is not None and meta[low]['smiles'] == meta[high]['smiles']
        changed = [j for j in range(5) if a[j] != b[j]]
        expected = {'V/ul': 2, 'PE/EA': 0}[row['axis']]
        assert changed == [expected]
        low_level = 1 / (1 + float(a[0])) if expected == 0 else float(a[2])
        high_level = 1 / (1 + float(b[0])) if expected == 0 else float(b[2])
        assert low_level < high_level
        assert np.isclose(float(row['low_level']), low_level, rtol=1e-12, atol=1e-12)
        assert np.isclose(float(row['high_level']), high_level, rtol=1e-12, atol=1e-12)
        rows.append({'al_seed': seed, 'label_budget': budget, 'axis': row['axis'],
                     'pair_key': pair_key(row['axis'], low, high), 'low_id': low, 'high_id': high,
                     'canonical_smiles': meta[low]['smiles'], 'low_level': low_level, 'high_level': high_level,
                     'al_history_membership_verified': True, 'checkpoint_sha256': identity['checkpoint_sha256'],
                     'source': str(path.relative_to(ROOT)), 'truth_values_projected': False})
    all_pairs = pd.DataFrame(rows)
    assert not all_pairs.duplicated(['al_seed', 'label_budget', 'pair_key']).any()
    all_pairs.to_csv(OUT / 'al_pair_identity_by_budget.csv', index=False)
    common = []
    for (seed, key), group in all_pairs.groupby(['al_seed', 'pair_key']):
        if set(group.label_budget) != {525, 1005}:
            continue
        first = group.iloc[0].to_dict()
        first.pop('label_budget')
        first.pop('checkpoint_sha256')
        first['common_budgets'] = '525|1005|1485'
        assert first['low_id'] in identities[(seed, 1485)]['source_history_ids']
        assert first['high_id'] in identities[(seed, 1485)]['source_history_ids']
        common.append(first)
    common = pd.DataFrame(common)
    common.to_csv(OUT / 'al_common_pair_identities.csv', index=False)
    all_pairs.groupby(['al_seed', 'label_budget', 'axis']).agg(pairs=('pair_key', 'size'), molecules=('canonical_smiles', 'nunique')).reset_index().to_csv(OUT / 'al_pair_coverage_by_budget.csv', index=False)
    return common


def exposure_class(low_role, high_role):
    roles = {low_role, high_role}
    if 'test' in roles:
        return 'source_test_involved'
    if roles == {'train'}:
        return 'both_source_train'
    if roles == {'validation'}:
        return 'both_source_validation'
    if roles == {'train', 'validation'}:
        return 'mixed_source_train_validation'
    return 'unmapped_or_other'


def scoring_permission(low_id, high_id, low_role, high_role, quarantined_ids):
    kind = exposure_class(low_role, high_role)
    quarantined = low_id in quarantined_ids or high_id in quarantined_ids
    allowed = kind in ['both_source_train', 'both_source_validation', 'mixed_source_train_validation'] and not quarantined
    reason = 'global_incident_quarantine' if quarantined else 'formal_source_test_conflict' if kind == 'source_test_involved' else 'role_unmapped' if not allowed else ''
    return kind, quarantined, allowed, reason


def load_quarantine(path):
    incident = js(path)
    if 'quarantined_ids' not in incident:
        raise ValueError('Incident must expose quarantined_ids explicitly before scoring permission can be granted')
    assert isinstance(incident['quarantined_ids'], list) and all(isinstance(sid, str) for sid in incident['quarantined_ids'])
    return set(incident['quarantined_ids'])


def pair_permissions(common, snapshot):
    role_path = OUT / 'formal_split_roles.csv'
    metadata_path = OUT / 'source_metadata.csv'
    quarantine_path = OUT / 'LABEL_BOUNDARY_INCIDENT.json'
    for path in [role_path, metadata_path, quarantine_path]:
        if not path.exists():
            raise FileNotFoundError('Pair permission is fail-closed until metadata, roles and quarantine are fixed: ' + str(path))
    roles = pd.read_csv(io.BytesIO(read(role_path)))
    metadata = pd.read_csv(io.BytesIO(read(metadata_path))).set_index('sample_id')
    quarantine = load_quarantine(quarantine_path)
    assert metadata.index.is_unique
    old_meta = snapshot['metadata']
    for sid, prior in old_meta.items():
        assert sid in metadata.index
        row = metadata.loc[sid]
        current = {'smiles': row.canonical_smiles, 'conditions': {key: row[key] for key in ['PE/EA', 'Density g/ml', 'V/ul', 'loading solvent', 'Volume of loading solvent/ul']}}
        assert prior['smiles'] == current['smiles'] and _canonical_conditions(prior) == _canonical_conditions(current)
    rows = []
    for formal_seed in FORMAL_SEEDS:
        selected = roles[roles.row_seed == formal_seed]
        assert selected.sample_id.is_unique and len(selected) == 4163
        assert selected.source_role.value_counts().to_dict() == {'train': 3330, 'test': 417, 'validation': 416}
        mapping = selected.set_index('sample_id').source_role.to_dict()
        for pair in common.to_dict('records'):
            low, high = pair['low_id'], pair['high_id']
            low_role, high_role = mapping.get(low, 'unmapped'), mapping.get(high, 'unmapped')
            kind, quarantined, authorized, reason = scoring_permission(low, high, low_role, high_role, quarantine)
            rows.append({**pair, 'formal_row_seed': formal_seed, 'low_source_role': low_role, 'high_source_role': high_role,
                         'formal_exposure_class': kind, 'global_incident_quarantine': quarantined,
                         'allow_new_scoring': authorized, 'exclusion_reason': reason,
                         'label_budget_is_not_formal_row_seed': True, 'truth_values_projected': False})
    result = pd.DataFrame(rows)
    global_roles = result.groupby(['al_seed', 'pair_key']).formal_exposure_class.apply(lambda x: 'source_test_involved' in set(x)).to_dict()
    global_legal = result.groupby(['al_seed', 'pair_key']).formal_exposure_class.apply(lambda x: len(x) == 3 and set(x) <= {'both_source_train', 'both_source_validation', 'mixed_source_train_validation'}).to_dict()
    result['any_formal_source_test'] = [global_roles[(row.al_seed, row.pair_key)] for row in result.itertuples()]
    result['global_all_three_non_test'] = [global_legal[(row.al_seed, row.pair_key)] and not row.global_incident_quarantine for row in result.itertuples()]
    assert not result[result.formal_exposure_class == 'source_test_involved'].allow_new_scoring.any()
    assert not result[result.global_incident_quarantine].allow_new_scoring.any()
    result.to_csv(OUT / 'al_common_pair_exposure.csv', index=False)
    result.groupby(['al_seed', 'formal_row_seed', 'axis', 'formal_exposure_class', 'allow_new_scoring']).agg(pairs=('pair_key', 'size'), molecules=('canonical_smiles', 'nunique')).reset_index().to_csv(OUT / 'al_common_pair_exposure_summary.csv', index=False)
    (OUT / 'al_permission_boundary.json').write_text(json.dumps({'status': 'PASS_ROLE_AND_QUARANTINE_GATED',
        'per_formal_comparison_rows': len(result), 'allowed': int(result.allow_new_scoring.sum()),
        'quarantined_ids_n': len(quarantine), 'quarantined_pair_rows': int(result.global_incident_quarantine.sum()),
        'source_test_pair_rows_excluded': int((result.formal_exposure_class == 'source_test_involved').sum()),
        'formal_domain_rows': len(metadata), 'AL_metadata_rows': len(old_meta),
        'formal_only_metadata_ids': sorted(set(metadata.index) - set(old_meta)),
        'target_values_read_by_this_module': 0, 'new_fits': 0, 'new_API_calls': 0}, indent=2) + '\n')
    return result


def write_report(inventory, common, exposure=None):
    table = common.groupby(['al_seed', 'axis']).agg(common_pairs=('pair_key', 'size'), molecules=('canonical_smiles', 'nunique')).reset_index()
    lines = ['| AL seed | 因素 | L525∩L1005共同对 | 分子数 |', '|---|---|---|---|']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in table.to_numpy()]
    permission = '角色/隔离名单未全部固定，权限阶段阻断；不授予新评分许可。' if exposure is None else f'共{len(exposure)}个AL seed×正式seed×同pair比较记录；允许{int(exposure.allow_new_scoring.sum())}，其余明确隔离。详见al_common_pair_exposure.csv。'
    text = f'''# 主动学习checkpoint与同pair身份/权限审计

## checkpoint资格

New CW的五个AL seed是157/887/2357/6101/12203，标签预算L525/L1005/L1485对应round06/21/36。已找到{int(inventory.exists.sum())}/15份已有checkpoint，官方`load_predictor_checkpoint`严格CPU eval加载通过{int(inventory.strict_load_pass.sum())}/15。无R2转换、无替代模型、无新增拟合，未执行forward；本模块不读取任何目标值。

**正式模型的Row seed525是随机划分种子，不是AL标签预算L525。** 文件中用`formal_row_seed`与`al_seed/label_budget`分开命名。完整checkpoint路径、SHA、schema、参数数、H身份、scaler与归一化hash见`al_checkpoint_inventory.csv`和`al_checkpoint_identities.json`。

H525/H1005由对应选前contract取得，H1485由最后一个实际H1453和已冻结接受32个ID构成，再核对最终fit/prediction receipt的train ID hash及行数。L1485五个AL历史checkpoint均存在，可作为同pair扩展，无需训练。每份保存自身preprocessing、target_scales、normalization与六输出顺序；不会向正式3330训练模型强套AL scaler，也不将不同模型状态的梯度或预测当作同一固定空间。

## 同pair交集与身份

只从旧`frozen_response_pairs.csv`投影seed/budget/因素/ID/条件水平/checkpoint SHA列，**没有将actual_delta_ml或任何真值列作数值解析或载入分析**；源文件全字节只用于SHA绑定。checkpoint本身没有用于新标签查询。以label-free AL metadata重新核对同分子、恰一个因素变化、物理单位与低高水平，且两ID均在对应H。按AL seed保留L525与L1005的相同pair ID交集，并确认端点也在H1485；可选1485只预测这些相同ID，不扩大标签范围。

{chr(10).join(lines)}

同一全量Row模型与不同AL seed的比较是跨模型预测比较，不是配对随机种子试验。多个实验对可共享分子/条件系列，不能把所有pair当独立科研重复。

## 正式曝光与隔离

{permission}

每个正式Row seed分别标注两端source-train、mixed train-validation、两端validation或涉及source-test。涉及该模型source-test的pair全部阻断新评分；AL中曾揭示并不覆盖正式source-test禁令。事故隔离名单中的任一ID在所有模型下都拒绝评分。另存任一正式seed涉及test的标记和三正式seed均非test的敏感性子集，不能用其它seed的train角色放宽当前模型的test边界。

完整正式X域为4163行，而旧AL label-free metadata为4162行；缺少的正式独有行不补造成AL历史，也不强行扩展共同pair。正式角色映射及元数据由独立模块提供，目标读取与评分由根分析严格按这些许可执行；本模块只产身份与权限，不读取正式train/validation/test truth。

## 接口与复现

`model, identity = load_al(al_seed, label_budget)`返回官方严格加载模型与该checkpoint自身scaler/目标尺度；不加载标签。`python -B .../al_checkpoint_audit.py --checkpoints-only`只运行15份身份与共同pair检查；默认进一步要求formal_split_roles.csv、source_metadata.csv和带quarantined_ids数组的LABEL_BOUNDARY_INCIDENT.json全部已固定后才产生许可。
'''
    (OUT / 'AL_CHECKPOINT_AUDIT.md').write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(lzma.decompress(read(STATES)))
    read(ROOT / 'studies/active_learning/nocw_free_selection_diagnostics_20261101/condition_metrics.py')
    inventory, identities = audit_checkpoints(snapshot)
    common = audit_pair_identities(snapshot, identities)
    exposure = None if args.checkpoints_only else pair_permissions(common, snapshot)
    write_report(inventory, common, exposure)
    (OUT / 'al_checkpoint_inputs.json').write_text(json.dumps({'sources': list(INPUTS.values()),
        'new_fits': 0, 'new_API_calls': 0, 'numeric_truth_values_projected': 0,
        'policy': 'Only identities/covariates and existing checkpoint parameters; no AL observed catalog/input truth or formal test predictions.'}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'checkpoints': len(inventory), 'strict_load_pass': int(inventory.strict_load_pass.sum()), 'common_pairs': len(common), 'permission_rows': None if exposure is None else len(exposure)}))


if __name__ == '__main__':
    main()
