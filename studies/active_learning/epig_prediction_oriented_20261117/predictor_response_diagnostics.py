"""Post-freeze, label-free predictor/selector matching appendix.

This script is not a new acquisition hypothesis or a modification to the sealed
science implementation. It opens feature-only inputs, trusted original L333
weights, zero-label graph caches and completed selection receipts. No target
store or saved feedback is read; no target-dependent selection is possible.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import gc
import hashlib
from itertools import combinations
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

import study_support as support

HERE=Path(__file__).resolve().parent
OUT=HERE/'predictor_match'
SEEDS=(157,887,2357,6101,12203)
FACTOR_COLUMNS={'sample_volume':'V/ul','loading_solvent_volume':'Volume of loading solvent/ul','eluent_EA_fraction':'EA_fraction'}
CONDITIONS=('Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul','EA_fraction')


def sha_id(values):
    return hashlib.sha256(repr(tuple(values)).encode()).hexdigest()


def ea_fraction(value):
    left,right=str(value).split('/')
    a,b=Fraction(left),Fraction(right)
    if a<0 or b<0 or a+b<=0:
        raise ValueError('invalid eluent ratio')
    return float(b/(a+b))


def feature_frame():
    frame=support.load_features()
    if set(frame)!=set(support.FEATURE_COLUMNS):
        raise PermissionError('unexpected source feature schema')
    frame['EA_fraction']=frame['PE/EA'].map(ea_fraction)
    return frame


def prediction_contract(seed):
    original=support.BASE/f'runtime/seed_{seed}'
    model=original/'lcmd/round_00/model/best.pt'
    cache=original/'scrubbed_graphs.pt'
    context_path=original/'context.json'
    context=support.read(context_path)['contract']
    cache_receipt=support.read(cache.with_suffix('.pt.contract.json'))
    if cache_receipt['contract']!=context or cache_receipt['contract_hash']!=support.stable_hash(context):
        raise RuntimeError('original scrubbed graph context contract differs')
    if support.digest(cache)!=cache_receipt['sha256']:
        raise RuntimeError('original scrubbed graph byte hash differs')
    legacy=support.ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/protocol.json'
    published=next(v for v in support.read(legacy)['initial_states'] if v['seed']==seed)
    relative=str(cache.relative_to(support.ROOT))
    if published['sources'][relative]!=support.digest(cache):
        raise RuntimeError('scrubbed graphs differ from published frozen original bytes')
    freeze=support.read(HERE/'freeze.json')
    for path in (model,context_path,support.BASE/f'splits/row_seed_{seed}.csv'):
        if support.digest(path)!=freeze['dependencies'][str(path.relative_to(support.ROOT))]:
            raise RuntimeError('baseline dependency differs from EPIG science freeze')
    if published['checkpoint_sha256']!=support.digest(model):
        raise RuntimeError('L333 checkpoint differs from published frozen original bytes')
    return {'seed':seed,'model':str(model),'model_sha256':support.digest(model),
        'cache':str(cache),'cache_sha256':support.digest(cache),
        'cache_contract_sha256':support.digest(cache.with_suffix('.pt.contract.json')),
        'context_sha256':support.digest(context_path),'legacy_protocol_sha256':support.digest(legacy),
        'split_sha256':support.digest(support.BASE/f'splits/row_seed_{seed}.csv'),
        'target_scales':context['preprocessing']['target_scales'],'dtype':'original float32',
        'prediction_inputs':'outer_train_only; original L333 model; no labels assigned',
        'predict_batch_size':256,'torch_threads':1}


def predict_seed(seed,frame):
    import torch
    from qgeognn_al.active_learning_v2.runner import predict_outputs
    from qgeognn_al.models import load_predictor_checkpoint
    from qgeognn_al.active_learning_v2.cache import verify_cache
    contract=prediction_contract(seed)
    target=OUT/f'predictions_seed_{seed}.npz'
    receipt=OUT/f'predictions_seed_{seed}.json'
    split=support.partition(seed)
    outer=split.loc[split.role.isin(['l0','u0']),'canonical_index'].to_numpy(int)
    if len(outer)!=3330 or frame.sample_id.astype(str).tolist()!=split.sample_id.astype(str).tolist():
        raise RuntimeError('outer identity drift')
    if target.exists() or receipt.exists():
        if not target.exists() or not receipt.exists():
            raise RuntimeError('partial diagnostic prediction cache; preserve before explicit repair')
        saved=support.read(receipt)
        if saved['input']!=contract or saved['npz_sha256']!=support.digest(target):
            raise RuntimeError('diagnostic prediction cache contract/hash drift')
        return saved
    started=time.perf_counter()
    cache=Path(contract['cache'])
    cache_receipt=support.read(cache.with_suffix('.pt.contract.json'))
    if not verify_cache(cache,cache_receipt['contract']):
        raise RuntimeError('scrubbed cache missing')
    atom,angle=torch.load(cache,weights_only=False)
    if len(atom)!=len(frame) or len(angle)!=len(frame):
        raise RuntimeError('scrubbed graph count drift')
    for i,item in enumerate(atom):
        if torch.count_nonzero(item.y).item()!=0 or int(item.canonical_position.reshape(-1)[0])!=i:
            raise PermissionError('graph labels are not zero sentinels or canonical index drifted')
    torch.set_num_threads(1)
    model=load_predictor_checkpoint(Path(contract['model']))
    if model.training or sum(p.numel() for p in model.parameters())!=458952:
        raise RuntimeError('original QGeoGNN V2 model mismatch')
    prediction,order=predict_outputs(model,atom,angle,outer,batch_size=256)
    if not np.array_equal(order,outer) or prediction.dtype!=np.float32:
        raise RuntimeError('prediction order/dtype drift')
    np.savez_compressed(target,canonical_indices=outer,
        sample_ids=frame.iloc[outer].sample_id.to_numpy(dtype=str),q50=prediction[:,[1,4]])
    record={'input':contract,'npz_sha256':support.digest(target),
        'source_labels_read':0,'test_labels_read':0,'validation_labels_read':0,'new_gnn_fits':0,
        'graph_labels_verified_zero':len(atom),'prediction_rows':len(outer),
        'inference_seconds':time.perf_counter()-started,
        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    support.write_once(receipt,record)
    del atom,angle,model,prediction
    gc.collect()
    return record


def enumerate_pairs(seed,frame):
    split=support.partition(seed)
    outer=split.loc[split.role.isin(['l0','u0']),'canonical_index'].to_numpy(int)
    base=frame.iloc[outer].copy()
    base['canonical_index']=outer
    base['role']=split.iloc[outer].role.to_numpy()
    stored=np.load(OUT/f'predictions_seed_{seed}.npz',allow_pickle=False)
    if not np.array_equal(stored['canonical_indices'],outer) or stored['sample_ids'].tolist()!=base.sample_id.astype(str).tolist():
        raise RuntimeError('frozen diagnostic prediction identities drifted')
    preds={int(i):p.astype(np.float64) for i,p in zip(outer,stored['q50'])}
    scales=support.read(OUT/f'predictions_seed_{seed}.json')['input']['target_scales']
    scale=np.asarray([scales['V1'],scales['V2']])
    rows=[]
    for factor,active in FACTOR_COLUMNS.items():
        fixed=['canonical_smiles']+[c for c in CONDITIONS if c!=active]
        for key,group in base.groupby(fixed,sort=True,dropna=False):
            records=list(group.sort_values([active,'sample_id']).itertuples(index=False,name=None))
            # Explicit column lookup keeps original names with spaces intact.
            values=[dict(zip(group.columns,rec)) for rec in records]
            series_id=sha_id([factor,*key])
            for a,b in combinations(values,2):
                if a[active]==b[active]:
                    continue
                ia,ib=int(a['canonical_index']),int(b['canonical_index'])
                delta=preds[ib]-preds[ia]
                standardized=delta/scale
                rows.append({'seed':seed,'factor':factor,'series_id':series_id,'molecule_id':sha_id([a['canonical_smiles']]),
                    'canonical_index_a':ia,'canonical_index_b':ib,'sample_id_a':str(a['sample_id']),
                    'sample_id_b':str(b['sample_id']),'role_a':a['role'],'role_b':b['role'],
                    'active_value_a':float(a[active]),'active_value_b':float(b[active]),
                    'pred_delta_V1_ml':float(delta[0]),'pred_delta_V2_ml':float(delta[1]),
                    'pred_abs_delta_V1_L333std':float(abs(standardized[0])),
                    'pred_abs_delta_V2_L333std':float(abs(standardized[1])),
                    'pred_combined_delta_L333std':float(np.sqrt(np.mean(standardized**2))),
                    'frozen_predictor_only':True})
    return pd.DataFrame(rows)


def describe_pairs(pairs,scope):
    if len(pairs)==0:
        return {'scope':scope,'pairs':0,'series':0,'molecules':0,'distinct_rows':0,
            'series_equal_mean_combined_delta_L333std':None,'molecule_equal_mean_combined_delta_L333std':None,
            'V1_mean_abs_delta_ml':None,'V2_mean_abs_delta_ml':None,
            'V1_mean_abs_delta_L333std':None,'V2_mean_abs_delta_L333std':None,
            'mean_combined_delta_L333std':None,'fraction_both_pred_delta_below_0p1ml':None}
    return {'scope':scope,'pairs':len(pairs),'series':pairs.series_id.nunique(),'molecules':pairs.molecule_id.nunique(),
        'series_equal_mean_combined_delta_L333std':float(pairs.groupby('series_id').pred_combined_delta_L333std.mean().mean()),
        'molecule_equal_mean_combined_delta_L333std':float(pairs.groupby('molecule_id').pred_combined_delta_L333std.mean().mean()),
        'distinct_rows':len(set(pairs.sample_id_a)|set(pairs.sample_id_b)),
        'V1_mean_abs_delta_ml':float(pairs.pred_delta_V1_ml.abs().mean()),
        'V2_mean_abs_delta_ml':float(pairs.pred_delta_V2_ml.abs().mean()),
        'V1_mean_abs_delta_L333std':float(pairs.pred_abs_delta_V1_L333std.mean()),
        'V2_mean_abs_delta_L333std':float(pairs.pred_abs_delta_V2_L333std.mean()),
        'mean_combined_delta_L333std':float(pairs.pred_combined_delta_L333std.mean()),
        'fraction_both_pred_delta_below_0p1ml':float((pairs.pred_delta_V1_ml.abs().lt(.1)&pairs.pred_delta_V2_ml.abs().lt(.1)).mean())}


def summarize():
    frame=feature_frame()
    all_pairs=[]
    for seed in SEEDS:
        receipt=support.read(OUT/f'predictions_seed_{seed}.json')
        if receipt['npz_sha256']!=support.digest(OUT/f'predictions_seed_{seed}.npz'):
            raise RuntimeError('diagnostic prediction NPZ drift')
        all_pairs.append(enumerate_pairs(seed,frame))
    pairs=pd.concat(all_pairs,ignore_index=True)
    pairs.to_csv(OUT/'input_matched_pairs.csv',index=False)
    amplitude_columns=['pred_abs_delta_V1_L333std','pred_abs_delta_V2_L333std','pred_combined_delta_L333std']
    for grouping,name in [('series_id','input_series_summary.csv'),('molecule_id','input_molecule_summary.csv')]:
        pairs.groupby(['seed','factor',grouping])[amplitude_columns].mean().reset_index().to_csv(OUT/name,index=False)
    pair_summary=[]
    for (seed,factor),group in pairs.groupby(['seed','factor']):
        for scope,mask in [('outer_all',np.ones(len(group),bool)),('candidate_to_L333',group.role_a.ne(group.role_b)),('current_U_to_U',group.role_a.eq('u0')&group.role_b.eq('u0'))]:
            pair_summary.append({'seed':int(seed),'factor':factor,**describe_pairs(group.loc[mask],scope)})
    pd.DataFrame(pair_summary).to_csv(OUT/'input_pair_summary.csv',index=False)
    selection_rows=[]
    batch_pair_rows=[]
    inputs={}
    for directory in sorted((HERE/'acquisition').glob('*_P1_M1024')):
        complete=directory/'complete.json'
        if not complete.exists():
            continue
        completed=support.read(complete)
        if completed['freeze_sha256']!=support.digest(HERE/'freeze.json'):
            raise RuntimeError('selection receipt uses different science freeze')
        for name,expected in completed['files'].items():
            if support.digest(directory/name)!=expected:
                raise RuntimeError('completed acquisition artifact drift')
        summary=support.read(directory/'summary.json')
        seed=int(summary['seed']);family=summary['family']
        if not family.startswith('gp_'):
            continue
        batches=support.read(directory/'batches.json')
        split=support.partition(seed)
        labeled=set(split.loc[split.role.eq('l0'),'sample_id'].astype(str))
        u=set(split.loc[split.role.eq('u0'),'sample_id'].astype(str))
        universe=pairs.loc[pairs.seed.eq(seed)]
        for name in ('epig_conditional_greedy','ivr_conditional_greedy'):
            selected=set(map(str,batches[name]['selected_row_ids']))
            if len(selected)!=32 or not selected<=u or selected&labeled:
                raise PermissionError('diagnostic selection is not a legal current-U B32')
            for factor,group in universe.groupby('factor'):
                a,b=group.sample_id_a,group.sample_id_b
                both=a.isin(selected)&b.isin(selected)
                to_history=(a.isin(selected)&b.isin(labeled))|(b.isin(selected)&a.isin(labeled))
                touches=a.isin(selected)|b.isin(selected)
                any_matched_rows=set(a[touches])|set(b[touches])
                eligible_candidates=set(group.loc[group.role_a.eq('u0'),'sample_id_a'])|set(group.loc[group.role_b.eq('u0'),'sample_id_b'])
                for scope,mask in [('selected_within_batch',both),('selected_to_L333',to_history),('any_selected_endpoint',touches)]:
                    selection_rows.append({'seed':seed,'family':family,'method':name,'factor':factor,
                        'posterior_qualified':bool(summary['posterior_qualified']),
                        'selected_rows_32':32,'selected_rows_with_any_matched_partner':len(selected&any_matched_rows),
                        'candidate_rows_with_any_matched_partner':len(eligible_candidates),
                        **describe_pairs(group.loc[mask],scope)})
                included=group.loc[both|to_history].copy()
                if len(included):
                    included['family']=family;included['method']=name
                    included['pair_scope']=np.where(both.loc[included.index],'selected_within_batch','selected_to_L333')
                    batch_pair_rows.append(included)
        inputs[str(complete.relative_to(support.ROOT))]=support.digest(complete)
        inputs[str((directory/'batches.json').relative_to(support.ROOT))]=support.digest(directory/'batches.json')
    selected=pd.DataFrame(selection_rows)
    selected.to_csv(OUT/'selected_pair_summary.csv',index=False)
    (pd.concat(batch_pair_rows,ignore_index=True) if batch_pair_rows else pd.DataFrame()).to_csv(OUT/'selected_pair_details.csv',index=False)
    decision=support.read(HERE/'qualification/decision.json')
    expected_gp_states=sum(1 if (decision['selected_family_by_seed'][str(seed)] or '').startswith('gp_') else 2 for seed in SEEDS)
    render_report(pd.DataFrame(pair_summary),selected,expected_gp_states)
    generated=[OUT/name for name in ('input_matched_pairs.csv','input_pair_summary.csv','selected_pair_summary.csv','selected_pair_details.csv','input_series_summary.csv','input_molecule_summary.csv')]+[HERE/'PREDICTOR_MATCH_DIAGNOSTICS.md']
    support.dump(OUT/'appendix_receipt.json',{'schema':'post-science-freeze-label-free-predictor-appendix-v1',
        'created_unix':time.time(),'script_sha256':support.digest(Path(__file__)),
        'science_freeze_sha256':support.digest(HERE/'freeze.json'),'original_science_freeze_unchanged':True,
        'new_gnn_fits':0,'new_surrogate_fits':0,'source_target_reads':0,'test_target_reads':0,
        'validation_target_reads':0,'oracle_information_gain_computed':False,
        'completed_GP_P1_M1024_states':len(inputs)//2,'expected_GP_P1_M1024_states':expected_gp_states,
        'candidate_pairs':len(pairs),'inputs':inputs,
        'prediction_receipts':{f'predictions_seed_{s}.json':support.digest(OUT/f'predictions_seed_{s}.json') for s in SEEDS},
        'outputs':{str(p.relative_to(support.ROOT)):support.digest(p) for p in generated}})
    print(f'PREDICTOR_APPENDIX: {len(pairs)} input pairs, {len(inputs)//2}/{expected_gp_states} completed GP P1/M1024 states; 0 target reads / 0 fits',flush=True)


def render_report(base,selected,expected_gp_states):
    lines=['# 预测器与 EPIG / IVR 条件信息的匹配诊断','',
        '这是科学预注册封存后新增的独立、无标签附录；只描述输入结构、原L333预测器响应和已完成同状态批次。它不改变模型族、资格门槛、目标分布或正式两臂，不能据此调参或挑选策略。',
        '', '## 输入与严格边界','',
        '五seed 157/887/2357/6101/12203，各使用其原L333 QGeoGNN-V2 checkpoint及该seed outer-train 3330输入。模型字节SHA同时匹配本研究science freeze和原Free协议；scrubbed graph cache同时核对原context contract、cache字节SHA、原Free保存SHA，并逐图验证y全零与canonical index。只载入feature-only字段，没有调用RestrictedLabelStore、读取候选/validation/test真值或历史feedback。原float32模型以CPU/1线程、inference batch256计算q50，不新增任何神经网络或辅助模型拟合。',
        '', '严格pair要求canonical molecule相同、Density以及除指定因子之外的全部记录条件相同，仅V/ul、加载溶剂体积或PE/EA的EA分数不同。相同比例的PE/EA字符串先按精确分数归为同一级。全部合法Row组合保留；重复row/重叠series使pair不独立，不能把pair数充当统计样本数。',
        '', '响应为|Δq50|，同时保留mL和各seed固定L333端点标准差单位。combined幅度是sqrt(mean((Δq50/s)²))；它是冻结预测幅度，**不是真实响应、学习收益、误差、模型不确定性或oracle信息增益**。','',
        '## 原L333预测响应','',
        '| seed | 因子 | pairs | V1平均预测幅度 / L333std | V2平均预测幅度 / L333std | 两端均<0.1mL比例 |',
        '|---|---|---:|---:|---:|---:|']
    for r in base.loc[base.scope.eq('outer_all')].itertuples():
        lines.append(f'| {r.seed} | {r.factor} | {r.pairs} | {r.V1_mean_abs_delta_L333std:.6f} | {r.V2_mean_abs_delta_L333std:.6f} | {r.fraction_both_pred_delta_below_0p1ml:.1%} |')
    lines += ['', '同一seed/因子内按pair、series和molecule分别等权（下表幅度均为combined / L333std），再跨seed解释；不把54k相关pair合并为独立证据：', '', '| seed | 因子 | series | molecules | pair等权 | series等权 | molecule等权 |', '|---|---|---:|---:|---:|---:|---:|']
    for r in base.loc[base.scope.eq('outer_all')].itertuples():
        lines.append(f'| {r.seed} | {r.factor} | {r.series} | {r.molecules} | {r.mean_combined_delta_L333std:.6f} | {r.series_equal_mean_combined_delta_L333std:.6f} | {r.molecule_equal_mean_combined_delta_L333std:.6f} |')
    lines += ['', '## 同状态 P1 / M1024 选点触及哪些条件对照','']
    if selected.empty:
        lines+=['当前尚无完成的GP P1/M1024 batch receipt，选点匹配表为空；不推测EPIG/IVR偏好。']
    else:
        states=selected[['seed','family']].drop_duplicates()
        lines += [f'已完成并校验 {len(states)}/{expected_gp_states} 个GP状态（按冻结选择规则：本seed所选GP，或无所选GP时两种预注册核）；缺少状态保持缺少。概率资格失败的模型只作明确标记的离线方法诊断，不能由可算score宣称可靠EPIG或正式学习收益。', '',
            '| seed | GP | 方法 | 因子 | 批内pair | 至L333 pair | 批内pair平均combined预测幅度 / L333std |',
            '|---|---|---|---|---:|---:|---:|']
        for (seed,family,method,factor),g in selected.groupby(['seed','family','method','factor']):
            batch=g.loc[g.scope.eq('selected_within_batch')].iloc[0]
            hist=g.loc[g.scope.eq('selected_to_L333')].iloc[0]
            amplitude='—' if pd.isna(batch.mean_combined_delta_L333std) else f'{batch.mean_combined_delta_L333std:.6f}'
            lines.append(f'| {seed} | {family} | {method} | {factor} | {int(batch.pairs)} | {int(hist.pairs)} | {amplitude} |')
    lines += ['', '每一选中行“触及pair”分别记录三种口径：批内两端均选中、选中行对原L333已有行、任一端选中且另端可为未选U。第三种只表示存在输入匹配邻居，不能称已设计或购买完整对照。CSV同时保留各选中scope的series/molecule等权响应，以及具有任意匹配邻居的候选基数，避免把原数据网格密度误称为策略独特性。',
        '', '## 能与不能解释什么','',
        '若EPIG选择的体积对照在原模型下预测幅度很小，只能说明选点涉及当前冻结预测器响应弱的输入方向；不能确定真实物理变化存在、不能确定模型随后难以利用，也不能认定EPIG失败。GP辅助模型使用描述符/条件，并不等于原GNN表示，二者响应不匹配是待检验线索。反过来，预测幅度较大也不证明批次有真实Row AULC收益。最终仍需原QGeoGNN scratch重训的完整五seed配对轨迹。',
        '', '既有上样响应审计曾在已揭示历史数据及充分训练模型中报告响应压缩，同时披露同X响应差、未记录条件、seed异质性及标签边界事故。它支持把预测器可学习性作为独立机制问题，不能替代本附录不存在的真实pair响应。不得把本次无标签小幅度与历史有标签幅度拼接为本次oracle效用。若做预测器结构干预，必须让EPIG与IVR/LCMD同时更换同一模型，作为另一个研究因素。',
        '', '## 文件与复现','',
        '`predictor_match/input_matched_pairs.csv`：五seed全部输入pair及原q50差分；`input_pair_summary.csv`：候选网格与L/U分层、pair/series/molecule等权；`input_series_summary.csv`与`input_molecule_summary.csv`：组内均值明细；`selected_pair_summary.csv` / `selected_pair_details.csv`：已完成GP批次匹配；`predictions_seed_*.npz/json`：全outer冻结q50与严格身份receipt；`appendix_receipt.json`：独立脚本SHA、science freeze SHA、输入输出SHA和零训练/零目标访问声明。',
        '', '```bash',
        'KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_20261117/predictor_response_diagnostics.py',
        '# acquisition全部完成后只刷新匹配表；不会再运行原模型预测：',
        '/Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_20261117/predictor_response_diagnostics.py --summarize-only',
        '```','']
    (HERE/'PREDICTOR_MATCH_DIAGNOSTICS.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    support.verify_freeze()
    OUT.mkdir(exist_ok=True)
    if not args.summarize_only:
        frame=feature_frame()
        for seed in SEEDS:
            predict_seed(seed,frame)
            print(f'PREDICTOR_FROZEN seed={seed} rows=3330 fits=0',flush=True)
    summarize()

if __name__=='__main__':
    main()
