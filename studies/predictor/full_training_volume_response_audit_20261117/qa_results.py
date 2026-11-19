#!/usr/bin/env python3
"""Independent scores/permissions QA from safe derived deltas and predictions."""
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
from io import StringIO
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]
SEEDS=[42,525,1101]; AL=[157,887,2357,6101,12203]
METHODS={'Fully_trained','AL_L525','AL_L1005','AL_L1485'}
FIELDS=['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']
SOURCES={}; CHECKS=[]

def read(name):
    path=(OUT/name).resolve(); data=path.read_bytes(); key=str(path.relative_to(ROOT))
    item=dict(path=key,sha256=sha256(data).hexdigest(),bytes=len(data))
    assert key not in SOURCES or SOURCES[key]==item, ('changed input',name)
    SOURCES[key]=item; return data

def csv(name): return pd.read_csv(StringIO(read(name).decode()))
def js(name): return json.loads(read(name))
def digest(value): return sha256(repr(value).encode()).hexdigest()

def statistics(g,threshold):
    a,p=g.delta_true.to_numpy(float),g.delta_pred.to_numpy(float)
    meaningful=np.abs(a)>threshold; flat=np.abs(p)<.1; correct=np.sign(a)==np.sign(p)
    def conditional(mask): return float(mask[meaningful].mean()) if meaningful.any() else np.nan
    actual=float(np.abs(a).mean()); predicted=float(np.abs(p).mean()); error=float(np.abs(p-a).mean())
    return dict(n_pairs=len(g),n_meaningful=int(meaningful.sum()),molecules=g.molecule.nunique(),series=g.series_key.nunique(),
        observed_abs_mean=actual,predicted_abs_mean=predicted,difference_mae=error,
        amplitude_error_mean=float(np.abs(np.abs(p)-np.abs(a)).mean()),direction_accuracy=conditional(correct),
        correct_nonflat_fraction=conditional(correct&~flat),wrong_nonflat_fraction=conditional((a*p<0)&~flat),
        flat_fraction=conditional(flat),actual_threshold_ml=threshold,predicted_flat_threshold_ml=.1,
        amplitude_ratio=predicted/actual,mae_reduction_vs_zero_response=1-error/actual)

def summarize(frame,keys):
    rows=[]
    for key,g in frame.groupby(keys):
        for threshold in [.1,.5,1.]: rows.append(dict(zip(keys,key),**statistics(g,threshold)))
    return pd.DataFrame(rows)

def compare(name,a,b,keys,columns,tolerance=2e-10):
    assert not a.duplicated(keys).any() and not b.duplicated(keys).any(),name
    a,b=a.set_index(keys).sort_index(),b.set_index(keys).sort_index(); assert a.index.equals(b.index),(name,'keys')
    maximum=0.
    for c in columns:
        x,y=a[c].to_numpy(float),b[c].to_numpy(float)
        assert np.allclose(x,y,atol=tolerance,rtol=0,equal_nan=True),(name,c,np.nanmax(np.abs(x-y)))
        finite=np.abs(x-y)[np.isfinite(x-y)]
        if len(finite): maximum=max(maximum,float(finite.max()))
    CHECKS.append(dict(check=name,rows=len(a),max_absolute_difference=maximum,tolerance=tolerance,status='PASS'))

def exposure(a,b,matched=False):
    roles={a,b}
    assert roles<={'train','validation'},('illegal scored role',roles)
    if roles=={'train'}: return 'both_source_train' if matched else 'train_train'
    if roles=={'validation'}: return 'both_source_validation' if matched else 'validation_validation'
    return 'mixed_source_train_validation' if matched else 'train_validation'

def main():
    incident=js('LABEL_BOUNDARY_INCIDENT.json'); quarantine=set(incident['quarantined_ids']); assert len(quarantine)==3
    metadata=csv('source_metadata.csv'); lookup=metadata.set_index('sample_id')
    split=csv('formal_split_roles.csv'); roles={s:g.set_index('sample_id').source_role.to_dict() for s,g in split.groupby('row_seed')}
    assert set(roles)==set(SEEDS)
    original_split_hashes={42:'a8a4a322cd9c75253b1a4ed8c5c91ededed74a933df2db2685015c8b3e008008',
        525:'58876dc5a9033a0584da964e1e52bd0fa5d1aeba8131410060cce947fe11d0a1',
        1101:'fb51a225dd2b802b4dbc2982fa45166a7c11962fef9add2015e6637a1a2a6bf1'}
    for seed in SEEDS:
        original_path=ROOT/f'studies/predictor/final_4g_qualification/splits/row_seed_{seed}.csv'
        original=csv(original_path)
        assert sha256(read(original_path)).hexdigest()==original_split_hashes[seed]
        derived=split[split.row_seed.eq(seed)]
        assert original.sample_id.tolist()==derived.sample_id.tolist()
        assert original['split'].tolist()==derived.source_role.tolist()
        assert original.canonical_smiles.tolist()==derived.canonical_smiles.tolist()
    def verify_pairs(frame,seed_column,exposure_column,matched=False):
        assert not set(frame.low_id)&quarantine and not set(frame.high_id)&quarantine
        unique=frame[[seed_column,'axis','low_id','high_id',exposure_column]].drop_duplicates()
        for row in unique.itertuples(index=False,name=None):
            seed,axis,a,b,reported=row
            assert reported==exposure(roles[seed][a],roles[seed][b],matched)
            x,y=lookup.loc[a],lookup.loc[b]
            assert x.canonical_smiles==y.canonical_smiles
            for field in FIELDS:
                if field!=axis: assert x[field]==y[field],(axis,field,a,b)
            if axis=='V/ul': assert x['V/ul']<y['V/ul']
            else:
                assert axis=='PE/EA'
                assert 1/(1+Fraction(x[axis]))<1/(1+Fraction(y[axis]))
        return len(unique)
    raw={}; prediction_rows=[]
    for seed in SEEDS:
        p=csv(f'formal_predictions_seed_{seed}.csv'); assert p.sample_id.is_unique
        legal=split[(split.row_seed==seed)&split.source_role.isin(['train','validation'])&~split.sample_id.isin(quarantine)]
        assert p.sample_id.tolist()==legal.sample_id.tolist()
        assert p.source_role.tolist()==legal.source_role.tolist()
        assert np.isfinite(p.filter(regex='^V[12]_q').to_numpy()).all()
        raw[('Fully_trained',seed)]=p.set_index('sample_id')
        prediction_rows.append(dict(seed=seed,rows=len(p),train=int(p.source_role.eq('train').sum()),validation=int(p.source_role.eq('validation').sum())))
    response=csv('formal_response_pairs.csv'); formal_pairs=csv('formal_pairs.csv')
    expected=formal_pairs[formal_pairs.scoring_allowed]
    wanted={(r.row_seed,r.pair_id,t) for r in expected.itertuples() for t in ['V1','V2']}
    got=set(zip(response.row_seed,response.pair_id,response.target)); assert got==wanted and len(got)==len(response)
    assert response.scoring_allowed.all() and response.label_authorized.all() and not response.global_quarantine.any()
    formal_checked=verify_pairs(response,'row_seed','exposure')
    maximum_prediction_delta_error=0.
    for (seed,target),g in response.groupby(['row_seed','target']):
        p=raw[('Fully_trained',seed)]; col=target+'_q50'
        dp=p.loc[g.high_id,col].to_numpy()-p.loc[g.low_id,col].to_numpy()
        maximum_prediction_delta_error=max(maximum_prediction_delta_error,float(np.max(np.abs(dp-g.delta_pred))))
    assert maximum_prediction_delta_error<2e-12
    # Legal delta_true is not rerevealed: check consistency between already-safe tables.
    consistency=response.groupby(['axis','low_id','high_id','target']).delta_true.agg(['min','max'])
    assert (consistency['max']-consistency['min']).max()<2e-10
    primary=summarize(response,['row_seed','exposure','axis','target'])
    local_columns=[c for c in primary.columns if c not in ['row_seed','exposure','axis','target','actual_threshold_ml']]
    compare('formal_response_per_model',primary,csv('formal_response_per_model.csv'),
            ['row_seed','exposure','axis','target','actual_threshold_ml'],local_columns)
    equal=[]; eqkeys=['exposure','axis','target','actual_threshold_ml']
    measures=[c for c in local_columns if c!='predicted_flat_threshold_ml']
    for key,g in primary.groupby(eqkeys):
        r=dict(zip(eqkeys,key),n_models=g.row_seed.nunique())
        for c in measures:
            r[c+'_mean']=g[c].mean();r[c+'_std']=g[c].std(ddof=1);r[c+'_models_nonmissing']=g[c].notna().sum()
        r['ratio_of_equal_model_amplitude_means']=g.predicted_abs_mean.mean()/g.observed_abs_mean.mean(); equal.append(r)
    equal=pd.DataFrame(equal)
    compare('formal_response_equal_models',equal,csv('formal_response_equal_models.csv'),eqkeys,[c for c in equal if c not in eqkeys])
    row_metrics=csv('formal_row_metrics.csv'); row_rms=[]
    for (seed,role),g in row_metrics.groupby(['row_seed','source_role']):
        assert set(g.target)=={'V1','V2'}
        assert np.allclose(g.rmse_ml/g.target_scale_ml,g.normalized_rmse,atol=1e-14,rtol=0)
        value=float(np.sqrt(np.mean((g.rmse_ml/g.target_scale_ml)**2)))
        assert np.allclose(g.combined_normalized_rmse,value,atol=1e-14,rtol=0)
        if role=='validation': assert set(g.n_rows)=={416}
        row_rms.append(dict(seed=seed,role=role,recomputed_official_RMS=value))
    metric_checks=csv('formal_validation_metric_checks.csv')
    assert len(metric_checks)==21 and metric_checks.passed.all()
    assert np.all(metric_checks.absolute_difference<=1e-6)
    js('formal_row_metric_R1_fix.json')
    matched=csv('matched_response_pairs.csv'); exp=csv('al_common_pair_exposure.csv'); safe=exp[exp.allow_new_scoring]
    matched_checked=verify_pairs(matched,'formal_seed','exposure',True)
    keys=['formal_seed','al_seed','axis','pair_key','target']
    assert not matched.duplicated(keys+['method']).any()
    assert matched.groupby(keys).method.apply(lambda x:set(x)==METHODS).all()
    assert matched.groupby(keys).delta_true.agg(lambda x:float(x.max()-x.min())).max()<2e-12
    expected_c={(r.formal_row_seed,r.al_seed,r.axis,r.pair_key,t) for r in safe.itertuples() for t in ['V1','V2']}
    assert set(map(tuple,matched[keys].drop_duplicates().to_numpy()))==expected_c
    for seed in AL:
        for budget in [525,1005,1485]:
            p=csv(f'al_predictions_seed_{seed}_L{budget}.csv'); assert p.sample_id.is_unique
            assert not set(p.sample_id)&quarantine and np.isfinite(p.filter(regex='^V[12]_q').to_numpy()).all()
            raw[(f'AL_L{budget}',seed)]=p.set_index('sample_id')
    maximum_matched_delta_error=0.; maximum_matched_wide_delta_error=0.
    for (method,formal_seed,al_seed,target),g in matched.groupby(['method','formal_seed','al_seed','target']):
        p=raw[(method,formal_seed if method=='Fully_trained' else al_seed)]; col=target+'_q50'
        high=p.loc[g.high_id,col].to_numpy(); low=p.loc[g.low_id,col].to_numpy()
        maximum_matched_wide_delta_error=max(maximum_matched_wide_delta_error,float(np.max(np.abs(high-low-g.delta_pred))))
        # AL tables were scored from in-memory float32 predictions before CSV
        # serialization; native-width subtraction exactly reproduces that path.
        dp=high-low if method=='Fully_trained' else high.astype(np.float32)-low.astype(np.float32)
        maximum_matched_delta_error=max(maximum_matched_delta_error,float(np.max(np.abs(dp-g.delta_pred))))
    assert maximum_matched_delta_error<2e-12
    truth_map=response.set_index(['row_seed','axis','low_id','high_id','target']).delta_true
    for (seed,axis,target),g in matched[matched.method=='Fully_trained'].groupby(['formal_seed','axis','target']):
        values=[truth_map.loc[(seed,axis,a,b,target)] for a,b in zip(g.low_id,g.high_id)]
        assert np.allclose(values,g.delta_true,atol=2e-12,rtol=0)
    for row in matched[['low_id','high_id','global_all_three_non_test']].drop_duplicates().itertuples():
        expected_global=all(roles[s][row.low_id]!='test' and roles[s][row.high_id]!='test' for s in SEEDS)
        assert row.global_all_three_non_test==expected_global
    ckeys=['formal_seed','al_seed','axis','exposure','method','target']
    msummary=summarize(matched,ckeys)
    ccols=[c for c in msummary if c not in ckeys+['actual_threshold_ml','amplitude_ratio','mae_reduction_vs_zero_response']]
    compare('matched_seed_metrics',msummary,csv('matched_seed_metrics.csv'),ckeys+['actual_threshold_ml'],ccols)
    global_summary=summarize(matched[matched.global_all_three_non_test],ckeys)
    compare('matched_global_nontest_metrics',global_summary,csv('matched_global_nontest_metrics.csv'),ckeys+['actual_threshold_ml'],ccols)
    pool=js('CURVE_POOL.json'); curves=csv('controlled_volume_curves.csv'); plan=js('ANALYSIS_PLAN.json')
    assert pool['input_metadata_sha256']==sha256(read('source_metadata.csv')).hexdigest()
    assert pool['frozen_unix']>=plan['frozen_unix']
    assert pool['requested_grid_ul']==plan['curve_grid_ul']==[30,60,90,120,150]
    def series(row): return digest((row['canonical_smiles'],)+tuple(str(row[k]) for k in FIELDS if k!='V/ul'))
    domain=defaultdict(list)
    for _,r in metadata.iterrows(): domain[series(r)].append(r)
    expected_cases=[]
    eligible=exp[(exp.axis=='V/ul')&exp.allow_new_scoring&exp.global_all_three_non_test]
    for seed,g in eligible.groupby('al_seed'):
        candidates=defaultdict(set)
        for r in g.drop_duplicates('pair_key').itertuples(): candidates[series(lookup.loc[r.low_id])].update([r.low_id,r.high_id])
        chosen=sorted(candidates,key=lambda key:digest(('full_training_volume_curves_20261117',int(seed),key)))[:3]
        expected_cases.extend((int(seed),key,sorted(candidates[key])) for key in chosen)
    assert len(expected_cases)==len(pool['cases'])==15
    for case,(seed,key,ids) in zip(pool['cases'],expected_cases):
        assert case['al_seed']==seed and case['series_key']==key and case['observed_ids']==ids
        assert case['base_id']==ids[0] and case['case_id']==digest((seed,key))[:12]
        assert not set(ids)&quarantine
        levels=[float(r['V/ul']) for r in domain[key]]
        assert case['local_min_ul']==min(levels) and case['local_max_ul']==max(levels)
    cases={c['case_id']:c for c in pool['cases']}
    assert len(curves)==900 and not curves.duplicated(['case_id','method','model_seed','target','volume_ul']).any()
    assert np.isfinite(curves.prediction_ml).all() and curves.volume_ul.between(30,150).all()
    assert not any(c in curves for c in ['delta_true','truth','error','difference_mae'])
    curve_raw_checks=[]
    for (case_id,target),g in curves.groupby(['case_id','target']):
        case=cases[case_id]; assert set(g.method)==METHODS and set(g.volume_ul)=={30,60,90,120,150}
        assert g.al_seed.eq(case['al_seed']).all() and g.molecule.eq(case['molecule']).all()
        expected_support=g.volume_ul.between(case['local_min_ul'],case['local_max_ul'])
        assert np.array_equal(g.locally_supported,expected_support)
        assert g.local_min_ul.eq(case['local_min_ul']).all() and g.local_max_ul.eq(case['local_max_ul']).all()
        for (method,seed),unit in g.groupby(['method','model_seed']):
            assert set(unit.volume_ul)=={30,60,90,120,150}
            assert seed in SEEDS if method=='Fully_trained' else seed==case['al_seed']
            p=raw[(method,seed)]
            for sid in case['observed_ids']:
                volume=float(lookup.loc[sid,'V/ul'])
                if volume not in set(unit.volume_ul) or sid not in p.index: continue
                v=float(unit.loc[unit.volume_ul.eq(volume),'prediction_ml'].iloc[0])
                curve_raw_checks.append(abs(v-float(p.loc[sid,target+'_q50'])))
    assert curve_raw_checks and max(curve_raw_checks)<5e-4
    assert float(metadata['V/ul'].min())==30 and float(metadata['V/ul'].max())==150
    overlay=csv('curve_authorized_observations.csv'); assert not set(overlay.sample_id)&quarantine
    for r in overlay.itertuples():
        case=cases[r.case_id]; assert r.sample_id in case['observed_ids'] and roles[42][r.sample_id]!='test'
        assert float(lookup.loc[r.sample_id,'V/ul'])==r.volume_ul
    read('matched_and_curves.py'); read('common.py'); read('analyze_formal.py')
    manifests={name:js(name) for name in ['FORMAL_PREDICTION_MANIFEST.json','MATCHED_PREDICTION_MANIFEST.json','formal_response_inputs.json']}
    for item in list(SOURCES.values()): assert sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    result=dict(status='PASS_WITH_DISCLOSED_LABEL_BOUNDARY_EXCEPTION',new_fits=0,new_inference_calls=0,API_calls=0,
        raw_target_files_read=[],legacy_formal_prediction_caches_read=[],label_interfaces_called=0,
        truth_scope='Only already-authorized derived delta_true in this study; no independent reopening of absolute targets.',
        boundary_exception='Initial text preview included source-test role values; three IDs globally quarantined. This QA does not claim the whole study had zero test-value visibility.',
        quarantine_ids=sorted(quarantine),formal_prediction_rows=prediction_rows,
        formal_scored_pair_endpoint_events=len(response),formal_unique_model_axis_pairs=formal_checked,
        matched_endpoint_method_events=len(matched),matched_unique_formal_axis_pairs=matched_checked,
        exported_roles_match_three_hash_locked_original_splits=True,
        four_method_identical_pair_sets=True,source_test_scored_events=0,quarantined_scored_events=0,
        formal_prediction_delta_max_error=maximum_prediction_delta_error,matched_prediction_delta_max_error=maximum_matched_delta_error,
        matched_CSV_float64_subtraction_max_error_ml=maximum_matched_wide_delta_error,
        matched_precision='Formal CSV float64 subtraction; AL native float32 subtraction. Casting saved AL predictions back to float32 reproduces every released AL delta exactly.',
        checks=CHECKS,row_RMS_checks=row_rms,official_validation_aggregate_checks=21,
        curve_pool_metadata_hash_reconstruction=True,curve_cases=15,curve_rows=900,
        curve_locally_supported_rows=int(curves.locally_supported.sum()),curve_local_extrapolation_rows=int((~curves.locally_supported).sum()),
        curve_overlap_prediction_checks=len(curve_raw_checks),curve_overlap_prediction_max_error_ml=max(curve_raw_checks),
        curve_scope='Local support is the whole source metadata min/max, including unlabelled covariates; interpolation inside bounds is not proof of experimental support at every synthetic level. Curves measure sensitivity, never true error.',
        remaining_limits=['No model forward was independently repeated here.', 'Row RMS is independently recomputed from released endpoint RMSE; absolute-label Row errors are not re-scored.', 'Pool order independently rebuilt using metadata hashes; temporal claim also relies on frozen plan, pool timestamp, and source code order, not a signed external timestamp.'],
        sources=sorted(SOURCES.values(),key=lambda r:r['path']),script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'RESULTS_QA.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# 充分训练体积响应审计：独立结果QA','',
        '**结论：通过，保留已披露的标签边界事故例外。** 仅使用本轮已有合法派生差分和无truth原始预测，未读绝对target、旧正式predictions.gz或调用label接口；0新训练、0推理、0API。不能把本轮全过程描述为从未看到test数值。初期文本预览的3ID在所有seed隔离，本QA逐ID核对无例外。','',
        f'正式评分覆盖{len(response)}个模型×pair×端点评分事件，同pair比较覆盖{len(matched)}个formal×AL×pair×endpoint×method事件。每个比较单元的Fully trained、AL L525/L1005/L1485使用完全相同ID与同一合法delta_true；四方法没有通过缺失pair改变分母。三个正式模型的原训练身份仍3330行，本次预测train为3327/3328/3328、validation各416；全部source-test和3个quarantine在评分中为0。','',
        '## 独立数值重算','',
        '不导入原common.metrics、summary或评分函数，直接从delta_true/delta_pred重算绝对真实/预测响应、差分MAE、幅度误差、方向/非平坦正确/非平坦反向/平坦率以及幅度比和零响应基线误差改善。真实有意义变化严格>0.1/0.5/1mL，预测平坦严格<0.1mL；主MAE使用全部pair。正式per-model、三个模型等权均值及sample std、matched逐formal×AL及全三formal非test子集汇总全部一致，容差2e−10。模型平均的幅度比与幅度均值之比均独立检查，未把两口径混用。','',
        f'正式raw q50预测重算pair差分最大误差{maximum_prediction_delta_error:.3g}mL；matched raw q50最大误差{maximum_matched_delta_error:.3g}mL。AL原实现对内存float32预测相减，重读CSV后转回float32可精确复现；若全用float64相减最大偏差为{maximum_matched_wide_delta_error:.6g}mL，属于舍入边界。正式模型按CSV float64相减。已有合法delta_true在正式表、matched表及交叉模型相同pair间一致。pair身份、只变一因素、PE/EA按EA比例上升与V按µL上升方向均从metadata独立核对。','',
        '整体Row NRMSE修复已核：从每端点RMSE/冻结尺度独立重算`sqrt(mean(normalized_RMSE²))`，不是算术均值；6个seed×role汇总均一致。完整416行validation的21项历史aggregate对照均通过1e−6。这里没有重新读取绝对标签，因此不把该部分称为独立逐行重算Row误差。','',
        '## 曲线pool与支持域','',
        f'15个预固定case按metadata与合法共同pair的hash排名完整重建，每AL seed取3个，ID/base/系列/全域局部体积min–max均一致；900曲线行严格覆盖三个正式模型及对应三预算AL的两个端点、五个水平。局部支持域内{int(curves.locally_supported.sum())}行，局部外推{int((~curves.locally_supported).sum())}行，标记逐点正确；30–150µL均在全源X域范围内，但这不表示每个分子系列都支持该区间。','',
        f'曲线和原始合法预测在相同真实输入水平的{len(curve_raw_checks)}个重合标量独立交叉检查，最大差{max(curve_raw_checks):.6g}mL，低于5e−4浮点图重建容差。曲线表不含真实误差列，真实overlay只来自预固定observed IDs且无test/quarantine。支持域min–max含全源无标签X，区间内新水平是插值敏感性，不等于每一级都有真实测量；外推点不能用于真实恢复结论。','',
        'pool的metadata排序不依赖预测或真实值；冻结plan早于pool，源码在曲线预测前写pool。此QA可验证排序与文件证据，不能把本地时间戳当作外部签名的预注册证明。若按全30–150µL span选恢复/平坦/seed差异示例，必须说明可能包含局部外推，只能作模型灵敏度描述。','',
        '## 复现与来源','',
        '运行`qa_results.py`即可从冻结派生表重建本MD/JSON，不进行模型加载或forward。所有实际读取的派生表、预测、manifest及相关源码SHA在RESULTS_QA.json中，结束前再次验证稳定。局部pair并非独立科研重复，不给伪独立显著性结果；三个正式模型与五个AL seed的交叉比较不称配对随机种子实验。']
    (OUT/'RESULTS_QA.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],comparisons=len(CHECKS),formal_events=len(response),matched_events=len(matched),
        curve_overlap_checks=len(curve_raw_checks),curve_overlap_max_error=max(curve_raw_checks),input_files=len(SOURCES))))

if __name__=='__main__': main()
