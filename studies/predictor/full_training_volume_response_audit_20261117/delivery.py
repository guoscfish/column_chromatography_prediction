#!/usr/bin/env python3
"""Write the final evidence-bound conclusions and seal this read-only audit."""
from pathlib import Path
import argparse,ast,hashlib,io,json,subprocess,sys,tokenize
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text())

def md(frame):
    def fmt(v):return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(map(fmt,row))+' |' for row in frame.itertuples(index=False,name=None))

def documents():
    f=pd.read_csv(OUT/'formal_response_per_model.csv')
    v=f[(f.axis=='V/ul')&(f.actual_threshold_ml==.5)&f.exposure.isin(['train_train','validation_validation'])]
    table=v[['row_seed','exposure','target','n_pairs','observed_abs_mean','predicted_abs_mean','difference_mae','amplitude_ratio','flat_fraction']]
    matched=pd.read_csv(OUT/'matched_plot_equal_models.csv');mt=matched[matched.exposure=='both_source_train']
    change=[]
    for target,g in mt.groupby('target'):
        g=g.set_index('method');full=g.loc['Fully_trained']
        for budget in [525,1005,1485]:
            al=g.loc[f'AL_L{budget}'];change.append(dict(target=target,AL_budget=budget,AL_difference_MAE=al.difference_mae,fully_difference_MAE=full.difference_mae,
                improvement_ml=al.difference_mae-full.difference_mae,relative_improvement=(al.difference_mae-full.difference_mae)/al.difference_mae))
    changes=pd.DataFrame(change);changes.to_csv(OUT/'final_matched_change_numbers.csv',index=False)
    rows=pd.read_csv(OUT/'formal_row_metrics.csv').query("source_role=='validation'").groupby('row_seed').combined_normalized_rmse.first()
    final=f'''# 最终科研判断

## 1. 充分训练的QGeoGNN能否学会上样体积变化？

**能表现出部分响应，但当前三个正式模型没有稳定、准确地学好。** 三者各真实训练3330行；训练内体积对数为2771/2767/2730，validation为48/44/48。训练内V1预测/真实响应幅度比依次5.58%/51.58%/0.71%，V2为5.13%/27.31%/0.79%；validation的V1为7.04%/37.85%/0.87%，V2为5.94%/35.26%/0.94%。Row525部分恢复，Row42仍弱，Row1101几乎平坦；不能只报最好的seed。Row seed525是初始化/划分种子，与AL预算L525不同。

下面先逐模型列原集合结果，单位mL；n_pairs不是独立科研重复。方向等完整指标见TRAIN_FIT_RESPONSE与VALIDATION_RESPONSE，均采用|Δtrue|>0.5mL、|Δpred|<0.1mL的历史定义。

{md(table)}

整体原validation NRMSE分别为{rows.loc[42]:.6f}/{rows.loc[525]:.6f}/{rows.loc[1101]:.6f}，与21项历史aggregate核验一致（最大误差1.86×10⁻⁷）。这说明整体Row误差可较好，但不保证体积差分准确。原validation参与历史checkpoint选择，因此不是新独立test；条件对数量和相关性也限制泛化论断。

## 2. 相比L525/L1005，充分训练后改善多少？

主比较严格使用相同ID pair，并将正式模型两端均source-train单独列出：三模型先各对AL contexts等权，再正式模型等权。共同体积真实幅度V1/V2为3.2854/3.2665mL；L525预测0.0049/0.0066，L1005为0.2145/0.1027，Fully为0.2664/0.1906mL。已有L1485扩展为0.4338/0.2261，并不低于Fully平均值。

{md(changes)}

Fully相对L1005的V1差分MAE仅改善0.0220mL，V2反而恶化约0.0029mL；不是普遍修复。比较已区分source-train两端、train–validation混合和validation两端；后者仅6个共同体积pair记录，不足以据此确认同pair泛化。完整source-validation实验B则有44–48对。三个正式seed与五个AL seed是交叉模型比较，不能视为配对随机seed；训练量、组成、初始化、标准化和best epoch同时不同，因此不能把所有变化归因于多买标签。PE/EA参照单列，见MATCHED_AL_COMPARISON.md。

## 3. 是否存在见过大量对照仍未拟合好的现象？

**存在，但它不是已经证明的网络容量缺陷。** 每个模型训练内有约2700个严格体积pair、194–195个分子、1092–1122个系列，预测幅度仍明显偏小；V1差分MAE相对零响应基线只改善0.40%–10.92%，V2仅0.21%–1.81%。同模型PE/EA训练幅度比V1约91%–99%、V2约87%–108%，其差分MAE改善约66%–83%，支持因素选择性现象。

同时，完全相同已记录分子与五条件的训练重复行，V1绝对差均值约1.25–1.44mL、V2约2.37–2.62mL，已接近甚至大于体积pair平均真实变化；个别差异可达21.55/30.83mL。不能据此直接算不可约噪声，因为可能包含未记录条件、批次或记录问题，但也不能把每个观测差都当成干净的物理体积效应，要求模型逐一重现。系列/分子等权结果均保留，未把相关pair当显著性样本。

## 4. 下一步优先选点还是模型本身？

**先做数据一致性和模型训练机制诊断，再决定选点改动。** 首先追查同X大差异的原始记录与隐藏条件；其次审阅已有模型条件分支、loss对局部信号的权衡和训练动力学。Row42/525/1101的best epoch为95/318/67，这与响应差异同时变化，尚不能据此断言多训练若干epoch就能解决。当前结果削弱了“仅因AL标签少”的单一解释，但不能在无新受控实验时区分表示、优化、数据覆盖与噪声。

这些优先项服务于Row-level标签效率：先确认新增体积对照有稳定可学习信号、能降低Row误差，再设计有限的单步选点增益对照。当前不换网络、不加差分loss、不扩模型搜索，也不自动重跑AL。训练内与validation都没有三seed一致恢复，且存在seed异质性，应按用户决策逻辑的情况3/4继续诊断，不能仅用全局R²或某个最好seed宣布问题解决。

## 5. 是否支持让LLM引导结构修改？

**没有新增支持证据。** 本轮只复用冻结模型，没有LLM提案、API、训练或结构干预；先前简单Skip也未通过门槛。现有结果可用于形成可证伪的机制假设，但没有证明LLM相比人、确定性规则或相同预算的有限搜索提供额外知识。优先数值诊断与数据核查，不将LLM架构Agent强行加入结论。

## 标签边界事故与证据范围

本轮没有新训练、LLM/API、新实验标签揭示或AL重跑。不过初期一次`head`文本查看误展示3行目标，其中两行分别属于另一个正式seed的test；另一次sed误开同文件未显示目标值。因此**不满足严格的全过程零test目标可见要求**，VERIFICATION不会写成无条件PASS。三ID随后在全部seed全局隔离，未写入派生目标、未用于数值指标/模型/案例选择；后续对应正式模型test与隔离ID数值投影、评分均为0。原checkpoint和split不变。隔离不影响本轮体积pair集合，PE部分pair另列排除；实际授权train数3327/3328/3328、validation均416。完整过程见LABEL_BOUNDARY_INCIDENT.json。

不同seed对同ID的角色不同，所有权限按对应模型检查，合法train数值不被带入另一模型test评分。源CSV由官方受限store流式经过字符串，只转换授权ID；文件hash经过字节，这与违规文本预览分开记录。受控曲线15个series在新预测前按metadata hash固定，900个标量中648个在局部支持范围、252个为明确标记的局部外推；不存在满足预定恢复/全三seed平坦阈值的案例时如实标缺，不追加挑选。仅把已授权真实水平叠图，不把合成网格当真实误差。所有旧模型、报告、loss、Prompt和AL流程保持不变，没有反馈本轮参数或预测到选点流程。
'''
    (OUT/'FINAL_DECISION.md').write_text(final)
    readme='''# 充分训练QGeoGNN的样品体积响应审计

三个正式Row模型各训练3330行后，体积响应仍未稳定恢复：Row525部分响应，Row42明显压缩，Row1101几乎平坦；原validation亦然。与AL相同pair比较没有显示充分训练普遍修复。先读 [FINAL_DECISION.md](FINAL_DECISION.md)，其中开头直接回答五个科研问题；三张主图位于`figures/`。

本轮0新模型训练、0LLM/API、0新实验标签揭示、0AL重跑。存在一次初始目标文本预览事故，涉及3ID随后全局隔离，另有一次未输出数值的文件误开；不能称严格全过程0test目标可见。完整例外和补救在`LABEL_BOUNDARY_INCIDENT.json`，数值评分只使用对应正式模型授权train/validation并剔除隔离ID，VERIFICATION明确保留事故状态。

| 文件 | 内容 |
|---|---|
| CHECKPOINT_AUDIT.md | 正式Row42/525/1101路径、SHA、3330训练身份及官方原生V2严格加载 |
| PAIR_COVERAGE.md | 单因素pair、系列/分子、水平/差值分布、重复行与权限门 |
| TRAIN_FIT_RESPONSE.md | 训练内volume/PE响应、逐模型与等权汇总、重复行差异 |
| VALIDATION_RESPONSE.md | 两端validation泛化、混合对单列、原Row指标重现 |
| MATCHED_AL_COMPARISON.md | L525/L1005/L1485与Fully同pair、曝光分层、分子差异 |
| MODEL_RESPONSE_EXAMPLES.md | 15个metadata固定曲线pool、案例规则、插值/外推范围 |
| FINAL_DECISION.md | 五问数值结论、局限与面向Row效率的下一步 |
| RESULTS_QA.md | 独立公式、raw预测、角色与曲线支持范围复算 |

正式model seed525与AL预算L525无关。三个正式checkpoint均本地存在、原生V2，无R2转换；seed42 SHA精确匹配指定fce9…544b0。正式模型分别使用自己的3330训练scaler，AL模型使用各自L333冻结scaler，比较响应统一为mL。完整域4163行从未被误称为3330训练行，AL旧metadata少1个非共同ID也没有补造。

## 复现

仓库根目录，解释器`/Users/fish/miniforge3/envs/chromatography/bin/python`，使用`-B`和`PYTHONDONTWRITEBYTECODE=1`，OMP/BLAS/MKL/VECLIB线程环境均1；无训练入口：

```sh
python -B studies/predictor/full_training_volume_response_audit_20261117/data_boundary.py
python -B studies/predictor/full_training_volume_response_audit_20261117/checkpoint_audit.py
python -B studies/predictor/full_training_volume_response_audit_20261117/al_checkpoint_audit.py
python -B studies/predictor/full_training_volume_response_audit_20261117/predict_formal.py
python -B studies/predictor/full_training_volume_response_audit_20261117/analyze_formal.py
python -B studies/predictor/full_training_volume_response_audit_20261117/matched_and_curves.py
python -B studies/predictor/full_training_volume_response_audit_20261117/matched_summary.py
python -B studies/predictor/full_training_volume_response_audit_20261117/make_figures.py
python -B studies/predictor/full_training_volume_response_audit_20261117/qa_results.py
python -B studies/predictor/full_training_volume_response_audit_20261117/delivery.py --check
```

此处`python`须指上述科学环境。正式与AL checkpoint、图cache及历史原日志为现有本地依赖；缺失时停止，不自动训练替代。全测试命令、实际结果和SHA在`TEST_RESULT.json`，输入/输出SHA在`VERIFICATION.json`。重复生成脚本会重写本目录派生表与时间戳，核验旧封存时只用`--check`，不要重生成后覆盖旧身份。

主要数据：`formal_response_per_model.csv`、`formal_response_per_series.csv`、`formal_response_per_molecule.csv`、`formal_row_metrics.csv`、`matched_seed_metrics.csv`、`matched_paired_improvements.csv`、`controlled_volume_curves.csv`。未请求source-test凑pair，未打开历史含test truth的predictions.csv.gz用于分析。Row NRMSE初期派生公式从算术均值改为官方RMS，完整21指标重现核验与R1修复审计保留；局部差分未受该修正影响。原始split与checkpoint未变。
'''
    (OUT/'README.md').write_text(readme)

def verify(check=False):
    if check:
        d=load(OUT/'VERIFICATION.json')
        for entry in d['inputs']+d['outputs']:assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
        print(json.dumps(dict(status=d['status'],numerical_checks=d['numerical_checks'],inputs=len(d['inputs']),outputs=len(d['outputs']))));return
    names=['README.md','CHECKPOINT_AUDIT.md','PAIR_COVERAGE.md','TRAIN_FIT_RESPONSE.md','VALIDATION_RESPONSE.md','MATCHED_AL_COMPARISON.md','MODEL_RESPONSE_EXAMPLES.md','FINAL_DECISION.md']
    assert all((OUT/n).is_file() for n in names)
    manifests=['checkpoint_audit_manifest.json','data_boundary_inputs.json','al_checkpoint_inputs.json','FORMAL_PREDICTION_MANIFEST.json','formal_response_inputs.json','MATCHED_PREDICTION_MANIFEST.json','matched_summary_inputs.json','FIGURE_MANIFEST.json','RESULTS_QA.json']
    sources={}
    for name in manifests:
        payload=load(OUT/name)
        for field in ['sources','inputs','input_sources']:
            entries=payload.get(field,[])
            if not isinstance(entries,list):continue
            for e in entries:
                if not isinstance(e,dict) or 'path' not in e or 'sha256' not in e:continue
                path=e['path'];digest=e['sha256']
                assert path not in sources or sources[path]==digest,('conflicting input identity',path)
                sources[path]=digest
    for path,digest in sources.items():assert sha(ROOT/path)==digest,('input changed',path)
    incident=load(OUT/'LABEL_BOUNDARY_INCIDENT.json');quarantine=set(incident['quarantined_ids']);assert len(quarantine)==3 and len(incident['affected_test_id_roles'])==2
    assert not incident['target_values_copied_into_artifacts'] and not incident['used_for_metrics_or_models_or_selection']
    roles=pd.read_csv(OUT/'formal_split_roles.csv')
    for name in ['formal_response_pairs.csv','matched_response_pairs.csv']:
        p=pd.read_csv(OUT/name);seedcol='row_seed' if 'row_seed' in p else 'formal_seed'
        assert not (set(p.low_id)|set(p.high_id))&quarantine
        for seed,g in p.groupby(seedcol):
            allowed=set(roles[(roles.row_seed==seed)&roles.source_role.isin(['train','validation'])].sample_id)
            assert (set(g.low_id)|set(g.high_id))<=allowed
    for seed in [42,525,1101]:
        pred=pd.read_csv(OUT/f'formal_predictions_seed_{seed}.csv');assert len(pred)==(3743 if seed==42 else 3744)
        assert not set(pred.sample_id)&quarantine and set(pred.source_role)=={'train','validation'}
    tests=load(OUT/'TEST_RESULT.json');assert tests['status']=='PASS' and tests['exit_code']==0
    assert sha(OUT/'test_stdout.txt')==tests['stdout_sha256']
    current={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'tests').glob('test_*.py')}
    assert current==tests['tests_sha256'],'Test suite changed after its recorded run'
    qa=load(OUT/'RESULTS_QA.json');assert qa['status'].startswith('PASS')
    for p in OUT.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    outputs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='VERIFICATION.json' and '__pycache__' not in p.parts]
    record=dict(status='COMPLETE_WITH_DISCLOSED_LABEL_BOUNDARY_EXCEPTION',numerical_checks='PASS',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        new_model_training_runs=0,new_LLM_API_calls=0,new_experimental_label_reveals=0,full_AL_rollouts=0,
        test_target_text_visibility_requirement_satisfied=False,initial_preview_rows=3,initial_preview_rows_with_test_role=2,
        quarantined_ids=sorted(quarantine),subsequent_own_model_test_numeric_targets_projected_or_scored=0,
        label_scope='Per-model original train/validation through existing official RestrictedLabelStore + explicit frozen-role adapter; no cross-model transfer into test scoring. CSV strings and hashing pass file bytes; initial text-preview incident separately disclosed.',
        original_splits_and_checkpoints_modified=False,formal_original_train_rows=[3330,3330,3330],formal_scoring_train_rows=[3327,3328,3328],formal_validation_rows=[416,416,416],
        tests=tests,independent_results_qa_status=qa['status'],
        inputs=[dict(path=p,sha256=h) for p,h in sorted(sources.items())],outputs=outputs,
        limits='No causal estimate of training size, no claim of universal architecture incapacity, no new independent test or LLM knowledge contribution; disclosed boundary exception prevents unqualified compliance PASS.')
    (OUT/'VERIFICATION.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status=record['status'],numerical_checks='PASS',inputs=len(sources),outputs=len(outputs),new_fits=0,new_API_calls=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if not args.check:documents()
    verify(args.check)
