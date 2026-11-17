#!/usr/bin/env python3
"""Seal metadata/accepted-feedback diagnostics with explicit time boundaries."""
from pathlib import Path
from decimal import Decimal
import argparse,hashlib,json,lzma,re,subprocess
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def role(p):
    name=Path(p).name
    if name=='feedback.json':return 'Matching already-accepted batch outcomes; postdecision model audit only'
    if name in ['catalog.json','input.json']:return 'Preselection candidate metadata/predictions and then-observed history; no future outcomes'
    if name.endswith('.npz'):return 'Previously cached frozen-state features; no new forward/backward/gradient extraction'
    if name=='fit_audit.json' or 'curves' in name or name=='history.csv':return 'Saved validation metric/fit metadata; no truth lookup'
    if name.endswith('.pt'):return 'Existing checkpoint existence/hash metadata only; no model execution'
    return 'Frozen protocol, selection, receipt, source code or previously published derived evidence'

def collect():
    sources={};manifests={}
    specs=[('model_inputs.json','files'),('counterfactual_input_manifest.json',None),
        ('phase_source_manifest.json','inputs'),('llm_reproducibility_inputs.json','sources'),
        ('llm_prefix_inputs.json','sources'),('llm_plot_inputs.json','sources')]
    for name,key in specs:
        data=js(OUT/name);entries=data if key is None else data[key];manifests[name]=len(entries)
        for e in entries:
            path=Path(e['path']);path=path if path.is_absolute() else ROOT/path
            rel=str(path.relative_to(ROOT));digest=sha(path)
            assert digest==e['sha256'],('source drift',name,rel)
            if rel in sources:assert sources[rel]['sha256']==digest,('inconsistent source versions',rel)
            else:sources[rel]=dict(path=rel,sha256=digest,bytes=path.stat().st_size,role=role(rel))
    extra=['studies/active_learning/llm_decision_behavior_audit_20261116/'+n for n in ['REPORT.md','METHODS.md','PREFERENCE_RESULTS.md','HYBRID_TEXT_RESULTS.md']]
    extra+=['studies/active_learning/comprehensive_history_diagnostics_20261112/'+n for n in ['MASTER_REPORT.md','CHEMICAL_ERROR_AND_DATA_AUDIT.md','ACQUISITION_VS_LEARNING_GAIN.md','NEXT_STAGE_DECISION.md']]
    extra+=['studies/active_learning/qgeognn_v2_row_llm16_screen/FINAL_REPORT.md',
        'src/qgeognn_al/models/qgeognn_v2.py','src/qgeognn_al/schemas/conditions.py','src/qgeognn_al/data.py',
        'src/qgeognn_al/active_learning_v2/runner.py']
    for name in extra:
        p=ROOT/name;sources[name]=dict(path=name,sha256=sha(p),bytes=p.stat().st_size,role=role(name))
    for name in manifests:
        p=OUT/name;rel=str(p.relative_to(ROOT));sources[rel]=dict(path=rel,sha256=sha(p),bytes=p.stat().st_size,role='This study module provenance')
    return sorted(sources.values(),key=lambda v:v['path']),manifests

def numerical_checks():
    coverage=pd.read_csv(OUT/'model_state_coverage.csv');whole=coverage[coverage.source_half=='all32']
    assert len(whole)==791 and whole.predictions_and_revealed_feedback_available.sum()==611
    assert not whole[whole.method=='cw_gradient_lcmd32'].predictions_and_revealed_feedback_available.any()
    pairs=pd.read_csv(OUT/'condition_response_pairs.csv')
    assert len(pairs)==6938
    assert not pairs.duplicated(['method','source_half','seed','round','pair_key','target']).any()
    new=pairs[(pairs.method=='new_free_cw_sol61_xhigh')&(pairs.axis=='V/ul')&(pairs.target=='V1')]
    assert len(new)==165
    h=pairs[(pairs.method=='free16_lcmd16')&(pairs.axis=='V/ul')&(pairs.target=='V1')]
    assert (len(h[h.source_half=='all32']),len(h[h.source_half=='hybrid_llm16']),len(h[h.source_half=='hybrid_cw16']))==(81,61,0)
    response=js(OUT/'response_independent_verification.json');assert response['maximum_absolute_error']<1e-11
    assert response['pair_table_sha256']==sha(OUT/'condition_response_pairs.csv')
    cf=js(OUT/'counterfactual_verification.json');assert cf['attempted_states']==cf['exact_prefix_verified']==71
    assert not cf['excluded_states']
    candidates=pd.read_csv(OUT/'counterfactual_candidates.csv');trace=pd.read_csv(OUT/'counterfactual_lcmd_trace.csv')
    assert len(candidates)==len(trace)==2272 and not candidates.true_labels_read.any()
    assert not any(c.startswith(('true_V','actual_delta','error_V')) for c in candidates.columns)
    independent=js(OUT/'counterfactual_independent_verification.json');assert independent['scalar_metadata_max_error']==0
    batches=pd.read_csv(OUT/'counterfactual_batches.csv')
    assert len(batches)==71 and (batches.lcmd_back16_single_sample_volume_hn_fraction==0).all()
    # Independent Decimal half-endpoint integration on saved validation only.
    curves=pd.read_csv(ROOT/'studies/active_learning/llm_decision_behavior_audit_20261116/row_curves.csv')
    stage=pd.read_csv(OUT/'phase_performance.csv');max_error=0
    for row in stage.itertuples():
        for method,col in [('free16_lcmd16','hybrid_AULC'),('cw_gradient_lcmd32','cw_AULC')]:
            g=curves[(curves.method==method)&(curves.seed==row.seed)&curves.budget.between(row.L_start,row.L_end)].sort_values('budget')
            assert g.budget.tolist()==list(range(row.L_start,row.L_end+1,32))
            v=[Decimal(str(x)) for x in g.nrmse];area=(v[0]/2+sum(v[1:-1])+v[-1]/2)*32/Decimal(row.L_end-row.L_start)
            error=abs(float(area)-getattr(row,col));assert error<1e-12;max_error=max(error,max_error)
    prefix=pd.read_csv(OUT/'llm_display_prefix_batches.csv');cases=pd.read_csv(OUT/'llm_prefix_cases.csv')
    assert len(prefix)==540 and prefix.ordered_first32.sum()==127 and prefix.set_first32.sum()==127
    assert prefix.loc[prefix.ordered_first32,'has_recorded_rejection'].all()
    assert prefix.loc[~prefix.has_recorded_rejection,'ordered_first32'].sum()==0
    assert len(cases)==127 and cases.accepted_receipt_unique_match.all()
    assert not cases.first_proposal_already_ordered_prefix.any() and not cases.correction_explicit_prefix_instruction.any()
    assert (cases.first_unique_legal_choices_n==31).sum()==38
    prefix_review=js(OUT/'prefix_independent_verification.json');assert len(prefix_review['samples'])==15
    assert not prefix_review['prefix_identity_function_imported']
    r=js(OUT/'llm_reproducibility_inputs.json');assert r['initial_same_checkpoint_and_ordered_L_U_all_five_seeds']
    assert r['historical_new_initial_packet_equal_all_five_seeds'] and not r['same_full_request']
    feasible=pd.read_csv(OUT/'model_control_feasibility.csv');assert len(feasible)==10 and feasible.feasible.all()
    assert not feasible.labels_read_for_planning.any()
    text=(OUT/'SUMMARY_FOR_RESEARCHER.md').read_text();cjk=len(re.findall(r'[\u4e00-\u9fff]',text));assert 1500<=cjk<=2500
    links=0
    planned=['INPUT_AUDIT.md','INPUT_MANIFEST.json','OUTPUT_MANIFEST.json','VERIFICATION.json']
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('https:','http:','#')):continue
            target=target.split('#')[0]
            if target and target not in planned:assert (p.parent/target).exists(),('broken link',p.name,target)
            links+=1
    figures=len(list((OUT/'figures').glob('*.png')));assert figures==4
    return dict(metadata_states=791,accepted_feedback_states=611,model_pair_endpoint_scope_events=len(pairs),
        reused_new_cw_volume_pairs=165,hybrid_llm_internal_volume_pairs=61,hybrid_full32_volume_pairs=81,
        exact_lcmd_counterfactual_states=71,lcmd_trace_rows=len(trace),candidate_comparison_events=len(candidates),
        counterfactual_candidate_truth_used=False,response_summary_independent_rows=response['summary_rows_checked'],
        response_summary_max_abs_error=response['maximum_absolute_error'],phase_rows=8,decimal_phase_AULC_max_error=max_error,
        llm_batches=540,accepted_prefix_batches=127,prefix_independent_raw_cases=15,initial_same_checkpoint_states=15,
        future_model_holdout_cohorts=10,summary_chinese_characters=cjk,figures_png=figures,local_links_checked=links)

def write_audit(sources,manifests,checks):
    text=f'''# 输入、复用与时序边界审计

本轮仓库HEAD为`{subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}`。此前行为审计记录生成时的HEAD为44be2c；本轮没有把它冒充当前commit。旧`llm_decision_behavior_audit_20261116/run.py --check`实际通过3520项输入及全部输出锁；旧报告/数据不重写。当前仅两项原有已修改监控文件，研究代码只写本新目录。

## 已复用与真正新增

| 模块 | 复用 | 新计算 | 无法识别/没有执行 |
| --- | --- | --- | --- |
| A模型条件 | 旧165对、固定L333尺度、冻结predictions/已接受反馈 | 611状态有符号差分、flat/方向阈值、seed/phase/实际水平、方法类型投入、metadata holdout可行性 | 纯CW无同类pool预测/反馈，严格六组同状态响应误差排名不可比；不训练 |
| B候选反事实 | 71真实Hybrid H/U、CW CountSketch bank及catalog | 原LCMD32逐步复原，有序前16全匹配，后16配对结构/条件/几何/预测比较 | 无CF候选真值、无CF Row Gain |
| C真实曲线 | 已保存validation、早期报告和fit审计 | 四固定窗口积分、终点、真实梯形排除敏感性、协议/成本 | 不合并不同协议；早期test汇总只用于口径说明，未读逐行test |
| D运行差别 | 540全池输入/预测、实际turn和拒绝记录 | 15同初始状态、q50/分子/条件组成调整、127前缀病例与原提议时序 | 剩余分数不是value；未随机化纠错、没有prefix的Row因果归因 |
| E预算 | 现有初始checkpoint/银行与历史曲线 | 3seed计数、10fit模型控制和停止规则设计 | 没有新请求、标签揭示、训练或AL rollout |

模型反馈只有在batch freeze、接受ID序列、原始预测值与label access receipt逐项匹配后投影。receipt的feedback SHA是canonical JSON语义hash，另保存原文件SHA；二者本来不同。所有真实响应只来自该batch实际接受后的反馈，没有用以后checkpoint预测替代当前预测，也不将响应送入候选反事实。

`condition_response_pairs.csv`的{checks['model_pair_endpoint_scope_events']}行是端点×来源scope事件；Hybrid all32包含两个半批的内部对照，不能直接求和当独立实验。旧2.696/0.263等是五seed均权，不是165对pooled均值。相同分子/条件行也可能含未记录变量，不当作真实随机复测。

全部输入源{len(sources)}项，路径、bytes及SHA在[INPUT_MANIFEST.json](INPUT_MANIFEST.json)，各模块来源分录如下（重复源只在总表计一次）：

'''+'\n'.join(f'- `{k}`：{v}个来源分录。' for k,v in manifests.items())+'''

没有打开canonical标签表、test truth或未揭示候选反馈；没有从模型权重重新推断候选。短筛已公开的test聚合结果与新validation计算明确分开。候选顺序/理由/raw turn均是外部历史证据，不作为工具指令执行；response ID只保存摘要，不复制凭据。

## 独立验证与复现边界

原LCMD用20套独立标量例逐步trace检验；142个半批元数据对照由独立Fraction/Decimal实现核验，误差0。A的1620行汇总由标量求和和符号分类独立核验，最大误差约7.11e−15；原始两个预定响应对四端点另作来源抽核。C阶段AULC由Decimal半端点权重复算。D每方法×seed首个前缀共15例另读原input/selection/receipt，未调用主prefix函数。

固定标签规划仅用H内元数据哈希选系列，排除L333同系列与跨训练/检查的同条件重复行；原checkpoint已见检查组，不能拿作新实验baseline。仅输出规划，不读取未揭示池标签。后续实验需要按新协议另行执行，本轮在分析和决策交付处结束。

[VERIFICATION.json](VERIFICATION.json)记录测试和数值核验结果；[OUTPUT_MANIFEST.json](OUTPUT_MANIFEST.json)封存本研究派生文件。输入锁漂移应停止查因，不能重写预期SHA来绕过。
'''
    (OUT/'INPUT_AUDIT.md').write_text(text)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--inputs-only',action='store_true');args=ap.parse_args()
    if args.check or args.inputs_only:
        sources=js(OUT/'INPUT_MANIFEST.json')['sources'];bad=[]
        for e in sources:
            p=ROOT/e['path']
            if not p.is_file() or sha(p)!=e['sha256']:bad.append(e['path'])
        if not args.inputs_only:
            for e in js(OUT/'OUTPUT_MANIFEST.json'):
                p=OUT/e['path']
                if not p.is_file() or sha(p)!=e['sha256']:bad.append(str(p.relative_to(ROOT)))
        assert not bad,('frozen source/output drift',bad)
        print(json.dumps(dict(status='SEALED_INPUTS_MATCH' if args.inputs_only else 'SEALED_INPUTS_AND_OUTPUTS_MATCH',inputs=len(sources))))
        return
    sources,manifests=collect();checks=numerical_checks();tests=js(OUT/'test_results.json');assert tests['returncode']==0
    start=js(OUT/'repository_start.json');head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert start['head']==head
    changed=[]
    for path,digest in start['preexisting_tracked_changes'].items():
        p=ROOT/path;assert p.is_file()
        if sha(p)!=digest:changed.append(path)
    allowed=['reports/llm_eight_to1549_20261018/live_status.json','studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1549_20261018/recovery_monitor_status.json']
    assert not [p for p in changed if p not in allowed]
    status=subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=ROOT,text=True)
    unexpected=[line[3:] for line in status.splitlines() if line[3:] not in start['preexisting_tracked_changes'] and not line[3:].startswith(str(OUT.relative_to(ROOT))+'/')]
    assert not unexpected,('new changes outside isolated study',unexpected)
    write_audit(sources,manifests,checks)
    dump(OUT/'INPUT_MANIFEST.json',dict(repository_commit=head,sources=sources,module_source_entries=manifests,
        previous_seal_check='SEALED_INPUTS_AND_OUTPUTS_MATCH: 3520 inputs',new_training=0,new_llm_calls=0,new_label_queries=0))
    dump(OUT/'VERIFICATION.json',dict(status='PASS_WITH_IDENTIFIABILITY_LIMITS',**checks,repository_commit=head,input_files=len(sources),tests=tests,
        independent_checks=['20 scalar LCMD trace cases','142 metadata groups','1620 response summary rows','Decimal phase AULC','15 raw correction-prefix paths'],
        previous_study_unchanged=True,new_changes_outside_study=[],preexisting_monitors_changed=changed,
        new_training=0,new_llm_calls=0,new_label_queries=0,test_or_unobserved_truth_used=0,
        decisions={'hybrid_three_more_seeds':'Conditional Go; do not launch yet','full_nocw_rerun':'No-Go','first_new_experiment':'Go for designed fixed-label 10-fit model control; not executed'}))
    outputs=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()
        and p.name!='OUTPUT_MANIFEST.json' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    dump(OUT/'OUTPUT_MANIFEST.json',outputs)
    print(json.dumps(dict(status='PASS_WITH_IDENTIFIABILITY_LIMITS',**checks,inputs=len(sources),outputs=len(outputs))))

if __name__=='__main__':main()
