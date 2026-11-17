#!/usr/bin/env python3
"""Fixed-sample human adjudications and focused Chinese reports; no new data access."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'comprehensive_history_diagnostics_20261112'
REVIEW_INPUTS={}
def load(n,old=False):
 p=(OLD if old else OUT)/'tables'/n
 REVIEW_INPUTS[str(p) ]={'path':str(p.relative_to(OUT.parents[2])),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':'reused previous-study result' if old else 'derived input within current study'}
 return pd.read_csv(p,float_precision='round_trip')
def md(d):
 def val(v):
  if pd.isna(v):return 'NA'
  return f'{v:.5f}' if isinstance(v,float) else str(v).replace('|','/')
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(val(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
def write(n,s):(OUT/n).write_text(s.strip()+'\n')
def main():
 b=load('nocw_batches.csv');h=load('nocw_hypotheses.csv');cases=load('nocw_gain_cases_with_behavior.csv');flags=load('nocw_text_flags.csv');r=load('nocw_repairs.csv');e=load('nocw_evidence.csv');c=load('nocw_science_gain_correlations.csv');p=load('nocw_test_pairs.csv')
 sample=json.loads((OUT/'tables/nocw_fixed_case_excerpts.json').read_text());s=json.loads((OUT/'verification/nocw_summary.json').read_text())
 bp=b[b.primary];hp=h[h.primary];rp=r[r.primary]
 cw=load('reasoning_batches.csv',True);cw_common=cw[cw['round']<21]
 assert len(cw_common)==105 and int(cw_common.first_complete_response_legal.sum())==50 and int(cw_common.complete_response_n.sum())==309
 cw_responses=load('reasoning_responses.csv',True)
 assert int(((cw_responses['round']<21)&(~cw_responses.valid_original_rules)&(cw_responses.first_rejection_category=='packet_binding')).sum())==115
 reliability=pd.DataFrame([
  {'method':'NoCW','window':'L333-1005 matched five seeds','batches':len(bp),'first_complete_legal':int(bp.first_complete_legal.sum()),'first_legal_fraction':float(bp.first_complete_legal.mean()),'complete_responses':int(bp.complete_response_n.sum())},
  {'method':'new CW-Free','window':'L333-1005 matched five seeds','batches':len(cw_common),'first_complete_legal':int(cw_common.first_complete_response_legal.sum()),'first_legal_fraction':float(cw_common.first_complete_response_legal.mean()),'complete_responses':int(cw_common.complete_response_n.sum())},
  {'method':'new CW-Free','window':'L333-1485 full trajectory; contextual only','batches':len(cw),'first_complete_legal':int(cw.first_complete_response_legal.sum()),'first_legal_fraction':float(cw.first_complete_response_legal.mean()),'complete_responses':int(cw.complete_response_n.sum())}])
 reliability.to_csv(OUT/'tables/nocw_reliability_common_window.csv',index=False,float_format='%.17g')
 # All 38 primary identity flags and 2 unknown prose occurrences were read.
 errors={(157,11,'86cdb37fc2caaa178856'),(2357,5,'ca7faa605430c91c04cc'),(2357,12,'48dba5c341ba12a96231'),
  (6101,1,'34aaca6f7a0768d743fe'),(6101,2,'5b7151c95c4e964a121b'),(6101,2,'45ee5106187dc3b895e1'),
  (6101,4,'904fa4a17e429d0616f5'),(6101,5,'45a1b11ae63f2dab2e48'),(12203,3,'17ff70c0f8a083dcf5f0'),
  (12203,10,'aef4c70d74081b04c15e'),(12203,17,'6a39aff909ea35989c74')}
 f=flags[flags.primary].copy();f['manual_assessment']=[('unknown_prose_reference_not_autocorrected' if x.flag=='unknown_prose_id' else 'explicit_selected_molecule_misidentification' if (x.seed,x['round'],x.id) in errors else 'valid_cross_molecule_reference') for _,x in f.iterrows()]
 f.to_csv(OUT/'tables/nocw_flag_adjudications.csv',index=False)
 # First hypothesis and question in all 52 prespecified case/grid batches were read.
 explicit={(157,0),(157,10),(2357,2),(6101,0),(6101,7),(12203,5),(12203,6),(887,0),(887,1)}
 named={(157,3),(157,4),(157,5),(157,11),(157,14),(887,5),(887,6),(887,7),(887,14),(887,18),(887,20),
  (2357,0),(2357,5),(2357,6),(2357,19),(6101,5),(6101,13),(6101,14),(6101,20),(12203,1),(12203,19)}
 prediction_descriptions={(157,9),(12203,14),(12203,19),(12203,20)}
 manual=[]
 for z in sample:
  key=z['seed'],z['round'];hyp=z['first_hypothesis'];pair=z['first_metadata_sorted_test_pair']
  cat='empty_hypothesis_list' if hyp is None else 'explicit_named_model_shape_or_transition_test' if key in explicit else 'concrete_condition_question_under_broad_claim' if key in named else 'broad_family_or_tail_question_without_decision_rule'
  if key in prediction_descriptions:cat='retrospective_model_description_or_acquisition_value_claim'
  note='First claim/question gives scope and an observable model-shape alternative; semantic outcome still requires exact relevant controls.' if key in explicit else 'A local response can be measured, but the broad scaffold-specific/nonuniversal proposition lacks a prespecified rejection threshold.'
  if not hyp:note='Final schema accepts zero hypotheses; prose feedback cannot reconstruct a missing typed scientific claim.'
  if key in prediction_descriptions:note='This reports a model-error pattern or calls anchors necessary; new measurements alone do not establish the claimed learning value.'
  manual.append({'seed':key[0],'round':key[1],'manual_scope':'first hypothesis and question, reasons, one metadata-fixed pair, next cited hypotheses',
   'category':cat,'metadata_test_pair_available':pair is not None,'outcome_of_broad_claim':'not_identifiable_without_prespecified_contrast_rule' if hyp else 'not_applicable',
   'next_cites_linked_experiments':bool(z['next_hypotheses_citing_linked']),'gain':z['gain'],'note':note})
 manual=pd.DataFrame(manual);manual.to_csv(OUT/'tables/nocw_fixed_sample_review.csv',index=False)
 chain_rows=[]
 for seed,round_,hyp_id,mol,axis,decision,next_meaning in [
  (157,0,'H1','m39','PE/EA','specific_frozen_V1_inversion_rejected; broad_family_claim_not_decided','next_round_explicitly_calls_previous_m39_inversion_prediction_error'),
  (6101,7,'H7_C','m95','PE/EA','specific_frozen_V2_plateau_or_reversal_rejected; structural_mechanism_not_decided','next_feedback_recognizes_methyl_acetoacetate_finish_rise_and_uses_new_carbonyl_evidence'),
  (6101,7,'H7_I','m125','Volume of loading solvent/ul','supports_local_recorded_volume_association; causal_or_reproducible_mechanism_not_decided','next_feedback_recognizes_1_to_1_finish_contrast_recurred_at_100mg_but_not_universal')]:
  hp0=h[(h.seed==seed)&(h['round']==round_)&(h.hypothesis_id==hyp_id)].iloc[0]
  pp=p[(p.seed==seed)&(p['round']==round_)&(p.hypothesis_id==hyp_id)&(p.molecule_id==mol)&(p.axis==axis)&(p.pair_kind=='same_batch')].iloc[0]
  chain_rows.append({'seed':seed,'round':round_,'hypothesis_id':hyp_id,'subtest_molecule':mol,'subtest_axis':axis,
   'id_a':pp.id_a,'id_b':pp.id_b,'conditions_a':pp.conditions_a,'conditions_b':pp.conditions_b,
   'actual_delta_V1_ml':pp.actual_delta_V1_ml,'actual_delta_V2_ml':pp.actual_delta_V2_ml,
   'frozen_delta_V1_ml':pp.pred_delta_V1_ml,'frozen_delta_V2_ml':pp.pred_delta_V2_ml,
   'scientific_subtest_verdict':decision,'next_information_use':next_meaning,
   'next_typed_reference_occurrences_to_linked_choices':hp0.next_cites_linked_selected_n,
   'evidence_grade':'E3_traceable_subtest_not_causal_learning_value','gain':float(bp[(bp.seed==seed)&(bp['round']==round_)].gain.iloc[0]),
   'selection_source':hp0.selection_source,'truth_role':'accepted_after_batch_freeze; feedback ID allowlist only'})
 pd.DataFrame(chain_rows).to_csv(OUT/'tables/nocw_scientific_chain_adjudications.csv',index=False,float_format='%.17g')
 # Add honest identity errors as a lower-bound batch indicator, not a full semantic error rate.
 errper=f[f.manual_assessment=='explicit_selected_molecule_misidentification'].groupby(['seed','round']).size().rename('confirmed_identity_error_lower_bound_n')
 reviewed=bp.merge(errper,on=['seed','round'],how='left');reviewed['confirmed_identity_error_lower_bound_n']=reviewed.confirmed_identity_error_lower_bound_n.fillna(0)
 reviewed.to_csv(OUT/'tables/nocw_reviewed_batches.csv',index=False,float_format='%.17g')
 error_comparison=[]
 for seed,z in reviewed.groupby('seed'):
  yes=z[z.confirmed_identity_error_lower_bound_n>0];no=z[z.confirmed_identity_error_lower_bound_n==0]
  error_comparison.append({'seed':int(seed),'detected_error_batches':len(yes),'no_detected_error_batches':len(no),
   'detected_error_gain_mean':yes.gain.mean(),'no_detected_error_gain_mean':no.gain.mean(),
   'detected_error_before_nrmse_mean':yes.nrmse_before.mean(),'no_detected_error_before_nrmse_mean':no.nrmse_before.mean(),
   'interpretation':'no-detected-error is not verified error-free; sparse observational comparison'})
 error_comparison=pd.DataFrame(error_comparison);error_comparison.to_csv(OUT/'tables/nocw_detected_error_gain.csv',index=False,float_format='%.17g')
 summaries=[]
 for (seed,ph),z in bp.groupby(['seed','phase']):
  summaries.append({'seed':seed,'phase':ph,'batches':len(z),'gain_mean':z.gain.mean(),'worsened_gt001':int((z.gain<-.001).sum()),
   'near_zero_001':int((z.gain.abs()<=.001).sum()),'hypothesis_n':int(z.hypothesis_n.sum()),'condition_test_fraction_mean':z.condition_test_fraction.mean(),
   'explicitly_cites_previous_batch_n':int((z.previous_batch_evidence_n>0).sum())})
 pd.DataFrame(summaries).to_csv(OUT/'tables/nocw_seed_phase_summary.csv',index=False)
 case_summary=[]
 for cat,z in cases.groupby('case_category'):
  case_summary.append({'case_rule':cat,'n':len(z),'actually_worsened_gt001':int((z.gain<-.001).sum()),'near_zero_001':int((z.gain.abs()<=.001).sum()),
   'positive_current_plus_next2':int((z.current_plus_next_2_gain>0).sum()),'next2_censored':int(z.next_2_right_censored.sum()),
   'positive_current_plus_next3':int((z.current_plus_next_3_gain>0).sum()),'next3_censored':int(z.next_3_right_censored.sum())})
 case_summary=pd.DataFrame(case_summary);case_summary.to_csv(OUT/'tables/nocw_case_followup_summary.csv',index=False)
 perf=load('stage_summary.csv',True);perf=perf[(perf.start==333)&(perf.end==1005)&(perf.seeds==5)&(perf.method!='random32')][['method','aulc_mean','aulc_sd','endpoint_mean','endpoint_sd']]
 corr_summary=c.groupby('feature').agg(n_seeds=('seed','size'),raw_mean=('spearman','mean'),partial_mean=('partial_budget_phase_before','mean'),partial_positive=('partial_budget_phase_before',lambda x:int((x>0).sum())),without_before_mean=('partial_budget_phase_only','mean')).reset_index()
 corr_summary.to_csv(OUT/'tables/nocw_science_gain_correlation_summary.csv',index=False)
 gs=load('gain_feature_summary.csv',True);gs=gs[(gs.method=='nocw')&(gs.metric=='partial_rank_correlation')&gs.feature.isin(['raw_gradient_reference_coverage_relative_drop_random_percentile','raw_gradient_selected_to_l_distance_mean_random_percentile','within_any_single_selected_fraction','new_molecule_slot_fraction'])][['feature','mean_seed_correlation','positive_seeds','negative_seeds']]
 family_section=''
 if (OUT/'tables/nocw_case_family_summary.csv').exists():
  fm=load('nocw_case_family_summary.csv');fc=load('nocw_case_family.csv')
  compact=fm[['case_category','batch_max_slot_share_mean','batch_hhi_mean','U_max_slot_share_mean','U_hhi_mean','batch_hhi_minus_expected_random32_mean']].copy()
  compact.columns=['固定病例组','批次max share','批次HHI','当轮U max share','当轮U HHI','批次HHI−随机32期望']
  family_section='''
### 病例分子族补检：较低收益更集中，但接近候选池本底

'''+md(compact)+'''

本表只在既定45个病例位置（41个唯一批次）上补充metadata-only Murcko指标；每seed三阶段等权，再对五seed等权，不重选病例。最小gain−高gain的HHI差在4/5seed为正，seed157反向−0.04297；near-zero−高gain的HHI差也为4/5正，但6101反向。因批次只有32点，原U的HHI不是随机32点的期望，另按无放回抽样的精确公式校正：高gain相对随机32的HHI差−0.07347，最小gain−0.00248。这更像高gain病例避开了候选池的主导核心，而不是低gain普遍超常锁定单一族；低gain的最高核心占比0.6063几乎等于其U的0.6082。前至多3个实际批次scaffold重复名额相对U的富集，高/低组约+0.0101/+0.0401，但低−高仅3/5seed为正，157和12203反向，不构成一致的持续锁定证据；首轮无历史者保持缺失，这一项仅在有历史的状态内平衡。完整45行及三组汇总见 `nocw_case_family.csv`、`nocw_case_family_summary.csv`。Murcko的c1ccccc1将醛、酮、酯等许多不同官能团环境归成泛苯环，空scaffold又合并全部无环分子，不能由本结果认定某个化学功能家族过采样；这些按gain选出的相关现象也不能导出新acquisition规则。来源单锁在 `verification/nocw_case_family_inputs.json`，未读新标签或改变H/模型。
'''
 repstats={'correction_batches':len(rp),'complete_responses':int(bp.complete_response_n.sum()),'first_legal':int(bp.first_complete_legal.sum()),
 'first_legal_ids':int(rp.first_legal_ids.sum()),'first_legal_retained':int(rp.first_legal_retained.sum()),'all_legal_replaced':int(rp.all_legal_replaced.sum()),
 'sets_changed':int(rp.set_changed.sum()),'evidence_to_zero':int(((rp.first_evidence_n>0)&(rp.final_evidence_n==0)).sum()),
 'hypotheses_to_zero':int(((rp.first_hypothesis_n>0)&(rp.final_hypothesis_n==0)).sum())}
 all_reps=hp.exact_text_repeat.sum();feedback_batches=int((bp.previous_batch_evidence_n>0).sum())
 write('NOCW_PROBLEM_AUDIT.md',f'''# NoCW 的真实问题：证据归属与低收益病例

## 结论与范围

NoCW最可靠的缺陷是**相对强数值基线的整体学习效率不足、部分科学命题难以形成明确验证决策、身份/整批纠错的方案保真不足**。这些不能都归因于缺少CW。最新CW-Free五seed在共同L333–1005的AULC反而全部高于NoCW；模型条件不敏感也不是NoCW独有。

复用上一轮综合诊断的性能、行为、几何和收益表；新增的是NoCW科学链。主范围五seed157/887/2357/6101/12203、每seed21批共105批；额外18个已完成批仅用于后续累计gain和next-hypothesis的可用性，不能组成完整五seed晚期比较。记录527条主范围假设、3152次typed引用、229个完整响应；共123批620假设被读取。没有读取原始标签CSV、test真值或未采集候选真值。

## 八类问题的归因表

| 问题 | 具体证据与方法/seed/预算 | 是否NoCW独有/新CW支持 | 证据等级与竞争解释 | 最小排除方式 |
|---|---|---|---|---|
| Row学习效率 | 五seed333–1005 NoCW AULC0.497741±0.065443，raw LCMD0.462549、数值Hybrid0.438420；NoCW终点0.405025 | 非独有；新CW AULC0.532876更差但1005终点更好；旧Free混模型 | 保存曲线事实；非同状态CW因果效应 | 同状态NoCW/Placebo/Critic/raw LCMD |
| 分子/条件行为 | 早期新分子名额NoCW27.81% vs新CW17.60%；中期批内单因素22.08% vs21.94%；813–1005 NoCW10.10% vsCW25.42% | 中期旧差异未在新CW重现；后期差异存在但异质 | 行为事实；批次分散、单因素匹配不等于信息增益 | 事前科学命题与真实设计操作数绑定 |
| 数值覆盖 | NoCW raw覆盖随机分位中期0.5028、813–1005约0.4104；新CW0.5153/0.5167；LCMD较高 | H2有线索但不独有，新CW中期也低 | 相对自身状态随机batch的代理，不是共同真实效用 | Critic提高几何后仍需真实gain胜Placebo/数值baseline |
| 假设可证伪性 | 527假设中305能找到关联的严格单因素测试结构、222找不到；固定样本常无明确拒绝阈值 | 新CW也有宽泛scaffold-specific命题；不能用结构缺失断定无科学价值 | 全量图结构＋52批首假设人工审查；跨分子设计可能不需同分子控制 | 明确范围/替代模型/预期差分后前瞻验证 |
| 反馈吸收 | {feedback_batches}/105批typed引用上一批新观测；m39预测反转被后轮明确否定，多个具体修订例 | 非独有，新CW也有真实修订；尚无反馈因果增益 | 引用/修订实例，不等于每条推理正确 | E1真实/隐藏/扰动反馈，重复提案校准随机性 |
| 证据/实体 | 3152 typed引用全属于选前L，但38分子flags人工确认11误写、27合法跨分子参照；2未知prose ID | 新CW也有25误写下界；并非NoCW独有 | 11为检出下界，不是所有命题的错误率 | typed row→canonical分子→变化轴核验 |
| 整批纠错 | 共同105批NoCW首完整答合法37/105、229完整响应；{repstats['correction_batches']}纠错批，合法ID仅{repstats['first_legal_retained']}/{repstats['first_legal_ids']}保留 | 新CW共同窗口50/105、309完整响应；全程95/180只作背景；旧Free/16+16均有漂移 | 首合法率较低但完整回答反而更少；CW极端packet绑定重答影响总量。不能判科学质量或总费用更差 | 确定性身份处理＋保留合法slot的等预算对照 |
| 条件信息吸收 | 前研究新CW165上样对真实Δ远大于预测；本轮C/D另做NoCW/冻结模型与小试 | 属于QGeoGNN/数据的共同候选瓶颈，不能凭CW结果认定NoCW每批如此 | 未识别编码、优化、噪声份额；需参考本轮条件审计与pilot | 固定已获标签，隔离H，唯一模型对照 |

## 性能与NoCW失效叙事的反证

{md(perf)}

上述各方法是五seed相同窗口，均值±样本SD。NoCW在AULC上优于历史Free及新CW，却在1005终点落后，不能简单称“去CW失效”。旧Free与NoCW的中期覆盖/局部对照差无法直接推出CW因果效应。NoCW五seed有30/105批gain<−.001，raw LCMD与新CW也各30/105；锯齿不是LLM特有故障。

### 纠错可靠性也使用共同窗口

{md(reliability)}

NoCW首完整答合法率35.24%，新CW共同窗口47.62%，但完整回答总数NoCW229、新CW309，方向相反。新CW共同窗口有115份完整回答首先因packet绑定不符被拒绝，极端单轮重答支配了总量；这些不是115个独立科学判断案例。CW全程95/180（52.78%）、430份回答保留为原完整轨迹证据，**不与NoCW共同105批的分母直接暗比**。上述事实既不能说明NoCW科学质量更差，也不能由完整回答更少认定总费用更低；token、失败传输、服务商账单和模型状态仍不同。

## 按规则选取病例，不以病例确认规则

规则先固定于 `verification/nocw_analysis_plan.json`：每seed×三阶段r0–5/r6–14/r15–20分别取最大gain、最小gain、最接近0；同值取早轮，共45个规则位置（可重合）。最小gain若仍为正，不称恶化。eps=.001仅区分描述类别。所有病例属于按结果挑选的解释样本，不能作为新的独立确认集。

未来2/3批累计分别保留：仅后续gain之和，以及当前＋后续gain之和。已有2/3批总窗也另列，避免“后续”歧义；缺失显示右删失，不补值。来源是已完成真实轮次，当前单批与后续batch共同形成变化，不能归因于最初那批。

{md(case_summary)}

15个最小gain位置中13个实际恶化，但当前＋后续2批有10/15累计为正，当前＋后续3批有11/15为正；不能把单轮下降直接判为无效实验。15个“最接近零”位置仅5个落入|gain|≤.001，名称是排序规则，不是实际零收益判定。

完整45位置见 `nocw_gain_cases_with_behavior.csv`，同时提供每seed预算、单轮/后续gain、最大分子名额、局部对照、raw几何随机分位及科学链结构。它们没有一致表现为“围住一个分子反复测”：NoCW每批主要覆盖多分子，单分子最大占比较小；高/低/近零收益病例都可能有严格条件测试，也可能缺少。

按seed×阶段平衡的15个高gain/最小gain位置，单分子最大名额比例均为0.07083；raw覆盖分位高gain0.54167、最小gain0.55833，甚至最小gain组略高。相反，hypothesis关联测试结构的比例分别约0.6587/0.4158。这些是按结果选取后的描述，不是新的独立效应估计；分子“家族”比单分子更宽，不能从上述比例排除家族偏好。

{family_section}

**必要解释限制：** 后续累计从负转正可能只是后来的其他实验起效；单轮负gain不能直接称标签浪费。反之，科学理由看起来正确也不保证整批32点训练后泛化改善。判定某一具体命题是否被验证，见下一份报告的真实链；把比例、覆盖或新分子数直接叫学习价值没有依据。

## NoCW数值代理与学习收益的旧结果复用

{md(gs)}

这是旧综合分析已经计算的NoCW控制预算/阶段/before NRMSE的关联，未重复生成。代理方向并不普遍稳定；gain和before共享分量，且同validation选checkpoint，不能将控制后的相关当因果。H2目前支持到“缺少显式诊断可能限制某些状态”，不能升级成“加critic必然提高AULC”。

## 可信度与输出边界

本次以实际NoCW五seedwrapper的validator纯函数AST重放，不导入runner或label store。全部接受批次32IDs与冻结及feedback一致；input/catalog/turn按batch_freeze原SHA核对，feedback绑定batch_freeze；observed=L、candidates=U，候选没有真实响应列。时间合法性全量通过。完整packet和原始模型回答不复制到交付；少量科学病例摘录来自已接受反馈并有ID/轮次来源。

主要竞争解释H1/H2/H3尚未被彼此排除：科学命题和几何诊断可有不足，但模型不敏感会限制科学标签转化，训练/验证检查点选择亦可改变gain。下一步依据本轮C/D的结果再决定E1或E2，不能仅用本表拼接新Agent。

复现与核验：`scripts/nocw_hypotheses.py`→`review_nocw_science.py`→`verify_nocw_hypotheses.py`；在仓库根用已有chromatography Python和`-B`运行。`nocw_inputs.json`锁定1012项源；固定20批独立枚举154pair、累计gain望远镜和、76个partial的null-space复算通过。先前59批审计精确复现33首答合法/99完整响应，说明本次重放与旧真实规则一致。缺失未来轮次保持右删失，未补跑。
''')
 write('HYPOTHESIS_TO_LEARNING.md',f'''# NoCW的科学假设—真实验证—模型学习链

## 回答核心问题

**存在“解释丰富、具体科学命题未被充分区分”的现象，但不是所有NoCW假设都停留在叙述。** 实际有可识别的预测反转/平台测试、明确反例和后续修订；同时大量“scaffold-specific、不能用统一修正”的宽命题没有预先规定拒绝条件。即使真实条件对照存在，它也可能只回答一个局部响应问题，未验证假设中“能改善泛化”的学习价值主张。

主范围105批、527假设、3152次typed引用。对全部假设建立选前证据与关联choice图；对按固定规则抽出的52批首个假设/问题、所关联理由、一个按元数据排序取出的条件对及下一轮引用文字作人工审查。52批来自45个gain规则位置与r0/r5/r14/r20网格的并集，既有按收益选取也有固定网格，均为解释性样本。不是527条命题的全量科学判真。

![NoCW科学测试结构与学习收益](figures/nocw_science_and_gain.png)

## 六步链与可判定边界

| 链条问题 | 本次核验 | 仍不可判定 |
|---|---|---|
| 具体可测命题？ | 人工分开命名的模型形状/转折问题、宽泛家族问题、事后模型/采集价值叙述、空假设 | 无数值边界的“long/ordinary/threshold”不能补写检验阈值 |
| 范围明确？ | 保存m编号/条件原文、实际linked分子数；canonical身份稳定键 | 明确命名不证明机制成立，不能把m编号跨轮直接当身份 |
| 证据是否合法/正确？ | 3152引用均在选前L；38身份flags逐条人工判分11错/27合法参照 | 证据属于L不等于支持命题；未逐条判定所有科学语义 |
| 选择能否区分？ | 305/527假设找到其关联choices或所引证据中的严格单因素测试结构，985pair-event | 222无这种结构不自动无效；跨分子比较可有价值，未记录变量仍存在 |
| 反馈支持/削弱？ | 针对明确模型形状的病例能观察原预测被支持/反驳 | 一个局部差分不能证明宽泛家族机制或采集效用 |
| 下一轮吸收？ | 保存新观测被typed引用、候选命题连接与具体修订 | 引用不等于正确吸收，Jaccard不是严格命题持续身份 |

527假设中305（57.87%）有可识别关联单因素测试结构、222（42.13%）没有。有假设的99批中，23批每个假设都有该结构，11批均没有，65批混合；另6批假设列表为空，单列而不填成0比例。结构是本次的可验证代理，不是科学假设“被验证率”。每个pair可能共享点、同一pair可能服务多命题；985不能当985独立实验。

### 固定样本人工类别

{md(manual.groupby('category').size().rename('sampled_first_hypotheses').reset_index())}

人工判分保留在 `nocw_fixed_sample_review.csv`。未给宽命题补上原文不存在的前向预测/阈值；52个首命题的语义判断不能转写成全体527命题错误率。命题编号H1会重用，只有文本/证据候选匹配，不声称掌握完整科学命题图谱。

三个具体子检验的操作数、条件、冻结差分、观测差分、支持/削弱/不可判定边界和下一轮信息使用单列于 `nocw_scientific_chain_adjudications.csv`。子问题被区分不等于整个宽命题被证实；其余未做语义判定的命题保留不可判定，不用模型自己的status代替。

## 病例1：m39预测反转被真实实验区分，下一轮明确修订

seed157/r0在相同m39、50µL样品和加载条件下，选5/1与20/1两点。冻结V1预测22.9875→16.5621mL（方向反转）；揭示为10.8833→33.5500mL，V2为22.8667→57.7167mL。5/1→20/1意味着EA分数下降，实际保留增加。r1明确写“previous m39 inversion was a prediction error”，并继续强调单点不建立完整曲线。

这条链有具体替代解释、元数据正确的同分子控制、选后观测及后续修订，证据强于关键词计数。该批gain仅+0.021410；它不是只改变m39两点的随机化对照，不能把整批收益归于此命题。科学验证价值和总体学习价值必须分别记账。

## 病例2：模型平台被打破，但下一轮验证误差仍恶化

seed6101/r7第一命题追问m82/m95是否真正平台化。一个同批严格PE/EA对的预测V2变化为−2.205mL，观测变化+54.363mL；V1预测+4.789、观测+13.200mL。它揭示一个真实局部模型形状失败，下一轮用前批证据继续强调carbonyl placement与endpoint-specific响应。

当批gain+0.099739，下一轮r8为−0.006306；不能将局部反例或文字修订作为每轮持续增益保证。当前＋后续两批仍可能累计为正，这又是单轮恶化不等于原实验无效的反例。本例未孤立化学结构机制、模型误差修复与其余30行的作用。

同一批另一个更接近模型可学习性的实例来自H7_I：m125、PE/EA=1/1、100µL样品，DCM加载溶剂300→500µL。冻结V1/V2差−0.000837/−0.000931mL，观测差+5.941667/+17.933334mL。r8反馈准确指出该dimethoxybenzaldehyde的1/1 finish contrast在100mg再次出现，同时5/1、150mg对照主要改变onset；它没有把现象泛化为普适体积机理。这是**科学对照被测量、反馈被解释，而当前模型事前对该轴不敏感**的具体链，支持H3候选机制；但两行不是随机复测，也没有证明下一模型学不到或改网络一定改善。

## 病例3：身份误写让“matched”论证缺乏正确操作数

NoCW中确认11条明确selected molecule误写，典型是seed6101/r1 `34aaca6f7a0768d743fe` 实际m128，理由却以“Measure m177”解释；r2又把另一个实际m82点称m177。seed12203/r10实际m122点被写成m177，称补全固定loading的10/20/50曲线。ID本身属于U，typed证据合法，并不能阻止这些语义错配。

其余27个“首m编号不同”的flags都是合法跨分子参照，不能错误判为幻觉。两个未知prose ID保留不可追溯，不猜修。下界11/3360不是完整错误率，也没有证据说NoCW的真实错误率高于新CW。错误病例可与gain联结，但错误并不随机分配，不能估计它造成多少误差损失。

## 病例4：宽命题可以反复被“支持”，而缺乏明确拒绝标准

seed887 r5–r12多次强调氮分子需要各自曲线、不能用统一氮修正；部分轮次确实选择同条件或局部响应实验，但“不同家族不共享一个offset/transition”没有定义何种新观测会推翻它。seed157 r15–r20和2357中后期的长尾/threshold命题同样经常使用“can、some、scaffold-specific”。这些命题可以产生有价值的测量，却不等于提出了可量化否证的机制。

主范围逐字重复{int(all_reps)}条；重复本身可以是未解决问题的合理持续跟踪，不能自动判零信息。无明确拒绝规则时，增加几个不同响应并继续标supported，无法证明知识状态更准确。更有辨识力的协议应预先指定操作数、变化轴、候选替代模型和方向/区间判断规则，而不是事后让文字顺应任意观测。

## 纠错与知识链丢失

{repstats['correction_batches']}个主范围纠错批中，最终集合改变{repstats['sets_changed']}批，{repstats['all_legal_replaced']}批原合法ID全部替换；首答合法ID保留{repstats['first_legal_retained']}/{repstats['first_legal_ids']}。{repstats['evidence_to_zero']}批结构化证据从有变零，{repstats['hypotheses_to_zero']}批从有假设变零。零假设批不能通过丰富feedback prose补造一个不存在的typed命题。

这些是方案保真问题，非直接科学效用损失估计。原始最终schema允许0–8假设，最终合法率不能替代假设链完整率；同类问题已在旧Free、新CW与16＋16观察到，不能归为NoCW独有。

直接方法比较限定共同105批：NoCW首完整答合法37、完整回答229；新CW首合法50、完整回答309。NoCW首合法率更低，但CW的packet绑定异常整批重答使回答总数反而更高，不能将任一数量解释为科学质量或费用优劣。新CW95/180与430仅为完整轨迹背景，详见 `nocw_reliability_common_window.csv`。

## “被检验”与下一次gain的关联

{md(corr_summary)}

全105批的condition_test_fraction等是可复算图结构代理；身份flags是待人工语义判分的扫描指标，不等于错误数。每seed独立计算Spearman与控制预算、阶段和before NRMSE的partial rank，另列不控制before的敏感性。无batch p值，五条轨迹不视作105独立重复；gain与before共享项、同validation选checkpoint带来数学耦合。

这里有必须保留的**正向线索**：condition_test_fraction与gain的partial均值约+0.245，4/5seed为正；仅控预算/阶段约+0.319。这支持继续检验“有可追踪局部测试的科学设计可能更有学习价值”，不支持将其判为无效。不过该指标只描述图结构，非命题语义被区分；控制不能消除完整状态、模型可学习性、验证选择和连续轮次混杂。

实际确认身份误写的batch另在 `nocw_detected_error_gain.csv` 按seed与“未检出错误”batch比较；后者不是已确认无错。仅10个被检出错误批、每seed0–4个，不足以稳健判断错误造成多少gain损失。具体反例修订有真实行为证据，却没有相同批次、只改变科学信息的反事实。固定样本中的明确模型形状测试和宽泛命题都出现正/负gain，三类结果仍须分开：科学命题有无被区分、模型是否响应条件轴、整体row预测是否改善。

## 对Hypothesis-Driven AL的研究判断

本次支持研究“Scientific Hypothesis Value与Model Learning Value可能脱节”，但尚未证明系统性因果脱节。可验证代理为：事前明确的替代命题、合法identity/condition graph、观测是否落入预定义区分区域、下一轮是否据此修订；学习价值另用同状态配对gain/AULC及训练噪声控制。一个二目标acquisition还没有资格直接实施。

H1得到的是“证据绑定/否证标准有缺口”的支持，不是NoCW所有选择无效；H2保留几何代理与强数值基线的证据，但新CW反例阻止将其解释为CW缺失的因果；H3是否被小试支持，应由本轮条件敏感性和唯一模型控制裁决。下一步E1识别的是显式反馈通道，E2识别的是数值内容，二者不能混成一个大Agent。

输入与独立核验见 `verification/nocw_inputs.json`、`nocw_checks.json`、`nocw_manual_review.json`、`nocw_independent_checks.json`。新图是结构代理、105个相关轮次与一个具体病例的并列，不把图形差异当科学判真或因果证据。
''')
 review={'status':'PASS','primary_identity_flags_reviewed':len(f),'explicit_molecule_errors_lower_bound':len(errors),'valid_cross_molecule_references':int((f.manual_assessment=='valid_cross_molecule_reference').sum()),
  'unknown_prose_id_occurrences':int((f.manual_assessment=='unknown_prose_reference_not_autocorrected').sum()),'fixed_first_hypotheses_reviewed':len(manual),
  'sample_limits':'first hypothesis per prespecified batch, no claim of full semantic truth audit','full_subtest_chains_adjudicated':len(chain_rows),'all_raw_values_from_authorized_selected_feedback':True,
  'new_llm_calls':0,'training_runs':0,'retrospective_case_selection':'explanatory only','repair_summary':repstats}
 (OUT/'verification/nocw_manual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 (OUT/'verification/nocw_review_inputs.json').write_text(json.dumps(list(REVIEW_INPUTS.values()),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(review,ensure_ascii=False))
if __name__=='__main__':main()
