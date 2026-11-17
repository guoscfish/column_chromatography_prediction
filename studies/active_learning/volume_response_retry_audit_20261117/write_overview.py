#!/usr/bin/env python3
"""Write final Chinese decisions from the complete, fixed experiment tables."""
from pathlib import Path
import json
import pandas as pd

OUT=Path(__file__).resolve().parent

def main():
    result=json.loads((OUT/'control_decision_numbers.json').read_text())
    repair=json.loads((OUT/'local_repair_verification.json').read_text())
    retry=json.loads((OUT/'retry_audit_verification.json').read_text())
    assert len(list((OUT/'fits').glob('seed_*/*/fit_audit.json')))==15
    means={r['kind']:r for r in result['summary']};g={r['comparison']:r for r in result['gates']}
    a,b,c=(means[k] for k in ['A_original','B_amount_skip','C_volume_skip'])
    def outcome(label,r,gate):
        return f"{label}：局部差分MAE从{a['difference_error_mean']:.5f}变为{r['difference_error_mean']:.5f}mL，相对改善{100*gate['local_improvement_ratio_equal_seed_means']:.2f}%，{gate['local_positive_seeds']}/5 seed改善；Row NRMSE为{r['combined_normalized_rmse']:.5f}，相对A变化{100*gate['row_relative_change_equal_seed_means']:+.2f}%。机制参考门槛{'通过' if gate['mechanism_reference_pass'] else '未通过'}。"
    passed=[name for name in ['B-A','C-A'] if g[name]['mechanism_reference_pass']]
    if not passed:
        model_decision='B/C均未通过预定机制参考门槛，停止这一组简单直接残差路线，不追加深度、seed或超参数搜索。结果不排除更一般的表示、优化或loss问题，但本轮没有支持立即替换原模型的证据。'
    elif g['C-A']['mechanism_reference_pass'] and g['C-B']['mechanism_reference_pass']:
        model_decision='独立体积输入在同规模直接通路下也通过C−B机制门槛，值得在独立数据上确认其信息贡献；尚不能推断体积的物理因果作用。'
    elif g['B-A']['mechanism_reference_pass']:
        model_decision='优先确认已有上样量通路的传递机制；当前结果未要求立即新增独立体积信息。独立体积是否额外有用仍应依据C−B逐seed表判断。'
    else:
        model_decision='C相对A有机制信号，但C−B未通过相同参考门槛；只能把独立体积通路列为待确认方案，不能宣称已经区分两种表示。'
    row_pass=[name for name in ['B-A','C-A'] if g[name]['row_relevance_reference_pass']]
    joint_pass=[name for name in passed if g[name]['row_relevance_reference_pass']]
    if joint_pass:
        row_decision='有方案同时通过局部与Row工程参考门槛，可进入独立确认阶段；仍不证明LLM选点优于LCMD，也不证明LLM贡献了模型知识。'
    elif row_pass:
        row_decision='部分Row指标改善，但未同时通过局部机制门槛，不能把整体改善归因于修复了体积响应压缩。'
    else:
        row_decision='没有方案通过预定Row relevance门槛，不能用本次局部表示干预解释整体AL收敛差距。'
    final=f'''# 最终判断与下一步

本轮完成10个原冻结state的零训练诊断、固定五seed×三模型的**15次拟合**、540个已接受Free32轨迹批次的重试审计及127批离线局部修复。没有新LLM/API/Judge请求，没有完整AL重跑，没有打开test评价，也未把未揭示候选标签投影进研究。旧模型、runner、prompt、checkpoint和历史selection均保持原文件。

## 1. 体积响应压缩主要在哪个环节？

信号从Density×V进入量条件补充分支，未进入原backbone的主要RBF路径。L1005原冻结模型的V1/V2真实绝对体积响应均值为3.367/3.558mL，预测仅0.220/0.107mL；同预算PE/EA则为真实10.224/22.172、预测9.819/21.259mL。审计系列q50 pre-ReLU非正比例为0，20个体积endpoint局部AD的backbone路径导数全为0，补充分支导数与最终导数一致。因而可观察现象主要是**体积通路学到的幅度偏小**，不能归咎于最终ReLU压零或整分支死亡。

217个分子仅2个有多个记录Density；同分子固定其他条件的量别名在0.1mg内为0。当前元数据没有证明大规模不可逆信息碰撞。冻结state检查是训练内机制诊断，不能作为隔离检查成绩；现有证据也未唯一排除loss或优化因素。详见`MODEL_ARCHITECTURE_AUDIT.md`、`VOLUME_SENSITIVITY.md`。

## 2. B、C分别比原模型好多少？

{outcome('B 上样量Skip',b,g['B-A'])}

{outcome('C 独立体积Skip',c,g['C-A'])}

C相对B的局部相对改善为{100*g['C-B']['local_improvement_ratio_equal_seed_means']:.2f}%，{g['C-B']['local_positive_seeds']}/5 seed同向。三者都由相同初始化重新训练，B/C仅多2个零初始化参数。逐seed先报告于`MODEL_COMPARISON.md`；pair不被当作独立科研重复。{model_decision}

## 3. 是否改善了Row Performance？

原Row validation五seed等权NRMSE：A={a['combined_normalized_rmse']:.5f}，B={b['combined_normalized_rmse']:.5f}，C={c['combined_normalized_rmse']:.5f}。B/C分别有{g['B-A']['row_nonworse_seeds']}/5、{g['C-A']['row_nonworse_seeds']}/5 seed不变差。{row_decision}

原validation始终按冻结规则选checkpoint，检查标签不参与训练或选择。需要保留的限制是：五seed分别2/3/2/0/2个检查系列与原validation条件关联，2357和12203各有一条完全相同记录条件的validation行；已另报剔除关联系列的固定预测敏感性。所有H内同系列及等价条件训练重叠为0，但不能称本次局部检查为完全独立确认集。

## 4. 127批前缀行为的主要可观察触发条件是什么？

全部127批均先出现校验失败，首份可解析方案均非前缀；首次触发错误为123个候选身份问题、4个packet绑定问题。38批原本已有31个合法唯一ID。实际纠错要求重新输出完整方案，没有锁定原合法选择；最终前缀方案仅保留原合法ID事件440/3634=12.11%。这支持优先修复方案保真协议，不证明模型故意偷懒。

分母应写540个Free32轨迹已接受批：539个精确匹配最终LLM原答，另1个有既有授权人工ID修正且不属于前缀样本。全记录率127/540=23.52%；严格原答子集127/539=23.56%。首个错误是validator短路首先触发的问题，不代表方案只错一处；前缀样本还有27例潜在未观测证据问题。候选顺序、长上下文、错误类型、日期/model/effort与政策的统计都是观察性证据，尚无随机顺序对照。详见`PREFIX_FAILURE_ANALYSIS.md`。

## 5. 是否发现实际API上下文丢失？

在可核对的客户端证据中没有发现：850个保存request加270个Historical精确重建上下文共1120个请求hash全部匹配；580个续问保留原packet、全部前答和记录错误。原候选表仍在消息历史内，并非只发送一个error。实际body通过input显式传入messages，未使用previous_response_id或conversation作为隐式记忆。

这只证明客户端记录与发送构造一致，不能证明远端内部没有裁剪；430条保存服务端metadata的truncation字段均未记录。Historical额外3个上下文还需当时授权反馈消息才精确重建，已显式恢复并记录来源。New CW后期实际persistent amendment令逻辑与网络重试均无次数上限，不能继续把整条New轨迹概括为8×4固定重试。详见`RETRY_CONTEXT_AUDIT.md`。

## 6. 是否有中转站模型质量问题的证据？

没有足够直接证据。已审计具体运行配置、请求model/effort、返回model、日期和endpoint host hash；这些可以核对配置与相关性，不能验证服务商权重、内部路由或质量替换。175723条网络attempt日志既不是175723次科学纠错，也不是可直接认定的计费调用数。未来官方/中转与候选顺序的独立2×2协议仅写在`future_provider_protocol.md`，本轮未执行任何对照API。

## 7. 局部纠错能保留多少原始科学方案？

原型保留全部3634个合法唯一ID选择事件，占4064原名额89.42%，留下430个明确待补空额；reason、hypothesis及批次科学文本逐值保留，没有前缀补位。4个根packet_sha字段可通过原请求hash、逐值packet及内部绑定校验确定性恢复，但修复后仍**0/127份完整方案通过validator**，27例证据成员资格问题保留阻断。两个原本32个ID合法的方案也被隐藏证据问题阻断。

这里保真的是ID和原科学文本，不能宣称89.42%的科学理由已正确或方案已恢复执行。离线dry-run说明可避免丢掉合法部分，未验证未来LLM填空成功率、成本或学习收益。详见`LOCAL_REPAIR_DESIGN.md`。

## 8. 下一步先改什么？

**先推进工程纠错保真流程的独立确认。** 将合法ID和原理由冻结，确定性修复有证明的工程字段；把430个空额和27例证据问题作为明确局部编辑任务。先通过冻结案例测试，再在未来明确授权的小规模交互中验证保留率和合法率。本轮原型没有写回真实selector。

模型方面：{model_decision} {row_decision} 不同时改采集函数、Prompt和模型，不自动启动新AL，也不让LLM自由生成架构。LLM知识贡献需要在独立确认中与人、确定性规则和同预算有限搜索比较；仅训练预先由人指定的A/B/C不能证明LLM发现能力。
'''
    (OUT/'FINAL_DECISION.md').write_text(final)
    tests=json.loads((OUT/'test_result.json').read_text())
    readme=f'''# 体积响应与LLM纠错策略审计

已完成用户授权的15次固定对照拟合，以及10个原冻结state的零训练诊断、540已接受轨迹批次/1120逻辑请求上下文审计和127批局部修复dry-run。**新LLM/API请求0，完整AL重跑0，额外拟合0。** 本目录为隔离研究产物，未修改旧模型、runner、prompt、checkpoint或历史selection。

请先读 [FINAL_DECISION.md](FINAL_DECISION.md)，逐seed模型数据见 [MODEL_COMPARISON.md](MODEL_COMPARISON.md)。A/B/C五seed等权局部差分MAE为{a['difference_error_mean']:.5f}/{b['difference_error_mean']:.5f}/{c['difference_error_mean']:.5f}mL，原Row validation NRMSE为{a['combined_normalized_rmse']:.5f}/{b['combined_normalized_rmse']:.5f}/{c['combined_normalized_rmse']:.5f}。工程原型保留3634/4064原名额，430空额显式待补；0份草稿可直接当完整selection执行。

## 报告索引

| 文件 | 内容 |
|---|---|
| MODEL_ARCHITECTURE_AUDIT.md | 真实条件输入、量单位、128维路径、输出与损失 |
| VOLUME_SENSITIVITY.md | 原10 checkpoint的构图、FD/AD、激活和体积/PE对照 |
| MODEL_COMPARISON.md | 15 fits逐seed/端点/配对结果、参考门槛与稳健性 |
| TRAINING_INDEPENDENT_REVIEW.md | 独立元数据/配置QA及原validation关联限制 |
| RESULTS_INDEPENDENT_QA.md | 从30份原始预测独立重算9项表，误差≤2e−12 |
| RETRY_CONTEXT_AUDIT.md | 各真实controller、messages、逻辑/网络重试与配置身份 |
| PREFIX_FAILURE_ANALYSIS.md | 127前缀批首错、合法ID保留及H1–H5 |
| LOCAL_REPAIR_DESIGN.md | 离线保真原型、确定性修复、科学阻断 |
| future_provider_protocol.md | 未来provider×顺序独立实验，仅设计 |
| LLM_MODEL_DISCOVERY_ROADMAP.md | 有限模型发现、知识贡献归因、防validation过拟合 |
| FINAL_DECISION.md | 八项问题的数值回答与可执行下一步 |

模型图在`figures/`，其余审计图在`audit_figures/`；核心表为`control_seed_summary.csv`、`control_endpoint_summary.csv`、`paired_model_differences.csv`、`paired_endpoint_differences.csv`、`mechanism_gates.csv`、`control_robustness.csv`、`retry_case_batches.csv`、`retry_request_contexts.csv`及`local_repair_cases.csv`。各模型原始history、预测、独立checkpoint和fit receipt在`fits/seed_*/模型/`。

## 数据边界和固定选择

历史H1005标签仅来自各seed当轮冻结catalog.observed，1005个ID严格匹配contract/freeze；体积检查集来自此前预定`model_control_holdouts.csv`，每seed完整隔离10系列。训练侧等价条件/同系列重叠为0。原validation416行保持冻结规则；其中的条件关联已披露，未伪称全新确认集。

协变量由canonical CSV的feature-only列读取；原validation仅通过现有RestrictedLabelStore按role核验后投影数值。其csv流会经过原文件行，hash检查也经过原文件字节，因此不宣称从未读取目标文件字节；未请求、转换为数值、输出或用于分析test/未揭示候选目标值。模型图来自全y=0盲cache，C的独立V通过稳定sample ID附加并逐batch核验。L333归一化尺度固定。

`ANALYSIS_PLAN.json`在无已完成fit时冻结主模型/seed/门槛；`ANALYSIS_SUPPLEMENT.json`记录metadata QA触发的series等权与剔除validation关联series描述性敏感性，声明时已完成1 fit，未查看其比较结果、未改变主门槛。预训练数据门、共享初始化和RNG hash在`DATA_FEASIBILITY.json`，冻结三训练源码未变。执行环境见`environment.json`。

540分母包含1个既有授权人工ID修正例外，严格原答为539；127前缀均属于精确原答子集。请求文件保存850份，Historical270份由实际版本/消息重建且hash匹配；不能把重建记录当作独立保存HTTP wire日志，也不能据本地上下文证明provider内部行为。

## 复现与验证

实际解释器：`/Users/fish/miniforge3/envs/chromatography/bin/python`。所有命令以仓库根目录运行并使用`-B`；线程环境见`run_tests.py`。分析脚本不发API、不训练：

```sh
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/volume_response_retry_audit_20261117/evaluate_controls.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/volume_response_retry_audit_20261117/report_controls.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/volume_response_retry_audit_20261117/write_overview.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/volume_response_retry_audit_20261117/verify.py --check
```

零训练A2由`sensitivity.py`、`summarize_sensitivity.py`复现，逐分子表由`summarize_molecules.py`汇总；C由`retry_context_audit.py`后接`finalize_retry_audit.py`、`local_repair.py`复现。首次拟合前的`prepare_data.py`会构造新数据门，本次已经完成并锁定，不应在已有fit时重写。`train_controls.py`只允许预定15组：已有完成receipt会复用，存在未完成start则拒绝重启，不自动追加或重试。

`run_tests.py`已完成28项测试，exit code={tests['exit_code']}，本轮未因门槛结果增加任何拟合。`VERIFICATION.json`包含输入/输出SHA、15 fit receipts、训练配置/选择核对和边界说明；`verify.py --check`只复核封存身份。若重跑生成脚本，输出可能重写，应保留原封存或重新明确封存版本，不用新seal掩盖历史输入变化。
'''
    (OUT/'README.md').write_text(readme)

if __name__=='__main__':main()
