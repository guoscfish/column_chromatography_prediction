"""Generate the stage decision and eight requested answers from real receipts."""
import json
import pandas as pd
from data_boundary import HERE


def main():
    gates=json.loads((HERE/'summary/phase1_gates.json').read_text())
    metrics=pd.read_csv(HERE/'summary/phase1_aggregate.csv')
    costs=json.loads((HERE/'summary/phase1_cost.json').read_text())
    completion=json.loads((HERE/'phase1_completion.json').read_text())
    if not completion['complete']:
        raise ValueError('cannot write final decision over an unfinished primary matrix')
    if any(g['pass'] for g in gates):
        raise ValueError('a prespecified gate passed: review the authorized expanded benchmark before stopping')
    def record(rep,method,cal='raw',role='check'):
        return metrics.loc[(metrics.representation==rep)&(metrics.method==method)&(metrics.calibration==cal)&(metrics.role==role)].iloc[0]
    q=record('qwen','original');qb=record('qwen','converged');qr=record('qwen','regularized');qt=record('qwen','original','temperature')
    rows=[]
    for rep in ('qwen','fusion'):
        for method in ('original','converged','regularized','noise_floor','staged'):
            a=record(rep,method);t=record(rep,method,'temperature')
            rows.append(f"| {rep} | {method} | {a.nlpd:.3f} | {t.nlpd:.3f} | {100*a.coverage_90:.1f}% | {100*t.coverage_90:.1f}% |")
    text=f'''# 最终阶段决定与八个研究问题

2026-11-06。**No-Go：本轮没有方法具备进入正式QGeoGNN主动学习实验的概率与表示证据。** Phase1和Phase3获授权的有界数值研究已完成；Phase2完成源码对齐、资源审计、实现准备与工程预飞，真实T5/LoRA性能实验尚未执行。保留完整负结果，不启动正式AL。

## 本轮真实执行范围

- 历史20状态/720fit及所有身份哈希复核，旧360次Projection均40步耗尽；旧/新真实闭包数值梯度、数据边界及概率代数测试通过。
- 主消融4状态×2表示×5方法×2初始化×2端点=160次Projection；同拆分6种Frozen表示×4状态×2端点=48次，共{costs['finite_fits']}个有限fit，失败{costs['failed_fits']}；其中Frozen Fusion的8次为独立补充回执。另8次全L_t数值复跑。没有重复运行旧720配置。
- Phase3共32次完整L_t冻结特征加性核/容量控制，全部SciPy bounded success；32次posterior冻结重建一致。
- Phase2是4结构×2初始化×3步的toy全批Exact MLL梯度预飞；T5与Qwen3实际离线加载各失败一次。toy不是预训练模型效果，不计入真实任务fit。
- Phase1的10个表示×方法候选全部未过训练前冻结的扩展门槛（完整矩阵分母16，缺项/失败不得被剔除）。因此停止5seed×4budget扩展；不是按某个预算效果挑选停止。

## Q1：此前GP严重欠覆盖的证据与可改善性

历史Qwen/Fusion Projection观测90%覆盖22.78%/21.79%，noise中位约0.0516/0.0512 mL²；360/360未收敛，185/360触边。数值差分没有发现梯度或方差混用错误。全L_t从40增至200步后，MLL提高、训练几乎插值、noise仍可触底，4/4组合的开发NLPD恶化、3/4组合覆盖下降，说明预算不足确实存在，但增加预算本身不能修复失配。高容量plug-in表示、噪声塌缩、长度尺度耦合及模型失配是证据支持的机制线索，不能认定唯一因果来源。

分子组GP check中，Qwen A40→B200的NLPD从{q.nlpd:.3f}变为{qb.nlpd:.3f}，90%覆盖从{100*q.coverage_90:.1f}%变为{100*qb.coverage_90:.1f}%；C正则化NLPD为{qr.nlpd:.3f}。独立calibration估计temperature后A40的check NLPD为{qt.nlpd:.3f}、覆盖{100*qt.coverage_90:.1f}%。这与旧Row validation评价总体不同；具体分子数/宽度/各seed差异见Phase1报告。

| 表示 | 训练消融 | raw NLPD | 独立temperature NLPD | raw 90%覆盖 | temperature 90%覆盖 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

校准只能被解释为观测方差尺度后处理，不能据此声称原始潜函数联合后验、IVR或BatchIG已校准。必须同时看宽度、NLPD、标准化RMS及独立核验；不能只把覆盖调到90%。本轮没有候选达到预设的综合放行条件。

## Q2：真正T5/LoRA是否优于旧Qwen

**尚不能判断。** 本地没有T5-base或Qwen3-Embedding-0.6B，真实模型离线加载回执为blocked。已固定GOLLuM官方commit并查真实源码，没有把Frozen Qwen+Projection冒充完整GOLLuM，也没有把toy预飞当成T5结果。需要下载授权、真实梯度/资源预飞和共同GP设置冻结，之后才可能执行真实配对Pilot。

## Q3：LoRA与Projection贡献及成本

旧Qwen Projection曾改善均值，却大幅恶化概率质量；本轮对其训练机制做了实际消融。LoRA独立效应、与Projection交互和真实微调收益/成本均**未测**。toy证明梯度能从full-batch exact MLL经float64 Cholesky回到adapter及Projection，冻结base不更新；它不回答科学收益或真实模型成本。官方P-LLM、LLM-phi和P-LLM-phi的区别见方法对齐报告。

## Q4：Additive Fusion是否更稳、语言核有无新增信息

32/32冻结加性/容量fit报告有界优化成功，比旧Projection的未收敛状态更容易优化；但这个低参数核对照不能被扩写成所有融合方式的优越性。90%观测覆盖仍约55%，learned lambda范围0.000123–0.035709，语言分量总体很小。learned加性核相对GNN nRMSE平均仅下降约0.000625，NLPD反升0.214；GNN双核容量control有相似量级均值变化。当前加性核不优于旧Frozen PCA+concat参照。

模板改写对learned fusion的MaxVar/IVR平均Top32重叠仅53.13%/39.45%，BatchIG为91.02%；方差绝对量也有明显变化。单模板下排序与GNN相近不代表跨模板可靠。没有可验证的可靠语言增量证据；不能因lambda>0就声称语言知识有效。Residual缺合法OOF/冻结训练外残差与充分互补证据，为Not Yet Testable；Factorized仅设计，不启动结构搜索。

## Q5：是否有比GNN Hidden更值得用于数值AL的表示

**本轮没有新表示获得升级资格。** Frozen Qwen覆盖较好伴随更宽区间和较差均值；Projection/加性Fusion均未通过综合概率检查。Structured、Raw/CW、GNN同拆分基线及完整L_t历史参照全部保留。不能从GP中间评分宣布某表示提高下游标签效率，也不能把当前GNN Hidden本身称为已校准专家。保持它为专业表示强基线，语言增量仍待真实T5/LoRA合法配对验证。

## Q6：GP方差、信息增益和实际候选排序是否足够可靠

绝对概率资格未通过；原始方差不能直接当可信置信度。排序是另一问题：同模板的加性方差与GNN秩相关约0.96–0.98，部分探索次序保留，但文本改写可显著改变MaxVar/IVR；Phase1另保留init17/29与A/B模板稳定性及批内冗余、分子/五条件覆盖。MaxVar、conditionalIVR、greedyBatchIG均在同一合法U256子池真实计算，不查询其响应。这些结果最多支持继续离线排序诊断，尚无独立证据证明排名能识别未来高误差或提高QGeoGNN AULC。GP自身IG值不能跨模型等同真实实验价值。

## Q7：是否可以进入真实QGeoGNN AL

**No-Go。** 本轮无新增正式AL轨迹、无新QGeoGNN训练、无测试真值或未来U标签查询、无付费API。Phase1的优化/概率门槛未通过，Phase3欠覆盖和模板脆弱性未解决，Phase2缺权重与真实Pilot。5seed×4budget扩展尚未启动；此为阶段门槛停止，不是最终证明所有LLM表示无效。

恢复科学推进需先解决以下条件：在合法fit内部选择/冻结训练正则与noise设置；取得原始GP训练收敛及真实训练外概率或排名信息价值证据；补齐采集前冻结的forward特征/历史已揭示响应身份，或另设计合法独立核验；T5/LoRA再按共同设置、多个初始化、模板A训练/B核验完成配对Pilot。任何正式AL都需另行授权，并固定同L333、batch32、同训练初始化/验证协议，主比较AULC、跨seed波动、最差轨迹及成本，不能仅看L1005。

## Q8：如何用于LLM数值引导而保持研究目标

下一步可将经过验证的数值模块作为LLM决策的可审计信息接口：实验文本同时保留SMILES与全部精确条件，数值/类别字段另外做确定性语义校验；候选必须属于冻结U_t并通过ID、去重、单位和两个体积字段检查。LLM接收冻结checkpoint身份、候选相对排序、邻域支持及对初始化/模板的敏感性；尚未资格化的GP只标注探索性rank，不输出校准置信度承诺。概率失配、模板位移或分歧过大时，按预先冻结规则回退到已验证LCMD/Hybrid数值策略；这些回退仍需未来真实AL验证，不能本轮假定有效。

若未来概率专家通过核验，可把它的预测、支持范围和可验证的批采集目标用于约束LLM可选集合与batch冗余，保留LLM对实验语境的解释与建议角色。对比Free LLM、相同数值采集、LLM+数值引导/约束时，应共享候选/预算/初始化，分别识别语言表示、概率模型和决策约束的贡献。这样研究目标仍是提高LLM科学决策的效率、稳定性与可靠性，而不是为了给GP或Fusion取得正面排名。

## 资源边界与恢复命令

最小新增模型资源是固定revision T5 safetensors **891,646,390 bytes** 及配置/快tokenizer数MB；机器24GB内存、约202GiB可用磁盘，当前会话CUDA/MPS不可用。真实CPU训练时间/RSS尚未知，必须先有界预飞；不申请云算力或付费服务。目标模型/依赖未自动下载或安装。是否安装PEFT/GPyTorch/BoTorch及是否使用原生Torch实现，需在恢复时明确；工程预飞不等于正式Pilot已经可直接复现。

本地测试、重放、汇总与核验命令见 [README.md](README.md)；资源恢复命令和官方来源见 [PHASE2_MODEL_AND_RESOURCE_PLAN.md](PHASE2_MODEL_AND_RESOURCE_PLAN.md)。所有原始CSV/JSON、配置、哈希、失败/未收敛项和测试回执保留；全部证据仍属开发性诊断，没有独立确认或AULC新结论。
'''
    if (HERE/'test_receipt.json').exists():
        receipt=json.loads((HERE/'test_receipt.json').read_text())
        text+=f"\n最终工程测试：整目录{receipt['passed']}项全部通过（{receipt['pytest_seconds']}秒），见 `test_results.txt` 与 `test_receipt.json`。这不替代科学概率门槛。\n"
    (HERE/'NEXT_STAGE_DECISION.md').write_text(text)

if __name__=='__main__':
    main()
