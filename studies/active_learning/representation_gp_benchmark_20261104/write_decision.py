"""Write a bounded research decision from actual completed-state summaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    path=HERE/"summary/summary.json"
    summary=json.loads(path.read_text())
    coverage=summary["coverage"]
    overall={(r["representation"],r["kernel"]):r for r in summary["overall_metrics"]}
    comparisons={r["comparison"]:r for r in summary["paired_comparisons"]}
    def metric(rep,kind,name):return overall[(rep,kind)][name+"_mean"]
    def n(x):return f"{x:.4f}"
    def pct(x):return f"{100*x:.1f}%"
    def pair(name):
        r=comparisons[name]
        wins=r.get("normalized_rmse_L0_improved_seed_count",r.get("improved_seed_count"))
        return r,wins
    complete=coverage["complete_matrix"]
    status="完整20状态开发矩阵" if complete else f"部分矩阵{coverage['completed_states']}/20"
    q=overall[("llm_conditions","projected")]
    g=overall[("gnn","matern52")]
    fq=overall[("llm_conditions","matern52")]
    fusion=overall[("fusion","projected")]
    optimizers=summary["optimizer"]
    nonconverged=sum(r["nonconverged_fit_count"] for r in optimizers)
    totalfits=coverage["endpoint_fits"]
    projection=next(r for r in summary["paired_comparisons"] if r["comparison"]=="llm_conditions: projection - frozen")
    fusion_pair=next(r for r in summary["paired_comparisons"] if r["comparison"]=="Fusion projected - GNN projected")
    robust=summary.get("text_robustness",[])
    robust=robust[0] if robust else {}
    source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    core=(f"本次完成{status}，共{totalfits}次端点GP拟合。所有结论来自已有LCMD轨迹、L_t训练与既有validation开发评价；"
          "四预算与五seed相关，validation参与过QGeoGNN checkpoint selection，不能称独立测试或AL确认。")
    decision="NO_GO_FOR_FORMAL_LLM_GP_ACTIVE_LEARNING"
    text=f"""# 下一阶段研究裁决

日期：2026-11-04。**{decision}**。本次停止于离线研究，不启动正式主动学习。

{core}

当前证据支持继续研究可验证的数值表示与概率模型，但不足以把本地LLM+GP升级为正式选样方法。该判定针对本次明确冻结的Qwen方案；不等于否定T5、其他encoder或整个GOLLuM思路。

## 主要依据

1. 固定Qwen conditions的主误差为{n(fq['normalized_rmse_L0_mean'])}，Structured为{n(metric('structured','matern52','normalized_rmse_L0'))}，GNN hidden为{n(g['normalized_rmse_L0_mean'])}。本地语言先验是否增加信息，应看这组非语言对照，而非只比SMILES-only。
2. Qwen Projection相对Frozen的配对主误差变化为{n(projection['normalized_rmse_L0_delta_mean'])}。适配后主误差为{n(q['normalized_rmse_L0_mean'])}，低于Structured的{n(metric('structured','matern52','normalized_rmse_L0'))}、仍高于GNN的{n(g['normalized_rmse_L0_mean'])}；这两个方向均在五个seed均值上一致。但其90%观测区间覆盖为{pct(q['coverage_90_mean'])}、NLPD={n(q['nlpd_mean'])}；Frozen对应{pct(fq['coverage_90_mean'])}、{n(fq['nlpd_mean'])}。均值收益不能抵消概率可靠性失败。
3. GNN fixed的90%覆盖也只有{pct(g['coverage_90_mean'])}，说明问题不限于语言模型。用监督训练后的表示拟合GP，得到的条件后验没有自动包含整个QGeoGNN预测器的泛化误差；这是与结果一致的机制解释，尚非唯一因果诊断。
4. Fusion Projection相对参数匹配GNN Projection的主误差变化为{n(fusion_pair['normalized_rmse_L0_delta_mean'])}；Fusion自身90%覆盖为{pct(fusion['coverage_90_mean'])}。两个小头均8,256参数，但Fusion多了train-only PCA与block scaling，不能把全部均值差唯一归于语言新增信息。
5. {nonconverged}/{totalfits}个有限拟合未在冻结优化预算内收敛；全部按规则保留，没有按validation增加迭代或重启。模型内信息量和采集排序差异不能弥补这一可靠性缺口。
6. 等义模板改写的V1/V2 MaxVar Top32重叠为{pct(robust.get('V1_maxvar_top32_overlap_mean',0))}/{pct(robust.get('V2_maxvar_top32_overlap_mean',0))}。冻结计算的重复性与表述鲁棒性是不同要求；本次只核验conditions A/B，未把模板B作为调参集。

## 八个研究问题的回答

**1. LLM是否比传统化学与条件编码提供更多价值？** 当前固定本地Qwen的比较见上述主误差及配对seed表。完整数值支持的收益与负结果都保留；本次没有T5结果，不能外推所有语言模型。SMILES加入条件有帮助也只证明它利用了一部分实验信息，未证明优于结构化编码。

**2. Hidden与Gradient各有什么优势和不足？** Hidden直接继承任务监督，128维成本低，均值预测信息强；它的GP覆盖仍不足。Raw/CW描述全参数输出敏感性、512维且可精确绑定缓存，适合审计覆盖与相似性，但不是响应预测值或可靠不确定性。上样量方向的压缩必须与真实差分及预测差分误差共同解释；加载溶剂体积仅3对/2分子，无法作稳定结论。

**3. MLL学习Projection是否带来可重复未见行收益？** 三初始化全部平均后，Qwen Projection相对Frozen在五个seed均值上均改善；适配后也均优于Structured、仍均逊于GNN。这是可重复的开发均值收益，但严重欠覆盖和未收敛阻止“更可靠概率模型”的结论；不能以训练NLL低或挑单初始化证明可复现。

**4. 复杂表示是否过度自信？** 本次结果支持该风险：Projection观测方差与真实误差不匹配、NLPD和覆盖恶化。noise是surrogate条件似然参数，不能直接宣称为真实实验测量噪声。没有事后用validation缩放方差。

**5. Fusion收益是否超过容量解释？** 已做同参数数目的小头控制，仍有PCA/scaling与输入组成差异；需共同查看Frozen Fusion、参数匹配Projection及seed方向。当前概率结果不足以将Fusion升级为可靠互补性证据。

**6. 是否足以进入LLM+GP正式AL？** 不足；当前为No-Go，未计算任何新AL学习曲线或AULC。

**7. 最小且最有价值的后续核验？** 优先修复和核验概率模型，而非先跑昂贵AL：固定既有早期checkpoint及当时L，仅使用已授权、历史上已观测且模型未拟合的后续实验，按时间严格区分拟合、校准与最终核验；先锁定一个保守Projection/噪声方案并与Structured/GNN同协议比较。不能用GNN已监督的L内留出来冒充端到端外样本校准。采集偏倚和validation曝光仍须披露。随后若另获明确资源许可，再补真正T5-base冻结对照；不在本次下载。

**8. 如何服务最终学习效率与稳定性？** 让实验身份、表示、核、噪声、预测方差与batch条件信息量都可复算，并先通过外样本概率和文本稳定性门槛；通过后才检验同初始状态、同预算的数值batch AL。数值可审计是研究工具优势，本次尚未证明实验标签效率提高。

## 本次停止边界

未调用付费生成式API、未下载模型、未新训QGeoGNN、未读取本状态test真值或未来U响应、未改现有Free/NoCW/CW-Free轨迹。数据来源、权限和工程日志字段修复见[AUDIT.md](AUDIT.md)、[execution_audit.json](execution_audit.json)与[summary/summary.json](summary/summary.json)。原始拟合日志保留；状态seed由冻结目录恢复，模型初始化seed另存，数值预测未改。

完整证据：[表示](REPRESENTATION_REPORT.md)、[校准](GP_CALIBRATION_REPORT.md)、[容量与融合](COMPARISON_REPORT.md)、[未来AL设计](AL_FEASIBILITY.md)。本裁决机器汇总SHA256：`{source_hash}`。
"""
    (HERE/"NEXT_STAGE_DECISION.md").write_text(text)
    al=f"""# 后续主动学习可行性：只设计、不执行

{core}

**结论：No-Go，暂不启动正式LLM Embedding + GP主动学习。** 原因不是只看谁RMSE最小，而是本次Projection概率欠覆盖、模板采集不稳定、有限优化预算未收敛及开发评价曝光。详细裁决见[NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)。

## 最小验证顺序

1. 先用既有早期冻结checkpoint与其未拟合、但历史已合法观测的实验作严格prequential开发核验；拟合/校准/最终核验按时间冻结，不用最终核验调噪声、核或Projection。只改一个预先指定的正则化/噪声问题，保留Structured、GNN与相同优化预算。该步骤也需要单独明确数据范围，不由本脚本自动执行。
2. 只有概率评分、覆盖-宽度、初始化与文本扰动都支持时，再补真正T5-base对照。T5当前缺本地权重，官方单份safetensors约892MB、Apache-2.0；资源与下载须另获明确许可。Qwen负结果不能代替这一步。
3. 只有通过上述验证的Frozen、Projection或Fusion才进入单独预注册的AL比较；不因本模型内BatchIG高而直接放行。

## 未来实验臂

| 实验臂 | 目的 |
| --- | --- |
| LCMD32 / Numerical Hybrid32 | 当前强数值参考，冻结其真实历史实现 |
| Frozen LLM + GP + batch acquisition | 检验固定语言表示 |
| Task-adapted LLM + GP + 同batch acquisition | 检验MLL适配是否有标签效率增益 |
| 经离线筛查的GNN+LLM Fusion + GP | 检验同容量控制后的互补性 |
| 已有Free LLM32历史参考 | 对照显式生成式决策的成本与可靠性；执行时期差异必须披露 |

GP-MaxVar、条件IVR和Batch Information Gain与预测学习目标一致；不直接照搬面向目标值最大化的EI。未来应使用相同初始L333、同合法候选、同标签预算、QGeoGNN scratch初始化/重训与checkpoint选择规则。各采集函数分别成臂，不把不同表示、核、容量和batch rule一起换后归因单因素。

未来两端点应共选一个32点batch，而非各选32点。可预先锁定独立端点GP下的等权归一化MaxVar `0.5×Σ_q Var(f_q)/s_L0,q²`；IVR使用相同两目标归一化平均方差缩减，BatchIG在明确的端点独立假设下相加。每加一点都对两个端点的共享batch同时条件化。该规则只作为未来预注册建议；本次按端点分别做诊断，没有运行上述合并选样，也未验证端点独立假设。

正式主指标为预冻结raw学习曲线的归一化梯形AULC；次指标为终点误差、跨seed方差、最差seed、标签及计算成本、合法运行成功率。各方法不能用不同完成端点或best-so-far曲线冒充公平比较。未来必须保留独立标签权限与prediction/batch freeze。

本次仅在同256个无标签合法U子池上比较数值采集排序、Top32交集、条件化批次与冗余；没有获取这些离线所选实验的响应，没有训练新QGeoGNN，没有新AL轨迹或AULC结果。不同GP的自评信息量不是共同真实收益标尺。
"""
    (HERE/"AL_FEASIBILITY.md").write_text(al)
    result={"decision":decision,"complete_development_matrix":complete,"states":coverage["completed_states"],
        "endpoint_fits":totalfits,"summary_sha256":source_hash,"formal_al_started":False,
        "interpretation":"No-Go for current frozen local-Qwen formal AL; continue bounded probability/robustness validation"}
    (HERE/"decision.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":main()
