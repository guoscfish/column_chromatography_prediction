# 短期筛查与长期Hybrid并不是同一个选点协议

**核验结论：不能把两者合并成“16＋16的平均性能”。** 相同的是QGeoGNN训练、CW-LCMD先选16、批量32和L333初始模型；不同的是LLM的合法选择范围、可见反馈、数值字段精度、模型接口及纠错规则。短期并非长期Hybrid的前三轮。

本轮新证据来自实际r0 packet、freeze、预测冻结回执与fit元数据；未打开test预测行或test truth。协议输入SHA见 [phase_source_manifest.json](phase_source_manifest.json)，r0逐seed检查见 [hybrid_protocol_packet_checks.csv](hybrid_protocol_packet_checks.csv)。

| 维度 | 早期 `qgeognn_v2_row_llm16_screen` | 长期 `cw16_llm16_scientist_v2` | 是否仅预算长度差异 |
|---|---|---|---|
| seed/horizon | 157/6101；L333→429，3批/seed | 157至1485、6101至1453；本轮共同L333–1453，35批/seed | horizon是长度差异，其他列不是 |
| LLM合法池 | 128行shortlist | 全部实际U扣除pending16；初始为2981可选＋16 pending | 真正的搜索空间差异 |
| shortlist构成 | 条件于L＋pending16：40 CW＋40 IVR＋40 MaxDet＋8随机，去重后独立随机补足128 | 没有预筛；完整lossless候选与observed表直接交付，query budget=0 | 真正的数值预选择差异 |
| 已观测反馈 | 不提供个体已测响应，禁止用旧packet/理由作额外证据，主要为避免共享对话串seed记忆 | 当前L真实已测记录、上一批预测/实测误差、假设/解释记忆、历史高误差例子及已选ID；仅自身轨迹 | 真正的信息/记忆差异 |
| CW信号 | 当前CW梯度到L最近距离分位，保留4位小数 | 同类当前CW距离分位，lossless精度 | 信号定义近似相同、交付精度不同 |
| 其他预测 | V1/V2 q50，保留3位小数；无完整q10/q90 | q10/q50/q90、宽度、新颖性、结构相似性等 | 真正的信息差异 |
| 模型/effort | 当前Codex对话；冻结协议明确snapshot未暴露，effort未可审核固定；不据产品名猜模型 | 初始Astra/high，L813起Sol/xhigh；独立API请求，确切后台版本仍不能由字符串保证 | 真正的配置差异 |
| 生成/校验 | 单次响应；只保留前16条中的合法唯一ID与非空理由，不足按预设随机序补位；实际6轮失败/补位均0 | final scientific plan、packet绑定、候选/观测证据ID和假设schema；无自动数值随机fallback；失败可纠错后接受 | 真正的策略包差异 |
| 科学对照配额 | 无；prompt偏好互补与多样性 | 无硬分子/对照/探索配额，允许局部条件对照 | 都不是强制单因素对照 |
| 数值16 | exact CW-gradient LCMD-TP先选16 | 同算法先选16，LLM不能重选pending | r0相同；后续因自身状态变化而不同 |
| checkpoint | 复用同seed已有L333；之后scratch | 复用同seed同L333；之后scratch | 初始共享，后续分叉 |
| 原报告评价 | 四条新轨迹全部冻结后统一test汇总，且历史CW test已暴露；开发筛查 | 本轮及11月16日主诊断使用保存的validation Row曲线 | 评价集也不同，不能数值拼接 |

## 初始状态的实证核验

两seed的早期r0 `prediction_freeze` 指向与长期/CW相同的L333 checkpoint，初始化hash、验证行hash均一致。早期实际 `packet.pending_CW16` 与长期冻结 `pending_ids` **16个ID及顺序完全相同**。早期候选128行；长期packet给出2981个可选候选、16个pending、333个observed记录。因此最初数值半批并未改变，选择包差异集中在LLM半批可见输入与接口；不能把后续性能差别仅归为预算长短。

两者的CW梯度都由当前模型q50的Center/Width变换、L333固定尺度、512维CountSketch构建；LCMD采用最近中心簇平方距离和最大簇、簇内最远点、逐点更新与稳定并列顺序。本轮未重算梯度或改变算法。

## 性能口径必须分开

早期 [FINAL_REPORT.md](../qgeognn_v2_row_llm16_screen/FINAL_REPORT.md) 的已公布test汇总为：LLM AULC0.811373，CW0.804957，随机补充0.839394，未通过“双控制均优且两个seed均胜”的门槛。这是**既有报告结论**，本轮没有重新评价test。

为避免把test与validation混在一起，本轮从原始保存fit_audit恢复早期16个validation评价点（包含两补充策略、两seed、四预算），并与对应CW validation、长期同长度prefix比较：

| 协议（均为validation，L333–429） | seed157 AULC | seed6101 AULC | 两seed均值 | 相对纯CW均值差 |
|---|---:|---:|---:|---:|
| 纯CW-LCMD32 | 0.723654 | 0.900900 | 0.812277 | 0 |
| 早期128 shortlist＋LLM16 | 0.725553 | 0.887701 | 0.806627 | −0.005650 |
| 早期128 shortlist＋Random16 | 0.739937 | 0.855713 | 0.797825 | −0.014453 |
| 长期fullpool Hybrid前三批 | 0.762397 | 0.848487 | 0.805442 | −0.006835 |

早期LLM validation平均略优于CW，却劣于同shortlist的随机补充0.008802；两个seed对随机的方向相反，仍不满足原一致性门槛。早期test与validation的相对随机排序不同，更说明不能从已公布test主表挑数值来支持长期策略。

这些曲线只能说明各自协议在两个已暴露开发split上的历史表现。早期与长期LLM的搜索空间、反馈及模型未受控，**不能把长期结果较好解释为“全候选交付/反馈记忆是唯一原因”**。也不能用短期未达门槛否定长期后期已有信号。完整逐seed新表见 [early_long_validation_comparison.csv](early_long_validation_comparison.csv)。

长期Hybrid的新阶段证据在 [PHASE_PERFORMANCE.md](PHASE_PERFORMANCE.md)。它在两seed后3段方向一致、对局部波动较稳健，具有值得检验的信号；但是否投入长跑仍需结合冻结状态后16替代机制与模型条件表达诊断，不能单凭此两条轨迹自动扩展。
