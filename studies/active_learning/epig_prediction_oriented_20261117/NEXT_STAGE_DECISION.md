# 下一阶段决策与六个研究问题

当前状态：**INCOMPLETE_POSTERIOR_QUALIFICATION；主五seedL1005配对未完成**。这是证据范围声明，不是普遍否定EPIG。

本文件首先报告stage1的实际证据。第二阶段已按独立freeze实际执行，当前资格状态：INCOMPLETE_POSTERIOR_QUALIFICATION。实际结果与新formal状态以[stage2结果](stage2_nested_calibration/STAGE2_RESULTS.md)及[stage2决策](stage2_nested_calibration/STAGE2_NEXT_STAGE_DECISION.md)为准；下面保留stage1原资格结论，不把stage2结果倒写成stage1通过。

1. **当前数据能否可靠计算EPIG？** 解析实现与规模已通过；887/6101尚无通过冻结校准门槛的本地代理，五seed完整正式门槛未满足。矩阵正定性及平均联合NLPD门槛本身通过，不能称联合协方差已被证伪。其他seed的合格状态单独报告，不能跨seed认证。
2. **EPIG信息是否超出现有IVR？** 已获得58个同后验决策/MI诊断；M1024状态的EPIG/IVR平均pointwise Top32重合=0.874，greedy批次重合=0.856（实际已完成状态，非独立重复）。可有不同决策，但这不是新增真实预测价值的证据，且每状态资格须分别核对。历史Kernel-IVR和新GP代理不同；只有同posterior IVR/EPIG配对才隔离objective因素。
3. **是否改善真实Row-level AULC与终点？** 未知：五seed完整L333→L1005主配对尚未完成，不能给Go/No-Go。
4. **若未改善，最支持的原因？** 当前最强证据是预注册后验资格不足（A）及真实性能证据不足（F）；不能把尚未测量的性能称为未改善，预测器瓶颈D仍是候选解释。
5. **是否值得作LLM数值辅助？** 当前不具备直接宣称有益的真实学习证据。可保留为待验证数值候选；只有概率资格、稳定决策和真实Row-level收益足够时才提出受控LLM辅助实验。本次未调用新的付费LLM API。
6. **仍不能确定什么？** 未执行的完整AL/目标分布性能消融、非可分离跨输出模型、GNN表征或异方差代理、匹配新预测器控制、长期L1485价值与独立确认均不能由当前诊断推定。

## 继续执行的依据

任一seed没有合格本地后验时，预注册完整正式门槛未满足，不能把不合格代理的210次昂贵重训当成已资格主实验，也不删掉失败seed。应先完整保存所有预注册族的资格/目标敏感性，再根据失败维度提出有限预算、预先冻结的下一代理族；该新资格设计须标明是结果后的第二阶段，不能伪装原预注册。当前研究结论为A/F，非EPIG NO-GO。

## 有限第二阶段nested calibration：明确区分阶段

该设计明确写于stage1资格结果揭示之后，不改stage1的任何阈值、模型选择或已有轨迹，也不能称为原stage1预注册。第二阶段已按独立freeze实际执行，当前资格状态：INCOMPLETE_POSTERIOR_QUALIFICATION。实际结果与新formal状态以[stage2结果](stage2_nested_calibration/STAGE2_RESULTS.md)及[stage2决策](stage2_nested_calibration/STAGE2_NEXT_STAGE_DECISION.md)为准；下面保留stage1原资格结论，不把stage2结果倒写成stage1通过。设计为先另立不可变stage2协议与hash，再一次性执行：保留现有四个模型族、三个outer OOF fold和原优化预算；每个outer训练fold内部再三折产生合法inner-OOF误差。只从inner-OOF拟合两个预定义校准候选——端点marginal Gaussian NLL温度与端点central80比例尺度——并保留identity作为控制。两种尺度可从同一inner预测计算，不重复拟合模型。outer held-out标签只用于最终资格评分，不用于估计该fold温度；五seed各自选择，不汇合其他seed真值。

若端点校准尺度为正对角D，则对所有latent/noise/cross blocks一致施加DΣD，保留正定性和原相关结构。该产物应另称**校准Gaussian联合代理**，不能声称仍是原GP先验/似然下未经修改的精确后验。温度仅改变幅度，不修复Gaussian的尖峰重尾、条件异方差、错误跨row相关或均值失配；中央80%覆盖即使改善，也不保证完整概率评分、95%尾部覆盖、相关结构或AL收益。小inner训练模型与outer模型样本量不同，校准迁移本身也需披露。

有限预算：四族×五seed×三outer×(三inner+一次outer拟合)=240次辅助拟合；若所有seed资格足够，再为各seed选定族五seed合计最多20次inner+全L333最终拟合，总计260次计划完成、最多263次含中断attempt、1800秒墙钟、逐fold缓存与attempt账本。模型族、两个校准候选、资格门槛、选择准则和预算均在stage2结果揭示前另行冻结，不用AL学习曲线挑温度。若仍无完整五seed合格后验，报告该有限校准族失败，不继续无限追加方法直到通过。是否启动正式210次QGeoGNN拟合，仍须满足新冻结资格与资源边界；**stage1门槛保持不变，实际stage2执行不属于原stage1结果**。

独立确认策略：目前五seed属于development，后续应在新冻结Row数据或新的独立实验批次预先指定选点策略、模型族、目标分布和主要指标；新测试真值只在全部方法与预算冻结后一次性评价。不得反复查看已有test来调整本轮代理。

全部结果均为重复暴露历史 Row 开发队列上的 development evidence；不是独立确认。统计单位为 outer seed，五个 seed 的划分仍重叠，不能把 batch/OOF 行数当独立实验重复。
