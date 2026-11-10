# 当前研究状态

状态：`CURRENT`。更新：2026-11-10。本页维护项目级结论与下一步；原始数值、预算、模型/Prompt 身份及标签消费以冻结协议与权威报告为准。工程设计见 [系统架构](ARCHITECTURE.md)，报告分类见 [reports 索引](../reports/README.md)。

## 当前优先事项

CW-visible Free LLM32 五 seed 研究按 `ACTIVE_RUNTIME` 保护，目标为每 seed L1485。本轮文档清理不判定其已完成、不接管运行或续跑、不修改中途模型头、loss、Prompt、预算和冻结规则。本轮开始时 main 为 `b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28`，与候选差异独立保留，不自动合并。

已发布的 [2026-11-09 20:06 快照](https://github.com/guoscfish/column_chromatography_prediction/tree/b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28/reports/free_cw_overview_20261109_2005) 使用当时共同完成预算，不是实时进度或 L1485 最终结果。[该版本研究入口](https://github.com/guoscfish/column_chromatography_prediction/blob/b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/README.md) 说明冻结协议及修订来源，其中更早进度也须按原日期阅读。

候选实现仍只开放合成 CLI，不是已验证的真实科学 runner。完整生产重训练、真实 K3 和历史 LLM 协议的验证范围见 [二期测试报告](repository/phase2/TEST_REPORT.md)。

## 已确认的证据

| 研究身份 | 结论和限制 | 权威入口 |
| --- | --- | --- |
| QGeoGNN-V2 4g 点预测 | 458,952 个有效参数；三固定 seed 的 row/compound 资格验证完成。Clean 为失败结构尝试。点预测资格不授予校准 UQ 资格。 | [工程验证](../studies/predictor/final_v2_engineering/README.md)、[资格报告](../studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md) |
| Sequential B32，L333→1005 | LCMD/Hybrid 对 Random 有收益；原分类 NO_CLEAR_SEQUENTIAL_AL_GAIN 未区分两者最佳/相近，不能改写为“对 Random 无收益”。 | [主动学习索引](../studies/active_learning/README.md) 中对应结果 |
| Static / Adaptive，固定 L653 | Adaptive 端点更好而 AULC 未改善；单一端点不证明全曲线优势。 | [固定预算对照](../studies/active_learning/qgeognn_v2_batch_adaptivity/FINAL_REPORT.md) |
| Free LLM32 / LCMD32，五 seed L1485 | 已完成十条同预算轨迹；历史模型前缀与显式 amendment 限制随结果解释。validation 平台期不等于无剩余泛化空间。 | [最终报告](../reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md) |
| IVR / 数值机制及 CW 探索 | NO_CLEAR_IVR_IMPROVEMENT、STOP_BEFORE_GATE_C 等否定/部分证据保留；开发性多臂/短程结果不替代当前五 seed 长程研究。 | [主动学习索引](../studies/active_learning/README.md) |
| 双专家历史扩展 | seed157 完成 L1005，seed6101 在 L525 后连接故障停止；不是两 seed L1005 完成。 | [固定终态核验](https://github.com/guoscfish/column_chromatography_prediction/blob/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/dual_expert_two_seeds_to1005_20261021/verified_final_status.json) |

以上结论属于各自模型、划分、人群与协议，不能混成跨方法通用排名。原始实验保持独立身份、采样序列及失败/授权记录。

## 分位数和损失：事实与假设

完整诊断按用户要求只保留在 main 的 [fcb51852 科学报告](https://github.com/guoscfish/column_chromatography_prediction/blob/fcb51852c8bb6e38403fe95a302287a0a1779d47/studies/diagnostics/quantile_loss_audit_20261109/SCIENCE_AUDIT.md)，候选不复制 CSV 或图。

- `NEW_OFFLINE_MEASURED`：五 seed 冻结 L333 eval 模型中，V2 的共享 backbone 梯度范数更大，条件分支有异质性；不等于整个 Adam 轨迹或 loss 权重因果试验。
- 原始六输出存在交叉，名义中央80%覆盖在 seed 间波动。q50 用 MSE 学习，不是已校准中位数；排序后处理不是硬约束。
- 固定辅助模型在可用 q50/gradient/latent/条件控制信号后，加入宽度和 crossing 标志未得总体增量收益；不证明所有预算和模型下永远无效。
- `NOT_MEASURED`：其他预算、当前 compound AL、匹配 seed/budget 的真实 K3、实际 Adam 更新竞争和正式 AL 效率增益。

`HYPOTHESIS`：loss 重加权、硬非交叉头或 crossing 严重度可能有价值，但当前不足以将 V2 权重从1改为0.5，也不足以把 crossing 解释为可减少的 epistemic uncertainty。后续先独立预注册预测器消融与增量误差信号检验，再决定是否开展匹配 seed/budget 的 AL 对照；不在当前运行中途换方法。

## 数据与后续门槛

曝光事实见 [数据消费登记](data_consumption_register.json)；已用于拟合、选模、测试评价或诊断的数据不能重新命名为盲确认集。row 插值、target-compound holdout、source-unseen OOD 是不同任务。测试真值不用于选点或事后挑模型，固定 validation 标签成本须单独披露。

迁移学习自 2026-10-28 归档停止维护，固定来源 [dadd2281](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b)。当前不安排跨柱后续训练，恢复路径见 [文档指南](README.md)。

当前先完成并审计 CW-Free，保留负结果和异常解释，再按新授权安排正式科学切换或消融。文档清理、工程测试和离线相关性不替代独立科学确认。
