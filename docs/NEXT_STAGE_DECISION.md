# 当前研究状态

更新：2026-11-11。此处区分已测事实、假设与运行保护；原始完整证据统一从 [archive973](https://github.com/guoscfish/column_chromatography_prediction/tree/973ccaf8cb436e129ba3facc79d42853263923ce) 恢复，精炼结果见 [research](../research/README.md)。

## 已确认事实

| 问题 | 当前证据与限定 |
| --- | --- |
| 当前预测器 | 独立QGeoGNN-V2的3seed×row/compound点预测资格；unseen-compound差距明显，不授予已校准UQ资格。[来源](../research/PREDICTOR.md) |
| 五seed Free/LCMD L1485 | 十条完整validation轨迹。Free最终误差均值较低，但全程AULC较高；Free有mixed-model历史前缀与授权例外，不能当纯配置一致因果比较。[来源](../research/FIVE_SEED_AL.md) |
| CW/NoCW | NoCW的共同L973历史快照与当前main CW-Free目标L1485分别保留，阶段进度不是最终结果；框架收敛不判断main任务已完成。[来源](../research/CW_NOCW.md) |
| GP/IVR | 开发GP未通过正式AL门槛，IVR完整历史探索未建立明确AULC收益，25态机制audit止于Gate C之前；不同指标/数据角色不混排。[来源](../research/NEGATIVE_RESULTS.md) |

当前框架只资格化合成CLI、多轮工程状态机、数值核心、已有checkpoint推理及受限历史回归。真实生产多轮训练、真实K3、LLM provider/Prompt端到端 **NOT_VERIFIED**。当前机器具备授权私有fixture，不代表fresh clone自动拥有模型、原始runtime或数据权限。

## 现阶段边界

main的CW-visible Free LLM32按独立冻结协议与运行目录保护，目标五seed各L1485。本轮不接管、不停/续跑、不改模型头/loss/Prompt、不自动merge。已发布 [b1a6ed7a的20:06快照](https://github.com/guoscfish/column_chromatography_prediction/tree/b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28/reports/free_cw_overview_20261109_2005) 是固定历史观察，不能当本日实时进度。

main的 [L333损失/分位数复盘](https://github.com/guoscfish/column_chromatography_prediction/blob/fcb51852c8bb6e38403fe95a302287a0a1779d47/studies/diagnostics/quantile_loss_audit_20261109/SCIENCE_AUDIT.md) 保持独立来源：冻结eval梯度不等于Adam轨迹或权重因果；宽度/crossing在已有信号控制后未建立普遍增量收益。loss重加权、硬非交叉或新概率模型仍是待预注册假设，不在当前运行中途修改。

先完成并审计CW-Free，再经新授权讨论科学切换、预测器消融或新AL实验。已使用test、validation选模或后验诊断的数据不能重新命名为盲确认集，见 [数据消费登记](../configs/data_consumption_register.json)。迁移学习保持历史归档状态，本轮不安排跨柱后续训练。
