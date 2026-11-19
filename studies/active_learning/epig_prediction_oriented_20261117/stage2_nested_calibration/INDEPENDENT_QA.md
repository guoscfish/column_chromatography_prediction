# 阶段 2 独立数值与边界复核

**PASS：20 个 family 状态的 identity OOF 与阶段 1 完全一致；240 次本地已完成拟合与全局预约账本吻合；180 个温度向量由合法 inner-OOF 独立复算，最大误差为 0。** 本复核新增模型拟合为 0，不读取原始数据源或 U/validation/test 真值。实际数值见 `INDEPENDENT_QA.json`。

这验证了第二阶段的实现和执行记录，不代表用户要求的充分 AL 效能研究已经完成。五 seed L333→L1005 的真实 QGeoGNN EPIG/IVR 配对仍未执行；核心 AULC 和终点研究问题仍然未知。

## identity 与阶段 1 的一致性

对五个 seed、四个固定 family 全部 20 个状态，核对各阶段完成 receipt、科学 seal、sample IDs、canonical indices 和合法 L333 真值。结果如下：

| 复核项目 | 20 状态最大绝对差 |
| --- | ---: |
| OOF 均值，mL | 0 |
| OOF 逐行协方差，mL² | 0 |
| 全部边际概率指标及条件校准指标 | 0 |
| 每折 full/diagonal/block-diagonal NLL、PSD 和白化残差指标 | 0 |

这不是通过重跑拟合获得的结果，而是直接比较两阶段已经保存的实际数组和数值。阶段 2 identity 的外层拟合与阶段 1 训练规则、输入、输出及划分一致，支持“校准控制没有悄悄换掉基础模型”的解释。

## 240 次拟合与断点记录

- 全局 `compute_budget.json` 为 240 条 `nested_qualification` 拟合预约，每个 seed×family 恰好 12 条。
- 本地 `nested_cache/` 恰好 240 个 attempt，全部标为 COMPLETE；没有中断或未完成 attempt。每状态九次 inner fit、三次 outer fit，共 240 次。全局记录本身是预约日志，完成性由对应本地 hash receipt 共同确认。
- 验证 300 个缓存完成 receipt：180 个 inner、60 个 outer 模型、60 个 outer 评价缓存；另外验证所有输出候选的完成 receipt。
- 阶段 2 科学 seal 绑定的七个代码/协议/配置文件通过 hash 校验。stage1 seal 和资格决策的链接一致。
- 未生成 finalization 拟合，未生成本阶段正式 QGeoGNN fit receipt；因此不能把未使用的最多 20 次 final fits 或 210 次正式神经拟合记为已完成。

## 温度没有跨 outer-held-out 真值旁路

全部 60 个外层边界逐一复核：outer train 222 行、outer held-out 111 行；其内三折各用 148 行训练、74 行预测。

复核从合法阶段 1 L333 导出的已保存真值重建 `y[outer_train]`，其 hash 与 inner/outer 两种拟合缓存 contract 完全一致。全部 inner 训练及预测位置映射回 L333 后，均与该 outer held-out 集不相交；inner-held-out 覆盖 outer train 的全部 222 行恰好一次。完整输入 X 的 hash 与阶段 1 相同，模型源码与配置由两个科学 seal 绑定。

独立从三个 inner fold 的保存 mean/covariance 重建 222 行 OOF，按冻结公式计算：

`D_identity = 1`

`D_nll = sqrt(mean(((y−mean)/sigma)^2))`

`D_central80 = quantile(abs((y−mean)/sigma), .8) / Φ⁻¹(.9)`

60 个外层状态 × 三种方案共 180 个二输出温度向量与实际保存值完全一致，最大差 0。导出候选的逐折温度也一致。复算只使用 outer train 的 inner-OOF，outer held-out 真值没有参与这些公式。冻结实现的合成干预测试另外验证过：只改变某一 outer-held-out 的 y，不改变该折温度、预测均值或协方差，只改变对应评价分数。

## NLL 与 central80 的实际失败维度

| 校准方案 | 候选数 | 合格数 | 80% 覆盖范围 | 95% 覆盖范围 | 标准化残差 RMS 范围 |
| --- | ---: | ---: | --- | --- | --- |
| identity | 20 | 3 | 0.85886–0.94294 | 0.93393–0.95796 | 0.97895–1.23719 |
| nll | 20 | 0 | 0.88288–0.96697 | 0.93994–0.99099 | 0.46385–1.03777 |
| central80 | 20 | 0 | 0.76577–0.82583 | 0.84384–0.93093 | 1.40838–2.19103 |

NLL 温度没有解决主体过覆盖：20 个 family 状态的 V1 全部超过 0.90 的 80% 覆盖上限，17 个状态的 V2 也超过。另有一个 V2 同时超过 95% 覆盖上限并低于 RMS 下限。温度向量元素范围 0.96155–10.66811。

最大 NLL 温度来自 seed12203、Matérn GP、outer fold2、V2。其合法 outer-training-side inner-OOF 中最大 |z|=157.442，该一个观测贡献约 98.11% 的该端点标准化平方误差，产生宽度温度 10.66811。该数值经独立复算一致，并非把 variance temperature 误当 width temperature，也没有隐式 clipping。这体现平方误差温度对少数极大标准化误差的敏感性，以及 148 行 inner 模型与 222 行 outer 模型之间的校准迁移风险；不能据此认定该观测标签错误或删除它。

central80 温度元素为 0.44428–0.94068，确实把全部 40 个端点的中央 80% 覆盖移入原 [0.70,0.90] 范围，但同时带来其他失格：20 个状态的 V1、18 个状态的 V2 超过 RMS 上限 1.50；三个 V1 和四个 V2 的 95% 覆盖低于 0.88；两个状态超过联合 NLL 相对 diagonal 的平均退化上限。把中心校准好不能修复同方差高斯的尖峰重尾形状。

上述计数中的 family、校准方案及端点共享基础拟合和重叠数据，不是额外独立实验样本。不能把 60 个校准候选视为 60 个 outer seeds 来计算 AL 显著性。

## 资格决策与解释边界

第二阶段选择与阶段 1 相同：157/12203 为 Matérn GP identity，2357 为 coupled BLR identity；887/6101 仍没有通过的注册候选。冻结的完整五 seed gate 因此不允许启动原注册正式对照。没有放宽阈值、删除困难 seed 或根据 AL 结果选择温度。

这次有限校准族已全部真实执行；它的负结果是“这两个预定义幅度校准方案未补齐五 seed 资格”，不是“EPIG 采集目标已被证伪”。一致的正对角 `DΣD` 不改变同一基础后验的 EPIG 分数及条件化批次，因此本阶段也没有试图通过温度调出新的 EPIG acquisition 效果；trace-IVR 的相对端点权重会改变。

最强机制证据仍是这批固定概率族的边际形状与条件校准问题。冻结 gate 的边界敏感性仍应保留；A/F 是适用范围声明，不能代替实际 QGeoGNN 学习曲线。B（目标函数）、C（批量）、D（预测器瓶颈）或 E（充分轨迹上的真实无收益）都未由本次嵌套校准复核证明。没有完整五 seed 21 轮实测，就不能声称已完成用户要求的充分效能验证。

## 复现与产物

```bash
KMP_DUPLICATE_LIB_OK=TRUE OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n fish python studies/active_learning/epig_prediction_oriented_20261117/stage2_nested_calibration/analysis/independent_qa.py
```

该命令只读取保存结果，重新生成 `INDEPENDENT_QA.json` 和以下独立数值表：`independent_qa_identity_comparison.csv`、`independent_qa_outer_boundaries.csv`、`independent_qa_temperature_recomputation.csv`。不调用模型 fit、不打开原始标签数据源、不改冻结门槛。QA 脚本位于阶段封存后的 `analysis/` 附录，自己的 SHA256 写入 JSON，不冒称阶段 2 预注册代码。
