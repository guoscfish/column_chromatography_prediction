# EPIG 迁移验证：预注册协议

2026-11-17，基线提交 `b1f54b5132f671ab1c6068127fe8cc313df5b3ac`。具体机器参数、有限模型集合、资格阈值及计算上限以本目录 `config.json` 为准。`freeze.json` 在任何新资格评分前保存配置和实现 SHA；资格结果不得反向修改该文件。修改实现必须记录修订原因，保留旧 seal 和结果。

这是在已反复公开、且验证集参与预测器 checkpoint selection 的五个历史 outer splits 上进行的 **development evidence**。真正独立确认需要后续未公开的实验/划分和事先锁定策略，不能以本次五 seed 充当独立测试。

## 科学比较

主比较是同一 GP 概率模型、同一目标集合、同样的 B32 条件化，只改变 EPIG 与 IVR 目标。历史 kernel-IVR 的单位线性特征先验/噪声不是新 GP。单点 Top32、条件化 greedy、独立端点聚合、跨输出联合模型均分别命名。数学和离线分数仅检验可计算性，不证明样本效率。

四个固定模型：RBF GP、Matérn-3/2 GP、独立 Bayesian linear regression、具有跨端点协方差的 matrix-normal BLR。每模型、每 seed、3 folds，共 60 次完成的辅助拟合，累计尝试上限63（至多3次中断恢复）；每次最多60迭代、1个预定义起点。辅助输入来自冻结分子描述符和实验条件，没有从全部 L333 监督训练的 GNN 中抽取特征来冒充 honest OOF。标准化、PCA、响应均值和噪声在每个 fold 的训练子集拟合。正式 GNN 仍严格保留原模型与训练规则。

资格预先要求每个 seed 内各端点的 OOF 80%覆盖在[.70,.90]、95%覆盖在[.88,.99]、标准残差RMS在[.70,1.50]；数值协方差必须有效。联合 held-out Gaussian NLPD 与其对角化模型对比，每 scalar 最多恶化 .10 nat。后者是联合依赖可信度门槛，不以 marginal coverage 代替 cross-row 验证。报告各 seed、fold、端点而不只汇总；五 seed 有重叠行，跨 seed 汇总仅描述，不参与另一 seed 的资格、模型选择或拟合。有限但未收敛的拟合照实保留并报告。

每个 seed 只按本 seed L333 OOF 在两个合格 GP 中选择 marginal NLPD 最低者；本 seed 无合格 GP 则使用其合格 BLR 中 NLPD 最低者作为预定义备选。BLR 明确按真实家族命名，不冒充 GP。五 seed 每个都必须有本地合格后验才能开始完整主比较，不能删除不合格 seed。若仍存在无合格后验的 seed，则完成四族及全部可运行数学、输入/排名诊断；将正式效能记为证据不足，不能用不合格模型给 EPIG 普遍 NO-GO。不得继续增加核、重启或调参直到出现正结果。

## 目标分布与机制

P1 是冻结 3330 个 outer-train 输入的经验 Row 分布（主）；P2 是当前合法 U 的均匀分布（次）。M=256/1024/2048 使用固定 SHA 排序的嵌套无放回样本。每次参考的是未来独立噪声标签；候选与同输入目标的噪声不共享。正式主 M=1024 事先固定，不能依结果挑 M 或分布。各 M 报告误差、rank、Top32/100、greedy batch、时间和峰值内存，必要时明确计算预算未完成部分。

离线同状态完整 U 计算 objective 差异、P1/P2、Top32/greedy、V1/V2以及真正跨输出 BLR。选点 ID 必须唯一、属于当前 U、与 L/validation/test 不交。未揭示标签的输入条件统计与冻结 GNN 响应可用于体积偏好诊断；不能计算 oracle utility。预测器条件学习控制属于独立第二因素，需要对 EPIG 与控制同换预测器，不包含在本次主因素中。

## 正式学习曲线

五 seeds=157/887/2357/6101/12203；L333→L1005，B32，21次后续scratch拟合/seed/arm，共210次。原4g数据、row IDs和split、458952参数QGeoGNN-V2、Adam(.001)、maximum_epochs1000、patience100、训练batch2048、原每seed初始化/训练seed及validation checkpoint selection不变；不启用smoke配置。原float32神经网络与L333 ddof0 endpoint scales保持不变，矩阵代理float64。测试标签始终封闭，不为资格或效能调参读取。

允许复用经hash、输入契约、初始化和训练配置核验的L333模型及完全匹配历史Random/LCMD/Kernel-IVR保存validation曲线。最新CW-Free五seed是Free LLM参考；CW-gradient LCMD若只有两seed，必须标明，不冒充五seed。历史Kernel-IVR测试指标与新validation指标不可混用。

主要指标是原非单调combined validation NRMSE曲线的梯形AULC/(1005−333)，combined定义为sqrt(mean(endpoint NRMSE²))。报告逐点及逐端点MAE/RMSE、L525/L813/L1005、early/mid/late局部AULC与终点，绝不使用best-so-far。统计单位outer seed；报告配对差、均值、样本SD、t4 95%区间及32种配对符号翻转的参考检验，小n推断限制明确。

L1485扩展条件事先固定：完整L1005比较后若EPIG在均值全程AULC、终点或L813–1005晚段AULC任一改善，两臂一起延长至36轮。其余情况先审查概率校准、晚期趋势与预测器瓶颈，再写事前修订。未执行长预算时仅限L1005窗口，未执行主比较时不回答效能。

## 状态、资源和恢复

不修改任何历史实验。新目录保存每轮配置/commit、L/U有序hash、selection落盘与hash，随后才通过RestrictedLabelStore授权揭示；保存checkpoint、fit_audit、history及标签审计。单worker/2torch线程，先检查既有研究进程/锁。断点重启先验证已完成fit和selection契约；部分fit迁移到新目录incomplete_attempts，禁止覆盖已完成模型。预算耗尽时留下明确状态，不能将未完成轨迹补齐或包装为结论。无付费API。
