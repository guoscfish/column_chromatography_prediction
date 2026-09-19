# 4G 主动学习：下一阶段实验建议

2026-09-18 后续更正：下文保留审计完成前的研究建议。最新
[IVR mechanism audit](../../studies/active_learning/qgeognn_v2_ivr_mechanism_audit/FINAL_REPORT.md)
已完成：ranking 相关性较高，未证明主要瓶颈是数值不稳定；BAIT-style forward/backward
对 surrogate 的中位收益为零，不能把 greedy optimizer 视为主要瓶颈。
当前 scalar IVR 已属于 regression V-optimal/BAIT-like objective。
后续优先完成
[Phase 0 评价重构](../../studies/active_learning/qgeognn_v2_efficiency_review/REPORT.md) 与
[固定总预算 Static/Adaptive 对照](../../studies/active_learning/qgeognn_v2_batch_adaptivity/PROTOCOL.md)，
再按 [Phase 2 方案](4G_PHASE2_IMPLEMENTATION_PLAN_2026-09-18.md) 做 matched one-step screen。
不直接执行下文原建议的完整新 sequential，也不继续扩展维度、prior、noise 网格。

研究日期：2026-09-18。本文基于已冻结的 row sequential 结果、Kernel-IVR exploratory extension，以及不读取新 test truth 的缓存审计。本文不改动既有 protocol，也不把 exploratory 结果当作独立确认。

## 结论先行

当前主动学习已经证明了标签效率价值，但还没有解决后期平台：

| 方法 | normalized AULC | NRMSE@1005 | AULC 胜 LCMD |
|---|---:|---:|---:|
| Random | 0.624260 | 0.543301 | - |
| LCMD | 0.479539 | 0.408614 | - |
| scalar Kernel-IVR | 0.500070 | **0.395913** | 1/5 |

因此 Kernel-IVR 的正确解读是：它在后期和终点有潜在优势，但没有改善整条学习曲线。它不应直接进入 COMPOUND 或大规模参数 sweep；下一步应先解决表示和数值机制，再用一个严格匹配的三臂 sequential 实验验证。

## 已有证据说明了什么

LCMD 的优势主要集中在早中期。它在约 493 个 active labels 达到 Random@1005，在约 651 个 active labels 达到 N80，但在 1005 内没有稳定达到 N90。Kernel-IVR 在约 491 个 labels 达到 Random@1005，约 815 个 labels 达到 N80，约 953 个 labels 达到 N90；它牺牲了前中期，换来较低的终点误差。五个 seed 的 IVR 相对 LCMD AULC 差值分别为 `-0.0344, +0.0412, +0.0541, +0.0252, +0.0166`，只有 seed 157 胜出。

这更像是 acquisition objective 与真实 scratch-retrain 误差下降不完全匹配，而不是单纯的工程实现错误。当前 IVR 的每轮梯度提取和条件更新都通过了已有数值测试；已保存的 trace 也显示 batch 内确实做了条件化，而非一次性取 top-32。

## 缓存审计发现

对五个 seed、五个代表性 round 的 512D gradient cache 做了谱审计。该审计只使用输入和梯度特征，不使用 test truth 选择参数。

* 协方差在 `1e-12 * max eigenvalue` 阈值下仍有 512 个正方向，但谱非常集中：effective rank 约为 `17.7–30.1`，participation rank 约为 `5.3–9.2`。
* 原始特征协方差的条件数约为 `3.0e4–3.8e5`；这说明 512D 里大部分方向是弱能量方向，不能把 512 当成 512 个同等可靠的信息方向。
* IVR 实际使用的 `I + X_L^T X_L` precision 条件数约从 round 1 的 `256` 上升到 round 20 的 `939`。它尚未表现为数值崩溃，但正则化已经成为选择行为的重要组成部分。
* IVR 与 LCMD 的 batch overlap 从首轮约 10.6/32 降到后期接近 0。两者确实在采不同区域，不是重复实现同一 selector。
* IVR 每个 batch 的 integrated-variance reduction 逐轮衰减；它与下一轮 observed NRMSE 改善的 pooled Spearman 相关约为 `0.34`，说明 surrogate 有信号但不是可靠的逐轮误差预测器。

这个结果支持先做 `sketch/conditioning/regularization audit`，但不支持直接宣布 512D CountSketch 是失败原因。precision 的数值状态仍可控，真正的风险是低有效秩和 surrogate 与神经网络重新训练之间的失配。

## 方法学判断

1. 当前 `CountSketch(concat(g_V1, g_V2))` 是一个有效的 row-level similarity feature，但不能作为精确的双输出 posterior。实现里的 `countsketch_mapping` 还为 V1/V2 的两段参数分别生成了独立 bucket/sign 映射；这对标量 concat kernel 是可以解释的，却无法恢复要求共享 parameter-space projection 的跨 endpoint inner products。一个实验同时观测 V1 和 V2，严格的更新应使用两个 endpoint 的 Jacobian block 和 2x2 observation covariance。
2. `prior_precision=1` 和 `noise_variance=1` 是固定假设，不是从当前 L0 数据估计的事实。它们会影响弱特征方向的权重，尤其是在有效秩很低时。
3. BAIT 的 forward-backward 2B 选择是已有文献方法，不应作为全新算法命名。它适合做 batch optimizer 的小型 mechanism audit，但不能代替多输出 representation 修正。
4. VTL/ITL 类方法把候选集合和目标预测集合分开，并在 batch 内用 conditional embeddings 保持多样性。这个思想与本项目的 fixed outer reference 相容，但其理论假设主要是 GP/RKHS；这里应称为 task-adapted surrogate，不应声称对 scratch-retrained QGeoGNN 有理论保证。

## 推荐的实验 gate

### Gate A：Kernel stability / representation audit，零新训练

对 round `1, 5, 10, 15, 20` 和所有五个现有 seed，重新计算或恢复：

* sketch dimension `256/512/1024/2048`；
* 至少三个固定 CountSketch seeds；
* eigenvalue spectrum、effective rank、participation rank、precision condition number；
* candidate score 的 Spearman correlation 和 top-32 overlap；
* IVR 与 LCMD 的 overlap；
* 原始 concat feature 与分 endpoint feature 的相似度。

这一步的选择规则只允许使用输入、梯度和预先冻结的数值稳定性阈值，不读取 test truth。若排名随 sketch seed 或 dimension 大幅变化，先修 representation；若排名稳定而曲线仍不改善，问题更可能是 IVR objective 与 scratch retraining 的偏差。

### Gate B：selection-only mechanism audit

保持模型、梯度、L0 scales、B=32 和 `I` noise 假设不变，只比较：

* scalar conditional IVR；
* BAIT-style forward `2B` + backward remove `B`；
* multi-output IVR：同一 parameter-space random projection，分别保留 `phi_1` 和 `phi_2`，使用 rank-2 Woodbury。

必须对小矩阵验证 Woodbury/Sherman–Morrison 与直接重算一致，检查 batch 内重复率、score monotonicity、端点贡献和运行成本。不要在这一阶段用 one-step test NRMSE 选实现。

建议优先顺序是 **multi-output IVR > BAIT-FB**：前者修正了当前 acquisition 的科学近似，后者只改变 batch optimizer。BAIT-FB 仍应保留为低成本对照，因为它能判断当前 forward greedy 是否造成 batch 质量损失。

### Gate C：一个匹配的三臂 sequential

只有 Gate A/B 没有数值或实现问题时，才跑：

`Random vs existing LCMD vs one pre-registered improved IVR`

固定 `5 seeds, L0=333, B=32, 333→1005, scratch retraining, identical initialization/training config, global test barrier`。Random 和 LCMD 可复用已有 frozen controls，但新方法必须使用新的 study directory 和 protocol。主报告同时给出 normalized AULC、逐 seed paired AULC、N80/N90/N95、labels-to-target、终点 V1/V2 RMSE、持续 crossing 和 acquisition/training cost。

Gate C 的建议通过条件：相对 LCMD 的 mean AULC 不劣、至少 4/5 seed 的 paired AULC 不明显退化、N80 不推迟，并且终点两个 endpoint 均无超过预注册容许回归。若只改善 NRMSE@1005 而 AULC 仍更差，应把它记录为 late-stage candidate，不把它升级为 row 主方法。

### Gate D：COMPOUND/OOD

只有某个改进方法在 ROW 上有清晰 label-efficiency 价值后再进入 COMPOUND。COMPOUND 只比较 Random、Raw Gradient LCMD 和最终胜出方法。若 gradient-only 表示在 OOD 上不足，再单独冻结 `K_gradient + alpha K_chemical`；优先使用 Morgan/ECFP、RDKit physicochemical descriptors 或固定 molecular embedding，暂不重复加入高度相关的普通 QGeoGNN latent。

## 暂不建议投入

暂不扩大 Center/Width、q10-q90 width sampling、ensemble、UCB/EI/TS、RL、lambda/noise 大 sweep 或手工固定的 LCMD→IVR 切换。现有 headroom 检查已经说明简单切换的潜在 AULC 收益很小；在 IVR 的 output structure、conditioning 和 objective 尚未验证前继续横向加方法，难以解释结果。

## 文献依据

* Holzmüller et al., *A Framework and Benchmark for Deep Batch Active Learning for Regression*, JMLR 24(164), 2023：sketched full-network gradient kernels、kernel transforms 和 LCMD 框架。
* Ash et al., *Gone Fishing: Neural Active Learning with Fisher Embeddings*, 2021：BAIT 的 Fisher objective、multi-output regression Fisher，以及 forward-backward `2B→B` batch selection。
* Hübotter et al., *Transductive Active Learning: Theory and Applications*, NeurIPS 2024：VTL/ITL 的 target-space uncertainty reduction 和 conditional batch embeddings；理论与本项目的 scratch QGeoGNN 之间有适用边界。
* Bickford Smith et al., *Prediction-Oriented Bayesian Active Learning*, AISTATS 2023：强调预测目标信息，而不仅是参数信息，支持本项目把 fixed outer reference 纳入 acquisition objective。

## 决策

当前下一步不是直接跑完整新 sequential，而是完成 Gate A，然后实现并审计 multi-output IVR 与 BAIT-FB 的 selection-only 版本。根据审计结果只选择一个候选进入 Gate C。这样可以把“IVR 后期可能有效”“512D 是否稳定”“双输出是否需要真实保留”三个问题分开回答。
