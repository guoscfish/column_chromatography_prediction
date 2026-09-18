# 4g QGeoGNN 主动学习：代码、实验与近期文献评估

研究日期：2026-09-17。预注册代码基线：`a419ea75e7894004aef393cb3734a7ae5e1356a1`；最新正式结果提交：`1e8fe69`，分支 `exp/qgeognn-v2-4g-row-al`。CW 结果另核对本地工作树 `/private/tmp/qgeognn_cw_performance`，提交 `6ac0f9e`。

本次工作是研究评估：阅读已有代码、结果、运行审计与公开文献；没有启动训练、调用 test reveal、修改 frozen protocol 或更换分支。研究期间其他运行任务完成并提交了 sequential 正式结果，本报告已纳入这些已发布结果。以下新实验均为建议，不是已执行结果。

## 1. 核心判断

**现有正式 sequential 结果已经支持 LCMD/Hybrid 在这个 row benchmark 上节省标签。下一步最值得投入的是突破约 650 标签后的性能平台，稳定接近 N90/N95，并单独减少反复训练成本。**

LCMD/Hybrid 的整段 AULC 比 Random 低约 23%，且两者均在 5/5 seeds 上优于 Random。按预注册插值，LCMD 用约 472 个标签达到 Random@1005 的水平；两种主动方法用约 650 个标签达到 N80。然而 LCMD 在本预算内未达到 N90，Hybrid 只有一次短暂穿越 N90，随后回退。因此“主动学习有效”与“稳定接近 full-data”现在是两个不同的证据问题。

建议优先级：

1. 以已完成的 sequential study 为基准，准确解释自动分类、不同目标的标签节省和后期平台，并补齐计算成本口径。
2. 后续只新增一个核心候选：现有 Raw Gradient kernel 上的、以当前已标注数据为条件的 batch integrated variance reduction，简称本文的 Kernel-IVR。这是已有实验设计思想的任务适配，不预先宣称新算法。
3. 独立比较 scratch 与经过验证的 warm-start/replay，在相同标签轨迹上先研究计算效率，再研究闭环选点是否受影响。
4. 有结果支持后，再考虑批量大小、目标分布权重和实验成本。暂不扩大 CW、sketch 维度、whitening 或混合 acquisition 的变体矩阵。

## 2. 实际状态比对话整理更新

对话整理停留在 `PENDING_FORMAL_RUN`，当前正式结果已在 `1e8fe69` 提交。完整矩阵为：

| 种子 | Random | Hybrid | LCMD | matched full-data reference |
| --- | --- | --- | --- | --- |
| 157 | 22 点，终点 1005 | 22 点，终点 1005 | 22 点，终点 1005 | 已冻结并评价 |
| 887 | 同上 | 同上 | 同上 | 已冻结 |
| 2357 | 同上 | 同上 | 同上 | 已冻结 |
| 6101 | 同上 | 同上 | 同上 | 已冻结 |
| 12203 | 同上 | 同上 | 同上 | 已冻结 |

所有 15 个 `trajectory_freeze.json` 均记录冻结时 `test_truth_access_count=0`；五个 full-data reference 也齐全。顶层全局冻结文件包含 330 个曲线点和 5 个 references。之后生成了正式 learning curves、labels-to-target、AULC 和 `decision.json`。冻结文件中的 0 是冻结时状态，不应误读为现在仍未进行最终评价。

无需重新启动这些轨迹。接下来应正确解释已经产生的正式数据，尤其不要只看 `NO_CLEAR_SEQUENTIAL_AL_GAIN` 这个分类名称就判定主动学习无效。

## 3. 已完成实验究竟说明什么

### 3.1 必须分开三个证据层次

| 实验 | 样本/预算 | 已核对结果 | 可支持的结论 |
| --- | --- | --- | --- |
| 大批量 LCMD pilot | development 5 seeds；333→666 | Random 0.644425，LCMD 0.440213；均值相对改善 31.69% | 大批量一次选点存在明显有效信号 |
| B32 primary confirmation | confirmation 5 seeds；333→365 | Random 0.694920，uncertainty 0.675275，CoreSet 0.673834，LCMD 0.663146 | LCMD 是当前最强的小批量 row acquisition 证据 |
| B16 sensitivity | 同一 confirmation cohort；333→349 | mean per-seed improvement 为 -0.57% | LCMD 优势依赖预算与模型状态 |
| Hybrid matched extension | confirmation 5 seeds；333→365 | Hybrid 0.672660；仅 3/5 胜 Random median | 均值改善，但未过原定稳健性门槛 |
| CW performance | development 5 seeds；333→365 | Raw 0.708760，CW 0.728810；CW 胜 2/5 | CW 第一轮即时收益未优于 Raw |
| sequential | established confirmation 5 seeds；333→1005 | AULC：Random 0.624260，Hybrid 0.479354，LCMD 0.479539；两个主动方法均胜 Random 5/5 | 完整曲线支持标签效率；后期接近全数据仍有瓶颈 |

这里 NRMSE 越低越好。confirmation B32 的“均值之比改善”为约 4.57%；原报告的 4.81% 是各 seed 相对改善的平均，二者不是同一个统计量。CW 和 Raw 的 development 数值也不能拿来与 confirmation 的 0.663146 直接排名。

CW 的负信号不排除曲线交叉，但目前没有值得立即支付完整 CW sequential 训练成本的正证据。B64 以前是用户取消，不能当成失败或零效果。

### 3.2 Full-data 上限与泛化范围

独立 predictor qualification 的 row test 平均 V1/V2 RMSE 为 2.919/5.525 mL；compound split 为 5.608/10.980 mL。它说明“同一分子家族/条件空间内补数据”和“推广到未见分子”是明显不同的难度。

这些 qualification seeds 为 42/525/1101，归一化也不是当前 L0-fixed 方式，不能把其中 0.359254 的 NRMSE 直接作为本轮 N90 阈值。当前 matched full-data reference 的 cohort mean NRMSE 为 0.370065，V1/V2 RMSE 为 3.001/4.244 mL，才是本轮同 split、同尺度的对照。

当前 row benchmark 支持的是既有数据分布内的实验行预测。它不自动支持未知化学骨架、任意新溶剂体系、未经筛选的长尾样本或新柱规格。

### 3.3 最新连续曲线：有效性已经成立，下一步是后期平台

| 指标 | Random | Hybrid | LCMD |
| --- | ---: | ---: | ---: |
| Mean normalized AULC | 0.624260 | 0.479354 | 0.479539 |
| AULC 相对 Random 改善 | 基线 | 23.21% | 23.18% |
| AULC 胜 Random 的 seeds | — | 5/5 | 5/5 |
| 达到 Random@1005 的插值标签数 | 1005，按协议固定 | 504.47 | 472.33 |
| 达到该阈值的首个实际预算点 | 941，描述性实际 crossing | 525 | 493 |
| 预注册 incremental label saving | 0 | 74.48% | 79.27% |
| N80，插值 | >1005 | 643.32 | 651.19 |
| N80，首个实际预算点 | >1005 | 653 | 653 |
| NRMSE@1005 | 0.543301 | 0.409709 | 0.408614 |
| 终点关闭 initial-to-full gap | 48.55% | 88.23% | 88.55% |

主目标阈值为 0.543301；N80 阈值为 0.437402。Random 采用一条新的固定 permutation trajectory，与此前 one-step 的五个 Random controls 均值不是同一条曲线，不能拼接旧 B32 数值。

对用户关心的 N90/N95，本报告只在已公开 cohort curve 上按给定 gap 公式做**事后描述性补算**，不改正式判定：

- T90 = 0.4037335。LCMD 和 Random 在 1005 内均未达到。Hybrid 首次插值 crossing 为 945.69，对应实际点 973；1005 时反弹到 0.409709，未稳定保持。
- T95 = 0.3868991。三个方法均未达到。

LCMD 在 461–621 预算段明显优于 Hybrid；653–973 期间 Hybrid 的 cohort mean NRMSE 每个点都略低于 LCMD；终点 LCMD 再次略低。两条曲线确有交叉，AULC 总差仅约 0.0386%，不足以支持稳健的方法优劣排序。

LCMD 从 333→653 将 NRMSE 从 0.706753 降到 0.436791，再增加 352 个标签到 1005 只降到 0.408614。这里是目前最明确的后续研究瓶颈：低预算优势已经转化为 sequential 收益，但稳定达到 N90/N95 仍未解决。

### 3.4 为什么自动结论是 NO_CLEAR_SEQUENTIAL_AL_GAIN

这次结果进入了冻结分类规则的剩余分支，而不是“没有对 Random 的收益”：

1. LCMD 未满足“mean AULC 低于 Hybrid”，所以不能判 LCMD best。
2. Hybrid 未满足“labels-to-T_R30 低于 LCMD”，所以不能判 Hybrid best。
3. 两者 AULC 满足相近条件，但插值标签差为 32.14335，大于原定 32，差 0.14335，因此不能判 comparable。
4. 最终落入 `NO_CLEAR_SEQUENTIAL_AL_GAIN`；同一 JSON 的 `active_gain.hybrid` 和 `active_gain.lcmd` 都为 `true`。

必须保留正式分类结果和原门槛，不事后把 32 改大。科学表述应同时写清：**本 cohort 上两种主动策略均明显优于 Random，但冻结规则未确定两者的最佳/相近分类。** 这是类别命名及覆盖范围的局限，不应把它传播为“主动学习失败”。未来协议可把“是否优于 Random”与“主动方法之间如何排名”设计为两个独立判断。

## 4. 代码阅读带来的关键发现

### 4.1 现有 acquisition 已经不是一个弱基线

`gradient_features.py` 对当前模型的 V1/V2 点预测输出求完整参数 Jacobian，并分别除以 L0 标准差，然后将两个梯度块的拼接 CountSketch 到 512 维。189 个可训练张量、458,952 个参数；不是 loss gradient，也不是只看最后一层。

`lcmd.py` 的 TP 语义是先安装全部当前 L_t centers，以每个 cluster 未覆盖平方距离之和选最大 cluster，再选其最远点。`sequential_acquisition.py` 和 `sequential_runner.py` 每轮都重新训练、重新计算当前梯度、移除已选行。

这与 Holzmüller 等 JMLR 2023 的有效设计方向一致 [1]。因此继续换距离/表示，预期边际收益应有更高证据门槛。

### 4.2 LCMD 还没有显式建模“标注后能降低多少预测方差”

LCMD 把已标注数据当作几何 centers。它没有由 L_t 的累计信息形成 posterior covariance，也没有计算候选样本与目标预测区域的条件协方差。离已标注样本很远，不等于对整体预测最有用；反过来，一个不算最远但能约束许多目标预测的样本可能被低估。

这是比 CW 缩放更实质的改进空间。但 posterior/IVR 也可能因局部线性化失准而失败，需要性能验证。

### 4.3 梯度范数存在强烈的预测保留体积倾向

已发布的无标签审计中，梯度范数与预测 V2 的平均 Spearman rho 为 0.8931，与预测 V1 为 0.7668，与最近 L0 梯度距离为 0.6397。LCMD 选中样本的梯度范数和预测洗脱体积均高于 Random。

这可能恰好服务于 RMSE 的长尾误差，也可能使后期过度关注高体积/高噪声区域。现有相关性不能区分这两者，更不能据此立即去掉 norm。已发布曲线显示后期进步放缓，值得进一步描述改善来自何种 error strata，但不能用 test 分层结果调本轮方法；由此设计的后续策略也必须承认开发数据已被使用。

### 4.4 不要把 q10–q90 宽度直接当成可消除的不确定性

`training/predictor.py:75` 的中间输出实际上用 MSE 训练，虽然名称叫 q50，但它并非严格按 0.5 pinball loss 定义的条件中位数。两侧分位数的区间宽度混合了条件噪声、模型误差与校准问题。

训练损失在原始体积单位上按两个端点相加，而 acquisition 和 checkpoint selection 使用 L0 标准差归一化。这些目标并不完全一致；后续解释 IVR 时应承认它优化的是风险 surrogate，不能直接宣称等于实际六输出网络的训练收益。改变训练损失属于独立 predictor 研究，当前不混入 acquisition 对照。

已有完整模型的 row 区间覆盖率约为 V1 72.2%、V2 69.1%，低于 q10–q90 对应的名义 80%；compound V2 约 46.3%。这些是历史完整模型的审计，不是本轮 L0 的直接 UQ 质量估计。

所以“区间越宽越值得采”缺少必要前提。ensemble disagreement、局部参数后验方差和观测噪声必须区分；单纯扩大 ensemble 还会增加训练成本。

### 4.5 重复输入既可能是冗余，也可能是实验噪声证据

历史 3330 行 outer train 只有约 3103–3113 个不同实际输入，精确梯度重复均由相同输入解释，没有观察到不同 X 的精确 sketch collapse。LCMD 没有在大批次重复采已覆盖输入，Random 约 0.90%–3.60% 是这类冗余。

目前不能据此认为重复数据应一律删除。若是文件复制，应去重；若是独立实验重复，它们能估计噪声，甚至值得在某些区域重复测量。应在已有可用标签内先查实验 provenance 和重复测量方差，不能提前读取整个 U_t 的真实结果来设置噪声权重。

### 4.6 正式目标与讨论中的目标尚未完全一致

`sequential_reporting.py:178` 实现的只有：

- 主目标 `T_R30`：cohort mean Random@1005 的误差。
- 次目标 `T_80 = E_full + 0.20(E0-E_full)`。

没有实现/预注册 N90、N95。本报告 3.3 的补算明确属于辅助描述，而非替换成功标准；未来协议可把 N80/N90/N95 以及工程上可接受的绝对 mL 误差预先写入。

还有三个解释边界：

1. 代码将 Random 达到 T_R30 的标签数固定为 1005。本次 Random 实际在约 940.31 插值标签、941 实际预算点已低于自己的终点。因此正式 saving 是“相对固定 Random@1005 预算”，并非随机策略真正首次达到同一误差所需标签的差值。按实际预算点，LCMD 493 相对 Random 941 少 448 条；相对固定 1005 少 512 条。口径不同，均不能混写为 532.67 条真实已执行节省。
2. 当前是 first crossing，加相邻点线性插值。曲线反弹时一次穿越未必稳定；小数标签数也不是实际执行的实验数。未来可预注册持续穿越，且同时报告下一实际预算点，不能事后选更好看的规则。
3. cohort mean crossing 不是每个 seed 的 crossing 的平均；五个重叠 split 也不是五个独立化学数据集。要保留逐 seed 曲线、配对差异、未达目标的删失状态，避免给出过强普适性声明。

`N90` 是离线 benchmark 指标：真实部署时全数据模型和 E_full 通常未知，因此不能把它直接当在线停止规则。在线停止更适合独立监测集或预设的工程误差容限。

## 5. 计算成本已经提供了独立研究方向

读取 15 个完整 `fit_audit.csv`，只统计正式轨迹列出的拟合，避开历史中断残留，得到以下累计数据。计时字段包含训练、验证及最终预测相关操作；它们是累计任务耗时，不是同时运行任务的自然日历耗时，也不是 GPU-hours。

| 方法，5 seeds 合计 | 模型拟合次数 | 累计 epochs | fit_audit 计时之和 | 另计梯度提取 |
| --- | ---: | ---: | ---: | ---: |
| Random | 110 | 22,559 | 19.18 h | 无 |
| Hybrid | 320 | 82,123 | 56.18 h | 无 |
| LCMD | 110 | 28,339 | 16.06 h | 3.39 h |

Hybrid 每 seed 为 22 次评估拟合 + 21×2 次额外 ensemble 拟合，即 64 次，不是每个点都三模型。它需要显著标签收益才足以抵消计算代价。不同运行负载会影响 wall-clock，因此不能把 LCMD 拟合计时小于 Random 解读为算法训练更快。

发现一个报告口径缺口：`sequential_runner.py:988` 的 `runtime_rows` 只取 member 0，未把额外 210 次 ensemble 拟合的 epochs/fit time 纳入该字段；`acquisition_seconds_included_in_round` 实际写入整个 round elapsed time，不能再与训练时间简单相加，或当成纯 acquisition time。正式成本分析应从完整 fit audit 和 gradient audit 单独重建，保留原始报告及冻结记录。

当前模型只通过 CPU runtime qualification。训练集预算不超过 1005，而 minibatch 为 2048，大部分 AL 轨迹每 epoch 只有一个参数更新；全数据 3330 行则有两个 minibatch。因此累计 epochs、optimizer steps、处理样本数、预测/验证耗时均有不同含义，不能只用 epochs 当唯一计算成本。

## 6. 文献筛选与结论

检索覆盖截至 2026-09-17 的公开来源，重点为 2024–2026 的回归/批量/目标分布方法，并保留直接相关的基础论文。下表按对本项目的用途排序，而非单按年份或期刊名排序。搜索引擎摘要不作为方法结论；NeurIPS 网站的当前品牌年份也不等于论文发表年份。

| 文献与来源 | 核验深度 | 核心价值 | 对 4g 的适用边界 |
| --- | --- | --- | --- |
| [1] Holzmüller et al., JMLR 2023, *A Framework and Benchmark for Deep Batch Active Learning for Regression* | 期刊摘要、方法结构与本地实现/既有审计对照 | sketched full-network gradient kernel、LCMD、kernel transform 与 selector 的统一框架；15 个大型表格回归数据集 | 当前方法有强先验基础，但原基准不是小样本柱色谱 GNN |
| [2] Hübotter et al., NeurIPS 2024, *Transductive Active Learning: Theory and Applications* | arXiv 正文的设置、VTL/ITL、神经网络 batch 方法与理论条件；Crossref 确认会议年 | 明确区分可采样集合与目标预测集合，按目标不确定性降低来采样；batch 内条件化以减少重复信息 | 定理依赖 GP/RKHS 等假设，不能直接保证每轮从头训练的 QGeoGNN；方差目标也不能无条件宣称 submodular |
| [3] Takeno et al., ICML 2025, *Distributionally Robust Active Learning for Gaussian Process Regression* | 正式 PMLR 267 目录；正文 related work、实验与 limitations | 目标分布未知时优化候选分布集合中的最坏期望误差；说明 variance 与 entropy 目标并不等价 | 实验主要是 GP、合成函数及表格数据；不意味着 4g 应立即增加复杂的 distributionally robust 求解器 |
| [4] Min et al., NeurIPS 2025, *Enhancing Deep Batch Active Learning for Regression with Imperfect Data Guided Selection* | 正式会议论文页面及作者摘要 | AGBAL 用分布校正后的辅助数据损失帮助估计候选价值，补足纯 sensitivity 的局限 | 本次未完整阅读全文；需要有标签的辅助数据和 density-ratio 假设，不能免费使用 8g/25g/40g 或验证集结果 |
| [5] Bickford Smith et al., AISTATS 2023, *Prediction-Oriented Bayesian Active Learning* | 原始论文摘要和 [2][3] 对其目标的讨论 | EPIG 强调获得预测信息，不仅是参数信息 | 概念上非常契合；entropy/log score 与当前 RMSE 并非完全一致，优先验证 IVR 更直接 |
| [6] Ash et al., 2021, *Gone Fishing: Neural Active Learning with Fisher Embeddings* | 原始论文摘要 | BAIT 以 Fisher 信息近似优化估计误差，在分类与回归中验证 | 是必要的相关工作；新方案若与其 criterion 等价，应承认是适配，不重复发明名称 |
| [7] Ash & Adams, NeurIPS 2020, *On Warm-Starting Neural Network Training* | 原始论文摘要与发表元数据 | 直接 warm-start 可在相近训练损失下产生泛化落差；shrink-and-perturb 是重要既有方向 | 非新论文，但对“增量训练必然更好”这一工程假设有直接警示意义 |
| [8] Yin et al., RSC Advances 2026, *Out-of-distribution evaluation of active learning pipelines for molecular property prediction* | Crossref 摘要及出版元数据，出版社全文有浏览器验证 | 分子性质 AL 应专门评价 OOD；该文报告 evidential AL 对未见化学空间的收益 | 相关应用旁证，不属于本报告优先级最高的方法学证据，也不支持直接照搬 evidential head |

另筛查了 2026 预印本：*A Mutual Information Lower Bound for Multimodal Regression Active Learning* [9]、*Active Learning for Gaussian Process Regression Under Self-Induced Boltzmann Weights* [10]、*The Approximation Ratio for the Risk of Myopic Bayesian Active Learning for Linear Regression* [11]。它们分别讨论多峰预测、由函数诱导的目标分布、短视 greedy 的理论限制。目前没有足够依据认定 4g 需要 mixture density head、Boltzmann 权重或长视规划；检索时未核实这些论文已有正式顶会录用，因此只列为观察项。

没有找到并核实可直接证明“某个 2025/2026 新方法一定优于本项目 LCMD”的柱色谱证据。高质量相邻领域结果应转化为可检验假设，而不是性能承诺。

## 7. 最值得尝试：Raw Gradient Kernel + 条件化 Batch-IVR

### 7.1 与 LCMD 的实质差别

用现有 512D 特征 `phi_t(x)` 构造行级核。给定 L_t，建立正则化线性/核 surrogate 的条件协方差。候选 x 的价值定义为：在目标参考集 R 上，观察它以后能减少多少加权预测方差。

标量 surrogate 下可写为：

\[
a_t(x)=\sum_{z\in R}w_z\,
\frac{k_t(z,x)^2}{k_t(x,x)+\sigma^2(x)}.
\]

其中 `k_t` 是已经以 L_t 为条件的 posterior kernel，而不是未经条件化的原始内积；`w_z` 是预先确定的目标权重。噪声项说明：总不确定性很高但主要来自测量噪声的点，未必具有同样高的可学习价值。

在有限特征形式下，可令：

\[
C_t=(\lambda I+\sum_{i\in L_t}\phi_i\phi_i^\top/\sigma_i^2)^{-1},\quad
M_R=\sum_{z\in R}w_z\phi_z\phi_z^\top,
\]

\[
a_t(x)=\frac{\phi_x^\top C_tM_RC_t\phi_x}
{\sigma_x^2+\phi_x^\top C_t\phi_x}.
\]

选择 batch 时，每选一个 x 就更新条件协方差，再算剩余候选的边际价值；不能只在初始分数上取 Top32。固定 Gaussian surrogate 内，这个协方差更新不需要未知标签。得到真实新标签后，下一轮仍重新训练 QGeoGNN、重新提取当前特征。

512D 允许在特征空间求解，避免对每个候选重新训练或建立百万维完整 Fisher。数值上使用 Cholesky/线性求解和稳定更新；是否真的更便宜仍需测量。

### 7.2 一个容易被忽略的多输出问题

现有特征是 `sketch(concat(g_V1/s1,g_V2/s2))`。它的内积近似两个同端点梯度内积之和，是有效的行相似度核，但**不能不加解释地把这个标量核叫作 V1/V2 联合观测的精确 Bayesian posterior**。

一个真实实验同时观测两个端点。严格的多输出 IVR 应使用 2×2 的候选协方差块、两端点的交叉协方差及测量噪声矩阵，并把整个实验作为不可拆分的 acquisition 单位：

\[
\Delta(x)=\sum_{z\in R}w_z\,
\mathrm{tr}\!\left[D\,K_t(z,x)
(K_t(x,x)+\Sigma_\epsilon(x))^{-1}K_t(x,z)\right].
\]

`D` 体现预先固定的端点误差权重。若使用已标准化输出，则不应再次重复缩放。现有合并 sketch 无法无损恢复全部跨输出信息。

**建议第一版明确称为“基于现有行级梯度核的 IVR surrogate”**，保持改动最小；这是近似 acquisition hypothesis，而非精确误差下降保证。只有该路线显示真实收益后，才讨论分端点 Jacobian/block covariance 是否值得增加成本。后者应另行冻结定义，不与第一版混成无边界的变体搜索。

### 7.3 参考分布怎样选

对于现有 row 风险，推荐预先固定整个 outer-training 输入集 X_outer 的均匀经验分布，作为 R；即使一些行后来被标注，它们仍代表部署分布。它只使用允许访问的输入，不使用未揭示输出。

不推荐直接把历史测试输入作为 R，这会把普通 held-out evaluation 改成 transductive test-covariate access。即使不读取 test label，也改变了协议。

若未来目标是“每个化合物同样重要”，可设 compound-balanced 权重，且同时采用 compound-balanced 评价。如果仍按实验行均匀计算误差，随意改成 compound-balanced acquisition 可能损害主指标。分布鲁棒方法 [3] 更适合在部署分布确有不确定性时作为后续方案。

### 7.4 第一轮最小实验矩阵

后续另行冻结一个连续实验，只比较 Random、现有 LCMD、一个 Kernel-IVR 候选。可在完全相同且通过合约核验的历史设定下复用已有两条 baseline，明确这是后续探索，不能把已用 cohort 重新称作独立确认。

保持预测器、L0、固定 validation、初始化、scratch 训练、输出尺度、B32 与预算网格匹配。仅在 L_t 内部通过预定规则估计正则化/噪声，或者冻结无输出的尺度规则；不要新增一个未计价的辅助标签集，也不要遍历一组参数后用 test 挑最好。

完整轨迹优先到已有 1005 ceiling。若 N90/N95 未达到，报告 `>1005`，不能外推或因看到 test 未达标临时延长。本轮允许曲线交叉；提前终止只依据数值错误、泄漏/合约失败或预先设置的计算预算，不以某个早期 test 点淘汰。

预期成立的判据应是更低的目标标签数、跨 seed 方向稳定、端点误差不失衡，且额外选点成本可接受。Overlap 只做实现审计。若 IVR 只改变选点但完整曲线不改善，先停止该方向，检查 surrogate 与实际误差的关系，不继续自动加 CW、whitening 和大 ensemble。

MaxDet/BAIT 是既有相关方法。若后续要发表方法学论文，需要选取真正不同且必要的强对照；若 IVR 与某个 BAIT/V-optimal 形式等价，应说明等价关系，避免仅换名制造多条 baseline。当前不必一次把全部方法都加入正式矩阵。

## 8. 第二条路线：降低训练与交互成本

### 8.1 先在固定标签轨迹上比较训练方式

复用正式已冻结的标签顺序，在同一个 N 比较：scratch；从上轮模型继续训练并使用全部 L_t 的 warm-start。这样首先回答“相同标签能否更便宜地达到相同性能”，不让 acquisition 变化干扰判断。

必须重新加载当前全部已知数据，不能只用最新 32 个样本训练。学习率、优化器是否重置、训练步数、验证频率和 checkpoint 规则均提前固定。以 validation/开发规则控制，不能在 test 上动态切换 scratch/warm-start。

若直接 warm-start 有泛化损失，再依据 [7] 在单独开发阶段评估一个 shrink-and-perturb 或定期重启方案；不同时展开多参数扫描。固定轨迹成功后，才闭环重算 acquisition，因为 warm-start 模型会改变 gradient geometry。

### 8.2 批量调度也应单独比较

固定 +32 要 21 次 acquisition。为减少交互，可以后续比较一个预先定义的更大批量日程，例如早期 B32、后期 B64，并保持可比的总标签终点；这个示例是工程假设，不是已有论文或当前结果证明的最优日程。

批次越大，选点过程中模型越陈旧；批次越小，重训和实验往返越频繁。B16 已无优势，说明不能默认“小批更主动就一定更省标签”。应比较到目标时的 `(labels, rounds, compute, lab time)`，展示折中关系，而非强行合为一个随意加权分数。

若 batch 有共同换溶剂/准备样品开销，实验成本还可能具有 setup sharing。只有设备时间、耗材和可行约束有数据时，才引入 cost-aware batch design。简单以预测 V2 除 acquisition score，容易系统性回避本来最影响 RMSE 的长保留样本。

### 8.3 固定 validation 成本值得后续研究

当前初始已观察非测试标签为 `333+416=749`，不是 333；B32 后为 781；终点为 1421。比较方法时，416 是共享常数，不影响同协议下的绝对标签差值，却会改变节省百分比和真实部署预算。

未来可在另一个协议中研究更小的固定监测集、冻结训练日程或成本受控的模型选择。不要直接将这 416 行改作 acquisition 的“免费目标标签/辅助数据”，也不要把 validation reduction 与新的 acquisition、warm-start 同时改掉。

## 9. 目前不优先投入的方向

- 更多 CW scaling、direction-only、latent/gradient selector 交叉消融：优先服务机制说明，尚无足够新 label-efficiency 证据。
- 直接按 q90−q10 采样、单纯扩大 ensemble：没有区分可消除不确定性与噪声，且额外成本明确。
- AGBAL 式跨柱辅助监督：有 2025 顶会依据，但柱规格存在 domain shift，必须计入辅助标签成本、比较相同辅助监督的 baselines。可作为后续单独研究，当前纯 4g 主线不混入。
- 强化学习 acquisition、长视规划、生成式实验设计：当前只有一个小型数据集，策略学习额外监督与计算代价可能超过收益。
- 用 EI/UCB 直接最大化保留体积或其他响应：那是优化某个实验结果，与学习整片空间的预测函数是不同科学目标。
- 直接换 foundation model/预训练骨架后声称 acquisition 改进：这是 predictor/representation 主线，需要重建其 Random、LCMD 和 full-data references，且检查预训练数据与 holdout 重叠。

## 10. 可执行的下一步顺序

| 阶段 | 工作 | 完成后回答什么 | 当前是否已执行 |
| --- | --- | --- | --- |
| A | 既有 15 轨迹 + 5 reference 的全局验证、按 frozen protocol 汇总 | LCMD 是否真的省标签，优势区间与平台在哪里 | 其他任务已完成并提交；本次已读取结果 |
| B | 在已公开曲线上做描述性 crossing、端点与成本审计 | 瓶颈主要是 acquisition、训练还是评价目标 | 本报告已完成主要解读及辅助 N90/N95 分析；细分误差机制尚未分析 |
| C | 冻结一个 Raw Kernel-IVR 定义，验证条件协方差和 batch 更新的数学/数值正确性 | 是否是可信的候选实现 | 建议，未实现 |
| D | 三臂匹配的完整 sequential 比较，明确探索性与确认性边界 | 是否降低 labels-to-target | 建议，未启动 |
| E | 固定标签轨迹的训练加速对照，再闭环验证 | 能否在保持标签效率时降低总计算成本 | 建议，未启动 |

新 split 或新随机 seed 只增加同一数据集上的稳健性证据，不会抹掉此前反复 test 暴露。最终较强的确认应来自新实验/时间留出数据，或从未用于设计的新化合物集合，并与预期部署范围一致。

比较 full-data-like performance 时，推荐未来同时报告 `N80/N90/N95`、normalized AULC、到目标的 acquisition 轮数、累计完整拟合成本和实际到达预算点。预算未达目标时保留删失；不能把 1005 预算内表现最好自动等同于已经接近全数据模型。

## 11. 本地证据索引

- [Sequential preregistration](../../studies/active_learning/qgeognn_v2_row_sequential_b32/PREREGISTRATION.md)。
- `studies/active_learning/qgeognn_v2_row_sequential_b32/{decision.json,global_pre_test_freeze.json,FINAL_REPORT.md}`。
- `studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_*/{random,hybrid,lcmd}/{trajectory_freeze.json,fit_audit.csv}`。
- `studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_*/full_data_reference/pre_test_freeze.json`。
- `studies/active_learning/qgeognn_v2_row_small_batch_benchmark/{FINAL_REPORT.md,AUDIT_INTERPRETATION.md}`。
- `studies/active_learning/qgeognn_v2_row_hybrid_extension/RESULTS.md`。
- `studies/active_learning/qgeognn_v2_row_lcmd/FINAL_REPORT.md`。
- CW 分支的 `studies/active_learning/qgeognn_v2_row_center_width_performance/FINAL_REPORT.md`。
- `studies/predictor/final_4g_qualification/{FINAL_4G_QUALIFICATION_REPORT.md,QUANTILE_AUDIT.md}`。
- `src/qgeognn_al/active_learning_v2/{gradient_features.py,lcmd.py,sequential_acquisition.py,sequential_runner.py,sequential_reporting.py,runner.py}`。
- `src/qgeognn_al/{models/qgeognn_v2.py,training/predictor.py}`。

## 12. 文献链接

1. Holzmüller, Zaverkin, Kästner, Steinwart. **A Framework and Benchmark for Deep Batch Active Learning for Regression**. JMLR 24(164), 2023. [期刊主页](https://www.jmlr.org/papers/v24/22-0937.html)；[作者实现](https://github.com/dholzmueller/bmdal_reg)。
2. Hübotter, Sukhija, Treven, As, Krause. **Transductive Active Learning: Theory and Applications**. NeurIPS 2024. [论文](https://arxiv.org/abs/2402.15898)；[本次阅读版本正文](https://arxiv.org/html/2402.15898v6)。2025 是所读 arXiv 版本更新时间，不是会议年份。
3. Takeno et al. **Distributionally Robust Active Learning for Gaussian Process Regression**. ICML 2025, PMLR 267:58339–58358. [论文](https://arxiv.org/abs/2502.16870)；[正文](https://arxiv.org/html/2502.16870v3)；[正式论文集](https://proceedings.mlr.press/v267/)。
4. Min, Xu, Li, Zou, Zhou. **Enhancing Deep Batch Active Learning for Regression with Imperfect Data Guided Selection**. NeurIPS 2025. [正式会议页](https://neurips.cc/virtual/2025/loc/san-diego/poster/121499)；[OpenReview](https://openreview.net/forum?id=i5rApSWC9E)；[公开代码](https://github.com/OswinMin/AGBAL)。
5. Bickford Smith et al. **Prediction-Oriented Bayesian Active Learning**. AISTATS 2023. [论文](https://arxiv.org/abs/2304.08151)。
6. Ash et al. **Gone Fishing: Neural Active Learning with Fisher Embeddings**. 2021. [论文](https://arxiv.org/abs/2106.09675)。
7. Ash, Adams. **On Warm-Starting Neural Network Training**. NeurIPS 2020. [论文](https://arxiv.org/abs/1910.08475)。
8. Yin, Gao, Panapitiya, Saldanha. **Out-of-distribution evaluation of active learning pipelines for molecular property prediction**. RSC Advances, 2026. [DOI](https://doi.org/10.1039/D5RA08055J)。
9. Guilhoto, Kaushal, Perdikaris. **A Mutual Information Lower Bound for Multimodal Regression Active Learning**. 2026，预印本状态按本次可核实信息记录。[论文](https://arxiv.org/abs/2605.14917)。
10. Qing, Moss, Sachs. **Active Learning for Gaussian Process Regression Under Self-Induced Boltzmann Weights**. 2026，预印本。[论文](https://arxiv.org/abs/2605.10654)。
11. Mussmann. **The Approximation Ratio for the Risk of Myopic Bayesian Active Learning for Linear Regression**. 2026，预印本。[论文](https://arxiv.org/abs/2607.06642)。

阅读限制：Chrome 连接因当前认证方式不可用；内置浏览器成功读取搜索及 NeurIPS 正式页面，但部分 OpenReview/RSC 页面要求验证，一次 NeurIPS 跳转异常停滞。后续使用公开 arXiv 正文、PMLR、Crossref 核验。未把无法阅读全文的文献描述为已经全文复核，也没有依据搜索引擎 AI 概览采纳理论结论。
