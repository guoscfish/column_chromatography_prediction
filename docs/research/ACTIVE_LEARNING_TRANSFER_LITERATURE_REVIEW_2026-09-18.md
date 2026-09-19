# 4g 主动学习与跨柱主动迁移：瓶颈及文献评估

日期：2026-09-18。阅读基线为工作区 HEAD `0e7b2ee` 及当前尚未提交的研究文档、Phase 0/1 代码与结果。本文件是研究建议，不是执行协议，也不替换 `docs/NEXT_STAGE_DECISION.md` 的项目状态。

本次核对了当前模型、训练损失、标签访问边界、梯度表示、LCMD、Hybrid、IVR、校准模型，以及已发布的主动学习、迁移、物理元数据和复现报告。没有启动训练、打开新的封存测试标签、改动模型或改写历史判定。以下建议均不代表已取得收益。

## 1. 最重要的判断

**课题 1 已经证明了分布内的标签效率，接下来需要研究后期精度、真实反馈价值和泛化范围。课题 2 的困难集中在迁移误差结构、数据覆盖与评价目标；主动选样本身仍可建立在简单而强的校准模型上。**

不建议把下一阶段定义成“再找一个在所有柱、所有划分、所有指标上获胜的复杂模型”。模型容量、选样方式、预算和应用场景应分别提出可证伪的问题。未通过项目级统一升级门槛，不等于所有局部科学信号都为零；局部信号也不能事后升级为独立确认。

| 问题 | 当前证据 | 对后续工作的含义 |
| --- | --- | --- |
| 主动学习是否比随机有效？ | 4g LCMD/Hybrid 的 NRMSE-AULC 均约改善 23%，各 5/5 个划分获胜 | 已经有正结果，不必继续重复证明同一个问题 |
| 能否更快接近全数据参考？ | LCMD 约 653 标签达到 N80，之后改善明显放缓 | 重点看后期同状态增量收益，不能仅做初始 B32 筛选 |
| 更好的终点是否等于更好的全过程？ | IVR 终点更好，AULC 更差 | 分别报告终点、曲线与达到目标所需预算 |
| 神经迁移是否普遍优于校准？ | 低标签匹配基准不支持；FULL-data 存在场景限定信号 | 校准是合理预测器；不要把神经模型胜出设为研究选样的逻辑前提 |
| 是否已经到实验噪声下限？ | 独立重复实验与批次信息不足 | 不能把平台直接称为不可约噪声 |
| 能否推断任意新分子、新柱规格？ | 当前 row 和 target-compound 证据不足以支持 | 必须另设源域也未见过分子、独立批次或新柱验证 |

## 2. 课题 1：已经取得的效果与真正卡点

### 2.1 在同一设置下看数字

下面是当前 V2、相同五个划分、333→1005 个训练标签的已发布结果。共享验证集另有 416 个标签。NRMSE/AULC 越低越好。

| 方法 | NRMSE-AULC | NRMSE@1005 | V1 RMSE@1005 / mL | V2 RMSE@1005 / mL |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.624260 | 0.543301 | 4.146 | 6.873 |
| Hybrid | 0.479354 | 0.409709 | 3.294 | 4.804 |
| LCMD | 0.479539 | 0.408614 | 3.241 | 4.923 |
| Kernel-IVR | 0.500070 | 0.395913 | 3.168 | 4.646 |

来源：[Phase 0 报告](/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/active_learning/qgeognn_v2_efficiency_review/REPORT.md)、[IVR 报告](/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/active_learning/qgeognn_v2_row_kernel_ivr_b32/FINAL_REPORT.md)。

LCMD 与 Hybrid 的 AULC 几乎一样，不能稳定排名。原始 `NO_CLEAR_SEQUENTIAL_AL_GAIN` 是分类规则未确定两者的排名，不是否定两者相对 Random 的收益。

LCMD 的 NRMSE 从 333 标签的约 0.707 降到 653 标签的约 0.437，之后再增加 352 个标签才降到约 0.409。匹配的全数据参考约为 0.370，是当前训练配方下的参考值，不能视为理论上限。IVR 在 cohort mean 曲线上从 973 到 1005 保持 N90，但未达到 N95；这只有两个末期观测点，也不是每个划分都稳定达到同一目标。

标签节省必须说明分母：达到 Random@1005 的同一误差阈值，实际预算格点为 LCMD 493、Random 941。增量节省是 `(941-493)/(941-333)=73.68%`；如果初始与验证标签也需要支付，总标签是 909 对 1357，减少约 33.0%。历史插值下的 79.27% 是另一口径，不能当作实验室已实现的总成本降幅。测试集评价成本还需另列。

### 2.2 实现给出的约束

- [gradient_features.py:84](/Users/fish/Documents/GitHub/column_chromatography_prediction/src/qgeognn_al/active_learning_v2/gradient_features.py:84) 使用两个点预测对全网络参数的梯度，按 L0 尺度标准化后投影为 512 维。已经是有文献依据的强表示，不是简单最后一层距离。
- [lcmd.py:33](/Users/fish/Documents/GitHub/column_chromatography_prediction/src/qgeognn_al/active_learning_v2/lcmd.py:33) 将当前全部已标注数据安装为中心，用未覆盖平方距离聚类选样。它衡量覆盖与敏感性，没有直接测量真实的“标注后误差减少量”。
- [sequential_acquisition.py](/Users/fish/Documents/GitHub/column_chromatography_prediction/src/qgeognn_al/active_learning_v2/sequential_acquisition.py) 的 Hybrid 是集成分歧前 25% 内再做覆盖。它已包含不确定性加多样性，不能把同样组合换个名字当新方案。
- [ivr.py:11](/Users/fish/Documents/GitHub/column_chromatography_prediction/src/qgeognn_al/active_learning_v2/ivr.py:11) 已实现已标注集条件化的方差减少，用统一行核、等权参考集、单位先验/噪声。后验是代理模型，不能等同于实际六输出 GNN 的预测后验。
- [training/predictor.py:75](/Users/fish/Documents/GitHub/column_chromatography_prediction/src/qgeognn_al/training/predictor.py:75) 的中间输出虽然叫 q50，实际上以 MSE 训练；严格说并非 pinball 定义的条件中位数。训练损失使用原始体积尺度，而选模型与选样使用标准化目标。这里有可研究的不一致，但不能仅凭形式不一致就断言它是平台原因。
- [QUANTILE_AUDIT.md](/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/predictor/final_4g_qualification/QUANTILE_AUDIT.md) 中名义 80% 区间的 row 覆盖率只有约 72%/69%，compound V2 约 46%。区间宽度不能直接当作可被补标签消除的不确定性；强制分位数有序也不等于获得校准或认识不确定性。

### 2.3 最值得做的下一项精度实验：后期共同状态分叉

建议先完成当前 Static/Adaptive 固定预算对照，然后只选一个共同状态，例如 LCMD 已标注数 653 的状态。每个划分使用完全相同的 L、U、预处理、当前模型和下一轮训练初始化，比较三条 +32 分叉：继续 LCMD、改用现有 IVR、固定 Random。

这能回答“在已经完成覆盖的同一批数据上，IVR 是否更会选择剩余的有用样本”。当前 LCMD@653 和 IVR@653 的训练集合不同，直接比较它们的下一轮曲线不能回答这个问题。

五个划分共 15 次后续拟合；若原 LCMD→685 的数据顺序、配置与哈希完全匹配，可复用其中 5 次，约新增 10 次。先只做一个状态，显示稳定收益后才检验第二状态或连续多轮。用开发验证集比较两端点 RMSE 与增量收益，测试结果只在规则冻结后统一报告；当前数据已用于设计，因此仍属于开发性证据。

同时可在固定 L653 上做少量额外初始化，测量训练波动。若换一次初始化的变化就超过不同选样的增益，应先处理训练稳定性；若不同初始化都稳定而所有选样分叉都很弱，再检查剩余样本覆盖、目标分布和模型偏差。额外初始化是诊断成本，不能混入主结果挑最好的一次。

**不能现在就宣布采用“653 后切 IVR”。** 653 来自已观察到的平台，属于开发假设；切换规则必须在后续开发阶段固定，并对完整可执行轨迹做新验证。一次 +32 的胜出也不保证连续切换有效。

### 2.4 如何调整现有 Phase 2 的优先级

[现有方案](/Users/fish/Documents/GitHub/column_chromatography_prediction/docs/research/4G_PHASE2_IMPLEMENTATION_PLAN_2026-09-18.md) 已列出 MaxDet、RD-EMCM、噪声加权、multi-output IVR 与预测协方差。这些是候选储备，不宜同时铺开。

如果主要目标是低预算 AULC，先做便宜的 Gradient-MaxDet 一步筛选合理。如果主要目标是突破后期 N90/N95，333→365 的筛选与研究问题不完全对应：一个早期不占优的方法仍可能是后期有效候选，现有 IVR 就提示了这种可能。因此我更建议把共同状态分叉排在更多新算子之前。

噪声感知选样应从已标注集的独立重复或交叉拟合诊断开始。高误差可能源于可学习的系统偏差，也可能源于测量噪声；直接用区间宽度惩罚样本，可能把最需要覆盖的高保留样本排除。现有 IVR 机制审计的排名相关性高而集合重合未达门槛，也不能解释为数值实现坏了，更不能据此无限增加 sketch 维度。

### 2.5 另外两项有价值但目的不同的工作

**计算效率。** 当前 [runner.py](/Users/fish/Documents/GitHub/column_chromatography_prediction/src/qgeognn_al/active_learning_v2/runner.py) 每轮从相同初始化重新训练。五个划分的历史 Hybrid 总拟合 320 次，LCMD 110 次。可在固定标签轨迹上比较 scratch 与使用全部已标注数据的 warm-start，先确认相同性能所需计算是否降低，再研究闭环选样。Warm-start 不保证泛化相同，文献 [3] 已直接指出这种风险。

**泛化价值。** 当前全数据资格验证的 row V1/V2 RMSE 约 2.92/5.53 mL，compound 约 5.61/10.98 mL。若论文要声称适用于新分子，应补一个有明确分子隔离的最小 AL 对照，至少 Random 与 LCMD；源模型、辅助表示与数据预处理也须遵守该隔离。必要性不应完全取决于某个新算子先超过 LCMD。不能把新随机种子当成新化学数据集。

## 3. 课题 2：为什么“不断改迁移模型”收益有限

### 3.1 两套数据问题不可混排

无阈值、B=30/50/70/100 的低标签迁移，与过滤后约 320–360 条训练行的 FULL-data 迁移，回答不同问题。下表仅列前者 B=100、target-compound 划分的 V1/V2 RMSE，单位 mL：

| 柱 | Conditional-EA | Local shrinkage | 当前 V2 paper-style |
| --- | --- | --- | --- |
| 8g | 6.10 / 8.64 | 5.99 / 8.59 | 6.85 / 10.27 |
| 25g | 13.01 / 19.76 | 13.95 / 19.76 | 19.56 / 26.27 |
| 40g | 31.84 / 43.46 | 33.57 / 40.97 | 37.40 / 48.17 |

来源：[匹配迁移报告](/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/matched_rmse_benchmark/MATCHED_RMSE_REPORT.md)。B 包含目标训练与验证标签，compound 分组预算还需以实际消耗数为准。8g head-only 也是强对照，不能在最终比较中省略。表中的 Conditional-EA 是该固定模型，并不等同于会在验证集上回退的 conditional policy。

这些结果表明源预测保留了有价值的色谱顺序/尺度信息，而低维修正有较低估计方差。但它们没有证明“跨柱关系纯粹线性”，更没证明简单校准已经满足实际收集窗口精度。

### 3.2 已测机制不能继续当成未测新方向

Loss normalization、弱分位数/点预测 loss、adaptive readout、FiLM、多任务共享、PCGrad、物理质量归一化、条件缩放和 Center/Width 等均已有受控实验。

PCGrad 将负梯度冲突负担降低约 99.28%，却只得到 25g -0.37%、40g +0.50% 的 inner NRMSE 变化。它说明“测得到梯度冲突”不等于“冲突是主要性能瓶颈”。继续换一串梯度协调器的依据较弱。

也不能把复杂方法全部描述成没有效果：过滤后 FULL-data 的 A1 共享表示在 40g row 上报告 V1/V2 RMSE 约 11.92/14.75，P0 为 13.75/16.40；但额外的相关柱监督、compound 表现和泛化范围需要同时交代。A2 FiLM 未稳定超过 A1，和 A1 在某个场景有正信号是两件事。[多任务报告](/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/column_conditioned_multitask/FINAL_REPORT.md) 支持场景限定的后续确认，不支持普适低标签优势。

### 3.3 数据限制比“再加特征”更具体

1. **柱规格与流速混杂。** 8g/25g/40g 分别固定为 10/15/30 mL/min。同一柱内流速不变，单柱模型加 flow 不增加区分信息；跨柱模型也不能仅凭这几个点分离柱重与流速效应。旧 25g/40g 几何元数据三元组相同且缺乏测量出处。[元数据审计](/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/physical_metadata_identifiability_audit/FINAL_REPORT.md)。
2. **源域没见过的分子很少。** 审计中 source-absent 仅 8g 的 1 个分子/7 行、25g 的 1 个分子/19 行，40g 为零。target-compound holdout 不能代表真正 source-unseen OOD。
3. **严格配对并不普遍成立。** 若要求分子、洗脱剂、上样条件和流速均匹配，25g/40g exact pair 为零；忽略流速才有 297/490、422/529 行，其中源训练标签可用为 240、358 行。它们可以研究复合域迁移，但不是纯柱重因果对照。
4. **噪声下限尚未识别。** 8g/25g/40g 的精确重复组仅 7/3/7 组，每组最多 2 次，而且实验 provenance 仍重要。不能靠这点信息推断全面异方差噪声模型。
5. **长尾重要，但不能只追极端点。** 按真实目标体积的训练 q80 切分，25g/40g 大量 SSE 集中在高体积组；按可获取的 source-prediction q90 定义尾部时，40g 尾部只解释约 20%–28% 的 scale SSE，主体区仍有大误差。两种分层变量不同，不能混为一谈。候选选样时未知真实目标体积，应使用源预测/条件定义区间；真值尾部只能用于事后评价。[缩放审计](/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/scaling_failure_audit/SCALING_FAILURE_AUDIT.md)。

## 4. 课题 2 的可行转向：主动校准与目标误差减少

### 4.1 调整问题定义

推荐问题是：**给定已训练的 4g 预测器和少量目标柱实验，如何选择下一批目标柱实验，最有效地修正目标柱预测？**

这仍然是主动迁移。迁移可以发生在预测校准、表示或参数层面，不要求必须解冻 GNN。科学上也没有“神经迁移必须先超过所有校准基线”这一必要条件。

现有项目将主动迁移延后，是当前治理与证据安排；建议另立一个范围明确的开发试验来检验主动校准，而非重开旧神经迁移实验。若策略只使用线性设计矩阵，可以先评价其样本效率，不依赖现有不合格分位数宽度。只有把后验方差解释为可信区间或用来在线停止，才需要相应的 UQ 验证。

### 4.2 已有代码足以搭建便宜而清楚的第一步

[conditional_scaling.py:34](/Users/fish/Documents/GitHub/column_chromatography_prediction/src/qgeognn_al/transfer/conditional_scaling.py:34) 已实现每个端点三个系数的正则校准，可概括为：

```text
u_j(x) = source_prediction_j(x) / source_scale_j
target_j(x) / (mass_ratio * source_scale_j)
    = a_j * u_j(x) + b_j + gamma_j * standardized[u_j(x) * (EA - mean_EA)]
```

第一项试验固定预测器与拟合规则，只比较 Random、按源预测体积与 EA 分层覆盖、面向目标预测集合的 V-optimal 选样。各柱内使用同一个预先固定的校准器；不要每一轮在多个模型中依照很小的验证集重新挑赢家。

对于线性高斯代理，单端点的一步方差减少可写为：

```text
C = (prior_precision + sum_l phi_l phi_l^T / noise_l)^(-1)
M = sum_z target_weight_z * phi_z phi_z^T
gain(x) = phi_x^T C M C phi_x / (noise_x + phi_x^T C phi_x)
```

两个端点同时被一次柱实验观察，要合并端点风险并按一个实验计费；有共享参数/相关观测时应采用一致的块更新。这里是经典实验设计的色谱适配，不是凭公式获得的新算法贡献，也不保证消除源模型偏差。

**实现上有两个不能忽略的细节。** 当前校准目标使用平均平方损失加 penalty，因此正规矩阵含 `n * penalty`；不能机械套用“固定 ridge 的 rank-one 后验更新”。当前 EA 与交互特征会随训练集合重估标准化；若选择固定初始特征，则需要为所有对照声明同样的新拟合约定，重新验证它与现有强基线的性能关系。历史 IVR 的单位噪声只是固定假设，不是实测实验方差。

另一个重要区别：固定特征、固定噪声和固定超参数的线性 V-optimal，其选样顺序本质上可预先计算，不依赖新获得的 y。它是主动实验设计基线，不能包装成已经证明“标签反馈驱动的自适应”。先证明设计优于 Random；只有已标注残差能够支持可靠更新时，再单独研究反馈价值。

### 4.3 最小开发试验与成功条件

沿用目标柱 30/50/70/100 的总标签预算作报告，固定共同初始标签、固定监测标签和嵌套查询顺序；新增一行同时获得 V1/V2。监测/调参标签计入预算，不能免费额外读取。若重设初始分配，必须重跑同设置 Random，不能直接拼接旧曲线。

查询单位建议先固定为实验行。最终测试按目标分子分组隔离，与“每次必须标注整组分子”是不同设计；如选择整组采集，应按实际实验行数/时间核算不等组成本。源预测缓存允许复用，任何源实测标签使用都需记录其原始训练角色和额外成本。

预先分别确定三个判断：选样是否优于相同校准器的 Random；是否优于便宜的分层覆盖；B=100 绝对误差是否改善。建议先用五个配对划分、报告每个端点与 paired 差异，给出有实际意义的最小效果阈值，再决定是否追加连续轨迹。5% AULC 改善、至少 4/5 同向可作为讨论起点，不是文献保证，也不是独立样本显著性检验。

如果 V-optimal 只改变参数方差而总体误差不降，检查剩余系统偏差；如果各方法在 30 标签之后曲线都很平，优先判断当前候选池是否还含有效信息，而不是强迫主动策略胜出。FULL-data 校准参考需在相同目标总体和划分下拟合，它衡量该模型家族的经验余量，不是物理极限。

早期 E4/D45 已有 Legacy 神经主动迁移的 null 结果，且不确定性、覆盖分数与单点真实收益的相关性很弱。新的建议改变了预测器和信息目标，并不能声称填补“从未做过任何主动迁移”的空白。[历史 D45](/Users/fish/Documents/GitHub/column_chromatography_prediction/experiments/d45_oracle_marginal_utility/README.md)。

## 5. 若能补实验，优先买到哪种信息

在预算未明确前，不指定需要多少条才“足够”。建议先根据可操作流速、已有覆盖和预计实验方差，做小型分块设计，再用先导方差确定正式样本量。

| 补实验类型 | 设计要点 | 能解决的问题 |
| --- | --- | --- |
| 同条件独立重复 | 同分子/EA/上样量/柱/流速，跨运行或批次重复；保留失败记录 | 区分随机测量波动与可学习偏差 |
| 跨柱配对 | 同一分子、尽量匹配其余条件，明确记录不可匹配因素 | 识别源误差与目标柱修正的关系 |
| 柱重与流速交叉 | 至少两个柱规格、各至少两个可行流速，优先重叠流速；分子和批次作区组 | 打破“一个柱对应一个流速”的混杂；两点不能识别任意非线性 |
| 条件与长保留覆盖 | 用已知 EA、上样条件和源预测分层，兼顾主体与稀疏区域 | 改善实际工作域覆盖，检验系统偏差 |
| 独立确认 | 新分子或新批次在设计前锁定，覆盖实际使用范围 | 检验经反复开发后的方法是否真正泛化 |

配对数据支持的后续假设可以是 `target = rho * measured_4g + delta(conditions, molecule, column)`，但仅在部署时也能获得 measured_4g 时成立。用实测 4g 输入与仅用预测 4g 输入的策略，测量成本和任务不同，不能直接排名。25g/40g 当前忽略 flow 的配对不能解释成纯质量差异；8g 已有配对误差锚点信号并不稳健，因此不建议立刻重跑泛化的 delta 模型。

4g、8g、25g、40g 是不同实验系统，不能天然按“低保真→高保真”排序。多保真/多任务文献可以提供建模结构，相关性与成本优势仍要由当前实验支持。

## 6. 文献如何对应到本项目

| 文献与本次查阅深度 | 可借鉴的内容 | 本项目的限制 |
| --- | --- | --- |
| [1] Holzmüller 等，JMLR 2023；核对官方摘要及本地梯度/LCMD实现 | 全网络梯度核、投影、条件变换和批量选样的统一框架 | 当前方法已采用其核心思路；换名称不构成创新，原基准也不是柱层析 |
| [2] Hübotter 等，NeurIPS 2024；阅读 HTML 第 2、4 节 | 让采样服务于指定预测目标；VTL/ITL 兼顾相关性与冗余 | 理论与神经网络线性近似有假设，不保证当前 GNN 的实际误差下降；目标集合须来自合法可用 X |
| [3] Ash 与 Adams，NeurIPS 2020；核对原始摘要 | warm-start 可节省重训成本，但也可能损害泛化 | 先固定轨迹对照，不把增量训练默认当成精度改进 |
| [4] Ramakrishnan 等，JCTC 2015；核对原始摘要与 DOI | 学习可靠低成本基线与目标之间的差值 | 原文研究量子化学；跨柱的配对、条件与源输入成本必须重新证明 |
| [5] Min 等，NeurIPS 2025 AGBAL；核对正式论文集摘要 | 用有标签辅助数据及密度比加权辅助估计选样信息 | 辅助标签必须可用；跨柱存在条件映射变化，不能仅靠输入密度比自动校正 |
| [6] Wu 等，Chem 2025；核对 DOI、仓库方法/复现审计 | 柱层析几何网络与迁移任务背景 | 出版商页面受访问验证限制，本次没有重新获取其全文；精确 source checkpoint、划分等复现缺口不能靠调到相同 R2 解决 |

AGBAL 是辅助数据驱动的不确定性代理，并非“把 q90-q10 当噪声再加权”的直接文献证明。密度比主要调整输入分布，不能自行恢复不同柱的条件输出规律；需要检查校准后残差的可迁移性。

本次还发现旧 4g 文献评估中的 NeurIPS poster `121499` 实际指向 TabArena，并非 AGBAL。本文使用已经核实标题、作者和年份的正式论文集链接；旧文件未在此次研究评估中改写。

### 参考链接

1. Holzmüller, Zaverkin, Kästner, Steinwart. *A Framework and Benchmark for Deep Batch Active Learning for Regression*. JMLR 24(164), 2023. [官方页面](https://jmlr.org/papers/v24/22-0937.html)。
2. Hübotter, Sukhija, Treven, As, Krause. *Transductive Active Learning: Theory and Applications*. NeurIPS 2024. [论文](https://arxiv.org/abs/2402.15898)，[本次所读正文](https://arxiv.org/html/2402.15898v6)。2025 是该版本修订时间。
3. Ash, Adams. *On Warm-Starting Neural Network Training*. NeurIPS 2020. [论文](https://arxiv.org/abs/1910.08475)。
4. Ramakrishnan, Dral, Rupp, von Lilienfeld. *Big Data Meets Quantum Chemistry Approximations: The Delta-Machine Learning Approach*. JCTC 2015. [论文](https://arxiv.org/abs/1503.04987)，[DOI](https://doi.org/10.1021/acs.jctc.5b00099)。
5. Min, Xu, Li, Zou, Zhou. *Enhancing Deep Batch Active Learning for Regression with Imperfect Data Guided Selection*. NeurIPS 2025. [正式论文集](https://papers.nips.cc/paper_files/paper/2025/hash/cbae8efcc23a0cb6d15a20f245514020-Abstract-Conference.html)。
6. Wu 等. *Intelligent column chromatography prediction model based on automation and machine learning*. Chem 11, 102598, 2025. [DOI](https://doi.org/10.1016/j.chempr.2025.102598)，[本地方法契约审计](/Users/fish/Documents/GitHub/column_chromatography_prediction/docs/model/PAPER_CODE_CONTRACT_AUDIT.md)，[复现差距分解](/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/paper_transfer_gap_decomposition/FINAL_REPORT.md)。

## 7. 建议的投入顺序

1. 完成已经冻结的 Static/Adaptive 对照，确定反复反馈是否值得当前计算与实验往返成本。
2. 4g 精度方向先做一个后期共同状态的 LCMD/IVR/Random 分叉；这是对当前卡点最直接、成本可控的检验。
3. 跨柱先制定一个固定预测器的主动校准开发试验，将“迁移模型选择”与“选样是否有效”拆开；冻结强 Random 和分层覆盖对照。
4. 能补实验时优先独立重复、跨柱配对、质量与流速交叉和新批次确认；暂时只能用现有数据时，将结论限制为开发性池内/分组验证结果，不承诺任意新柱外推。
5. 只有上述诊断显示明确剩余结构，才增加一个针对该结构的新模型或选样机制。降低计算成本、降低标签成本与降低绝对误差分别评价。

可形成的两条研究主线分别是“几何梯度选样的阶段性标签效率及泛化边界”和“少量目标实验驱动的跨柱校准与主动实验设计”。具体机制和实验能构成贡献；把现成 LCMD、IVR 或线性设计改名，不构成新的方法贡献。
