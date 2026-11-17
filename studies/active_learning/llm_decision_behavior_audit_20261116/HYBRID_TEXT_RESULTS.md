# Hybrid 互补性、可见理由与行动偏好动态

## 证据范围与来源核对

本模块分析 seed 157 的36批（L333→1485）和6101的35批（L333→1453），合计71批、2272个接受名额。`contract.json`提供每轮历史与未标注候选ID；`batch_freeze.json.pending_ids`是数值半批，最终`selection.json.choices`是LLM半批。两者各16、无交集且并集等于32个最终ID，71批均为数值先冻结、LLM后补充。来源划分来自显式字段，不是依据位置猜测。最终结果未见自动补位或去重导致的来源混合。

`hybrid_proposal_audit.csv`保存93个可见结构化selection答复，较71批多22个；其中16个答复的choice顺序列表不同于最终接受列表，累计28个ID提议事件不在对应非pending U内、0个批内重复ID。列表相同也不保证其科学证据字段合法，因此这些数字不能逐一等同于特定拒绝原因。可见最终理由中有ID拼写修订；未把它解释成静默自动修复。原始推理未恢复。

## 同状态半批行为

下表是逐批均值再按seed汇总，单位：分子数/批、比例、池内CW最近历史距离百分位、两个端点q90−q10宽度的平均mL。

| seed | llm_new_molecules | numerical_new_molecules | llm_same_molecule_single_factor_history_fraction | numerical_same_molecule_single_factor_history_fraction | llm_cw_distance_percentile_mean | numerical_cw_distance_percentile_mean | llm_q90_q10_endpoint_average_ml | numerical_q90_q10_endpoint_average_ml |
|---|---|---|---|---|---|---|---|---|
| 157 | 0.83333 | 0.5 | 0.85417 | 0.69618 | 0.58351 | 0.93429 | 9.3509 | 6.9836 |
| 6101 | 1.1143 | 0.31429 | 0.85893 | 0.68571 | 0.52904 | 0.94571 | 5.3916 | 3.6814 |

LLM半批比数值半批选择更多首次出现的分子，也有更高的同分子单因素历史对照比例。这两件事可以同时成立：16个名额既可包含少量新分子，也可对历史已知分子进行局部条件实验。CW距离百分位说明数值半批更远离当前梯度历史；宽度不同只涉及单模型预测区间，不是经校准的认知不确定性。

## 梯度几何与覆盖顺序归因

每轮复用已有raw/CW CountSketch缓存，核对checkpoint、canonical index映射及L/U集合。扩展目录边界只在同checkpoint、同L/U及同轮索引全部匹配后引用原owner保存的bank；共142个空间状态，缺失见`hybrid_gradient_missing.json`。不把跨轮原始梯度距离当作同一个固定表示空间。

固定参考集为每seed的U0中按旧脚本SHA规则确定的128个ID。定义覆盖为参考候选到历史集合的平均最近距离的负值，增量是距离下降；分别计算L单独、N单独、L给定N、N给定L。顺序平均贡献为两个进入顺序的平均；两份贡献相加等于联合增量。这是几何归因，绝非学习性能贡献。下表将每轮距离下降除以该轮参考集起始平均距离，再按seed汇总。

| seed | llm_order_average_relative | numerical_order_average_relative | union_gain_relative | shared_coverage_gain_relative | llm_gradient_norm_pool_percentile | numerical_gradient_norm_pool_percentile |
|---|---|---|---|---|---|---|
| 157 | 0.017315 | 0.068711 | 0.086026 | 0.0041615 | 0.72604 | 0.67927 |
| 6101 | 0.011925 | 0.062635 | 0.07456 | 0.00038841 | 0.77506 | 0.70305 |

在CW梯度空间，数值半批的顺序平均相对覆盖贡献约为LLM半批的4–5倍；但是LLM梯度范数池内百分位反而更高。梯度范数大与集合几何覆盖高是不同性质。固定Morgan分子空间与固定条件空间的顺序平均贡献可查`hybrid_attribution.csv`：例如联合结构—条件空间中seed157数值稍大，6101则LLM稍大，不能将梯度结果泛化为所有空间。

静态条件距离使用EA体积分数、密度、上样体积、加载体积的固定候选元数据范围归一化，再与加载溶剂不等指示按五因素平均；分子距离是Morgan radius2/2048 Tanimoto距离；联合距离各占一半。范围来自无标签元数据并固定，不因训练更新变化。密度是记录因素且可能与分子绑定，单因素对照只对现有五个记录字段负责。几何共同参考覆盖的重叠不等于实验无效冗余。

## 保留真实数值16的条件随机基线

每个真实历史状态固定实际CW16，在当轮未标注池排除这16个pending后，不放回随机16个，重复500次，固定种子`20261116 + seed*100 + round`。同样排除已测量ID且保持预算；原Prompt没有分子配额、对照配额或固定探索比例。检索工具的已查看集合是策略行动的结果，基线以查询前完整候选宇宙作为可行域，未声称重现同一查询过程。

| seed | joint_nearest_history_random_delta | joint_coverage_after_numeric_random_delta | joint_coverage_after_numeric_random_percentile | molecular_coverage_after_numeric_random_delta | condition_coverage_after_numeric_random_delta |
|---|---|---|---|---|---|
| 157 | -0.001665 | -0.00015384 | 0.39369 | 0.0002633 | -9.5328e-05 |
| 6101 | -0.0013724 | 0.00025049 | 0.43529 | 0.0014866 | -1.5683e-05 |

LLM的联合邻历史距离相对条件随机均值在两seed都偏低；但LLM给定CW的联合覆盖增量相对随机差值跨seed异号。LLM补充分子覆盖的平均差值为正，条件覆盖差值为负。这支持空间分工的描述，无法支持一致的几何协同，更无法支持Row Performance协同。

## 可见理由与实际行动

仅使用最终接受的`batch_strategy`、`batch_rationale`、假设文本和候选reason；关键词规则在脚本`PATTERNS`中公开。它检测探索相关措辞，不是原论文Judge估计的[0,1]探索意图。没有外部Judge、没有新LLM调用，不能恢复内部思维。最终选择文件与freeze哈希绑定。

| method | batches | exploration_keyword_fraction | local_keyword_fraction |
|---|---|---|---|
| free16_lcmd16 | 71 | 0.92958 | 1 |
| historical_free_mixed_model | 180 | 0.96667 | 1 |
| new_free_cw_sol61_xhigh | 180 | 0.96111 | 1 |
| nocw | 180 | 0.9 | 0.99444 |

按候选历史状态随机校正指标计算每个campaign内的Spearman关联；文本变量为每1000可见字符的探索关键词数。下表限L333–1005，每seed先单独计算，再列seed均值和范围。没有把候选行作为独立重复，没有强显著性检验，也没有声称严格复现论文的cycle correction。

| method | action_metric | mean | min | max |
|---|---|---|---|---|
| free16_lcmd16 | hamming_novelty_random_delta | 0.032245 | -0.12277 | 0.18726 |
| free16_lcmd16 | joint_coverage_gain_random_delta | 0.26075 | -0.051965 | 0.57347 |
| free16_lcmd16 | rarity_random_delta | 0.064452 | -0.16889 | 0.29779 |
| historical_free_mixed_model | hamming_novelty_random_delta | 0.32884 | 0.0064935 | 0.65714 |
| historical_free_mixed_model | joint_coverage_gain_random_delta | 0.26922 | -0.12597 | 0.5039 |
| historical_free_mixed_model | rarity_random_delta | 0.24533 | -0.24805 | 0.53377 |
| new_free_cw_sol61_xhigh | hamming_novelty_random_delta | 0.093104 | -0.20779 | 0.44994 |
| new_free_cw_sol61_xhigh | joint_coverage_gain_random_delta | 0.16821 | -0.27792 | 0.61769 |
| new_free_cw_sol61_xhigh | rarity_random_delta | -0.099437 | -0.36234 | 0.16645 |
| nocw | hamming_novelty_random_delta | 0.26187 | -0.034427 | 0.65216 |
| nocw | joint_coverage_gain_random_delta | 0.25095 | 0.11108 | 0.38831 |
| nocw | rarity_random_delta | 0.13965 | -0.18831 | 0.54953 |

New CW的关联方向随seed及探索代理变化；Historical和NoCW有部分正向关联。结果反对“探索措辞与行动绝对无关”的预设，也不能将其读作校准意图的测量。候选reason本身可能在描述已经选择的点，词频关联还存在文本长度、批次内容和后验叙述混杂。

`decision_cases.csv`按方法×阶段选取随机校正Hamming Novelty最小与最大批，合计32例，保存原可见英文摘录、候选ID、具体理由和全套数值。它明确展示反例而非挑选单向故事：

| 方法/seed/round | 可见设计 | Hamming随机百分位 | 联合Coverage随机百分位 | 含义 |
|---|---|---:|---:|---|
| Historical /6101/0 | condition-response curves、failure repair及novel-region exploration | 0 | 0.022 | 有探索措辞而选点接近历史，符合局部实验设计 |
| New CW /157/4 | 阈值条件扫点、matched loading、late-tail transfer anchors | 0.001 | 0.002 | 局部对照为明确设计，不能据此判断低效 |
| New CW /12203/3 | 未覆盖功能团转移并配对照 | 1 | 0.998 | 探索措辞与高探索代理一致的反例 |
| NoCW /2357/15 | tail thresholds、loading interactions与central controls | 0.001 | 0.982 | Hamming局部程度与覆盖贡献不等价 |

## Action Preference Dynamics

定义每批分子名额的经验分布，比较相邻批的Jensen–Shannon散度（log2，范围0–1）及与上一批分子重叠的名额比例。它们是行动分布的变化，不能当作信念校准或反馈因果效应。

| method | transitions | molecule_action_js | previous_batch_molecule_slots |
|---|---|---|---|
| cw_gradient_lcmd32 | 175 | 0.85533 | 0.15089 |
| free16_lcmd16 | 69 | 0.72355 | 0.28623 |
| historical_free_mixed_model | 175 | 0.69314 | 0.32464 |
| new_free_cw_sol61_xhigh | 175 | 0.72396 | 0.295 |
| nocw | 175 | 0.75854 | 0.25839 |

LLM轨迹比CW-LCMD保留更多上一批分子名额，但不代表过度反应；科学对照、工具可见性和当轮训练状态均能解释这种持久性。当前可见假设不是统一定义、跨轮可比的概率信念，无法评估测量噪声，故不计算Martingale Score或Excess Belief Movement。

## 边界

两seedHybrid只支持描述性统计；与独立纯策略逐步分叉的历史不是同一冻结状态下的反事实。真实Row曲线必须由主分析审计，以上覆盖贡献均不作为性能贡献。分子新增数量随历史增长自然下降，原始距离也会随模型特征变换；因此所有强机制判断应限于同轮来源对照。零新训练、零新付费LLM、零未来标签查询。
