# 选点行为分解：适用性、实现与结果

原论文的 V/U 不能在此严格同义复现。本项目最大化的是后续行级预测精度，预测保留体积越大并不意味着实验价值越高。当前没有跨全部历史状态、决策前可得且经验证的候选学习收益预测。因此，本分析不将 V1/V2 的预测均值命名为 Value，也没有新拟合 GP 来构造一个看似完整的 V/U 坐标。

## 方法与时间边界

参考[论文 Methods 4.11](https://arxiv.org/html/2609.33061#S4.SS11)的条件选择建模思路：每个正候选匹配 64 个真实可行的未选负候选，排除整个最终接受 batch；ridge 系数为 1；将批内各点视为独立选择。该近似不能恢复 LCMD 的顺序更新，也不能把混合批视为单一决策者。论文使用共享 GP 特征；这里改为以下可审计的色谱任务代理特征。

| 特征 | 决策前定义 | 解释限制 |
|---|---|---|
| G | 现存 CW-gradient 表示中，到当轮历史集合的最近距离的全候选池百分位 | 不是梯度范数，不是后续学习价值，也不是 LCMD 的完整逐步打分 |
| U_proxy | 分别计算 V1/V2 的原始有符号 q90−q10 的候选池百分位，取平均后再取候选池百分位 | 保留负宽度的最低排名；不等于校准后的 epistemic uncertainty；响应尺度仍可能混杂 |
| D | 分子、PE/EA 比值、样品密度、样品体积、加载溶剂、加载体积六因素的最近历史 Hamming 距离 | 离散指标；连续变量只要非零变化即视为不同，仅作论文式离散视角，连续距离见主分析 |
| C | 对 128 个固定参考样本，加入单一候选所减少的平均最近历史 Hamming 距离 | 单点静态覆盖代理；不是实际批内 LCMD 边际增益，也不是信息增益 |

D/C 的表示只由候选元数据定义，跨轮固定。参考集按 seed 固定，在 L333 初始训练集与候选池的并集中按哈希选 128 点，不查看任何真实响应。G 仅在同一个冻结 checkpoint、同一个历史集合及正确的缓存行序内求距离；各轮原始距离不混合。所有特征先转为各自当轮合法候选池的 average-rank 百分位，再使用对方法、campaign、状态依次等权的共同均值/标准差。因而系数表示在本状态排序中偏好更高/更低分位，不是原论文 GP 均值/标准差的每单位效应。

原始 catalog/contract/batch_freeze 逐文件哈希核对。缓存重建只执行现存 npz 的距离计算，未重算梯度、前向预测或训练。NoCW 的 CW 距离是离线诊断，不代表 LLM 当时看到了这列。没有读取 model/predictions.csv.gz、test truth 或未观测标签。

## 候选级状态完整率

共 791 个实际接受 batch、22 条 campaign、25,312 个选择事件。D/C 全部可计算。U_proxy 在 Historical CW-Free 180 批、New CW-Free 180 批、NoCW 180 批、Hybrid 71 批均完整；纯数值 CW-LCMD 没有同类 catalog，因此 U 留空，未从可能含真实响应的预测总表中补齐。

G 直接复用 Historical 180 批、New 180 批、Hybrid 71 批的选前 catalog。为核心共同区间 seed 157/6101、L333–1005，额外从现有可信缓存计算 NoCW 42 批和 CW-LCMD 42 批的全池 G。其余 NoCW/CW-LCMD G 未计算，明确记录为本次缓存审计范围外，不能解释成历史必然缺失。

核心共同区间五方法各有 42 批、1,344 个选择事件；其均值如下，随机候选的池平均百分位约为 0.5002。每个 seed 含相同的 21 批，表中方法均值也是两个 seed 等权均值。

| 方法 | G 平均选中百分位 | U_proxy 平均选中百分位 | D 平均选中百分位 | C 平均选中百分位 |
|---|---:|---:|---:|---:|
| Historical CW-Free | 0.62655 | 0.76512 | 0.47740 | 0.48665 |
| New CW-Free | 0.58756 | 0.70100 | 0.48796 | 0.49196 |
| NoCW-Free | 0.55083 | 0.63754 | 0.49742 | 0.49350 |
| CW-LCMD | 0.93270 | 缺失 | 0.54329 | 0.49489 |
| LLM16 + CW-LCMD16 | 0.77806 | 0.73614 | 0.50756 | 0.49523 |

这组数据同时否定几个过强解释。LLM 并非在所有意义上回避较大模型区间宽度：三个纯 LLM 的 U_proxy 都高于候选池均值。CW-LCMD 在自己使用的 CW 几何上选得明显更远，但在静态六因素覆盖代理 C 上并没有同样排序；空间选择会影响结论。Historical 到 New 再到 NoCW 的 G/U 逐步下降，是选点排序的描述证据；Historical 的模型与 Prompt 历史不同，New/NoCW 的历史训练状态逐轮分岔，都不能将差异直接归因为显式 CW 一列。

## 条件 Logit 与稳健性

结果表 `behavior_preferences.csv` 分开保留共同 L1005、共同 L1453 和全部现有状态。单特征模型之后才拟合双特征及完整模型；完整模型要求所有 cohort 状态完整且设计条件数不超过 100。所有模型保留逐 campaign 系数，不用 25,312 个候选当作独立科研重复。只有五 seed 且完整的双特征全轨迹模型进行 100 次整 campaign 重采样重新拟合，见 `preference_campaign_bootstrap.csv`；它是有限 Monte Carlo 的探索性区间，不是论文 10,000 次重采样的精确复现。两个 seed 的核心比较不给显著性结论。

负样本随机种子为 7001/7002/7003；共同 L1005 的全部模型重抽样，见 `preference_negative_sensitivity.csv`。池内特征相关保留在 `preference_collinearity.csv`。若 L1453 或全部状态中某项 G 仅覆盖核心阶段，表中 `state_coverage` 会低于 1；这些系数只描述实际可用子集，禁止当作整段模型。

核心共同区间的 G+U 双特征拟合如下。全部四个 cohort 均为 42 个完整状态、两个 seed；Hybrid 是整批两种来源的混合偏好，而非 LLM 半批的系数。

| 方法 | βG | βU_proxy |
|---|---:|---:|
| Historical CW-Free | 0.22335 | 1.12199 |
| New CW-Free | 0.09520 | 0.75748 |
| NoCW-Free | 0.04193 | 0.49733 |
| LLM16 + CW-LCMD16 | 0.93803 | 0.90795 |

在完整 G/U/D/C 模型中，四者的 βU_proxy 仍依次为 1.10880、0.75202、0.50166、0.89808，方向保持为正。加入静态距离与覆盖后，没有出现论文式“回避不确定性”的同义证据；但区间宽度尚未控制所有预测响应尺度，因此不能把这一结果反过来叫作认知不确定性寻求。

G+D 是五方法都完整的几何行为双特征对照：CW-LCMD 的 βG=4.41517，Hybrid=1.23749，Historical=0.51185，New=0.34261，NoCW=0.19758。相应 βD 为 −0.00764、−0.13833、−0.21042、−0.13478、−0.05925。纯数值方法确实更强地选择 CW 距离高分位；LLM 在固定离散因素空间中的局部偏好与其在梯度空间中的远距偏好能够同时存在。LCMD 逐点模型仍是集合选择近似，系数大小不是 LCMD 算法参数。

核心 G+U 模型的两次额外负样本重抽样中，各系数方向都保持不变；βG 范围分别为 Historical 0.2203–0.2251、New 0.0990–0.0997、NoCW 0.0298–0.0320、Hybrid 0.9367–0.9458，主抽样估计见上表。负抽样稳定性只说明这一步 Monte Carlo 误差较小，不减少模型版本、Prompt 和轨迹状态的混杂。

系数只估计行为关联。正 U_proxy 不能证明 LLM 偏好贝叶斯认知不确定性，负 D 也不能证明局部选点低效；模型未控制所有条件对照、Prompt 要求、响应大小和历史路径。混合策略整体系数是两种来源的混合分布，来源拆分见 Hybrid 归因表。

## 复现与验证

```sh
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/llm_decision_behavior_audit_20261116/preferences.py --states studies/active_learning/llm_decision_behavior_audit_20261116/states.json.xz --bootstrap 100 --negative-seeds 7001 7002 7003
/Users/fish/miniforge3/envs/chromatography/bin/python -B -m unittest discover -s studies/active_learning/llm_decision_behavior_audit_20261116/tests -p 'test_preferences.py' -v
```

8 个单元测试覆盖解析梯度与数值差分、极端效用稳定性、均匀选择、已知偏好方向恢复、Newton 数值回退、整个已选批次的负样本排除、平局百分位、覆盖单调性。所有实际输入及哈希、缓存覆盖与零训练/零 API 数据流声明见 `preference_verification.json`。这些检查验证计算与历史绑定，不能验证代理特征的科学充分性。
