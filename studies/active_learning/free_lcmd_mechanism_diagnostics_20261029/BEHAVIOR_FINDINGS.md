# 五 seed 冻结选择行为、历史反馈与文本依赖诊断

本报告覆盖真实冻结的 **360 批、11520 次选择**。比较单位是相同 seed 内两种轨迹的阶段均值，再汇总五个配对 seed。逐轮相关只作描述，不把 180 轮当作 180 个独立实验。

## 1. 数据及可访问边界

- `selection_records.csv`：11520 条，真实 sample_id、原始五条件、SMILES、选择源路径及 SHA256、Free 当时可见的预测和 CW 分位数；没有真实响应或残差列。
- `behavior_profiles.csv`：360 行，`seed,method,round,labels_before` 是合并键。round=0 表示在 L333 时冻结第一批；其后模型验证改善是 L333→L365。
- `behavior_data_audit.json`：全部实际读取来源的路径和 SHA256。样本映射严格来自冻结 catalog 的 ID→SMILES/条件，未按表格行号推测。五 seed 合法 catalog 的元数据并集有 4162 个 ID、217 个分子。
- Free 每轮历史残差仅取该轮 catalog `observed`；已采集历史记录数逐轮严格等于 `32*round`。残差是 acquisition-time 预测减揭示值，**不是当前模型 residual**。排序与 scientist_selector `scientific_memory` 一致：`max(abs(error_V1),abs(error_V2))`，取前 8 条。
- LCMD 冻结 receipt 含所选 ID 和 revealed-values hash，不含响应/残差载荷；本分析不从全量响应源表或其他轨迹重建，因此 LCMD 的历史残差特征均为 NA。不能据此宣称 LCMD 不关注误差或 Free 比 LCMD 更关注误差。
- 没有训练、推理、LLM 调用、新标签请求或测试集真值读取。小案例只读取自己轨迹冻结选择后已揭示的 feedback。

Free 真实选择来源按 `scripts/report_free_lcmd.py.owner(seed,round+1)` 拼接版本，避免把某轮重试分支当作正式结果。接受的 turn provenance 能直接对应 164 批 Sol、15 批 Astra；seed 157 round 5 没有可直接对应的 served-model 记录，保留 `unknown`。这不覆盖历史审计对 Astra/high 前缀的说明，不能把 owner revision 的配置模型反推成实测服务模型。

## 2. 指标定义及一个关键区分

“同分子”按冻结 canonical SMILES。新分子指选择前 L_t 没有这个分子；批内同分子对数为各分子 `n*(n-1)/2` 的和。

“matched contrast”指：同一分子在选择前有一行，除目标条件变量外的四个条件全部完全一致，目标变量不同。目标分别为 PE/EA、sample amount `V/ul`、loading-solvent volume。这个定义可以包含对已测条件 tuple 的重复采样。

“curve extension / 真正缺失条件点补全”进一步要求：当前 **SMILES+完整五条件 tuple 尚未被观测**。新 ID 重复已测 tuple 不算补全。interpolation 还要求已有观测值在目标变量两侧。所有比较只看选择前观测，不用本批后续揭示记录为本批制造对照；批内共同形成的新设计可在病例中单独审计。

Morgan 使用 radius=2、2048 bits、Tanimoto，按 217 个分子缓存。批内均值包含同分子相似度 1；另报 unique-molecule 均值以排除这一机械影响。Murcko scaffold 只是粗家族 proxy：无环分子逐分子保留，不把所有无环分子合为一个家族。SMARTS 功能团可以重叠，并不构成完整化学分类。

## 3. Free 更集中，但不是更偏爱 PE/EA 曲线补全

以下为每个 seed 阶段均值的五 seed 平均；两种方法每批均为 32 行。

| 选择前预算阶段 | 不同分子 Free / LCMD | 新 PE/EA 条件点补全 Free / LCMD | 新 amount 条件点补全 Free / LCMD |
|---|---:|---:|---:|
| 333–525 | 24.00 / 29.37 | 47.29% / 49.17% | 19.90% / 0.00% |
| 525–653 | 23.40 / 29.50 | 59.22% / 60.47% | 23.13% / 0.00% |
| 653–813 | 23.32 / 29.32 | 67.38% / 68.38% | 25.88% / 0.00% |
| 813–1005 | 22.20 / 30.13 | 73.96% / 77.92% | 36.15% / 0.00% |
| 1005–1197 | 22.47 / 29.67 | 74.38% / 83.02% | 39.90% / 0.00% |
| 1197–1485 | 23.11 / 28.04 | 80.14% / 87.57% | 50.00% / 26.60% |
| 全程 | **23.06 / 29.22** | **68.58% / 73.13%** | **34.65% / 6.65%** |

- Free 每批少 6.16 个不同分子；批内同分子对数为 10.32 vs 3.36，五 seed 均同向。这是强、稳定的行为差异。
- **PE/EA 补全不是 Free 的独有优势**。全程 Free 比 LCMD 低 4.55 个百分点；PE interpolation 也非常接近，不能把所有局部条件设计都称为 Free 更强。
- 显著的描述性差别在 amount 维度。包含已测 tuple 重复的 amount matched contrast 为 35.78% vs 6.65%；排除重复后真正 amount 缺失点补全仍为 **34.65% vs 6.65%**，差约 28.00 个百分点。LCMD 首次出现严格 amount 对照是在 seed 12203 的 L1357；其他 seed 为 L1389。这不表示 LCMD 前期不选 50/100/150µl，而是所选行几乎不与先前观测形成“其他四条件固定”的 amount 对照。
- Free 的新分子行比例在最早阶段反而更高（23.02% vs 17.50%），到 653–813 转为更低（1.13% vs 4.13%），813 后 Free 所选分子均已观测。不能将“每批分子少”直接等同于“始终不探索新分子”。
- Free 选择已测完整条件 tuple 的独立 ID 约占 1.65%，LCMD 为 0；这类是重复测量/tuple 冗余，应与缺失条件补全分开。
- 批内 Morgan 平均相似度 0.1807 vs 0.1602；只对批内不同分子计算后为 0.1636 vs 0.1550，差异缩小。Free 更集中部分来自同分子多条件设计。Murcko 同 scaffold 对比例 Free 0.3432、LCMD 0.3601，**不支持“Free 对所有家族定义都更密集”**。功能团分布存在差别（例如 Free alcohol 行比例 8.65% vs 4.79%、halogen 25.47% vs 41.28%），这些是候选/轨迹状态共同作用下的描述。

模型的 condition schema 确实包含 density×V、loading solvent 和 loading-solvent volume，不能从 amount 采样差异推断模型没有这些输入。该行为是否源于当前 raw 梯度对某些条件变化不敏感，需要与几何诊断结合。

## 4. 历史失败区域有真实探索，但改善收益未被证明

Free 全历史 top-8 高误差分子，平均占本批 6.70%，占当时候选池 1.59%，相差 **+5.11 个百分点**。选点到 top-8 的最大 Morgan 相似度均值比候选池基线高 **0.0494**。候选池 baseline 必须使用自己的该轮 U_t，避免把分子剩余行数差异误当偏好。

这种对固定全历史 top-8 的偏好并不恒定：同分子 enrichment 在 333–525 为 +17.42pp，653–813 仅 +0.81pp，813–1005 为 +5.68pp。它并非足以单独解释中期优势的稳定信号。

如果只看前一批 32 个已揭示结果中的高误差 top-8，五 seed 的同分子 enrichment 分别为 +18.27、+14.09、+13.54、+18.96、+16.74pp；各阶段平均约 +14–21pp。约 32.46% 的新选行与前一批某个分子相同。因此“近期反馈附近继续探索”有实际 ID 支持，强于“反复追逐固定全历史最差八点”的简单解释。

但行为与后续验证收益的关系不稳健。控制预算和本轮起始 validation NRMSE 后，Free 全史高误差同分子 enrichment 与下一拟合改善的 rank-residual correlation 为约 `[0.070,0.299,-0.009,-0.023,-0.359]`，方向不一致。amount 补全与下一拟合改善也不能给出跨 seed 稳定正相关。此类控制仍不是因果识别，且 validation NRMSE 是报告分析变量，不是声称 selector 曾看到的输入。

## 5. 文本依赖：有数值关联，不能识别因果贡献

`llm_text_features.csv` 为每批 combined narrative；`text_field_mentions.csv` 为 choices、hypotheses、batch_strategy、batch_rationale、feedback_interpretation 五字段独立统计（900 行）。匹配仅递归提取叙述字符串 **values**，排除字段键和 ID/reference/hash 值；不能由 schema 的 `supporting_observed_ids` 自动制造 observed 命中。failure-repair 仅使用明确模型误差词（under/overpredict、residual、blind spot、systematic error、model/predictor failure 等），不用泛化 `fail/correct`。

| 信息类别 | 180 批中出现比例 |
|---|---:|
| CW/距离分位数 | 66.11% |
| quantile/分位数宽度 | 75.56% |
| 分子相似性/analog | 73.89% |
| 化学先验 | 100.00% |
| 实测响应/历史误差 | 99.44% |
| 局部条件曲线 | 100.00% |
| diversity/redundancy/coverage | 97.78% |
| 明确模型误差修复语义 | 88.89% |

这是关键词 proxy，可能含否定句、自我限定或多义词；例如“不能仅凭 CW”同样计为 CW mention，不能解释为赞成或依赖。

CW mention 与实际选中 CW 分位数的每 seed Spearman rho 为 0.359–0.544；控制预算和起始验证状态后的描述性 rank-residual correlation 仍约 0.508–0.563。分位数宽度 mention 与选中 V2 宽度在当轮候选池的分位数也同向（未调整 rho 0.119–0.624）。这些是文本与行为的一致性证据，**不是去掉数值信息会损失多少性能的估计**。

实际 CW 选择偏好随阶段变化：选中 CW 百分位均值从 333–525 的 0.630、525–653 的 0.679、653–813 的 0.675，下降到 1197–1485 的 0.525；top-10% CW 行占比相应从中早期约 45% 降到后期约 19%。Free 不能称为没有数值信息的纯化学 agent。

化学、局部条件曲线几乎每批都有提及，在 seed 内是常数，故其 mention→行为相关往往为 NA；这不是缺失结果，也不是证明没有作用。局部条件行为同时对照 PE 和 amount 两个目标，避免只看 PE 而遗漏真正区别。模型版本/提示词、训练状态、可选池以及其他数值特征均可能共同产生关联。NoCW、No-feedback、No-chemistry 的贡献仍需专门消融识别。

## 6. 三个具体科学假设案例

完整 source、hypothesis、真实 choice ID、事前已测同分子记录、冻结预测、揭示结果、下一轮解释及实际同分子选择见 `hypothesis_feedback_cases.json`。以下不是事后按预测误差筛选，而是固定选 seed 887 r0、seed 2357 r10、seed 887 r35 的第一条 hypothesis 下前三个 choices。

**中期 2×2 条件设计（seed 2357，r10，L653 选择）**：m125 `COc1cccc(OC)c1C=O`、PE/EA=1/1、DCM。事前已有 50µl sample/300µl DCM 的 `00043aebb2540e369a7b`，V2=18.350；150/500 的 `9bd998a328b109ef639e`，V2=39.383，且历史 V2 underprediction=21.706ml。LLM 假设 loading 与 solvent volume 可能共同影响终点，明确承认实验变异替代解释。它真实选择了缺失两角：

| 新冻结 ID | sample / DCM volume (µl) | 冻结 V2 预测 (ml) | 揭示 V2 (ml) |
|---|---:|---:|---:|
| `03744855d469f4cca7e6` | 150 / 300 | 23.652 | 20.225 |
| `f4fb5c03b8e8996e75f6` | 50 / 500 | 23.644 | 36.142 |

两个 sample amount 水平都显示记录中的 DCM-volume 条件差别大于这两角之间的 amount 差别，符合“检查被模型压缩的条件效应”的选点意图；由于原数据不是本任务重新随机化实验，不能因而宣称溶剂体积因果效应。下一轮 feedback_interpretation 明确引用 m125 的结果并对 m207 irregularity 保留意见，但该下一批没有继续选这三个案例分子的相同行；是解释更新的实例，不是“所有失败分子都会连续重采”的证据。

**非单调 amount 检查（同 seed/r10）**：m207 `OCc1cccc(Br)c1` 已测 50µl 的 V2=34.025、100µl 的 V2=10.417。选择 150µl 的 `a7906197994af5c86745`，冻结预测 V2=22.935、揭示 V2=36.408。它补上了一个真实条件点，却没有支持通用单调 loading 曲线；测量变异或未记录条件依旧是替代解释。

**后期 amount midpoint（seed 887，r35）**：`716d04ed1943f0708912` 是 m135、PE/EA=5/1 的 100µl 点，先前已测 50/150µl 边界。LLM 明确测试 amount curvature；冻结 V1/V2=22.256/34.574，揭示20.925/34.017。这个点与补全意图一致，但它属于最后一批，后面没有新选点行为可检验，不能宣称由此证明持续反馈收益。

## 7. 对机制假设的含义及复现

H3 可以支持为“Free 更常把一个批次用于同一分子的多维条件设计，amount 维度差别突出”，不能支持为“Free 的 PE/EA 曲线补全比 LCMD 更多”或“局部设计导致中期改善”。H4 的**行为部分**有近期反馈邻近性证据；**有助于改善部分**未得到稳定关联支持。H5 有输入、文本和实际数值偏好的关联证据，因果贡献仍需要 NoCW；不能由 mention 频率宣称因果收益。

复现：

```bash
MPLCONFIGDIR=/tmp/free_lcmd_diagnostics_mpl /Users/fish/miniforge3/envs/chromatography/bin/python studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/selection_behavior_analysis.py
MPLCONFIGDIR=/tmp/free_lcmd_diagnostics_mpl /Users/fish/miniforge3/envs/chromatography/bin/python -m unittest discover -s studies/active_learning/free_lcmd_mechanism_diagnostics_20261029 -p 'test_selection_behavior.py' -v
```

轻量检查验证：完整 360 批/11520 行、每批 source 中真实 ID 顺序、输出无真值列、历史计数等于32×已完成批数、matched 条件约束、interpolation 时间边界、重复 tuple 不冒充补全、schema 键不冒充文本证据、物理 failure 不冒充模型错误。图为 `chemical_condition_behavior.png`、`historical_error_neighborhoods.png`，均已人工检查。
