# 冻结梯度几何诊断（B1）

本分析覆盖五 seed、两方法、360 个真实冻结批次（11,520 次选点）。本地原始 q50 CountSketch bank 为 360/360；Free 的 CW bank 为 180/180。全部 bank 的文件哈希、逐行 canonical index、各自 checkpoint 哈希及选中 ID 已核验。Free 全候选 CW 分位与冻结 CW bank 重算的一致性为 180/180，通过容差 5.1e-6（展示舍入）。LCMD 没有沿该原始梯度轨迹冻结的 CW bank，全部 CW 指标保留 NA。没有新训练、推理、梯度提取、selector 重跑或真实标签表读取。

## 直接结果

Free 并非均匀覆盖采集器。其选中点的 CW 距离平均分位为 0.59254；30.35% 选点位于自身当轮候选池的最高距离十分位。该偏好主要集中在 525–813，随后减弱，不能把所有轮次归结为固定的高 CW 贪心策略。Free 选点的 raw 与 CW 分位高度相似：各 seed 的选点级描述性 Spearman 为 0.987–0.991；两种信息及化学/条件特征可能共线，观察到高 CW 不能识别 CW 输入的因果作用。

| 选点时预算阶段 | Free raw 平均分位 | LCMD raw 平均分位 | Free CW 平均分位 | Free CW 最高十分位占比 | Free 剩余 U 覆盖相对改善 | LCMD 剩余 U 覆盖相对改善 |
|---|---:|---:|---:|---:|---:|---:|
| 333–525 | 0.6266 | 0.9650 | 0.6301 | 0.3302 | 0.0364 | 0.0763 |
| 525–653 | 0.6773 | 0.9374 | 0.6787 | 0.4563 | 0.0359 | 0.0578 |
| 653–813 | 0.6766 | 0.9168 | 0.6748 | 0.4538 | 0.0403 | 0.0596 |
| 813–1005 | 0.5723 | 0.9235 | 0.5753 | 0.3125 | 0.0301 | 0.0730 |
| 1005–1197 | 0.5434 | 0.9491 | 0.5474 | 0.2125 | 0.0265 | 0.1043 |
| 1197–1485 | 0.5216 | 0.9791 | 0.5251 | 0.1889 | 0.0287 | 0.2627 |

表内先取各 seed 的阶段批次均值，再以五 seed 等权平均；阶段为选点前预算的左闭右开区间。数值都是各自模型状态定义的分位或相对量。Free 与 LCMD 不在同一训练状态，表格不是固定模型下的策略因果比较。

LCMD 的 raw 选中分位全程 0.94921，Free 为 0.59000。LCMD 的排除选中点后覆盖相对改善全程 0.12263，Free 为 0.03226。这两个指标在全部 30 个 seed×阶段配对中均为 LCMD 更大；这不是 30 次独立重复实验，统计独立单位仍为五 seed。

Free 的 raw 批次 centered effective rank 全程均值为 21.09，LCMD 为 24.33；30 个 seed×阶段配对中 29 个为 LCMD 更高。Free 中 4.65% 的选点有批内邻居距小于其当前 L 最近距离的一半，LCMD 为 0.0347%；该差异在 29/30 个 seed×阶段为正，另一个为平局。因此 Free 的批次通常更集中，但大多数 Free 点也未落入这一“很近”的几何冗余判据，不能把集中直接称作无效重复。

## 653 饱和假设需要收窄

以“剩余 U 平均最近距离的相对下降”定义，LCMD 在 653 后没有饱和：0.0578（525–653）、0.0596（653–813）、0.0730（813–1005）。后期相对改善还显著上升，因此不能据此解释 653–1005 的性能反转。

相对改善的分母会越来越小。为避免该分母效应，另以**各自当前 bank 的 centered feature RMS**归一化几何覆盖基线和覆盖增量。LCMD 的剩余 U 覆盖增量 / bank RMS 六阶段为 0.02343、0.01272、0.00967、0.00804、0.00601、0.00219；Free 为 0.01368、0.01121、0.01005、0.00653、0.00473、0.00425。这支持 LCMD 的几何改善渐减，但没有证据支持 653 附近的突然断点。653–813 两方法该增量已经接近，且仍是在不同模型状态上比较。

LCMD 的覆盖残距 / bank RMS 六阶段为 0.3162、0.2284、0.1714、0.1160、0.0635、0.0125；在选点预算 1389、1421、1453 时降至 0.000775、0.000165、0.000177。此时“每批相对下降很大”可以和“剩余可覆盖距离很小”并存。批内 NN / pool 平均距离在这一时期出现极大值，主要受很小的 pool 分母影响；报告不以该比值断言后期多样性爆增。所有归一化仍使用当前模型几何，不能认为消除了表示变化的影响。

现有证据更适合 H1 的限定表述：“LCMD 在早期具有更强的自身 raw 梯度覆盖，且同时拥有验证优势”；不能进一步断言覆盖导致优势。H2 的“逐渐覆盖、很后期接近几何穷尽”有描述性依据，“653 附近突然饱和导致性能反转”无依据。H5 的因果贡献必须用 NoCW 消融，文本引用、高分位偏好及信息共线都不能替代消融。

## 定义、来源和复现

`geometry_profiles.csv` 为 360 行；`geometry_selected_points.csv` 为 11,520 行，raw/CW 各自列前缀；`geometry_stage_by_seed.csv` 保留 seed 单位，`geometry_stage_summary.csv` 给五 seed 均值与样本标准差。`geometry_availability.json` 保留逐批来源路径、校验结果和输入 SHA-256。

原始距离定义为 512D bank 欧氏距离；rank 为当轮 U 中距离的 average-tie rank / |U|。覆盖相对改善以当轮同一 U 比较 `d(U,L)` 与 `d(U,L∪B)`，并单列排除 B 的敏感性版本。平方距离改善亦保留。有效秩为 centered bank 奇异值归一化后的 entropy effective rank；另给 covariance participation rank 和首特征值占比，不能将两种秩混为一谈。批次居中后最大秩为 31。所有绝对距离列仅供当轮审计，不作跨模型的绝对尺度比较。

LCMD r00–20 acquisition 在 `qgeognn_v2_row_sequential_b32/runtime`，r21–35 在 `qgeognn_v2_row_lcmd_to1485_20261026/runtime`；r21 模型仍由前一目录持有。Free selection r 跟随产生 fit r+1 的 revision；部分阶段边界的 bank 仍由 fit r 的前一 revision 持有，脚本以 checkpoint SHA 和有序 canonical index 同时匹配，避免误用其他轨迹或同预算状态。

源码核验：LCMD 当前 `lcmd.py` SHA-256 与原 protocol 相同（`9c0ede83a898394aa501815f7ae3641bc5f16c7c0343e75ac0a72cecf726e67d`）。冻结 raw gradient 文件 SHA-256 为 `d5de5c2ecdbcd57a4152c7be72f6bfc82028e9c1cfd2ff502c09ac78a5329320`，与原 protocol 相同。冻结版与当前 main 的五个共同函数 AST 全同，当前版本另增加 linear-output sketch 函数及 Mapping import。LCMD 确实是 corrected TP：当前 L 初始为 centers，以 cluster 内平方距离之和选最大 cluster，再选其最远点，批内逐个加入新 center，稳定输入序处理平局。它不是 CW-LCMD。

CW 是先在两个 q50 输出上作 L0-scaled center/width 线性变换，再提取 endpoint-specific CountSketch。不能从已经混合并压缩的一个 512D raw bank 逆推出 CW bank；沿 LCMD 轨迹补 CW 要重新提取梯度，本任务未执行。

复现：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/geometry_analysis.py
python -m pytest studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/test_geometry_analysis.py -q
```

轻量数值测试 3/3 通过：尺度不变性、覆盖排除选点的手算敏感性、effective-rank 与 tied-distance midrank。两幅主图和覆盖分母敏感性图已生成；runtime bank/checkpoint 为本机保留证据，新克隆未必具备，脚本缺失时会失败而不重建。

## 补充：loading amount 的梯度几何弱敏感性

为解释条件补全差异，预先固定选取 seeds 157、887 及 rounds 0、10、20、30（预算 333、653、973、1293），仅用其现存 bank 做 matched-condition probe。每对具有相同 canonical SMILES，且除目标条件以外的所有已记录条件完全相同；目标条件值必须不同。对比三个轴：`V/ul`、`PE/EA`、`Volume of loading solvent/ul`。没有重算梯度或模型输出。

结果跨这组代表状态相当一致：仅改变 amount (`V/ul`，density 已匹配) 的 pair 距离 / bank RMS 中位数约为 0.000127（LCMD raw）、0.000209（Free raw）、0.000239（Free CW）；仅改变 PE/EA 的对应中位数约为 0.631、0.645、0.729。各状态“PE pair 中位距离 / amount pair 中位距离”为 767–14,189 倍，中位 7,814 倍。LCMD raw amount pair 距离小于 1e-3 bank RMS 的比例，按代表状态等权平均，为 99.73%；Free raw 为 93.60%，Free CW 为 92.65%。PE pair 在全部代表状态中这一比例均为零。loading solvent volume 的 matched pair 距离也很小。所有 amount 对比向量距离均非严格零，不能称为模型完全忽略 amount。

这说明现有 full-network 梯度范数度量中，matched amount/volume 对比的几何尺度远小于 matched PE 对比，CW 也没有消除这一尺度差异。该证据可以解释 raw LCMD 对已有同分子同 PE 行的 loading 补点兴趣很弱，以及 Free 仍会采集很多低梯度距离的 amount 对比；它不是后续验证收益的因果证明。这里比较的是数据中实际存在的有限条件差异，**没有按物理单位或扰动幅度转换成导数敏感性**，因此不能将倍数解释为 amount 与 PE 的单位因果效应比。pair 和模型状态相关，round 0 两方法还共享初始模型；不作 pair-level 显著性推断。

将全部真实选点与 B2 的严格 matched amount-extension 标志连接后，Free 的 amount-extension 选点 raw/CW 平均分位均约 0.2116，其余选点 raw 为 0.7907、CW 为 0.7946。这一混合结构解释了 Free 全体平均 CW 分位 0.5925 的局限：它同时包含高距离采集与低距离条件补全，不能仅按总体分位给出单一机制。LCMD 的 amount 补点主要发生在很后期，此时池内梯度残距已非常小，所以仍可能获得较高的**池内相对**距离分位。

补充文件为 `matched_condition_geometry_probe.csv/json`、`behavior_geometry_amount_link.csv` 和 `condition_geometry_probe.py`。JSON 保留各状态逐轴数量、距离分位、完全零比例及带真实冻结 ID/条件的代表 pair。复现：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/condition_geometry_probe.py
```

补充 matched-pair 判据测试亦通过，当前轻量测试合计 4/4。所有 PNG 图均已人工视觉检查。

## 最后稳健性核验

matched-condition 距离采用 `np.linalg.norm(x[i]-x[j], axis=1)/rms`，直接对向量差求范数，小距离没有使用 norm-dot 相减。`lcmd_inputs` 已取消只比较 checkpoint 的宽松 fallback：现在 acquisition input 的全部字段必须逐项与 bank contract 一致，并核对 seed、轮次、L/U 哈希、checkpoint state/hash、L0 scales、batch size、512D、sketch seed、禁用 test 的计数、有序 canonical indices 及 metadata contract hash。对 180 个真实 LCMD contract 的 metadata-only 复核全部通过，未重新计算几何指标。

`behavior_geometry_amount_link.csv` 及 probe JSON 已按最新 `selection_records.csv` 的 **new-only `amount_curve_extension`** 刷新，Free 为 1996/5760；它与 `amount_matched_contrast` 标志不同。probe JSON 的代表 pair 已带 `id_a/id_b` 和 seed/method/round/bank path，可直接定位冻结选择状态；该几何诊断不需要读取 pair 的响应。最终轻量测试 5/5 通过，新增反例测试确保即便 checkpoint 相同，轮次、L 哈希或 scales 不一致也会拒绝。证据见 `geometry_contract_robustness_audit.json`。
