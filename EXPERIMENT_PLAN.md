# QGeoGNN 主动学习与跨柱迁移实验计划

版本：V1.4（E2 row机制审计完成、compound正式pilot启动）

日期：2026-08-19

依据说明：本文件是当前阶段顺序、Gate、验收标准与仓库执行状态的统一入口。最新研究口径优先于旧版阶段编号；未获数据或代码证据支持的规则不得写成既定结论。

## 0. 阅读导航与当前进度

### 0.1 一句话路线

先完成 Gate 0 并冻结 Predictor，再证明 acquisition signal 能识别真实高误差样本；随后在 4g 跑通逐轮重训，最后把主要结论放在 4g→8g 主动迁移。任何复杂方法都必须通过前一阶段 Gate 才进入。

### 0.2 阶段地图

| 阶段 | 核心问题 | 主要产物 | 进入下一阶段的条件 | 当前状态 |
|---|---|---|---|---|
| Step 1 / E0-1 | 到底用了哪些数据，过滤后还剩多少？ | manifest、异常明细、口径报告 | 数据版本和过滤规则可追溯 | **当前仓库口径已冻结** |
| E0-2/E0-3 | 当前代码能否复现 4g 与 4g→8g？ | 固定 split、scaler、best checkpoint、基线表 | Gate 0 | **完成；第一构象、末两层+头与V1/V2等权已由后续Gate 0冻结** |
| Gate 0 | Predictor 的分位数、校准、阈值与迁移结构是否可冻结？ | G0-1～G0-4 配对证据、冻结清单 | 冻结 Predictor；此后禁止按 AL test 回调 | **完全通过：科学配置冻结，D28统一接口/10k+推理/身份/resume/paired partition均通过** |
| E1 | acquisition signal 是否真的指向高误差？ | signal-error、hard-error enrichment、risk-coverage 图 | Gate 1 | **完成；Ensemble 12/16恰好过门，Latent Distance支持Coverage** |
| E2 | 主动选点并重训是否优于 Random？ | 4g 学习曲线与 AULC | Gate 2 | **row与compound三seed正式pilot均完成；compound Gate suggestive（Hybrid/Coverage 2/3胜Random），本阶段STOP** |
| E4 | 能否节省 8g 标签？ | 主结果、paired CI、标签节省 | 主结论成立 | Protocol A 三seed正式pilot已完成；主动策略相对Random为null，Protocol B未启动 |
| E3/E5 | 为什么有效、在哪里失效、是否改善 SQ？ | 消融、OOD、下游排序 | 机制和边界清楚 | 待实施 |
| E6 | 是否值得增加方法创新？ | Task-aware/B³AL/跨柱扩展 | 基础链路稳定 | 条件性开展 |

### 0.3 每个阶段统一执行方式

每个实验模块都按下面六项执行，缺一项不算完成：

1. **输入锁定**：数据 stage、split hash、L0、seed、anchor checkpoint。
2. **配置冻结**：训练预算、early stopping、ensemble members、acquisition 参数。
3. **运行前 smoke test**：1 seed、2 rounds，检查索引、训练和日志。
4. **正式运行**：paired seeds，共享测试集与初始集。
5. **验收**：检查产物完整性、主指标和 Gate 条件。
6. **失败处理**：按预注册分支排查，不事后挑 seed、改主指标或只汇报有利预算点。

## 1. 研究目标与主结论口径

### 1.1 当前真正研究问题

通过主动学习选择最有价值的新实验标签，逐轮重新训练 QGeoGNN，能否在有限标签预算下提高 V1/V2 预测能力？在已有 4g 预训练模型的前提下，能否通过主动选择少量 8g 数据，比 Random 使用更少的 8g 标签达到接近全量迁移模型的性能？

### 1.2 主线与边界

- 方法验证：在 4g 离线池上验证不确定性、批量去冗余和“查询后重训”的闭环。
- 主实验：4g→8g 批量主动迁移。
- 机制验证：UQ 质量、表示空间、批量大小、OOD 和训练重启策略。
- 下游验证：held-out 化合物对的分离质量（SQ）排序与推荐后悔值。
- 扩展：25g/40g、B³AL、AGBAL、PBNN、成本感知与真实自动化实验，仅在主线通过决策门后开展。
- 固定模型下寻找最大 SQ 条件属于主动搜索/贝叶斯优化，不作为“QGeoGNN 预测能力提高”的证据。

### 1.3 预注册主指标

- Primary metric：两个输出分别标准化后的 RMSE 学习曲线下面积（normalized RMSE-AULC，越低越好）。
- Secondary metrics：labels-to-90%-ceiling、labels-to-95%-ceiling、MAE、R²、UQ 校准、pair-SQ NDCG@K。
- Primary comparison：Hybrid vs Random transfer。
- 主结论成立条件：Hybrid 或预注册的最佳主动策略在 paired seeds 下显著降低 AULC，并减少达到全量 8g 迁移上界 90% 性能所需的目标域标签数。

## 2. 当前仓库审计结论

### 2.1 已具备的能力

- `GINGraphPooling` 已能输出 V1、V2 各自的 q10/q50/q90，共 6 个输出。
- 前向传播同时返回 `h_graph`；默认 `emb_dim=128`，可以直接作为 coverage、k-center、LCMD 和 Hybrid 的联合表示基础。
- 已有 4g、8g、25g 等独立训练/迁移函数及分位数损失，能作为重构后的模型内核。
- 已有 SQ/分离概率计算逻辑，可重构成独立的下游评价模块。

### 2.2 Gate 0 前必须解决的问题

1. 原入口数据文件口径不一致：仓库提供英文列名 CSV，但 `utils.py` 读取缺失的 `*_with_index.xlsx`，并使用中文列名；E0-2 已增加独立 CSV 入口，原函数暂未改写。
2. 仓库未包含原论文运行时使用的 `dataset_save/*.npy`、3D 构象缓存、预训练 checkpoint、固定 split；E0-2/E0-3 已按当前仓库 CSV 为4g与8g重新生成并冻结所需资产，其中8g只持久化4g缓存中不存在的新结构。
3. 数据缩放泄漏：构图函数在划分前使用全数据 min/max；test/pool 信息进入 scaler。
4. 连续描述符被转为 `int64`：Min-Max 后的连续值会大量退化为 0/1，必须保持 `float32`。
5. 训练阶段周期性计算 test 指标，且测试时加载硬编码 epoch；应仅用 validation 选择 best checkpoint，test 只在最终评价时运行一次。
6. 8g 迁移学习率代码为 `1e-5`，方案建议主值为 `1e-4`；必须在 Gate 0 中固定唯一配置并记录依据。
7. 训练 DataLoader 使用 `shuffle=False`，缺少真正的 early stopping、断点续跑、统一配置和运行级元数据。
8. 4g 与 8g 的训练/迁移函数高度重复且由全局变量控制，不适合逐轮 AL、paired seeds 和策略公平比较。
9. legacy 独立 q10/q50/q90 即使加入 crossing penalty 仍有 crossing，不能直接把区间宽度当成合法 acquisition signal。
10. 80% 区间尤其是 V2 存在 undercoverage；任何 inflation factor 必须只用 validation 估计。
11. legacy 60/120 mL 阈值可能预先删除主动学习最需要的尾部难例，必须与 no-threshold 做敏感性对照。
12. 当前 `last2+head` 对照不等于论文描述的“柱规格输入 + 新任务/输出头适配”，需克制地补齐 paper-style transfer 对照。

### 2.3 E0-1 部分数据审计结果（2026-08-07）

| 数据集 | 原始行 | 当前 reader | 当前 QGeoGNN 实际行 | 主要问题 |
|---|---:|---:|---:|---|
| 4g | 4,243 | 4,229 | 4,163 | 14 行核心缺失、1 行负标签、8 行 t1>t2；构图阈值排除66条 reader 记录 |
| 8g | 574 | 574 | 552 | 2 行 t1>t2；构图阈值排除22条 |
| 25g | 569 | 490 | 408 | 6 行核心缺失、73 行负标签；固定体积阈值额外排除82条 reader 记录 |
| 40g | 531 | 529 | 456 | 2 行负标签、2 行 t1>t2；代码使用150/200 mL阈值并排除73条 reader 记录 |

8g 口径已经部分解释：

- 582：实验方案引用的论文口径；当前仓库没有对应原始表，暂不能逐行复核。
- 574：当前 `dataset_8g.csv` 原始行数。
- 552：574 行经过现有构图规则 `V1=t1×flow/1200≤60 mL`、`V2=t2×flow/1200≤120 mL` 后的数量。
- 574→552 删除22行，其中 V1 超阈值21行、V2超阈值12行，两类有重叠。

当前审计产物位于 `experiments/data_audit/`：

- `data_manifest.csv`：8个数据集的原始、reader兼容和当前QGeoGNN口径；
- `dataset_summary.csv`：数据集级异常与阈值统计；
- `issue_details.csv`：需核查记录及其源行号；
- `AUDIT_REPORT.md`：口径结论与 Gate 0 未决问题；

已确认的实验口径：以当前仓库 CSV 为唯一数据源；基线按原代码启用体积阈值；重复实验代表重复测量并全部保留；论文4g/8g行数不再作为阻塞项。4g 已完成逐行决策、单位换算显式记录、RDKit/3D/图构建检查，并导出 `experiments/e0_4g_baseline/canonical_4g.csv`。

### 2.4 E0-2 4g 单seed基线复现（2026-08-13）

- 环境：conda `fish`；CPU（当前机器无CUDA/MPS）。
- canonical数据：4,163行、217个唯一canonical SMILES；重复实验104行保留。
- 图构建：217/217成功，其中216个MMFF、1个UFF回退。
- 固定行级80/10/10 split，seed=42：train/valid/test = 3330/416/417。
- scaler仅在train上拟合；best checkpoint仅按validation选择；test训练结束后只评价一次。
- 最佳epoch=91，早停于191轮。
- 固定test：V1 MAE=1.6040、RMSE=2.9713、R²=0.8669；V2 MAE=2.8682、RMSE=4.9691、R²=0.9029。
- 与方案引用目标R²(0.859/0.913)相比，V1达到，V2低约0.010；当前本地结果作为后续paired实验基线。
- test mean pinball loss：V1=0.6736、V2=1.0551；名义80%区间覆盖率：V1=0.8321、V2=0.6163。
- test quantile crossing rate=0.0504，且V2区间覆盖不足；该问题现已分别由G0-1结构性单调输出与G0-2 validation-only校准处理，但Quantile Width仍须通过E1资格测试。

当前状态：已在 conda `fish` 的 RDKit 环境完成4g canonicalization、SMILES parse、3D conformer与图构建审计；8g将在E0-3开始前按同一流程补齐。阈值按当前代码逐数据集复现：4g/8g/25g/DCM/C18/NH2/CN 为60/120 mL，40g为150/200 mL；记录该规则不代表认可其科学合理性。

数据处理原则：不盲目去重。相同化合物—条件的多条记录先判定为真实重复实验还是重复录入；真实重复可用于估计实验噪声，但同组记录必须整体分配到同一 split，避免泄漏。

## 3. 实验工程底座（先完成，后跑主实验）

建议新增独立实验层，不直接在 3,900 行的 `QGeoGNN.py` 中继续堆分支：

```text
experiments/
  configs/                 # baseline、uq、4g_al、8g_transfer
  data.py                  # 统一 schema、过滤、单位换算、manifest
  splits.py                # ID/compound/scaffold/condition/column split
  graph_cache.py           # 图、角图、静态描述符缓存
  trainer.py               # fit(labeled_idx, init_ckpt, seed)
  predictor.py             # 分位数、q50、h_graph、ensemble covariance
  active_loop.py           # reveal、重训、评价、断点续跑
  acquisition/
    random.py
    coverage.py
    ensemble.py
    lcmd.py
    hybrid.py
  metrics.py               # 回归、UQ、AULC、统计检验
  downstream_sq.py         # pair-SQ、NDCG、regret
```

### 3.1 统一样本定义

- `sample_id`：数据版本内稳定且唯一，不能依赖过滤后的行号。
- 查询单位：单化合物 × 完整实验条件 × 柱规格。
- 标签：`y=(V1,V2)`；明确从原始 t 到 V 的单位换算并写入 manifest。
- `condition_key`：至少包含柱规格、洗脱剂比例、流速、上样溶剂、上样溶剂体积和上样质量；用于条件 OOD 和重复组识别。
- `compound_id`：优先用标准化结构标识（canonical SMILES/InChIKey），CAS 仅作追溯字段。

### 3.2 数据 manifest

`data_manifest.csv` 至少记录：

- source file、SHA256、原始行数；
- 每步过滤原因与行数；
- 缺失值、负标签、V1>V2、构象失败、重复键处理；
- 单位换算公式和输出单位；
- 最终样本数、化合物数、scaffold 数、condition_key 数；
- split 文件路径与 split hash。

### 3.3 训练与预测接口约束

- scaler不得使用validation/test。E2 source-free在固定L0-train上拟合一次并跨轮次/策略冻结；E4迁移复用Gate 0冻结的4g source-train scaler。
- 每轮从同一 anchor 重启；warm-start 仅作消融。
- 只用 validation early stopping；禁止按 test 结果选 epoch。
- `predict()` 返回：两个输出的 q10/q50/q90、128 维 h_graph、sample_id。
- 每个 ensemble 成员保存固定 `member_id` 和 seed；成员顺序不可随轮次变化。
- 所有方法共用相同 split、L0、validation、pool、test、训练 epoch 上限和 early-stopping 规则。

### 3.4 每轮必须保存

- `config.yaml`、代码 commit、环境信息、数据/划分 hash；
- `labeled_indices.csv`、`selected_indices.csv`；
- `pool_predictions.parquet`（成员预测、均值、协方差、UQ、embedding 引用）；
- best checkpoint、scaler、validation 轨迹、test 指标；
- acquisition 耗时、训练耗时、GPU 峰值显存；
- `state.json`，支持从任意 round 断点续跑。

## 4. 所有主动学习实验的公平协议

1. 对每个 outer seed，预先固定 test、L0、target validation 和 U0。
2. 同一 outer seed 下所有策略使用完全相同的初始索引和测试集。
3. 第 t 轮仅使用 Lt 拟合模型和 scaler；对 Ut 预测后一次性选择 B 个样本。
4. 离线仿真从历史表 reveal 标签，更新 Lt+1 和 Ut+1。
5. 从同一初始化方案重新训练，并在固定 test 上评价。
6. 横轴使用累计目标域标签数，不使用 round；validation 标签计入预算。
7. Random、Coverage、Ensemble、Hybrid 均使用相同数量的 ensemble 成员做最终预测，以免把“模型集成收益”误当成“采样策略收益”。各策略只在 acquisition 规则上不同。
8. Pilot：3-member ensemble、3 paired outer seeds；Final：5-member ensemble、至少 5 paired outer seeds。
9. outer seed 与 member seed 分离，例如 `member_seed = outer_seed * 100 + member_id`。

## 5. 分阶段实验

下面每个模块先给“实验卡片”，再给具体设置。实验卡片用于日常执行和验收，具体设置用于写配置文件与预注册。

### E0：数据审计与基线复现（P0，必做）

| 项目 | 内容 |
|---|---|
| 目标 | 建立唯一、可复现的 4g/8g 数据与训练基线 |
| 输入 | 当前 CSV、论文/历史数据版本、QGeoGNN 代码、可用 checkpoints |
| 核心执行 | 数据 manifest → canonical reader → 固定 split → 4g 复现 → 4g→8g 复现 |
| 必备产物 | manifest、异常决策表、split 文件、scaler、best checkpoint、基线指标表 |
| 通过 | 数据差异可解释；同 seed 可复现；test 不参与模型选择；迁移基线达到或解释论文差异 |
| 失败处理 | 暂停主动学习；定位数据版本、单位、特征类型、scaler、checkpoint 或微调层问题 |

#### E0-1 数据口径

- 为 4g、8g 建立 canonical CSV 和 manifest。
- 解释 4g 的 4,684/4,243/4,096 三种口径；8g 的 574→552 已解释，继续追查 582→574。
- 确认负标签、缺失值、V1>V2、构象失败和重复实验的处理规则。
- 为每种异常增加 `decision`、`decision_reason` 和 `reviewer` 字段，不能只记录计数。

#### E0-2 4g 复现

- 固定 row-level 80/10/10 split，先复现原论文可比结果。
- 另存 compound-group split，暂不用于替代论文可比基线。
- 报告 V1/V2 的 MAE、RMSE、R²、pinball loss、80% 区间覆盖率和 quantile crossing rate。

#### E0-3 4g→8g 基线复现

比较：

- 8g scratch；
- 4g zero-shot；
- 4g pretrained 后仅预测头微调；
- 末 1–2 层 + 预测头微调；
- 全模型微调；
- 学习率 `1e-5` vs `1e-4`，其他条件固定。

确定后续唯一迁移配置，不在主动学习主实验中继续调参。

E0-3b 已完成 seed=42/525/1101、row-level/compound-group 两种划分下的24组核心配对训练，并在row/seed=42增加4组因素预实验。按跨6个运行的train方差归一化validation score，末两层+预测头、学习率`1e-4`当时为暂定候选（0.619±0.301），优于scratch（0.975±0.532）。compound-group三种子test R²均值为V1=0.901、V2=0.892；row-level均值为0.472、0.637，其中seed=42 test受待复核源行224强烈支配。正式指标保留该行，排除结果仅作敏感性。该段记录E0-3b阶段性证据；后续D04与G0-1～G0-4已完成并在`experiments/PREDICTOR_FREEZE.md`给出最终冻结配置。

E0-3c 已完成3种损失×3 seeds×2 splits共18组。等权损失的跨运行normalized validation最低（0.604），legacy 0.5为0.619，train-SD normalized为0.611；两种改法相对legacy都只赢4/6个split/seed，按3个独立seed计算的paired区间均跨0。操作上将V2等权作为下一阶段暂定候选，但不声称显著优于legacy；standardized loss曾把平均crossing从0.180降至0.080，但G0-1已证明只有结构性单调参数化能把crossing稳定降为0。损失主口径继续保留V1/V2等权，下一步执行G0-3阈值开/关与G0-4 paper-style迁移。

D04 已完成第一构象与最低能构象对照。217个4g结构中仅19个默认第一构象就是最低能构象，198个图的键长/键角发生变化，descriptor全部不变。最低能4g source的validation和crossing改善，但test点误差略差；4g→8g跨6组validation平均改善0.012，但test MAE平均恶化V1=0.460 mL、V2=0.279 mL，且compound三个seed均恶化、paired区间跨0。最低能构象没有稳定收益，后续主口径继续使用原代码`first_embedded`，最低能结果作为论文口径敏感性保留。

#### Gate 0：Predictor qualification 与冻结

Gate 0 必须按 G0-1→G0-2→G0-3→G0-4 的顺序执行；在四项完成前，`first_embedded + last2+head + lr=1e-4 + V1/V2 equal loss`仅是 provisional baseline。

**G0-1 Quantile monotonicity**

- Legacy：独立输出 q10/q50/q90，并保留 crossing penalty。
- Monotonic：`q50=m`、`q10=m-softplus(d_low)`、`q90=m+softplus(d_high)`。
- 固定其他因素，比较 q50 MAE/RMSE/R²、pinball、80% coverage、mean interval width 和 crossing。
- 预注册冻结标准：monotonic 的平均 validation normalized score 与 V1/V2 RMSE 均不比 legacy 恶化超过 5%，且 validation/test crossing 为 0。
- 正式结果：3 seeds × row/compound × 2 参数化共12次训练已完成。Monotonic 相对 legacy 的 validation normalized score/V1 RMSE/V2 RMSE 分别变化 +0.70%/+0.57%/+0.11%，均在5%阈值内；crossing 从 validation/test 13.6%/17.7% 降为0。G0-1通过，冻结结构性单调输出。

**G0-2 Validation-only interval calibration**

- 分别在 validation 为 V1/V2 估计 `alpha_V1`、`alpha_V2`，按中位数向两侧缩放区间。
- test 严禁参与 alpha 选择，只最终报告 coverage、width、AUCE、crossing。
- G0-1 的 validation/test 逐样本预测必须同时保存，以保证校准可审计。
- 已完成：平均 test coverage 从 V1/V2=0.709/0.519 校准到0.824/0.854，crossing保持0；alpha中位数为1.396/1.709。row/seed=1101因区间塌缩需要16.11/55.91，说明 Quantile Width 跨seed稳定性不足，必须在E1作为待否证信号而不是默认主 acquisition。

**G0-3 Threshold sensitivity**

- 对照 legacy `V1≤60, V2≤120` 与 no-threshold。
- 重点报告 tail error、signal-error relationship、calibration 和 high-score 样本分布，不按单个 R² 决策。
- 阈值结论必须先于正式 AL pool 冻结，防止提前删除最值得查询的样本。
- 已完成12次配对训练：validation-only规则选择no-threshold；common validation normalized RMSE改善3.4%，tail normalized absolute error改善23.4%。独立test的common RMSE仅恶化1.3%，但tail error平均恶化40.1%且5/6 contexts更差，登记为小tail test高方差与尾部拟合不稳定，不用test反向改选。
- Tail在test中仅占约4.3%，却在calibrated width最高10%中富集8.1～8.7倍。正式AL pool保留全部574行，不再用60/120 mL阈值提前删除难例；E1必须把tail作为单独失败切片。

**G0-4 Paper-style transfer**

- 只比较 last2+head、full fine-tune 与 paper-style transfer，不做大规模 architecture search。
- paper-style 必须显式包含柱规格输入、新任务适配与输出头适配，不能用“随机重置同构头”代替。
- 论文只规定新柱规格输入、更新输出层与 `lr=1e-4` 微调，并未公开逐层冻结图；本项目采用仓库已预留的 `column_dia/column_len/column_den` edge-level RBF 输入，迁移形状兼容的4g权重，将新增线性适配器零初始化以严格保持source初始函数，并训练新增输入适配器、末两层GNN与单调输出头。
- 预注册选择规则：paper-style 的平均 paired validation normalized score 至少改善2%、至少赢4/6个`split×seed` context，且row或compound任一split-mode均值不恶化超过5%，才替换last2+head；否则保留last2+head。Full fine-tune只作诊断对照，不用于放宽该规则。
- 已完成：新增适配器随机初始化的首轮诊断破坏source初始函数，结果完整保留但不用于论文方案结论；改为零初始化后按原规则重跑。修正版paper-style平均validation normalized score为0.295，较last2的0.256恶化15.3%，仅赢1/6；full为0.333，较last2恶化29.9%。因此冻结last2+head。独立test上paper-style较last2小幅改善1.3%，但不用于反向选择。

四项科学对照已完成，冻结清单见`experiments/PREDICTOR_FREEZE.md`。主动学习阶段不允许依据 AL test 结果返回调 Predictor。

工程验收已完成：固定split/seed可复现；test不参与scaler、early stopping、checkpoint或calibration选择；10240候选跨batch/chunk逐值一致；索引顺序稳定；round resume与连续运行的RNG和集合状态一致。详见`experiments/d28_al_engineering/`。

### E1：Acquisition Signal Qualification（P0，必做）

| 项目 | 内容 |
|---|---|
| 目标 | 判断候选 signal 是否与真实、可消除误差相关；本阶段不是主动学习主实验 |
| 输入 | Gate 0 固定模型、ID/OOD tests、3–5个独立模型成员 |
| 核心执行 | 计算 Quantile width、Ensemble covariance、latent density，并与真实误差逐样本对齐 |
| 必备产物 | `uq_predictions.parquet`、UQ-error 图、hard-error enrichment、校准与 risk-coverage 图 |
| 通过 | Ensemble 在多数关键切片优于 quantile width，并能富集真实高误差样本 |
| 失败处理 | 放弃纯 uncertainty top-B 主线，转向 Coverage/LCMD；检查 ensemble 是否真正独立 |

#### 比较信号

- Quantile width：两个输出先按训练集尺度标准化，再平均 `(q90-q10)`。
- Deep Ensemble：成员 q50 的双输出经验协方差；主分数为标准化 covariance trace。
- Latent density/distance：`h_graph+conditions` 到训练集的 kNN 距离；它是 representation-based novelty/coverage signal，不称为严格 uncertainty。
- Random：无信息 sanity baseline。

可选的 MC Dropout/MVE/Evidential 不进入第一轮主比较。

#### 评价切片

- 4g ID test；
- 4g compound OOD；
- 4g scaffold OOD；
- 4g condition OOD；
- 8g zero-shot。

#### 指标

- Spearman(UQ, 标准化绝对误差/平方误差)；
- Top-10% hard-error enrichment：最高 score 10% 中真实最高 error 10% 的富集；
- risk-coverage：依次删除最高 score 样本后，剩余预测风险是否持续降低，并报告 AUSE；
- 80% coverage、mean interval width、AUCE/ENCE、crossing rate。

#### Gate 1

若 Ensemble 在多数关键切片上稳定识别高误差并优于 quantile width，则保留 Ensemble uncertainty；若 Quantile Width 与误差几乎无相关，只保留为 QGeoGNN legacy UQ baseline；若 latent distance 有效，进入 coverage/diversity 主线。若所有 uncertainty 都弱，不强行做 uncertainty AL，E2 以 Coverage/LCMD 为主线。

E1预注册工程判据：primary error固定为训练尺度标准化后的V1/V2绝对误差均值；raw与calibrated Quantile Width不拆成两种ranking。关键聚合切片为row/full、row/common、compound/full、compound/common。Ensemble相对Quantile Width在Spearman、AUROC、enrichment（越高越好）与AUSE（越低越好）的16个聚合比较中至少赢12个，且关键切片平均Spearman>0.1、enrichment>1.5，才进入E2主uncertainty策略。Quantile Width或Latent Distance只有在至少3/4关键切片Spearman为正、关键均值Spearman>0.1且enrichment>1.5时才作为主信号；否则降为secondary baseline或仅保留coverage动机。Tail因样本少只作失败/机制切片，不单独触发方法入选。

E1已完成：K=3真独立Ensemble相对Quantile Width在16项聚合比较中赢12项，恰好达到门槛；四个关键切片的平均Spearman/enrichment/AUSE分别为0.481/5.28/0.033。Latent Distance为0.499/5.69/0.033，支持Coverage进入E2。Quantile Width也满足最低资格（0.434/4.44/0.052），但其row seed1101明显失效，且Ensemble已占用唯一主uncertainty入口，因此仅作secondary/legacy诊断。12/16不解释为普遍优势：Ensemble包揽row两个切片8项胜利，但在compound/full的Spearman、AUROC、enrichment均略输Quantile Width。Tail每个run仅2--3个样本，不作方法选择。完整证据见`experiments/e1_signal_qualification/`。

### E2：4g 主动学习闭环（P1，方法验证）

| 项目 | 内容 |
|---|---|
| 目标 | 验证查询后重训、批量选点和 paired 比较链路正确 |
| 输入 | 固定 4g test/L0/U0、source-free 初始化方案、E1 UQ 结果 |
| 核心执行 | Random、Coverage、Ensemble、Hybrid 从同一 L0 逐轮 reveal 与重训 |
| 必备产物 | 每轮索引、pool prediction、checkpoint、学习曲线、AULC、批内相似度 |
| 通过 | 至少一种主动策略稳定优于 Random；机制解释使用固定共同参考系诊断并保持非因果口径 |
| 失败处理 | 检查模型是否更新、UQ是否有效、L0是否失衡、batch是否过大和标签噪声 |

#### Pilot 配置

- Test：固定 10% ID；另做 compound-group 测试。
- L0：训练可用池的 10%；初始 validation 从 L0 中固定划出 15%，并计入预算。
- Batch：B=25，8 轮；总新增标签 200。
- Seeds：3 paired outer seeds；3 members。
- Source-free初始化：E2禁止加载用完整4g标签训练的anchor，否则会泄漏U0/test标签。每个member按固定seed随机初始化QGeoGNN与单调头，全模型可训练；其独立配置hash与Gate 0的4g→8g `last2+head`迁移配置分开。输入scaler只用固定L0-train拟合一次，随后跨轮次与策略冻结；validation、U0和test不参与拟合。

#### 方法

第一轮主比较严格限制为四种：

1. Random；
2. Coverage（h_graph+conditions 上 farthest-first/k-center）；
3. Ensemble covariance-trace top-B；
4. Hybrid：先保留 Ensemble score 前 25% 候选，再用 LCMD/farthest-first 选 B。

Quantile Width保留为secondary/legacy离线诊断，不作为第五个主策略；Stratified Random延后到主链路稳定后的敏感性分析。

E2 Round-0 source-free正式诊断已完成。Ensemble、Latent Distance、Quantile Width在3/3 seeds的Spearman均为正，跨seed均值分别为0.592、0.436、0.503，hard-error enrichment分别为4.36、3.82、5.04；Random均值Spearman为0.012。E1主信号没有在source-free regime失效，但该结果不是主动学习科学结论。E2 Round-0中Ensemble–Latent Spearman为0.373/0.666/0.387，Top-10% overlap为0.396/0.396/0.467，显示中等相关而非高度重合，为Hybrid提供机制动机，但不能预判Hybrid的AULC。

Coverage主实现冻结为：当前非validation已标注训练集拟合`h_graph`与9维condition各自的feature-wise z-score，拼接后使用Euclidean；首中心为距当前训练表示最远的pool样本，之后逐点更新到已选batch的最小距离，tie按`sample_id`字典序。Hybrid固定先取Ensemble score Top-25%，再复用完全相同的farthest-first实现选择B=25。Ensemble固定K=3、独立seeded random initialization及标准化V1/V2 q50经验协方差trace。

#### 目的与 Gate 2

- 先验证每轮 reveal 后确实重新训练，模型和预测会随 Lt 改变。
- Hybrid 同时提高 fixed/common-reference batch diversity 并改善 AULC，只能说明结果与去冗余机制一致；只有 uncertainty-filter-random causal control 通过后，才允许把 diversity selection 作为更直接的原因证据。
- 若所有策略均不稳定优于 Random，先审查 split、UQ、初始集与训练噪声，不进入 B³AL/AGBAL/PBNN。

### E3：批量与表示消融（P2，有条件开展）

| 项目 | 内容 |
|---|---|
| 目标 | 解释收益来自 UQ、表示、批量去冗余还是训练路径 |
| 输入 | E2 中最佳两种主动策略 |
| 核心执行 | 表示、batch、L0、多输出 UQ、warm-start 的单因素消融 |
| 必备产物 | 消融 AULC 表、批内距离、计算成本、paired effect |
| 通过 | 主要机制跨 seed 一致，且不是单预算点偶然领先 |
| 失败处理 | 缩小方法结论，不把复杂表示包装成稳定创新 |

仅对 E2 中最佳 2 个主动方法开展：

- 表示：MorganFP、conditions-only、h_graph、h_graph+conditions；
- Batch size：B=5/10/25，总新增标签保持一致；
- L0：5%/10%/20%；
- 重启：retrain-from-anchor vs warm-start；
- 多输出 UQ：Trace、Worst-output、LogDet；
- 可选 LCMD vs B³AL-LCMD/MaxDet。

选择标准不是单个预算点最高，而是 paired AULC、labels-to-90%-ceiling 和跨 seed 方差。

### E4：4g→8g 批量主动迁移（P1，主实验）

| 项目 | 内容 |
|---|---|
| 目标 | 用更少 8g 标签达到接近全量 8g transfer 的性能 |
| 输入 | 固定 4g anchors、8g target pool/test/L0、E2 冻结的方法与超参数 |
| 核心执行 | Random transfer、Coverage/LCMD、Ensemble、Hybrid 的 paired active transfer |
| 必备产物 | normalized RMSE-AULC、labels-to-ceiling、paired CI、上下界与计算成本 |
| 通过 | Hybrid/最佳方法降低 AULC，并减少达到90%上界所需标签数 |
| 失败处理 | 分解迁移收益与采样收益；按 Protocol A/B 限定结论；检查负迁移 |

#### 主协议

- Source：完整 4g 训练集；每个 outer seed 固定一组 source checkpoints。
- Target：Gate 0 后的 8g canonical 数据。
- Test：Protocol A 固定约 10%；Protocol B primary 按 canonical compound 分组固定 15–20%；scaffold OOD 延后到 E5。
- L0：50 个目标标签，其中固定 8 个作 target validation，全部计入预算。
- 每轮：B=10，共 15 轮；累计目标域标签从 50 增至 200。
- 重训：每轮从相同 4g anchor 重新微调当前 Lt。
- Pilot：3 outer seeds × 3 members；Final：至少 5 outer seeds × 5 members。

#### 必备比较

| 类型 | 方法 |
|---|---|
| 下界/迁移拆分 | 4g zero-shot；8g scratch+Random；4g pretrained+Random |
| 旧信号 | pretrained+Quantile-Width |
| 覆盖 | pretrained+Coverage/LCMD |
| UQ | pretrained+Ensemble |
| 主方法 | pretrained+Hybrid |
| 上界 | full-data 8g transfer；同时报告 full-data 8g scratch |

#### 两个结论口径

- Protocol A — calibration：test 与 pool 共享化学空间，回答“少量 8g 实验能否校准新柱”。
- Protocol B — novel-compound：按 canonical compound 隔离 test，回答“校准后能否泛化到目标柱新化合物”；Bemis–Murcko scaffold OOD 是独立的 E5 分析。

如果只在 Protocol A 有效，结论必须限定为“同类化学空间内的跨柱校准”。

### E5：OOD 与下游 SQ 验证（P2）

| 项目 | 内容 |
|---|---|
| 目标 | 判断方法适用边界，并验证误差下降是否改善真实分离排序 |
| 输入 | E4 最佳方法、compound/scaffold/condition splits、审计后的 pair pool |
| 核心执行 | OOD 学习曲线 + held-out pair-SQ 排序和 regret |
| 必备产物 | 分层指标、NDCG@K、top-K precision/recall、recommended-condition regret |
| 通过 | 模型改进能在至少一个严格 held-out 下游任务中稳定传递 |
| 失败处理 | 将结论限制为全局回归或域内校准，不声称改善实验推荐 |

#### OOD

- Compound OOD：同一标准化化合物的全部条件只属于一个集合。
- Scaffold OOD：Bemis–Murcko scaffold 分组。
- Condition OOD：按 condition_key 或洗脱比例/流速域整体留出。
- Column OOD：4g source、8g target。

#### pair-SQ 数据准备

当前仓库没有文档所述的 37,963 对 pair pool，需先生成并审计：

- 每行包含 `pair_id, sample_id_A, sample_id_B, condition_key, true_V1/V2_A, true_V1/V2_B, true_SQ`；
- A/B 必须对应共同可执行条件；
- 训练样本、模型选择与下游 pair test 之间不得通过化合物或条件泄漏；
- true SQ 只能由历史真值构造，不能把同一模型预测当真值。

#### 下游指标

- pair-SQ MAE/Spearman；
- NDCG@K、top-K precision/recall；
- recommended-condition regret；
- 全局 RMSE 与下游排序改善的相关性。

### E6：进阶创新与扩展（P3，仅通过前述 Gates 后）

| 项目 | 内容 |
|---|---|
| 目标 | 在不破坏基础公平协议的前提下增加一个清晰的方法创新 |
| 输入 | E4/E5 稳定结果、已定位的失败模式或效率瓶颈 |
| 核心执行 | 只选择一个方向：Task-aware、B³AL 或跨柱扩展 |
| 必备产物 | 与基础 Hybrid 的公平增量比较、额外计算成本和停止结论 |
| 通过 | AULC、最终误差和跨 seed 稳定性同时不劣于基础方法 |
| 失败处理 | 保留为负结果，不继续堆叠 AGBAL/PBNN/成本模型 |

优先级：

1. Task-aware SQ uncertainty；
2. B³AL-LCMD/MaxDet；
3. 25g/40g 复现；
4. 4g residual error predictor；
5. AGBAL；
6. PBNN transfer prior；
7. 成本感知跨柱预算与真实自动化验证。

停止规则：任何进阶方法若未同时改善 AULC、最终误差和跨 seed 稳定性，不升级为主方法；短暂领先单个预算点只作为探索结果。

## 6. 统计分析与绘图规范

- 主比较使用 paired outer seeds。
- 学习曲线按累计目标标签数对齐，并用梯形积分计算 AULC。
- 报告均值、标准差、bootstrap 95% CI、paired difference 和 effect size。
- 主检验：AULC 的 paired permutation test；样本较少时补充 Wilcoxon signed-rank。
- 多个次要指标控制 FDR；Primary metric/Primary comparison 不做事后更换。
- 曲线展示所有 seed 淡线、均值和 95% CI。
- 同时报告标签效率与计算成本，避免用 5 倍训练成本换来的集成收益被描述成免费收益。

## 7. 运行规模与算力决策

公平的 ensemble 主实验训练量近似为：

```text
fits = methods × outer_seeds × (rounds + 1) × ensemble_members
```

E4 若保留 4 个主动方法，Final 约为 `4 × 5 × 16 × 5 = 1600` 次 8g 微调，另加 scratch、zero-shot 和 full-data 上下界。Gate 0 必须先实测单次微调 GPU 小时，再决定是否并行或减少非核心方法；不能通过减少某一策略的 ensemble 成员来制造不公平比较。

## 8. 八周执行表

| 周 | 工作 | 交付物 | 决策门 |
|---|---|---|---|
| 1 | canonical schema、manifest、异常/重复审计、图缓存 | manifest、数据报告、缓存索引 | 数据口径确认 |
| 2 | trainer/predictor/split 重构；4g 与 4g→8g 复现 | 固定 splits、best checkpoints、baseline 表 | Gate 0 |
| 3 | E1 UQ 诊断 | UQ-error、校准、risk-coverage、OOD 图 | Gate 1 |
| 4 | E2 4g AL pilot | Random/Coverage/Ensemble/Hybrid 学习曲线 | Gate 2 |
| 5 | E4 8g active-transfer pilot | 主协议 smoke/pilot、运行时间估算 | Final 方法冻结 |
| 6 | E4 Final | AULC、标签节省、paired CI、上下界 | 主结论 |
| 7 | E3/E5 | 表示/批量消融、compound OOD、pair-SQ | 机制与适用边界 |
| 8 | E6 中只选一项 | Task-aware、B³AL 或 25g/40g 扩展 | 是否进入下一篇/下一阶段 |

若只有 1 张 GPU，Final 周期应按实测单次 fine-tune 时间扩展，优先保证 paired seeds 和公平协议，不优先增加方法数量。

## 9. 最小可发表结果包

### 必须完成

- E0 数据审计与 4g/8g 迁移复现；
- Quantile width vs Deep Ensemble 的 UQ 证据；
- 4g 真正逐轮重训的 AL 闭环；
- 4g→8g 的 Random、Coverage/LCMD、Ensemble、Hybrid；
- Protocol A + compound/scaffold OOD；
- normalized RMSE-AULC、labels-to-90%-ceiling、paired CI；
- held-out pair-SQ 排序或 regret 验证。

### 第一版不做

- 同时实现 AGBAL、PBNN、3D QP 和跨柱成本分配；
- 把 EI/UCB/Thompson sampling 混入模型 AL 主实验；
- 在基础 Hybrid 未优于 Random 前增加复杂 acquisition；
- 用单 seed 或单轮 R² 作为主要证据。

## 10. 当前实施队列

- [x] 建立 `data_manifest.csv`，完成当前仓库8个数据集的首轮口径审计。
- [x] 解释 8g 的 574→552：现有 60/120 mL 构图阈值删除22条。
- [x] 按已确认口径忽略论文4g/8g行数，以当前仓库CSV作为唯一数据源。
- [x] 在 conda `fish` 中完成4g SMILES canonicalization、parse、3D conformer和图审计；8g随E0-3补齐。
- [x] 为4g逐行记录增加 `decision/decision_reason/reviewer`；8g随E0-3补齐。
- [x] 新建4g canonical CSV reader，使仓库提供的 CSV 可直接运行；8g随E0-3扩展。
- [x] 在E0-2实验层修复全数据缩放泄漏并保持连续特征为 `float32`。
- [x] 在E0-2中把 test 从训练循环移出，实现 validation-best checkpoint。
- [x] 完成E0-2单seed 4g正式复现，并保存row-level及compound-group split。
- [x] 完成E0-3的8g canonical数据、scratch/zero-shot/三种微调范围与双学习率单种子矩阵。
- [x] 完成E0-3b三种子×row/compound稳健性实验，以及损失权重、尺度归一化和预测头重置预实验。
- [x] 完成 G0-1 分位数单调性正式配对实验；按 validation 判据通过并冻结 monotonic softplus 输出。
- [x] 完成 G0-2 validation-only V1/V2 区间校准与独立 test 的 coverage/width/AUCE/crossing 报告；记录 row/seed=1101 极端 inflation failure mode。
- [x] 完成 G0-3 legacy threshold vs no-threshold 尾部/UQ敏感性实验；validation-only选择no-threshold，并记录独立tail test未复现改善。
- [x] 完成 G0-4 last2+head/full/paper-style 克制对照；validation-only冻结last2+head，保留随机初始化诊断与test排序不一致证据。
- [x] 完成10240候选分块推理、跨batch逐值一致、稳定身份/顺序和round resume检查。
- [x] 抽出统一`fit/predict`接口，冻结E2/E4共12份paired test/L0/U0 partition。
- [x] 完成 1 seed、2 rounds、2 members 的Random source-free smoke；50次query无重复，round状态可恢复，checkpoint与固定test预测逐轮变化。
- [x] 增加 Ensemble 与 Coverage/Hybrid，确认每轮索引、模型和预测都发生预期变化。
- [x] 完成E0-3c多种子双split损失尺度对照；V2等权为暂定候选，legacy保留为原代码对照。
- [x] 完成D04最低能构象3-seed双split对照；无稳定收益，保留原代码第一构象主口径。
- [x] Gate 0科学与工程检查全部通过；进入E1 acquisition-signal qualification，不得用最终AL RMSE反向挑选Predictor或signal定义。
- [x] 完成E1三seed×双split信号资格测试；Ensemble以12/16恰好过门，Latent Distance支持Coverage，Tail因每run仅2--3例不作选择。
- [x] 完成E2 row四策略三seed正式pilot；Hybrid/Coverage对Random均3/3 seeds胜出，Ensemble仅2/3且paired CI跨0。
- [x] 完成E2三seed source-free Round-0 signal sanity与E1/E2 signal agreement；Ensemble/Latent无regime failure，Hybrid互补性动机成立。
- [x] 完成E2 row四策略×三seed×九预算×K=3正式paired AULC与fixed common-reference机制审计。
- [x] 完成E2 compound seed42 preflight及正式三seedpilot；无test compound leakage，Hybrid平均最好但仅2/3胜Random，Gate为suggestive，按D37停止。
- [x] 完成D40R语义纠正、corrected smoke与E4 Protocol A三seed正式pilot；Random平均AULC最佳，主动采样证据为null，按协议停止且未启动Protocol B。
