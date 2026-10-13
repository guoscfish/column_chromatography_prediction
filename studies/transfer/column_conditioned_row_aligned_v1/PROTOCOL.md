# A1/A2 Row-aligned v1：Phase 1 冻结协议

日期：2026-10-13（Asia/Shanghai）。主要问题：在统一 Row-CV 模型选择下，A2-FiLM 相对 A1 是否改善 25g/40g 记录级插值预测？本轮只执行 outer seed **769539383** 的五折、两方法，共十次拟合；另外四个 seed、FULL-data 重训和外层评分均需后续确认。本目录的 runner 不提供这些动作。

## 设计审查与唯一实验干预

历史研究的 Row 外层也使用跨目标柱 canonical-SMILES GroupKFold 内层选择。新研究仅将这一内层划分替换为 Row KFold；不修改架构、损失、优化器、选择指标、patience、epoch 聚合或外层身份。新文件、输入白名单、诊断和完整性检查属于执行与证据边界改进，不是新候选。历史研究仍称 Group-CV，不重命名、不覆盖其任何产物。

Row-CV 评价目标柱已见分子条件下的记录级插值，允许 canonical 分子跨 train/validation；它不保证每个验证分子已见，也不是未见化学结构 OOD。Group-CV 评价目标柱未见分子，4g 源柱仍可能见过该分子。两种 CV 验证记录不同，只比较**各协议内部** A2 相对 A1 的收益和 epoch 选择；不以绝对误差比较协议优劣。

独立对每柱 FULL gradient_train 使用 sklearn KFold(n_splits=5, shuffle=True, random_state=42)，输入先按 sample_id 排序，以免 CSV 行顺序改变身份划分。折号按 KFold 顺序同步；25g/40g 每折训练与验证都非空。五折覆盖每个合格训练 ID 恰好一次验证；每折所有目标柱的 sample_id train/validation 交集必须为空。预审当前两目标柱训练 ID 无跨柱交集；若未来数据出现共享 ID，该实现拒绝独立划分并停止，不能静默将同一身份分配到两侧。不同 ID 的相同模型输入允许跨侧，单独披露；它们不是身份重复。

五折训练集高度重叠，五个 outer seed 也在同一批数据上重复划分；fold wins/seed wins 为描述性稳定性指标，不把 25 个折或五个 seed 当作独立物理重复，不做未经支持的显著性推断。单 seed 没有 FiLM 成败判定门槛。原 gate 不迁移为单 seed 的通过/失败规则。已曝光外层测试集无法形成独立确认。

## 数据边界

沿用历史 protocol.json 中 4g checkpoint、数据、graph cache、预处理和源 split 的 SHA256。沿用父 filtered_full_data_benchmark/split_manifest.csv 的五个 Row outer seeds 和原外层 train/validation/test 身份。25g/40g 固定过滤人口为 408/456；FULL 指所有不属于固定外层 validation/test 的合格 gradient_train，包括原 pool。试点训练人口 320/364，源 row-seed-42 train 3330、诊断 validation 416。8g 排除。

feature CSV 和源 CSV 仅解析以下白名单：sample_id、canonical_smiles、PE/EA、loading solvent、Density g/ml、V/ul、Volume of loading solvent/ul。明确排除 t1/t2、verified times、Flow、Ending time、V1/V2 等冗余列。标签读取先只解析 sample_id，再使用 skiprows 与端点 usecols，目标只解析本 seed 的 gradient_train 标签；源只解析 train 和 diagnostic validation 标签。外层标签没有读取请求，外层记录不构建训练/验证图。哈希文件完整字节不等于解析标签；独立保存访问授权清单。

跨柱身份、canonical 分子、不同 ID 的相同图输入分别审计。相同输入依据实际 build_model_data 的张量字节（atom/bond/angle 图与条件特征，排除 y/位置），另记录含 task identity 的键。相同骨干输入在不同柱下并非完全相同的 A1/A2 输入，因为模型还接收柱 task ID。记录内和跨折重合分别披露。元数据 header 缺少明确实验批次/独立重复标识；source_row_1based 仅为来源行号，不能当作实验批次。Row 指标的独立性仅到记录身份，独立物理实验数未知。

## 严格配对与选择

直接使用原 src/qgeognn_al/transfer/column_conditioned_multitask.py：A1 共享 128 维 QGeoGNN 与三个独立 source-head 复制；A2 加 3×16 embedding、block 3/4 的两个 Linear(16,256)，零 gamma/beta 初始化。共享初始化张量逐字节验证相同。

仅训练 node blocks 3/4、index 3 的最后 edge/angle update、condition branch 和 heads，A2 另训练 FiLM/embedding。Adam pretrained/source-head lr=1e-4，new target-head/FiLM lr=3e-4，weight_decay=1e-5。原 raw_quantile 损失，三个 task 等权均值。early frozen BN eval，late trainable BN 在一次混合 forward 中更新；冻结前缀缓存复用源 eval blocks 0–2。最大 500 epochs、patience 80；每折 seed=outer_seed+fold×1009，两方法相同。

原 BalancedTaskBatcher：size=min(682,最大目标 inner_train 人口)，steps=ceil(最大目标人口/size)，确定性 epoch/task permutation cycles；4g 每轮只抽同样数量，不能描述成遍历全部源数据。保存每轮样本调度 digest、累计真实 optimizer.step 次数、best checkpoint 时 steps 和 total steps。

每折使用各目标 inner_train 的 V1/V2 标准差(ddof=0)归一化。Combined NRMSE=sqrt(((RMSE_V1/SD_V1)^2+(RMSE_V2/SD_V2)^2)/2)。joint selector 为两柱 Combined NRMSE 算术平均，严格小于更新（并列保留最早 epoch）。源 validation 只诊断。每方法/seed 五个 joint best epoch 的 round-half-up 中位数为后续 FULL 重训固定 epoch；此次仅保存选择，**不重训**。

## 诊断和报告

每轮记录 train raw loss、validation raw loss、各柱 point metrics、Center/Width RMSE、joint selector 和实际步数。源指标不参与选择。最佳 checkpoint 保存各柱 V1/V2 RMSE、MAE、R2、Center=(V1+V2)/2 和 Width=V2−V1 误差；保存盲 OOF 预测和真实身份。每 endpoint tail=outer gradient_train 的 80%分位数；center low/mid/high=outer gradient_train 三分位。阈值只用于诊断，沿用历史定义，不选择 checkpoint。tail 用五折 OOF SSE/count 汇总后开根号，避免平均不等大小或空 tail 折。

paired gain=1−NRMSE_A2/NRMSE_A1，正值为改善；报告均值、每折、每柱 fold wins、单 seed win 和 epoch 分布。joint gain 从配对折两柱 selector 计算。单 seed 不能推断完整 5-seed 稳定性。与历史 Group-CV 同 seed 对比仅用冻结 INNER_RESULTS/ROW_INNER_RESULTS 及保存的历史诊断，不读取外层评分标签。

不得因负结果改网络、调 LR、扩候选或引入 PCGrad/GradNorm/CAGrad。Phase 1 完成并审计后停止。若未来确认 Phase 2，再按同一冻结协议执行其余 seed；Phase 3 另行确认，先冻结盲预测再独立一次评分，标记 DEVELOPMENTAL CONFIRMATION。真正的泛化确认需要未曝光独立数据。

## Phase 2 授权与冻结补充（2026-10-13）

用户已确认继续其余四个 outer seeds：536279090、1362771960、1425370602、2767143051，共40次拟合；Phase 1 的10次拟合通过原始 receipt 哈希核验后复用，不重新训练。原 Phase 1 文档、报告、聚合表及完整来源记录已按字节保存在 phase1_archive/，其原 protocol.json、代码和 runtime receipt 保持不变。本补充不改科学训练协议。

**Phase 1 结果以及历史 Row 外层结果均已被研究者观察。** 后续研究属于开发性方法诊断，不能宣称独立外部确认。全部五个 seeds 来自共享历史人口，25折不是25次独立实验；不进行独立样本显著性检验。只比较各CV内部A2对A1收益及训练选择，不比较不同CV验证集的绝对误差。

本次目标标签权限仅为各seed的gradient_train；4g仅为固定source_train。与Phase1相比，停止解析source validation真实标签及输出源验证误差/损失；其身份和零标签输入可供有限性检查。这一由用户直接要求的诊断权限收紧不改变source训练人口、目标选择指标、优化器、batch、BN、最大500/patience80或epoch聚合。新增入口保留原fit更新代码，仅跳过源验证标签指标；合成标签等价性测试须验证A1/A2更新、目标历史及best epoch一致。源验证诊断缺失明确标记，不将零占位值当真实指标。

每seed/方法候选refit epoch仍为五折joint最佳epoch的中位数；本阶段仅记录候选，不重训。mean paired-fold gain与gain of means分别计算并命名。各seed尾部先合并OOF SSE/count，跨seed报告均值/离散及每seed收益；重复测量独立性未知。最大绝对端点/Center/Width误差作为新增只读诊断，从已获准inner验证标签计算，不改选择或门槛。

Phase2入口只接收四个新增seed、固定42五折和A1/A2，不含FULL refit、outer score、优化器干预或参数搜索。每seed完成后核验身份覆盖、配对、真实步骤、最早argmin、BN/冻结张量、哈希及零标签checkpoint重放，全部通过后汇总50次拟合。遇到科学协议差异停止；明确执行故障可以修复并留记录。Phase2完成后停止，等待下一步确认。
