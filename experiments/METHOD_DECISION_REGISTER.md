# QGeoGNN 方法决策与待验证事项登记表

更新日期：2026-08-15

本表用于区分三类内容：已经冻结的 legacy 基线、已有证据但尚不能冻结的候选方案、必须通过控制变量实验才能决定的争议项。任何 P0 项未解决前，不进入正式主动学习主实验。

| ID | 优先级 | 事项 | 当前证据/不确定性 | 当前决定 | 后续验证 |
|---|---|---|---|---|---|
| D01 | P0 | V1/V2 多任务尺度 | E0-3c完成18组：跨6个paired validation为等权0.604、standardized 0.611、legacy 0.619；两种改法均赢4/6但3-seed paired区间跨0 | 按预注册validation规则暂选等权；不声称显著优于legacy，legacy继续作原代码对照 | E0-4同时报告等权点模型与standardized UQ信号；更多seeds确认微小效应，不再用test调权重 |
| D02 | P0 | checkpoint选择分数 | 原代码没有自动best规则；E0-2用 legacy加权MSE，E0-3改用 train-variance normalized MSE | E0-3暂用 normalized score，不声称是论文口径 | 在相同训练轨迹报告 legacy/normalized/UQ-aware 三种选择，确认最终排名稳定性 |
| D03 | P0 | 体积阈值60/120 mL | 阈值来自原代码，论文没有证明其为物理边界；4g删除66行，8g删除22行 | legacy基线启用，不称为推荐清洗 | 有阈值 vs 无阈值成对实验，报告尾部区间、误差与校准 |
| D04 | P0 | 多构象选择 | 已完成217结构审计和4g source+6组8g transfer；最低能validation略好但paired区间跨0，8g test MAE平均恶化0.460/0.279 mL，compound三seed均差 | 主实验保留原代码第一构象；最低能作为论文口径敏感性，不覆盖legacy缓存 | 后续若论文固定seed/非收敛规则可得，再做严格paper复现；当前不继续调构象以追test |
| D05 | P0 | 8g极端不连续记录 | 源行224，同一化合物PE/EA从20/1到50/1时V1/V2从45.41/54.58突降至2.85/6.10；该点贡献当前test约80% SSE | 保留，不以提高指标为由删除 | 回查原UV文件/实验记录；报告含/不含该点的敏感性分析，但主数据修改需证据 |
| D06 | P0 | 分位数单调性 | E0-3c跨6组crossing：legacy 18.0%、等权17.7%、standardized 8.0%；compound standardized降至2.3%，但row seed1101仍39.3% | 损失缩放不足以保证顺序，Gate 0前必须结构性解决 | 比较累积正增量参数化、排序后处理和惩罚权重；以后处理不改变q50为优先 |
| D07 | P0 | 区间校准 | 4g V2名义80%覆盖仅61.63%；8g覆盖对配置高度敏感 | 当前区间不能直接用于AL不确定性 | validation-only conformal/温度式宽度校准；在独立test报告coverage-width |
| D08 | P0 | 8g单seed/小test波动 | E0-3b已完成3 seeds×双split；row seed42仍被源行224支配，跨运行标准差较大 | 不再用单split描述模型整体好坏；3-seed结果仍属pilot | 增加seeds/paired CI、分化合物宏平均和scaffold OOD后才冻结效应大小 |
| D09 | P0 | row-level泄漏 | E0-3b已完成compound-group复现；last2三种子test R²均值V1/V2=0.901/0.892，但小样本test组成影响仍大 | row-level只作代码/论文可比；compound-group作为主要泛化证据之一 | 核对重复条件整体分组实现；增加scaffold OOD，不把当前高分直接外推 |
| D10 | P0 | 8g迁移范围 | 跨6个paired运行normalized validation：last2+head 0.619，full 0.683，head-only 0.750，scratch 0.975 | `last2_head_lr1e-4`升为多seed暂定候选，仍不冻结 | 完成D01/D04/D29后只用validation复核排序；test不调参 |
| D11 | P1 | 迁移时scaler策略 | 预训练权重应匹配4g source scaler；scratch应使用8g-train scaler；原代码8g会在全数据上重拟合并覆盖文件 | 当前pretrained用source，scratch用target-train | source scaler vs target-train scaler vs target-train重算后输入适配；禁止使用valid/test拟合 |
| D12 | P1 | 模型未使用的模块/表示 | `NN_descriptor`从未在forward调用；键角分支产生`h_node_ba`但最终预测只pool `h_node` | 暂不删除，保证checkpoint兼容 | 消融或修复融合路径，核对论文图示与补充方法；新架构用新实验ID |
| D13 | P1 | 输出ReLU与训练/推理差异 | `graph_pred_linear`含ReLU，eval又clamp；非负性合理但可能压制分位数梯度 | legacy保留 | softplus、无约束q50+正增量区间与legacy对照 |
| D14 | P1 | 训练loss与UQ模型选择不一致 | checkpoint只按q50点误差选，不直接优化coverage/pinball/crossing | 暂以点预测主分数选择并完整报告UQ | 预注册UQ-aware tie-break或复合分数，不能查看test后定权重 |
| D15 | P1 | 早停与最大epoch | E0-3c row seed1101三种损失best epoch=468–489，其他split多在28–124；固定500对部分row split可能偏短 | 当前按代码500/1000+patience100，不因单split临时加epoch | 对候选做1500+validation早停的paired对照，检查是否改变配置排序 |
| D16 | P1 | BatchNorm冻结策略 | 部分微调时冻结参数但BN running stats是否更新会改变语义 | 当前冻结模块BN保持eval | 与“更新BN统计”做对照并记录实现 |
| D17 | P1 | DataLoader不shuffle | 原代码shuffle=False；当前每epoch次序固定且4g/8g整批训练近似full-batch | legacy保留 | shuffle=True/更小batch与full-batch对照，paired seed |
| D18 | P1 | 重复实验与噪声建模 | 重复测量保留，但row split可拆散；52个4g组、7个8g组 | 数据保留，严格split整组分配 | 估计组内方差；可用heteroscedastic/重复加权但不先平均删除 |
| D19 | P1 | 图缓存跨数据集复用 | 8g 88个结构中87个复用4g缓存，保证相同结构输入一致；1个8g新结构 | 当前方案合理 | 最低能构象版本中仍保持跨数据集同结构同图 |
| D20 | P1 | UFF回退 | 4g有1个结构缺少MMFF参数而用UFF；论文只写MMFF94 | 保留并记录 | 该结构单独敏感性；考虑MMFF失败时2D/其他力场的统一策略 |
| D21 | P1 | SMILES标准化与立体化学 | 当前canonicalization保留RDKit可识别立体信息，但盐、互变异构体、质子化状态未统一 | 不擅自标准化 | 统计盐/多组分/手性；按化学规则预注册标准化消融 |
| D22 | P1 | 目标异常 | 4g保留1个负标签和8个`t1>t2`，8g保留2个`t1>t2` | legacy保留并标记 | 人工复核；含/不含异常成对敏感性，不能凭指标删除 |
| D23 | P1 | 8g zero-shot语义 | 4g没有显式柱规格输入；8g与4g条件域不同，因此zero-shot很差是可预期的 | 仅作下界 | 若启用`Use_column_info`，必须重训source并单独评估跨柱外推 |
| D24 | P2 | 质量/纯度特征 | CSV含Purity，但当前QGeoGNN输入没有使用；质量用density×V而未乘Purity | 遵照原代码暂不加 | 机制合理性审查后做 purity-aware mass 消融 |
| D25 | P2 | PE/EA描述编码 | 当前只使用前两种溶剂权重，其他三个槽恒为0；纯EA为0/1 | 遵照原代码 | 与直接比例、log-ratio、可学习溶剂embedding对照 |
| D26 | P2 | 评价指标稳健性 | 小test的R2被极端点主导 | 保留MAE/RMSE/R2但不单独决策 | 增加median AE、trimmed敏感性、bootstrap CI、分化合物宏平均 |
| D27 | P2 | 计算确定性 | E0-3c完整重跑legacy 6组，与E0-3b的validation/test/R²/best epoch逐项零差异 | 当前CPU、固定seed和图缓存流程在已检查指标上可复现 | 后续新构象缓存记录torch/RDKit版本与哈希；必要时检查参数级bitwise一致 |
| D28 | P2 | 10k+推理/AL接口 | 尚未完成分块推理、fit/predict与round resume测试 | Gate 0未通过 | E0-4实现batch inference、索引一致性、断点续跑与内存测试 |
| D29 | P0 | 论文迁移结构与原代码不一致 | 论文描述新输入/输出层和迁移隐藏层；E0-3b仅随机重置同构头时crossing达50%/96.4%，并非paper-style实现 | 两种方法不能混称；随机新头失败不能否定论文方案 | 实现“柱规格输入适配+新输出头+隐藏层迁移”，新头采用合理初始化/单调分位数参数化，再与full及last2做paired对照 |
| D30 | P1 | 构象优化非收敛 | D04中21个结构共有50/2170个构象优化返回非收敛；原代码仍保留，论文未说明筛选规则 | D04保持原代码行为并完整记录，不事后删除 | 比较“全部构象最低能”与“仅收敛构象最低能”；若最低能恰为非收敛构象需单列敏感性 |

## 当前阶段结论

- E0-2 legacy 4g基线保留，用于代码口径可比。
- E0-3b/E0-3c/D04已完成；迁移范围暂选末两层+头，损失暂选V2等权，构象保留原代码第一构象。
- 下一步优先处理 D03、D05、D06、D07、D29，并补足D08/D09的CI与OOD证据。否则任何主动学习收益都可能建立在阈值偏差、异常记录、不稳定划分或不可用不确定性上。
