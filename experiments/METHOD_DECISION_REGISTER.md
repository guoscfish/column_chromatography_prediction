# QGeoGNN 方法决策与待验证事项登记表

更新日期：2026-08-19

本表用于区分三类内容：已经冻结的 legacy 基线、已有证据但尚不能冻结的候选方案、必须通过控制变量实验才能决定的争议项。任何 P0 项未解决前，不进入正式主动学习主实验。

| ID | 优先级 | 事项 | 当前证据/不确定性 | 当前决定 | 后续验证 |
|---|---|---|---|---|---|
| D01 | P0 | V1/V2 多任务尺度 | E0-3c完成18组：跨6个paired validation为等权0.604、standardized 0.611、legacy 0.619；G0-1在等权损失上完成单调头验证 | 按预注册validation规则保留等权；不声称显著优于legacy，legacy继续作原代码对照 | 不再用test调权重；standardized只保留历史UQ对照，crossing由结构参数化解决 |
| D02 | P0 | checkpoint选择分数 | 原代码没有自动best规则；E0-2用 legacy加权MSE，E0-3改用 train-variance normalized MSE | E0-3暂用 normalized score，不声称是论文口径 | 在相同训练轨迹报告 legacy/normalized/UQ-aware 三种选择，确认最终排名稳定性 |
| D03 | P0 | 体积阈值60/120 mL | G0-3完成12次配对训练；validation上no-threshold的common RMSE改善3.4%、tail error改善23.4%，据预注册规则胜出；tail在高width top10%富集约8倍。独立test的common仅恶化1.3%，但tail error恶化40.1%且5/6 contexts更差 | AL pool与Predictor数据协议冻结为no-threshold（574行），不再提前删除22条难例；test反向结果作为尾部不稳定失败模式，不据此回调 | E1对tail单独报告signal-error、risk-coverage与enrichment；如Quantile Width不稳则降级，但不恢复无物理依据的标签阈值 |
| D04 | P0 | 多构象选择 | 已完成217结构审计和4g source+6组8g transfer；最低能validation略好但paired区间跨0，8g test MAE平均恶化0.460/0.279 mL，compound三seed均差 | 主实验保留原代码第一构象；最低能作为论文口径敏感性，不覆盖legacy缓存 | 后续若论文固定seed/非收敛规则可得，再做严格paper复现；当前不继续调构象以追test |
| D05 | P0 | 8g极端不连续记录 | 源行224，同一化合物PE/EA从20/1到50/1时V1/V2从45.41/54.58突降至2.85/6.10；该点贡献当前test约80% SSE | 保留，不以提高指标为由删除 | 回查原UV文件/实验记录；报告含/不含该点的敏感性分析，但主数据修改需证据 |
| D06 | P0 | 分位数单调性 | G0-1完成12次配对训练；monotonic相对legacy的validation normalized/V1 RMSE/V2 RMSE为+0.70%/+0.57%/+0.11%，crossing由validation/test 13.6%/17.7%降为0 | G0-1按预注册validation标准通过，冻结`q50=m, q10/q90=m±softplus(d)`；legacy只保留为历史对照 | 后续不再按AL test回调参数化；G0-2/E1继续检验区间校准与signal-error关系 |
| D07 | P0 | 区间校准 | G0-2仅用validation估计alpha；平均test coverage V1/V2由0.709/0.519升至0.824/0.854，crossing=0；alpha中位数1.396/1.709，但row seed1101需16.11/55.91 | 冻结validation-only、per-target、per-run split-conformal inflation协议；不把校准后的平均coverage误写成跨split稳定 | E1必须报告Quantile Width的跨seed signal-error与risk-coverage；row seed1101区间塌缩作为明确失败切片，若信号弱则降级为legacy UQ baseline |
| D08 | P0 | 8g单seed/小test波动 | E0-3b已完成3 seeds×双split；row seed42仍被源行224支配，跨运行标准差较大 | 不再用单split描述模型整体好坏；3-seed结果仍属pilot | 增加seeds/paired CI、分化合物宏平均和scaffold OOD后才冻结效应大小 |
| D09 | P0 | row-level泄漏 | E0-3b已完成compound-group复现；last2三种子test R²均值V1/V2=0.901/0.892，但小样本test组成影响仍大 | row-level只作代码/论文可比；compound-group作为主要泛化证据之一 | 核对重复条件整体分组实现；增加scaffold OOD，不把当前高分直接外推 |
| D10 | P0 | 8g迁移范围 | G0-4在no-threshold、单调头、等权loss下复核6组paired validation：last2=0.256，full=0.333（+29.9%），paper-style=0.295（+15.3%） | 冻结`last2_head_lr1e-4`；full与paper-style只保留为诊断对照 | 主动学习阶段不得按AL test重开迁移范围；若未来更换数据域或架构，必须使用新实验ID |
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
| D28 | P2 | 10k+推理/AL接口 | 统一`fit/predict`已实现；10240候选在batch64/chunk512与batch257/chunk1024下prediction/embedding最大差均为0，身份顺序一致；2轮连续与resume状态完全一致、重复query=0；12份E2/E4 partition已冻结 | D28通过，Gate 0完全通过；`sample_id`为长期主身份，临时DataFrame行号禁止进入状态 | E1/E2/E4复用`scripts/al_engine.py`；状态必须校验split/config hash，任何接口变更使用新decision ID |
| D29 | P0 | 论文迁移结构与原代码不一致 | 论文描述新柱规格输入、更新输出层与`lr=1e-4`微调，但未给逐层冻结图；仓库预留直径/柱长/装填密度RBF输入，旧迁移入口却关闭它们。首轮随机初始化适配器使6/6 validation变差（+160.9%），诊断为未保持source初始函数并完整保留 | 零初始化新增线性适配器后按原规则重跑：paper-style validation均值0.295，较last2恶化15.3%，只赢1/6；row/compound分别恶化20.7%/12.8%，故不替换last2。Test虽小幅改善1.3%，不反向选择 | 随机轮保存在`g0_4_paper_style_transfer_random_init_diagnostic`；paper-style作为跨柱输入的负结果保留，不继续做架构搜索 |
| D30 | P1 | 构象优化非收敛 | D04中21个结构共有50/2170个构象优化返回非收敛；原代码仍保留，论文未说明筛选规则 | D04保持原代码行为并完整记录，不事后删除 | 比较“全部构象最低能”与“仅收敛构象最低能”；若最低能恰为非收敛构象需单列敏感性 |
| D31 | P0 | E1 acquisition signal资格 | 三seed×row/compound、K=3真独立成员：Ensemble相对Quantile Width赢12/16项，关键切片平均Spearman/enrichment/AUSE为0.481/5.28/0.033；Latent Distance为0.499/5.69/0.033；Quantile Width为0.434/4.44/0.052。Ensemble的胜利集中在row切片，compound/full有3/4指标由Quantile Width更好；tail每run仅2--3例 | Ensemble作为E2唯一主uncertainty，Latent Distance支持Coverage；Quantile Width只作secondary/legacy诊断。E2首轮固定Random/Coverage/Ensemble/Hybrid四种，不因E1 test重开Predictor | 先做1 seed×2 rounds×2 members链路smoke，再做三seed paired pilot；单seed、tail或12/16边界结果均不得包装为稳定优势 |
| D32 | P0 | E2 4g source-free初始化 | 若复用完整4g baseline checkpoint，U0/test标签已经进入初始化权重，主动学习曲线无效；而随机初始化后只训练last2会冻结随机早期层，同样不合理。最小smoke用seeded-random单调QGeoGNN、全模型训练、固定L0-train scaler与独立config hash，2轮×25条无重复且状态、checkpoint与预测变化审计通过 | E2专用source-free协议冻结为随机初始化+全模型训练+L0-train-only scaler；这不改变E4的4g→8g `last2+head`合同。2-epoch smoke只验链路，不解释RMSE | 四策略必须共享同一outer seed的L0/U0/test、scaler、member seeds和轮预算；正式三seed pilot使用完整训练预算并报告计算成本 |
| D33 | P0 | E2 Round-0 regime与Hybrid机制 | source-free L0下Ensemble/Latent/Quantile三seed平均Spearman为0.592/0.436/0.503、enrichment为4.36/3.82/5.04，均3/3为正；Ensemble–Latent Top-10% overlap为0.396/0.396/0.467 | 无regime failure，不调整E1定义；Coverage冻结为标准化`h_graph+conditions` sequential farthest-first，Hybrid冻结Ensemble Top-25%后同一farthest-first。互补性只作机制动机，不预判AULC | 完成row三seed完整paired AULC；若Hybrid未超过单策略，不包装成主方法；Quantile保留给E4 Protocol B legacy baseline |
| D34 | P0 | E2 row正式结果与机制边界 | row三seed平均AULC Hybrid=0.543、Coverage=0.562、Ensemble=0.627、Random=0.645；Coverage/Hybrid对Random均3/3胜，Ensemble仅2/3且CI跨0。Hybrid与更高batch diversity同时出现，但selected set多个属性共同变化 | 冻结row主结论；只说结果与去冗余假设一致，不作因果声称。登记Top-25% uncertainty-filter-random control但本阶段不运行 | 完成row机制审计与compound seed42 preflight；完整compound pilot另行批准 |
| D35 | P1 | Quantile Width后续采样角色 | E2 source-free Round-0 Quantile mean Spearman=0.503、AUROC=0.849、enrichment=5.04，而正式Ensemble acquisition仅小幅且不稳定优于Random | Quantile仍不属于预注册E2四主策略；允许未来作为明确标记的post-hoc secondary acquisition control，E4 Protocol B继续保留legacy baseline | 本阶段不运行Quantile AL，不修改row primary conclusion |
| D36 | P0 | E2 batch-diversity坐标可比性 | 后续round的strategy-native latent来自各策略自己的模型，跨策略绝对距离混入representation geometry变化；固定Round-0 member_0、仅以L0-train拟合标准化后，row Hybrid/Ensemble与Coverage/Ensemble的平均距离差均3/3 seeds为正 | 保留native diagnostics，同时以fixed Round-0 common-reference作为跨策略可比机制诊断；不反向修改选样。该结果只与去冗余机制一致，不构成因果证明 | compound正式pilot复用同一审计；causal uncertainty-filter-random control仅登记到roadmap，本阶段不运行 |
| D37 | P0 | E2 compound正式结果 | 三seed primary AULC均值Hybrid=0.761、Coverage=0.777、Random=0.789、Ensemble=0.802；Hybrid/ Coverage各2/3胜Random，Ensemble仅1/3。macro AULC均值Hybrid=0.587、Coverage=0.605、Ensemble=0.637、Random=0.641，方向一致。Round-0 Quantile/Latent/Ensemble Spearman均值0.591/0.523/0.511且3/3为正。Ensemble每批仅8.5个compound、HHI=0.158；Hybrid fixed-reference distance相对Ensemble 3/3更高。独立fits late/max为16/297=5.39% | Gate定为suggestive而非strong：Hybrid平均最好但没有active策略3/3胜Random，seed42三种active均输Random。保留Coverage在novel-compound场景有2/3改善的有限证据；Quantile只进入roadmap，不运行第五条曲线。无convergence problem | 本阶段STOP；不得自动进入E3/E4、Quantile AL、因果control、表示消融或高级方法 |
| D38R | P0 | Corrected E2 compound failure audit | Seed42 Round8 coverage gain严格取seed42 CSV；gradient rows=318→518，不含57 validation；condition distance只用L0_train标准化；compound error按同compound Random paired；coverage与centroid shift拆分 | 少数compound贡献较大但worsened compounds广泛，localization保持inconclusive；没有发现实现/泄漏/评价bug，也没有单一稳定机制解释 | D38到此结束，不新增test-driven diagnostics |
| D40 | P0 | E4 Protocol A engineering smoke | Protocol A/B partition compatibility、三source真实加载、单调转换、freeze/source-reset、五策略dry-run、Random/Hybrid两轮和resume审计 | 仅记录工程pass/fail，no scientific conclusion；不启动formal pilot | 若无blocking issue，下一阶段仅可人工启动Protocol A 3-seed formal pilot |
| D42 | P0 | E4 Protocol A headroom与首轮shock | 离线审计重算L0 recovery：seed42=0.742、seed525=0.930、seed1101=0.912；seed525/1101四种active在50→60全部恶化而Random改善。Active首批source residual与label extremeness均显著高于Random；diversity三seed均高却效果方向不同，convergence为混合弱线索 | 仅支持headroom hypothesis的描述性解释，不证明因果；Protocol A active=null保持冻结，不反向修改predictor/acquisition，不进行method fishing | 可把低L0 E4-A2作为独立secondary sensitivity提案，但本阶段不执行；当前不启动Protocol B |

## 当前阶段结论

### D41 — E4 Protocol A formal pilot

- D40R corrected smoke通过后完成seeds 42/525/1101、K=3、budgets 50–200的五策略正式pilot及zero/full/scratch controls。
- Mean normalized AULC最低者为pretrained Random（0.8433）；Coverage/Ensemble/Hybrid/Quantile Width相对Random均仅1/3 seed胜出，证据分类为null。
- Pretrained Random跨预算平均NRMSE较scratch Random低0.5158，支持迁移收益，但不支持复杂active acquisition优势。按预注册停止；Protocol B、E3、E5及其他扩展未启动。

- E0-2 legacy 4g基线保留，用于代码口径可比。
- E0-3b/E0-3c/D04与G0-1～G0-4科学对照已完成；冻结末两层+头、V1/V2等权、第一构象、no-threshold、单调头和validation-only校准，完整合同见`experiments/PREDICTOR_FREEZE.md`。
- D28已通过，Gate 0科学与工程配置全部冻结。
- E1已完成：Ensemble以12/16恰好过门并进入E2主uncertainty，Latent Distance支持Coverage；Quantile Width降为secondary/legacy诊断。
- E2 row四策略三seed正式pilot已完成：Hybrid/Coverage对Random均3/3胜；fixed common-reference机制审计同样显示Hybrid与Coverage相对Ensemble的batch diversity均3/3更高，但不作因果声称。
- E2 compound正式三seedpilot已完成：Hybrid平均AULC最低，Hybrid/Coverage均2/3胜Random，Gate为suggestive而非stable；三个partition无test compound leakage，macro结论一致，且无convergence problem。本阶段按D37停止。
