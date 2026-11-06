# 表示与概率模型开发基准：冻结协议

日期：2026-11-04，Asia/Shanghai。协议在正式效果评分前固定；真实执行清单、配置文件和哈希由 runner 保存。本研究只做已有状态的离线表示、GP 和采集诊断，不新训 QGeoGNN、不产生新选点轨迹、不调用生成式 API、不下载权重、不改环境、不访问 test 真值。

## 研究问题及证据等级

比较结构化化学条件、QGeoGNN 隐藏层、Raw/CW 梯度、本地预训练语言模型及融合表示，回答未见行预测、概率校准、单因素差分和模板稳定性问题。低训练 NLL、低表示距离或高模型内信息增益均不作为下游主动学习收益。

本地没有 T5 权重。D/E/F 明确采用已有 **Qwen1.5-0.5B-Chat decoder 的冻结 hidden state**，属于探索性本地预训练 LLM 对照；不称为 T5 复现、专用语义 encoder 或完整 GOLLuM 主变体。T5 对照保留为 blocked，不能以本次结果概括所有 LLM 或否定 T5。参数、pooling、revision 和模型文件哈希均须落盘。

## 数据及状态矩阵

- 来源只选统一 LCMD32 历史轨迹。种子固定 157、887、2357、6101、12203；预算固定 L333、L525、L813、L1005，对应 round 0、6、15、21，共 20 个开发状态。
- 实现预检固定 157/887 × L333/L525，检查身份、单位、维度、数值及资源；不依据效果删除基线或改变核/指标/拆分。完成后在资源边界内执行完整矩阵。
- 每状态所有表示绑定同一 checkpoint、有序 L_t/U_t、split 和训练记录。缺 checkpoint、身份不符或缺合法特征时报告该状态/表示缺失，不能借另一方法后期模型或未来标签回填。
- 仅 L_t 的响应进入 GP、Projection、响应标准化和任何标签相关超参数。评价固定为该 seed 既有 416 行 validation；其参与过 QGeoGNN checkpoint selection，因此是已暴露开发证据，不是独立确认测试。test 417 行真值不数值化、不返回、不评分。
- 原始表若只能流式读取，先按 sample ID 权限及 split 筛选，再将允许的 L_t/validation 响应转换成数字。U_t 只保留无标签元数据；不读取其他轨迹后期响应。写 label-access receipt 及请求 ID 集合。
- 既有 QGeoGNN 已使用 L_t 监督，GP 在 L_t 内部留出不等于端到端无泄漏验证；本次以 QGeoGNN 未拟合但用于其选 checkpoint 的 validation 为开发核验。五 seed 及四预算相关，不称为 20 次独立重复。

## 冻结表示及容量

| 名称 | 内容 | GP / 对照 |
| --- | --- | --- |
| Structured | Morgan radius 2 指纹与实际五条件；数值条件仅按 L_t 标准化、溶剂 one-hot，保留 PE/EA 组成语义 | Matérn-5/2 |
| GNN Hidden | `QGeoGNNV2.extract_representation`，图节点 add pooling + condition residual，N×128，冻结模型 | Matérn-5/2；以及 128→64 Projection |
| Raw / CW | 各自 512 维全参数 CountSketch；精确同 checkpoint 缓存优先 | 分开 Matérn-5/2；各有 Phase0 linear 核控制 |
| Frozen Qwen SMILES | 完整 canonical SMILES，主 A 模板 | Matérn-5/2；与含条件文本比较 |
| Frozen Qwen Conditions | SMILES、4g silica、五实际条件、准确单位，A/B 完整数值等义文本 | 主 A 模板 Matérn-5/2；B 做鲁棒性 |
| Qwen Projection | 主 A conditions hidden 1024→64、ELU | GP 边际似然联合学习 |
| Fusion | GNN 与 Qwen 各按 L_t 处理后各自 PCA64，拼接128维 | 固定表示 GP，以及 128→64 Projection GP |

Fusion 与 GNN Projection 同为 128→64，含偏置均有 8,256 个投影参数。PCA 只在 L_t covariates 上拟合；不同输入预处理也是比较的一部分，融合增益不能直接归因语言信息。主 comparison 同时保留不经 PCA 的 GNN128。LLM 输出维度从真实模型核验，禁止硬编码 768。B 模板不另训练/选择超参数，使用主 A 上拟合的 GP/Projection 直接评估输入改写。

## 数值设置

所有 GP 输入统一按 L_t 的向量均值中心化，再除以单一 global centered RMS：`sqrt(mean((X−mean_L X)^2))`；常数尺度设为1并记录。Structured 连续条件可在此之前训练侧 zscore，Morgan bit 不逐位 zscore。Matérn 距离按维数平均平方欧氏距离，避免高维数本身改变初始尺度。

每端点独立 GP，y 按 L_t mean/std 定标，常量 y 的 scale=1；均值固定为训练 y 均值。Matérn-5/2 isotropic：log lengthscale ∈[-5,5]，log standardized signal/noise variance ∈[-9,9]；初值按实现参数次序为 lengthscale=1、signal variance=1、noise variance=exp(−2)。单起点 exact MLL，SciPy L-BFGS-B + torch autograd，maxiter=40、maxls=20、maxfun=400、ftol=1e−9、gtol=1e−5、maxcor=10。核、损失、后验与 Projection 为 float64，Cholesky/三角求解，不显式逆矩阵。40步上限达到时记录未收敛，不把有限结果冒称最优。

Projection 为 Linear(input_dim,64)+ELU，W 独立 N(0,1/sqrt(input_dim))、bias=0，无额外惩罚；固定初始化 seed17、29、43，均完整报告。联合优化参数是 Projection、核幅度/lengthscale/噪声；使用相同40步上限。不能用 validation 从三种初始化挑最好者作为主结果。

Raw/CW linear 控制直接复用 Phase0 三个固定训练似然起点及[-9,9] log variance界；此优化预算与 Matérn 单起点不同，必须说明，不能把核差异视作完全等预算因果识别。复用 Phase0 stable_cholesky、probability_metrics、batch_information；数值 jitter 与测量噪声分开记账。全部输出包括训练原尺度 MLL、目标尺度、signal/noise、lengthscale、mean、优化状态、实际 jitter、训练/推理耗时、潜变量及观测预测方差。

## 科学与概率评价

主指标按冻结config为两个端点 `RMSE_q / historical_L0_std_q` 的均值，并作同seed同budget的配对比较。另固定逐状态、逐端点 validation RMSE/MAE、NLPD、90%/95% 覆盖及宽度，及标准化误差与不确定性关系。RMSE/MAE 使用 mL，NLPD 注明原响应尺度。均值表现与概率可靠性分别判断；不在评价集上回调噪声、缩放区间或选择核。当前不做后验覆盖校准，故无需引入可被误用的 validation 校准集。

条件对只从当前 validation 的合法观测行构造：同 canonical SMILES、除目标轴外其他全部记录条件严格一致，目标轴分别为 V/ul、PE/EA、loading-solvent volume。density×V 是派生质量，density 固定时 V 变化才是这里的上样量对比；样品体积与加载溶剂体积不能混淆。报告 pair 数、分子数、表示距离、GP/QGeoGNN 预测差、真实响应差与 `abs(Δy−Δpred)`。零对或不足时明确不可评；行对共享端点/分子，不按 pair 数声称独立显著性。小距离本身不是失败，需同时联系真实变化与差分误差。

文本鲁棒性对完全相同行的 A/B 模板计算嵌入相对距离、最近邻 Top-k 重叠、GP 均值差、方差秩相关及采集分数变化；完整原始数字不得舍入。语言-only 重复分子导致相同向量和邻居并列属方法特征，tie 用 sample ID确定性处理。

## 离线采集

每状态按无标签且确定的 ID/hash规则，从 U_t 冻结至多256点；保存候选 ID 与 SHA256，同状态各表示完全共用。计算 MaxVar、uniform-reference conditional IVR、greedy Batch Information Gain，batch32，均使用拟合后潜变量联合协方差和训练噪声。IVR reference 固定同256点。报告全排序相关、Top32交集、所选批内相关/冗余和实际联合信息量；候选不足按实际数量记账。此为子池诊断，不称全2997候选结果，不揭示选中候选标签，不进行 QGeoGNN重训，不推导真实 AULC。

## 资源与停止

单进程、单 CPU 线程，初始执行预算2小时；保留正在运行 CW-Free/NoCW 目录和进程。LLM 按文本内容跨状态去重缓存；完整模板超过512 tokens立即失败并留证，不截断、不按效果删样本。缓存绑定模型/tokenizer/文本/模板哈希。

预检估计资源后，执行清单记录确定的计算顺序和覆盖缺项。若预算不足，保存已完成的有效矩阵与未完成原因，不能按好坏决定继续哪个表示。NaN/Inf、真实权重缺失、权限/身份错误、严重非PSD或需要新依赖/下载时停止相应任务，不随机替代、不自动增加重启或算力预算。2×2预检与部分完成的5×4矩阵不得冒充完整确认。

结束必须给出 Go/No-Go、证据不足项和最小后续验证；本次没有任何自动启动正式五seed AL的路径。未来 AL 若获授权，才固定 LCMD32/Numerical Hybrid32、Frozen/Projection/Fusion + 同数值 batch acquisition，并用 AULC、最差 seed与成本检验学习效率。
