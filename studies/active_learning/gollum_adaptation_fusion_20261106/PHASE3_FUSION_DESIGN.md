# Phase 3：GNN–LLM 融合定义及执行边界

本轮实际执行受控的冻结表示加性核开发实验。语言表示仍是旧缓存中的 Qwen1.5-0.5B-Chat 最后一层 attention-mask mean pooling（1024维）；不是 T5、LoRA 或已通过 Phase 2 的新表示。结果只能检验这一特定缓存上的核组合，不能外推到完整 GOLLuM。GNN 为同一条历史 corrected-raw LCMD32 轨迹中当前预算的冻结 128维 hidden；不重训 GNN、不重提特征。

## 数学定义

每个模态独立用当前训练集合 L_t 的逐坐标均值中心化，再除以该模态的单个 centered global RMS。只使用训练特征拟合变换，不使用 validation 或 U 的分布估计。两个核的距离均为每坐标均方距离的平方根；这样维度128与1024不会仅因维度不同而改变核距离量级。

\[
r_b(x,x')=\sqrt{\|\tilde x_b-\tilde x_b'\|^2/d_b}/\ell_b,\qquad
\bar k_b=(1+\sqrt5r_b+5r_b^2/3)\exp(-\sqrt5r_b),\quad b\in\{G,L\}.
\]

\[
k_f=\sigma_f^2[\bar k_G+\lambda\bar k_L],\quad \lambda\ge0,\qquad
K_y=K_f+\sigma_n^2 I.
\]

两项均为单位对角的 Matérn-5/2 PSD 核，非负加权和保持 PSD。只有一个公共 signal variance；语言核没有第二个可自由训练的 amplitude，避免语言 amplitude 与 λ 的重复参数化。该约定下 λ 是相对先验协方差权重，不是“语言信息有效”的充分证据。若两核几何非常相似，有限数据仍可能难以识别权重，需同时检查参数、训练 MLL 与训练外预测。

目标仅用 L_t 均值与人口标准差标准化；常数均值为 L_t 经验均值。最终均值及 latent/observation variance 转回 ml 和 ml²。Observation variance 明确为 latent variance + 学到的 noise variance；数值 jitter 仅稳定矩阵分解，不冒充观测噪声。

## 冻结对照矩阵

| 方法 | 核 | 可训练标量 | 语言输入 |
|---|---|---:|---|
| gnn_lambda0 | σ² k_G；λ严格等于0 | signal、noise、ℓG，共3 | 不影响拟合/预测 |
| additive_fixed1 | σ²(k_G+k_L)；λ固定为1 | signal、noise、ℓG、ℓL，共4 | 旧Qwen |
| additive_learned | σ²(k_G+λk_L) | signal、noise、ℓG、ℓL、λ，共5 | 旧Qwen |
| gnn_capacity_control | σ²[k_G(ℓG)+λk_G(ℓ2)] | 与 learned fusion 相同的5个 | 不影响拟合/预测 |

所有表示冻结，不训练 projection。容量对照与 additive fusion 使用相同的 GNN 预处理及 MLL/数值预算；两个分量均来自 GNN，为防止初始化时两个长度尺度完全对称，第二个 GNN 长度尺度固定初始化为0.5，其余长度尺度为1。该对照控制本实验新增核自由度，不能替代对 LoRA/projection 的参数容量对照。

λ=0 分支不包含闲置语言长度尺度，目标函数与旧 GNN-only Frozen Matérn 实现严格相同；数值测试检查同一训练/查询集均值和方差一致。λ learned 使用 logλ∈[-9,9]，因此可趋近0但不能达到精确0；严格零基线单独保留。两个 log lengthscale∈[-5,5]，log signal/noise（标准化目标单位）∈[-9,9]，与旧 GP 一致。边界命中保留，不删解。

历史 early fusion 只读复用旧 PCA64+PCA64、训练集 block RMS 后 concat128 的 Frozen 和64维ELU Projection（初始化17/29/43、max_iter40）。新加性核没有 PCA；因此与历史 concat 的差别同时包含融合位置与输入预处理，不能把所有差异唯一归因于“加性核更好”。原始 Projection 全部未收敛，旧结果不当作训练充分的架构因果对照。

## 数据、优化与成本约束

- 预先固定 seeds157/887 × L333/L813 × V1/V2，4方法共32次 exact GP fit。先跑 seed157/L333/V1 四次 preflight；随后复用结果完成全矩阵，不根据评价效果选状态或方法。
- 拟合只接收当前 L_t 的合法缓存响应。416行旧 validation 已用于 GNN checkpoint selection，属于开发证据；不用于选择 λ、长度尺度、噪声、优化预算或方法。
- 独立复核每个历史 checkpoint、state contract、L/U/validation IDs、raw cache hash，以及旧prepared cache和LLM缓存的SHA256与索引顺序。仅把当前合法行映射到LLM缓存；缓存中其他行不参与计算。
- 单次 L-BFGS-B 最大200迭代，ftol1e-9、gtol1e-5、maxls30；torch提供 exact float64 MLL 梯度，SciPy控制有界优化。单初始化17；冻结核本身不依赖随机投影。测试用中心差分逐项核验五个核标量的梯度。
- 训练 Cholesky 相对 jitter预算为0、1e-12、1e-10、1e-8，超限失败；保存 loss/gradient轨迹、优化终止原因、迭代数、boundary、noise、signal、λ、核/观测condition及最小特征值。
- 固定与旧研究相同的合法 U_t 256候选池、batch32。MaxVar、conditional IVR、greedy BatchIG直接复用旧已验证实现。无候选响应查询，无新 AL 状态转换。
- 报告候选方差秩、Top32 overlap、批内后验相关、unique molecules、精确条件组合与selected/selected、selected/L_t单因素对照计数。模板A训练，模板B仅做推断与相同三种采集；不在B上训练或选参数。

## 条件性后续路线

Residual Fusion 暂不训练：定义为 y=f_G+f_L,residual。必须先出现跨seed/预算稳定的外样本增量信号，并准备合法 OOF 或采集前冻结的历史 GNN 预测误差作为训练/核验数据。当前加性实验仅有独立于当前 GP 的开发验证响应，不足以用GNN训练内残差证明泛化增益。需要按分子组审计 residual support、时间泄漏、条件差分误差、selection bias 与残差观测噪声。合法残差协议缺失时状态为 Not Yet Testable。

Factorized Fusion 仅设计：分子 GNN/化学编码器核与精确数值条件核分开，候选形式为 k_G + λL k_L,mol + λC k_C + λI(k_G k_C)，各权重非负、分量单位对角。PE/EA、density、sample volume、loading solvent、loading volume单独保留，不能混淆两个体积字段。正式比较应固定纯结构化、纯GNN、GNN+条件核、GNN+语言分子核、三者组合，控制新增核自由度。只有前面的独立消融显示条件表达确实丢失且存在互补信号，才冻结新协议并申请实施；本轮不自动开训。

本实验可判断冻结Qwen的开发性增量、概率质量和候选排序敏感性；不能证明 GP 信息增益会降低下游 QGeoGNN 误差，也不能给出 AULC 或闭环AL结论。最终进入真实AL还需 Phase1/2概率门槛、冻结协议与配对真实学习轨迹验证。
