# GP 数值专家：模型、概率目标与证据边界

日期：2026-11-03。实现：[gp.py](gp.py)、[offline.py](offline.py)。这是独立低成本侧车；没有改变 QGeoGNN、梯度提取器、正式 selector 或现有运行。当前响应 GP 未通过概率可靠性筛查，不能作为已校准正式引导。

## 1. 冻结表示与合法监督

使用历史 seed157/6101 的 L333 QGeoGNN-V2 checkpoint 对应 Raw/CW 512 维 CountSketch bank，3330 行严格等于冻结的有序 L0+U0。与新 CW-Free 的 r00 checkpoint、contract、候选顺序逐项对齐；表示来自相同 checkpoint，因此允许复用历史 bank，不由另一方法后期 checkpoint 补缺。

每种表示独立处理，不拼接、不学习组合权重。令 `s_g²=mean_(i∈L0)||g_i||²`，`x_i=g_i/s_g`；全程固定 L0 RMS。该缩放是新 GP 的明确模型定义，**不同于历史 IVR 的全3330行 RMS**。没有重新提梯度、加载模型或接触原始数据表。

响应目标为 `z_ij=(V_ij−mean_L0(V_j))/std_L0(V_j)`，j 为 V1/V2。L0 均值和总体标准差固定。r 批预测前只拟合 L0 与前 r 批已观测响应；其后32点仅评分。6个历史批次按原获取顺序形成每 seed 192 点 forward 核验。既有 bank 的监督只含 L0，评价点均在 U0，未使用“当前监督全部 L 后再切 GP 留出”的伪端到端 OOF。

风险目标选择固定 r00 QGeoGNN 的诚实外样本误差：

`ell_i = 0.5 Σ_(j=1,2) [(q50_j^(r00)(i)−V_ij)/std_L0(V_j)]²`

`h_i = log(1+ell_i)`。

q50 来自 r00 全候选 catalog 已保存预测；V 来自已完成历史批次的反馈，读取前校验已有 label-access receipt、batch freeze 和反馈哈希。前3批96点训练风险 GP，后3批96点 forward 评分。所有这些点均不在固定 r00 checkpoint 的训练 L333 中；没有使用训练内误差。侧车风险模型的“训练时点”在前96点已观测以后，不能将它回送到尚无这些标签的历史 r00 决策。

这是固定旧模型的回顾性、按时间留出的风险检验，**不是后续当前 checkpoint 的风险检验**。后续选择由原轨迹的更新模型和 LLM 产生，存在选择偏倚。两个 seed 是已有开发 seed，且划分来自同一数据集；不构成外部确认或独立384个实验单位。

原反馈中的 acquisition-time residual 属于各自当轮 checkpoint。逐条来源保存于 [results/residual_provenance.csv](results/residual_provenance.csv)，不与固定 r00 残差混训。未生成 QGeoGNN OOF checkpoint，不用训练内残差填补 L0 的风险监督。

## 2. 精确低成本 GP

每个响应端点独立建模，风险另外建模。线性核为：

`f(x) ~ GP(m, a² xᵀx')`，`y_i=f(x_i)+epsilon_i`，`epsilon_i~N(0,sigma_n²)`。

响应 m 固定为标准化空间的0；风险 m 为风险训练96点的均值。均值作为经验估计常数，未积分其参数不确定性。线性 GP 等价于给固定梯度表示上的线性权重加 Gaussian 先验；没有 LoRA、Deep Kernel 或新表征训练。

幅度方差 a²、观测噪声方差 sigma_n² 由**当前训练数据的负对数边际似然**拟合。对数方差界为[-9,9]，使用三个固定起点(0,−2)/(2,−1)/(−2,0)，L-BFGS-B 成功结果中仅按训练似然选择。没有用留出标签择核、调幅度、噪声或控制阈值。特征缩放、核族与拆分先保存于 [configs/offline_plan.json](configs/offline_plan.json)，首次评分前锁定其哈希。此计划是本次开发筛查预设，不是独立确认性预注册。

令 `C_L=K_LL+sigma_n² I`：

`mu_B=m_B+K_BL C_L^(-1)(y_L−m_L)`

`Sigma_(B|L)=K_BB−K_BL C_L^(-1)K_LB`。

潜变量方差为 diag(Sigma)；未来**观测**预测方差为 `diag(Sigma)+sigma_n²`。NLPD 和区间覆盖用后者，不把潜变量置信区间冒充响应预测区间。响应 GP 方差表示该 GP 响应模型的未知性，不天然等于 QGeoGNN 的预测误差。风险 GP 的 Gaussian 目标是 h，不是 V，也不是经校准的“QGeoGNN 出错概率”。

`exp(mu_h)−1` 只是一种转换后的中位数估计，可能落到负值；`exp(mu_h+v_h/2)−1` 在未约束 Gaussian 模型下同样不能直接保证合理的非负风险。当前原型输出 h 空间分布及排序，不伪装成已校准的候选失败概率或直接用于盲区配额。

## 3. 数值稳定性与失败

全部核、求解和概率计算使用 float64。训练似然用训练核的特征分解；微小负特征值只在核 Gram 矩阵的数值舍入范围处理。最终后验用 Cholesky/三角求解，不显式求逆。

Cholesky 依次尝试0、`1e−12*s`、`1e−10*s`、`1e−8*s` 的数值 jitter，`s=max(1,max(diag(C)))`；每次实际 jitter 独立记账。实质非 PSD、非有限数值、形状/状态错误或优化全部失败立即报错。jitter 不是观测噪声，也不进入噪声校准；当前52次拟合均成功、无边界命中、实际 jitter 全为0。没有把严重负特征值修剪成“合法概率”。

## 4. 条件信息增益

对完整32点 batch，在 GP 已条件化 L 之后，给定正定观测噪声 R_B：

`IG(B|D_L)=I(y_B; f_B | D_L)=0.5 log det(I+R_B^(-1/2) Sigma_(B|L) R_B^(-1/2))`。

这是指定 GP 中观测该批对其潜函数值的信息，单位 nats；不是 QGeoGNN 重训后 NRMSE 的期望下降。它也不等于“所选 B 与未选 U\B 的互信息”；若后续研究后者，必须另写条件协方差定义。没有借用传感器位置论文的保证来证明此处的学习收益。

通过 Cholesky 计算 `sum(log(diag(chol(...))))`。必须保留 batch 的协方差非对角项；相关候选通常降低新增联合信息。独立边际 `0.5 Σlog(1+var_i/noise_i)` 仅可作上界诊断；单点方差之和完全不是该指标。实现先检查输入后验子协方差 PSD，再求信息矩阵，避免 `I+Sigma/R` 看似正定掩盖原 Sigma 非法。

当前 V1/V2 独立建模，所以每端点单独报告；未估计跨端点联合协方差，不将它们汇总成已确认的多目标信息量。风险 GP 的监督域是诚实残差样本，条件化集合不含无残差的 L0；不得把“响应已标注”误作“风险已标注”。

## 5. 与现有固定核 IVR 的区别

| 方面 | 现有 `active_learning_v2/ivr.py` | 本次响应/风险 GP |
| --- | --- | --- |
| 表示 | 历史 q50 梯度，按方法冻结 | 复用同 checkpoint Raw/CW bank，分别试验 |
| 定标 | 原3330参考行的全局 RMS | 固定 L0 RMS |
| 目标 | 无 y、无 residual | 已观测响应，或合法固定模型外样本 log-risk |
| 先验/噪声 | 单位先验与单位噪声，固定设计假设 | 训练边际似然估计幅度/噪声，仍需外样本可靠性 |
| 指标 | 全参考池均匀加权的 integrated variance reduction | GP 均值、潜/观测方差、评分、joint IG |
| 选择 | 32次贪心协方差更新 | 本次不选择新实验；将来给LLM完整信息 |
| 可信性 | 几何 surrogate，单位噪声未校准 | 数值可解不等于概率校准，目前响应覆盖未通过 |

为比较而复算的固定核首步分数与原 IVR 一致：`S=(I+X_LᵀX_L)^(-1)`，`M=X_allᵀX_all/N`，`s_i=(x_iᵀS M S x_i)/(1+x_iᵀSx_i)`。单元测试将第一胜出点、score、条件方差直接与既有函数对齐，未重新实现 IVR selector。候选池上的 GP 方差与固定 IVR 分数相关系数0.791–0.848，说明大量共同几何；不能据非完全一致便认定新增效益。

## 6. 概率可靠性评价与正式放行

报告标准化响应的 RMSE、NLPD、50/80/90/95% 覆盖和90%宽度、标准化误差 RMS、PIT 十分位。风险另报告 heldout h 的相同概率指标及 Spearman 排序；常量 Gaussian 基准只由训练数据确定。本次未实现 CRPS、block bootstrap 或端点条件分层校准，正式放行仍需补足，不得由当前简单筛查代替。

拟合的同方差项可能吸收结构失配，超参数 plug-in 也忽略参数不确定性。尾部/局部条件、不同 checkpoint、选点导致的分布变化均可能使校准失效。应先用已有诚实误差划分独立开发校准与冻结核验 block，比较可解释的噪声/尺度校准或简单核替代；不在本次后96点评分上反复调到通过。

正式 GP 引导至少需证据证明所提供信号在**适用 checkpoint 与候选分布**上可信，并与只用已有固定 IVR 的重复信息区分。当前只能支持继续离线研究，结论见 [OFFLINE_REPORT.md](OFFLINE_REPORT.md) 和 [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)。

数学参考：[Rasmussen 与 Williams，GPML 原著第2章及第5章](https://gaussianprocess.org/gpml/chapters/)给出 Gaussian 条件分布和边际似然；[Krause、Singh、Guestrin，JMLR 2008](https://jmlr.org/papers/v9/krause08a.html)研究 GP 观测设计。这里的应用结论仍以本仓库实测为准。
