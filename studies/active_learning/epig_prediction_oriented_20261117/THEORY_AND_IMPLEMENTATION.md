# Gaussian EPIG：定义、实现与独立数值复核

本研究实现的是**给定明确联合 Gaussian 概率代理后的解析 EPIG**。解析公式正确不等于代理的联合预测可信，更不等于 QGeoGNN 重新训练后的性能改善；后两点分别由资格验证与完整主动学习轨迹检验。参考：[Bickford Smith et al., AISTATS 2023](https://proceedings.mlr.press/v206/bickfordsmith23a.html)，[论文 PDF](https://proceedings.mlr.press/v206/bickfordsmith23a/bickfordsmith23a.pdf)。本文以下 Gaussian 推导与数值实现为本仓库针对两输出回归的具体实现。

## 1. 随机变量、潜在函数与标签噪声

给定已合法揭示的训练数据 D，令 f(x) 为潜在两输出函数，观测标签 Y_x=f(x)+ε_x。概率代理提供 K_D(a,b)=Cov(f(a),f(b)|D)，以及每个输入的观测噪声协方差 N_x。候选标签与参考标签的边缘协方差为

\[
S_{xx}=K_D(x,x)+N_x,\quad S_{tt}=K_D(t,t)+N_t,
\quad S_{tx}=K_D(t,x).
\]

**同输入的语义已预先声明：目标是未来独立噪声重复测量的标签。** 即使参考输入 t 与候选 x 的 row ID 相同，ε_t 与 ε_x 也独立，cross block 仍仅含 latent K_D(t,x)。若把同一个有限池观测随机变量同时作为目标与候选，其自身互信息在连续分布下可以发散，不能悄悄把这两种语义混合。P1/P2 目标都采用前述未来重复标签语义。

`posterior.cov(a,b,noisy=True)` 则表示 row ID 相交时的**同一个已定义观测变量**，用于构造一组已选择候选的联合观测协方差；其中每个 row 的噪声只加一次。acquisition 中分开请求 latent cross 和 noisy marginal，避免错误共享未来噪声。

QGeoGNN 的 q10/q50/q90 只限定单输入边缘的三个分位数，不能决定 K_D(t,x)。本实现没有从分位数独立采样跨样本标签来制造联合分布，也未把训练种子 ensemble 称为严格贝叶斯后验。

## 2. EPIG 与同后验 IVR

对权重 w_t≥0、Σ_t w_t=1 的预定义参考输入分布，Gaussian EPIG 为

\[
\mathrm{EPIG}(x)=\frac12\sum_t w_t
\log\frac{\det S_{tt}}
{\det(S_{tt}-S_{tx}S_{xx}^{-1}S_{xt})}.
\]

相同概率后验、相同参考集上的 IVR 使用

\[
\mathrm{IVR}(x)=\sum_t w_t\operatorname{tr}
\{S_{tx}S_{xx}^{-1}S_{xt}\}.
\]

IVR 是绝对均方风险下降；EPIG 是相对熵下降。单输出情况下令 r²=S_tx²/(S_xx S_tt)，则 MI=−log(1−r²)/2，IVR=S_tx²/S_xx。很弱相关时 MI≈IVR/(2 S_tt)，因此即使后验完全相同，目标的不确定性归一化与非线性也可改变排序。这一数学差异本身不是新增真实标签效率的证据。

历史 `ivr.py` 使用全训练侧 feature 的单一 RMS 归一化、线性核、单位参数先验精度、单位观测噪声，且不监督拟合概率参数。该方法是历史 Kernel-IVR 对照。它与经训练标签拟合核尺度、端点噪声的 GP-IVR 是不同的代理；将后者与 GP-EPIG 配对才能只改变 acquisition objective。未经资格验证的旧核 Gaussian 形式若计算 EPIG，只能另名为 `EPIG-inspired` 或明确说明其代理假设；本研究没有将其充当主 GP-EPIG。

## 3. 多输出的含义与单位

`gaussian_pair_information` 接受真正的 q×q 边缘和 cross block，并直接计算 block Schur complement。`output_indices=[0]`、`[1]` 分别计算 V1/V2 单任务；全部输出才计算该 posterior 所声明的联合 MI。

当前独立输出 GP 的跨输出 covariance 为零，其 block 联合 MI 在数学上等于两个单输出 MI 之和。这是**独立 GP 下的正确联合分布**，不能宣称有跨输出依赖。matrix-normal BLR 的 coregionalized 模式存在非零跨输出 covariance，但仍只是一个明确的可分离模型。由于它在潜在先验与观测噪声中共享同一正定 coregion B，完整观测两个输出时可逆 output whitening 可消除 B 对联合 MI 的直接影响；它的相关性和联合 NLL 改变不保证 EPIG 排名改变。不能将这种结构限制解释成所有多输出 EPIG 都无效。

RestrictedLabelStore 揭示的 V1/V2 单位为 mL，GP/BLR 内部使用训练折标准差缩放后，posterior 将均值、latent covariance、noise covariance 全部恢复至输入标签物理单位。EPIG 对一致应用于所有相关块的可逆线性输出变换不变；IVR 的 trace 对输出单位敏感。因此正式 GP-IVR 应与冻结的 combined NRMSE 尺度保持一致（若由上层传入标准化 posterior，则全部块和噪声必须同步转换），不能只缩放均值或只缩放一端 covariance。神经网络训练原 dtype 和既有标签变换不由本模块改变。

## 4. 稳定数值算法

`epig.py` 的计算均为 float64。设 S_xx=L_x L_x^T、S_tt=L_t L_t^T，计算

\[
B=S_{tx}L_x^{-T},\quad W=L_t^{-1}B,
\quad I(Y_x;Y_t)=-\tfrac12\log\det(I-WW^T).
\]

实现使用 Cholesky 与三角求解，避免显式 covariance inverse。1/2 输出使用同样 Cholesky 的标量展开以避免数百万个微型 LAPACK solve；通过 `log1p` 保留接近独立时的微小信息量。两个 target output 的 `det(I-WWᵀ)=1−(a+d−ad+c²)`，其中 a、d 为两行 W 的平方范数，c 为其内积，并同时检验首主子式正性；这与通用 Schur/Cholesky 路径解析等价。3 个以上输出使用通用 batched Cholesky/log-diagonal 路径。

非有限数、明显非对称 covariance、负 latent variance、非正定标签边缘、非正定 Schur complement、负噪声等会明确失败。没有默认 jitter、相关性截断或把非法 covariance 自动修成正定矩阵的操作。若未来确需加 jitter，必须作为另行记录的模型/数值参数，不能隐式改变概率模型。

## 5. 条件化 B32 与单点 Top-32

公开方法名分别为：

| 方法名 | 实际选择规则 |
| --- | --- |
| `gaussian_epig_pointwise_topk` | 仅按初始单点 EPIG 排序，stable tie-break |
| `gaussian_epig_conditional_greedy` | 每选一个点后条件化其 noisy label，再重算剩余 EPIG |
| `gaussian_ivr_conditional_greedy` | 同一 posterior/参考集，目标改为条件化 IVR |

设已选择集合 B，条件化 latent covariance 为

\[
K_{D,B}(a,b)=K_D(a,b)-K_D(a,B)
[K_D(B,B)+N_B]^{-1}K_D(B,b).
\]

Gaussian 条件 covariance 不依赖尚未观测的标签值，因此 B32 可在不揭示任何新标签的情况下选完。rank-q 更新目标/候选 diagonal blocks 与 reference×candidate cross covariance。贪心各步的条件 MI 求和等于每个参考目标与整个批次的联合 MI（chain rule），但贪心不保证获得全局最优批次。pointwise Top-32 也报告其冻结批次的实际条件增益总和，不能把 32 个原始单点分数相加冒充联合信息。

选择器拒绝重复候选位置、越界位置、候选与已标记集合交集、无效 batch size、重复输出索引等；每个 winner 选一次后从可用集合移除。`selected_positions` 是 posterior 行位置，`selected_row_ids` 是映射后的真实 ID。上层必须只提供 outer-train universe 并冻结该映射；数学模块无法从一个任意字符串 ID 推断其是否属于 validation/test，不能把这一责任隐藏在数值 API 中。标签揭示仍须通过原 RestrictedLabelStore，在最终所选 ID 冻结之后执行。

## 6. API 与规模

posterior 最小接口：`row_ids`、`n_outputs`、`cov(left_positions,right_positions,noisy=False)`（row-major/output-minor 的扁平块矩阵）、`noise_cov` 或 `noise_blocks(positions)`，并建议提供 `latent_diag(positions)`。数学入口为 `score_gaussian_acquisition`、`select_gaussian_batch`；可复用状态为 `GaussianAcquisitionState`。

缓存为 M×C×q² 标量，而非候选方阵。初始 cross covariance 分候选 chunk 请求；每一步只额外请求所有候选到 winner 的 cross covariance，并用 rank-q 更新缓存。默认 cross cache 上限 768 MiB，超过时显式报 MemoryError，不静默减少 M。该预算只计算主缓存；实际进程峰值还包含临时数组、posterior cache 和模型训练对象，必须以运行测量为准。

## 7. 已执行数学与接口验证

执行命令（已有 Conda fish 环境）：

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish \
  pytest -q tests/active_learning_v2/test_epig.py
```

结果：**41 passed，0.32 秒**。测试覆盖单输出解析式、零相关、极弱相关与接近奇异但合法情况、1/2/3 输出和矩形 cross block、逐候选 direct dense Schur 对照、每一步 greedy ID/增益与 dense 重新条件化对照、batch chain rule、多输出依赖与独立边缘求和的区别、物理单位变换、目标权重/重复参考抽样等价性、Top-k 冗余、非法 ID/协方差/噪声与 cache 限制。

额外使用真实 `epig_models.py` 接口，但全部标签为固定 RNG 的合成数据（seed 20261117），独立构建完整所选批次的 noisy covariance 并直接求解 Schur 条件协方差：

| 辅助模型 | 缓存 greedy 的 batch MI | 独立 dense batch MI | 绝对误差 |
| --- | ---: | ---: | ---: |
| 已拟合 exact independent GP | 0.7388797878057944 | 0.7388797878034684 | 2.33e-12 |
| 已拟合 coupled matrix-normal BLR | 0.1426321721387239 | 0.14263217213872395 | 5.55e-17 |

单线程规模检验：已拟合 coupled BLR，3330 个输入、2997 候选、M=2048、q=2、B32，耗时 **11.36 秒**；主 cross cache **196,411,392 bytes**，macOS 进程峰值 RSS **790,200,320 bytes**。共 56 次 `cov` 调用，单次最大请求 1,048,576 个标量，从未请求完整候选方阵。该计时只说明 acquisition 数值实现可计算；它不是色谱校准、GP 训练耗时或真实 AL 效果证据，也不代表所有模型的相同运行时间。

单元测试和这些合成接口验证均不读取候选真实标签、validation 标签或 test 标签。真实 five-seed 性能结论只能读取后续冻结、完整重训的学习曲线，不能从上述数值正确性推出。
