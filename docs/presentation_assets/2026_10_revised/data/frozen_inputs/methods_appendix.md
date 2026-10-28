# 当前代码的方法定义与适用边界

核查基准：GitHub `guoscfish/column_chromatography_prediction` main，commit `42ab12600a353bba4a063ef5d1b623ee2f7b976a`（2026-10-11 23:29:25，Asia/Shanghai）。以下定义取自该提交真实代码。对应源码副本在本目录 `frozen_inputs/src/`；路径与 SHA-256 见 `input_sources.csv`。本次没有执行梯度提取、选样、拟合、预测或 test reveal。

## 1. 共同模型、预算与指标

源码：`src/qgeognn_al/models/qgeognn_v2.py`、`active_learning_v2/benchmark_protocol.py`、`active_learning_v2/sequential_protocol.py`、`evaluation/point.py`。

- QGeoGNN-V2 有 458,952 个有效参数，5 次 node 更新、4 次有效 edge/angle 更新。128 维表示接 `Linear(128,6) + ReLU`；eval 输出还裁剪到 `[0, 1e8]`。
- 六输出顺序是 `[V1_q10, V1_q50, V1_q90, V2_q10, V2_q50, V2_q90]`。梯度与主指标使用第 1、4 列（Python 从零索引），即两个 q50 输出。
- 本次比较固定外层 split seed 157、6101，初始训练标签 L0=333；B=32，21 轮，终点 1005。共享 validation=416 行；test=417 行；outer train=3330 行。横轴只计主动训练标签，不把固定 validation 标签计入。
- 历史 predictor fit 冻结配置：Adam，lr=0.001，weight_decay=0，训练 batch_size=2048，maximum_epochs=1000，patience=100，V1/V2 loss 权重均为 1。模型按 validation combined NRMSE 选 checkpoint。每轮历史 scratch fit；初始化 seed=`outer_seed+2,000,003+member×1,000,033`，训练随机 seed=`outer_seed+3,000,017+member×1,000,033`；CPU threads=2。实际早停 epoch 会变化，不等于给不同方法不同的最大训练预算。

当前真实 combined NRMSE 定义为：

$$e=\sqrt{\frac12\left[(\mathrm{RMSE}_{V1}/s_{V1})^2+(\mathrm{RMSE}_{V2}/s_{V2})^2\right]}.$$

不能改写成两个 normalized RMSE 的算术平均。仓库另有历史 `metrics.py` 中不同的 NRMSE 定义，本次没有使用它。尺度来自各 seed 冻结的 L0 训练标签，不从 validation/test 重新估计。

完整区间 AULC 是预算横轴下的归一化梯形积分：

$$\mathrm{AULC}_{a:b}=\frac{1}{b-a}\sum_i(n_{i+1}-n_i)\frac{e_i+e_{i+1}}2.$$

两 seed 的报告均值是算术平均；没有置信区间、显著性检验或 seed 外推。

## 2. Latent 提取

源码：`src/qgeognn_al/models/qgeognn_v2.py::QGeoGNNV2.extract_representation`；`src/qgeognn_al/active_learning_v2/coverage.py::extract_representations`。

$$h(x)=\sum_{v\in\mathrm{nodes}(x)}h_v^{(5)}+r_{\mathrm{condition}}(x)\in\mathbb R^{128}.$$

这里是 `global_add_pool` 的 node **求和**，加 `TypedConditionCompletionBranch` 返回的 128 维 residual；不是 node 均值，也不是六维预测输出。条件 residual 使用 checkpoint 冻结的条件预处理。

提取时 `model.eval()`、`torch.no_grad()`；默认 batch_size=2048，验证 canonical_position 与输入 indices 完全同序，输出必须有限。原始 latent 提取本身没有逐行 L2 归一化、中心化或 whitening。Hybrid 直接在该 latent 上计算欧氏距离；Fusion 的块归一化另见第 8 节。

## 3. Raw / CW 输出变换及梯度

源码：`src/qgeognn_al/active_learning_v2/gradient_transforms.py::{raw_output_transform,center_width_transform}`；`gradient_features.py::{extract_q50_gradient_sketches,extract_linear_output_gradient_sketches}`。

设当前六输出网络的两个 q50 为 $f=(f_{V1},f_{V2})^T$，全部可训练参数为 $\theta$，线性输出坐标为 $z=T f$。

Raw 使用冻结 L0 population std（ddof=0）：

$$T_{\mathrm{Raw}}=\mathrm{diag}(1/s_{V1},1/s_{V2}),\quad z_{\mathrm{Raw}}=(f_{V1}/s_{V1}, f_{V2}/s_{V2})^T.$$

CW 定义 $C=(V1+V2)/2$、$W=V2-V1$，只由 L0 真值计算 $s_C=\mathrm{std}(C,ddof=0)$、$s_W=\mathrm{std}(W,ddof=0)$：

$$T_{\mathrm{CW}}=\begin{bmatrix}1/(2s_C)&1/(2s_C)\\-1/s_W&1/s_W\end{bmatrix},\quad z_{\mathrm{CW}}=\left(\frac{f_{V1}+f_{V2}}{2s_C},\frac{f_{V2}-f_{V1}}{s_W}\right)^T.$$

没有减输出均值；固定常量平移的导数本来也为零。尺度要求严格为正且有限，不对零尺度加入任意 epsilon。

由 `factorial/protocol.json` 提取的冻结参数如下（不是本次访问训练标签重新计算）：

| seed | sV1 | sV2 | sC | sW | CountSketch seed |
|---|---:|---:|---:|---:|---:|
| 157 | 6.502957820892334 | 13.158523559570312 | 9.627325088483525 | 7.754280767170251 | 4000194 |
| 6101 | 5.462151050567627 | 12.266972541809082 | 8.419568612367186 | 8.779007719505096 | 4006138 |

每行提取 $g(x)=\mathrm{concat}(\nabla_\theta z_1,\nabla_\theta z_2)$。`model.eval()`，loader batch_size=1；`autograd.grad(create_graph=False, allow_unused=True)`，第一输出 retain_graph=True，未参与输出的参数梯度补零；参数顺序遵循 named_parameters 中 requires_grad=True 的顺序。当前 ReLU/clamp 的导数属于实际实现，不能替换成未截断线性头的理论 Jacobian。

Raw 是 **输出梯度**，不是 Latent。CW 只改变 acquisition representation 的输出坐标，模型架构、六输出头和训练目标保持相同；各方法后续标签集合不同，训练出的权重自然可能不同。不能说 CW 训练了一个新的 center/width 输出头，也不能说后续轨迹使用相同权重。

## 4. CountSketch

源码：`src/qgeognn_al/active_learning_v2/gradient_features.py::{countsketch_mapping,sketch_flat_gradient}`；seed 规则在 `benchmark_protocol.py::sketch_seed`。

$$\phi_k(x)=\sum_{j=1}^{2P}\mathbf1[b_j=k]a_jg_j(x),\quad k=1,\ldots,d.$$

- $d=512$；`default_rng(outer_seed+4,000,037)` 产生长度 2P 的 bucket 与 sign。bucket 均匀取整数 `[0,d)`；sign 为 ±1。
- 两个输出梯度使用同一条长度 2P 映射的各自片段，分别 scatter_add 后相加到同一 512 维向量；不是每个输出各 512 维再拼成 1024 维。
- 没有乘 `1/sqrt(d)`，没有逐行 L2 归一化。梯度和 scatter 累加为 float32，后续 selector 用 float64。
- 每次只物化一行的全梯度，不形成 N×P Jacobian。kernel 后续使用 $K=\Phi\Phi^T$，有限维碰撞会近似并扰动原始梯度内积。
- 未压缩的输出坐标等距旋转与有限 sketch 后的数值行为不能混同。真实映射对两个输出坐标采用不同随机片段，旋转等距性不保证逐次精确保留；factorial 的 finite sketch orientation 是已披露机制因素。

## 5. LCMD（TP 模式）

源码：`src/qgeognn_al/active_learning_v2/lcmd.py::lcmd_tp_select`，历史主方法调用在 `sequential_acquisition.py::acquire_sequential_batch`。

初始中心 $\mathcal C=L_t$（全部当前 labeled，不仅 L0）。对尚未选中的 $u\in U_t$：

$$c(u)=\arg\min_{c\in\mathcal C}\|\phi_u-\phi_c\|^2,\quad d_u=\|\phi_u-\phi_{c(u)}\|^2,$$
$$j^*=\arg\max_j\sum_{u:c(u)=j}d_u,\qquad u^*=\arg\max_{u:c(u)=j^*}d_u.$$

把 $u^*$ 加为新中心，删除其候选资格并更新最近距离，重复 B=32 次。这里的“最大簇”按 **平方距离总和** 定义，不按簇内样本个数。

没有额外 RMS、逐行单位化或 whitening。距离转为 float64，通过范数展开计算，负舍入截到 0；同分遵循稳定输入顺序，最近中心只有严格 `<` 才替换。若退化导致选中空簇，实现显式报错，没有随机修复或 epsilon。

LCMD32 / Raw-LCMD 均使用 512D Raw 输出梯度；CW32 / CW-LCMD 使用 512D CW 输出梯度。这一页没有完整 Latent-LCMD 轨迹。

## 6. Conditional MaxDet

源码：`src/qgeognn_al/active_learning_v2/maxdet.py::conditional_gradient_maxdet`。

只由当前 labeled 计算一个整体尺度：

$$s_L=\sqrt{|L_t|^{-1}\sum_{l\in L_t}\|\phi_l\|^2},\quad x_i=\phi_i/s_L,\quad A_L=I+X_L^TX_L.$$

选集 $S$ 的条件 D-optimal 目标为：

$$\log\det(A_L+X_S^TX_S)-\log\det A_L.$$

令 $A_{L\cup S}=I+X_L^TX_L+X_S^TX_S$，候选 $i$ 的边际分数是：

$$\Delta_i=\log(1+x_i^T A_{L\cup S}^{-1}x_i).$$

代码 greedy 最大 leverage（与最大 log1p 同序），不是搜索全局最优子集。数值实现：float64 Cholesky 分解 $A_L$，用 `solve_triangular` 得到 conditioned whitened 坐标，然后从 I 开始做 Sherman–Morrison rank-one inverse 更新。每轮对称化；leverage 轻微负舍入截为 0，超容差失去 PSD 则报错；要求 positive finite labeled energy，不加任意尺度 epsilon。

正则化为固定单位 I。完整主图的 Gradient MaxDet 为 **全池**候选版本；不是额外 ensemble Top50 gate 的 U50 版本。本次 Raw/CW-MaxDet 沿用同一个 objective 与 current-L RMS 策略。

## 7. Kernel-IVR

源码：`src/qgeognn_al/active_learning_v2/ivr.py::conditional_batch_ivr`。

reference universe 是冻结 outer-train $R=L_t\cup U_t$，包含全部 reference 行；不包含 validation/test X。每行参考权重为 $1/N$，L 和 U 必须严格 partition R。

$$s_R=\sqrt{N^{-1}\sum_{r\in R}\|\phi_r\|^2},\quad x_r=\phi_r/s_R,\quad M=X_R^TX_R/N,$$
$$C_L=(I+X_L^TX_L)^{-1},\quad v_i=x_i^TC_Lx_i,$$
$$\mathrm{IVR}(i)=\frac{x_i^TC_L M C_L x_i}{1+v_i}.$$

选择使 uniformly integrated row-kernel conditional variance 减少最大的候选。单位先验 precision=1，单位观测噪声 variance=1；不使用标签残差、不拟合噪声、不以 V1/V2 单独建立精确 Bayesian posterior。

数值实现：float64 `cho_factor/cho_solve`；初始投影 $Q=XC_L$、$H=QM$。选择 $i$ 后，设 $v=Q_i^T$、$h=H_i^T$、$b=Qx_i$、$d=1+v_i$：

$$Q\leftarrow Q-bv^T/d,\quad H\leftarrow H-bh^T/d.$$

这相当于 greedy batch 内不断条件化，不重复以同一个初始协方差评分。每次检查平均方差的真实下降与分数一致（rtol=1e-7、atol=1e-11）；负方差/分子超 `1e-9` 报错，微小负数截零。同分按 canonical candidate 顺序取 argmax。

**IVR 是固定当前模型梯度特征上的线性核代理方差缩减，不是精确 QGeoGNN 后验。** CountSketch、当前特征固定、单位先验/噪声、uniform reference weight 和 greedy batch 都是近似或建模假设。RMS 使用 L+U，MaxDet 使用 L；所以跨 selector 差异不能完全归因于 acquisition objective。本次展示 Raw/CW 配对以保持同 selector 下归一化规则相同。

## 8. Hybrid 与 Gradient + Latent Fusion

### Hybrid traditional

源码：`src/qgeognn_al/active_learning_v2/sequential_acquisition.py::v2_hybrid_select_current`；`uncertainty.py::ensemble_uncertainty_select`；`coverage.py::coreset_tp_select`；`sequential_runner.py`。

K=3 个独立初始化的当前模型，取 V1/V2 q50 的 ensemble population std：

$$u_i=\left\|\left(\mathrm{std}_k(f_{V1,k}(i))/s_{V1},\mathrm{std}_k(f_{V2,k}(i))/s_{V2}\right)\right\|_2.$$

按 descending $u_i$ 稳定排序，保留 `max(B,ceil(0.25×|U_t|))` 个候选；随后在 **当前单个 member-0 模型的 128D Latent** 上 farthest-first。全部 L_t 为初始中心，每次选与现有中心最近距离最大的候选并更新距离，共 B=32。它不是 LCMD 最大簇，也不是 gradient 与 latent 的加权 fusion。

主模型 member 0 作为本轮预测器，member 1/2 在 acquisition 阶段另做同配置 scratch fit。因而这里的比较是 **相同标签预算、相同 predictor 配置** 的效果描述；Hybrid 的训练计算总量更大，不能据图宣称相同 compute budget 的优势，也不对训练秒数/总模型拟合数直接排名。不确定性是 ensemble disagreement 代理，不是精确后验方差。

### Gradient + Latent Fusion

源码：`src/qgeognn_al/active_learning_v2/fused_features.py::fuse_gradient_latent_features`；`fused_maxdet_runner.py` 的实现调用已在代码检查时核对（未执行）。

对当前 `[L_t,U_t]` 的 512D gradient $g_i$ 与 128D latent $h_i$，分别从当前 L_t 估计块能量尺度：

$$s_g=\sqrt{|L_t|^{-1}\sum_{l\in L_t}\|g_l\|^2},\quad s_h=\sqrt{|L_t|^{-1}\sum_{l\in L_t}\|h_l\|^2},$$
$$f_i=[\sqrt\alpha g_i/s_g,\sqrt{1-\alpha}h_i/s_h]\in\mathbb R^{640},$$
$$K_f=\alpha K_{g/s_g}+(1-\alpha)K_{h/s_h}.$$

float64 算尺度与拼接，最终转 float32；默认 alpha=0.5，另有冻结的 alpha=0.80 screen。拒绝零/非有限 labeled energy，不逐行单位化、不减均值、不用 U 的标签；输入 API 不接收 target 数组。Fusion 后接相同 `conditional_gradient_maxdet`（还会应用自己的 current-L RMS，理想算术下融合后 L 平均能量已经为 1）。

`latent_common_mode_diagnostics` 中的 L_t 均值中心化仅用于诊断，**不改变 acquisition**。代码 `representation_diagnostics` 的 effective_rank 是 eigenvalue entropy 的 exp；不能与 same-state 报告中 participation ratio `(sum λ)^2/sum λ²` 混写。

Fusion 没有进入本次六方法主排名；alpha=0.80 等截断 screen 不能外推 1005。二层示意图仅表示 representation 可接不同 objective，不表示全部 Latent/Raw/CW 组合都有完整验证。

## 9. Same-state branching 与本次图 12 的 split

源码：`src/qgeognn_al/active_learning_v2/same_state_branching.py` 的 source/anchor 定义与 historical report 构造。这里只阅读源码，不调用 `reveal_test_and_report`。

历史 2 seeds × 2 sources（gradient_lcmd/kernel_ivr）× 2 anchors（429/653）=8 个状态。每个状态固定同一 `(L_t,U_t,checkpoint)`，分别接 Raw-gradient LCMD、Kernel-IVR、Gradient-MaxDet。每条分支进行 2 轮 B=32，历史中新增 48 次 scratch fit；锚点严格复用，不重训。状态 A 与状态 B 之间 L/U/checkpoint 不同，“same state”只限定各状态自己的三条分支。

$$\mathrm{short\mbox{-}AULC}=\frac{e_0+2e_{32}+e_{64}}4=\mathrm{AULC}_{n:n+64}.$$

`branch_results.csv` 的 NRMSE 与 short_AULC 是 **历史所有分支 acquisition、fit、test-X predictions 冻结后，统一揭示的 test 指标**。本次仅读取该既有数值表，不读取 test truth、不重算 test prediction。主图 10/11 为 validation，不能与图 12 test 数值直接排名。

seed157/budget429 的准确 short-AULC 原值保留于 `fig12_reversal_values.csv`（显示图按 6 位小数舍入）。8 状态中 LCMD 与 IVR 各胜 4 次、MaxDet 0 次。validation+64 与 test+64 完整排序仅 1/8 状态一致，因此不能把 test reversal 当作 validation controller 已成功。证据仅为 development/mechanism，当前数据不足以训练可靠 controller。

## 10. 方法映射与阶段边界

`method_mapping_audit.csv` 保存六个 factorial arm 的 representation/selector/归一化/历史路径。原 333–525 是已冻结的开发 factorial；用户看到结果后延伸 525–1005，原延伸主指标为 AULC525–1005。PPT 为展示全轨迹同时给 AULC333–1005，但没有改写原 gate 或把两阶段变成独立确认实验。

CW-LCMD 开发 gate 符合工程门槛，CW-IVR/CW-MaxDet 没有形成新的确认候选。2% 是资源分配工程阈值，不是统计显著性。Whitening 已按用户要求停在 525，不能纳入完整 1005 排名。历史 test exposure 已披露；这不是新的 blind test。
