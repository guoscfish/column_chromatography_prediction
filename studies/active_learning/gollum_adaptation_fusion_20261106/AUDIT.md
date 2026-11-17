# 旧研究、真实训练目标与数据身份审计

日期：2026-11-06，Asia/Shanghai。本审计只读旧目录；新增文件全部位于本研究目录。未训练 QGeoGNN、未下载权重、未调用付费 API、未读取 test 真值、未改写 Free/NoCW/CW-Free 或旧 GP 结果。

## 核心结论

旧 Representation Benchmark 的 **20 个状态、720 次端点 GP 拟合和其中 360 次 Projection 拟合均真实存在**。360 次 Projection 全部达到 40 次迭代上限且 `success=False`；其余 280 次 Frozen Matérn 和 80 次线性 GP 均报告优化成功。数值运算成功不能代替收敛或概率质量。

对旧代码直接交给 SciPy 的真实 Torch objective 做逐坐标数值差分，并核对最终 NumPy 后验，6 项合成测试全通过。最大梯度绝对误差为 **2.86×10⁻⁸**；训练协方差最大误差 **4.44×10⁻¹⁶**。因此本次没有发现能够解释严重欠覆盖的梯度、训练/推理变换或 latent/observation 混用 bug。这个结果仅覆盖声明的合成数值系统，不证明每个历史解都稳定或已经收敛。

更直接的历史证据是：所有 360 个 Projection 端点/初始化相对对应 Frozen 表示都降低训练 NLL，同时降低估计 noise；102 次命中标准化 noise 下界，96 次命中 lengthscale 上界。噪声下降、较大投影容量、未收敛及 plug-in 同方差 GP 是需要消融的机制线索；不能仅由这些相关现象断言单一原因，也不能直接归因模型大小。

机器可读结果见 [audit_results.json](audit_results.json)，真实闭包检查见 [legacy_gradient_results.json](legacy_gradient_results.json)。审计可从项目根目录重放：

```bash
python studies/active_learning/gollum_adaptation_fusion_20261106/audit_legacy.py
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python studies/active_learning/gollum_adaptation_fusion_20261106/test_legacy_gradients.py
```

## 源码实际定义

旧 `gp_models.py` 的 SHA256 为 `86eaaccd67da3679ea08e095f7f81cc54d5265b1a20fb219aa55e48cffd49118`，与旧 artifact manifest 一致；`run_benchmark.py`、配置、fits/metrics CSV 和 summary JSON 也均与旧 manifest 一致。

- 每个端点独立建模；训练响应减经验均值、除 population std，固定标准化零均值。signal/noise 在标准化空间优化，再乘响应尺度平方还原，预测与 NLPD 使用 mL/mL²。
- 特征先减各坐标的训练均值，再除一个训练侧 global centered RMS；**不是每维各除自己的 std**。Projection 为真实输入维数→64 的线性层与 ELU，随机初始化 `N(0,1/p)`、bias=0。
- Matérn-5/2 使用 `r=sqrt(||z−z'||²/d_z)/ell`。`log_signal`、`log_noise` 都是**方差**的对数，边界为 [-9,9]；`log_ell` 边界 [-5,5]。noise 下界因此是标准化方差 `exp(-9)=1.23409804×10⁻⁴`。
- Matérn/Projection 由 SciPy L-BFGS-B 调用 Torch float64 精确 MLL 和 autograd；每个已声明初始化单独拟合，`maxiter=40, maxls=20, maxfun=400, ftol=1e-9, gtol=1e-5, maxcor=10`。原实现不加投影正则或噪声先验，也不积分参数不确定性。
- 线性对照复用 Phase0 `LinearGP` 的三个固定起点，不能把它解释为同一套单起点 40 步预算。其 `max_iter` 显示字段不控制委托实现的迭代上限。
- Torch 与 NumPy 都有明确 Cholesky jitter 预算；全部 720 次最终 jitter 和有记录的训练 jitter 都为 0。**这次欠覆盖不能由“jitter 被当成 noise”解释。** 非零 jitter 分支的梯度与跨尺度一致性未由本审计确认。
- 后验 latent variance 是函数不确定性；观测方差为 latent+noise，旧评分确实使用后者。合成后验测试确认观测噪声只加一次。
- Frozen Fusion 先在 L_t covariates 上分别处理 GNN128/Qwen1024、PCA64、block RMS，再 concat128。Fusion/GNN 的 128→64 各 8,256 参数，Qwen1024→64 为 65,600 参数；容量不可混称相同。

核只依赖 `z/ell`，同时缩放 z 与 ell 会保持核不变。但 ELU 的负分支不齐次，直接把 W/b 同比放大不构成全局严格对称；全正激活区域存在严格对称，其余区域应描述为尺度耦合或弱可辨识，不能把原审计中的简写推广为处处精确不可辨识。

## 重算的概率与优化统计

以下对每个表示的五 seed×四预算×两端点平衡汇总；Projection 每个状态包含 17/29/43 三初始化。它们是计算诊断计数及描述性均值，不是独立重复。

| 表示/变体 | 拟合数 | success | 参数边界命中 | 90%观测覆盖 | 95%观测覆盖 | noise 方差中位数 mL² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen Conditions Frozen | 40 | 40 | 0 | 95.43% | 96.63% | 68.0772 |
| Qwen Conditions Projection | 120 | 0 | 63 | 22.78% | 26.90% | 0.05158 |
| GNN Frozen | 40 | 40 | 0 | 68.56% | 72.76% | 2.1759 |
| GNN Projection | 120 | 0 | 57 | 45.04% | 50.39% | 0.27508 |
| Frozen Fusion | 40 | 40 | 0 | 70.13% | 74.33% | 2.1044 |
| Fusion Projection | 120 | 0 | 65 | 21.79% | 25.45% | 0.05116 |

跨端点混合的 mL² 中位数仅是数值摘要，端点尺度不同；机器结果另保存标准化 noise 与 noise/signal。Qwen Projection 标准化 noise 中位数就是下界；Fusion 为 2.7247×10⁻⁴、GNN 为 1.7910×10⁻³。Projection 相对相同数据 seed/预算/端点 Frozen noise 的配对比值中位数分别为 Qwen **0.000268**、Fusion **0.03284**、GNN **0.2690**。全部 120/120 配对均下降，训练 NLL 也各 120/120 下降。MLL 改善不构成训练外校准保证。

边界细分：Qwen 63/120 为 noise 下界；GNN 57/120 为 ell 上界；Fusion 39/120 为 noise 下界、39/120 为 ell 上界，其中 13 次同时命中，联合 65 次。signal 上下界、noise 上界和 ell 下界均未命中。总体为 185/360 个 Projection 解触边；不能笼统写成“全部噪声触底”。

旧报告 96 条预算×表示/核×端点的 seed 均值覆盖范围为 **14.78%–97.26%**，72/96 低于名义 90%。Qwen Projection 的均值误差改善与概率恶化并存；GNN/Fusion Projection 连开发平均均值误差也未形成稳健改善。Frozen Qwen 较高覆盖伴随更宽区间和更差均值，不能以覆盖率单独宣布更优。

旧 JSON 保存了 objective evaluation 的 NLL 序列；L-BFGS 线搜索的中间 evaluation 不保证单调，不能把它当逐次接受迭代的学习曲线。历史结果未保存每个投影 W/b、完整优化向量、最终梯度范数或训练 kernel condition number，因此不能从现有 JSON 证明训练距离具体压缩了多少、是否每个解都在驻点，或完全分解初始化差异。需由冻结重放与新受控消融补证。

## 真正检查了什么身份

20 个状态均来自统一历史 Raw LCMD 轨迹：seed 157/887/2357/6101/12203，L333/L525/L813/L1005 对应 round 0/6/15/21。重新双读 SHA256 检查了每个状态的 checkpoint、Raw cache、partition、prepared NPZ 及全局 LLM cache；全部匹配旧冻结记录。prepared NPZ 的有序 indices 摘要、行数、状态合同也全匹配；没有反序列化模型或读取其中响应数组。

720 个原始 fits.json 的数值与修复后的 summary/fits.csv 逐项一致。一个已被旧汇总器处理的**元数据缺陷**必须继续防范：`{**base, **model.audit}` 会把数据 seed 覆盖成优化 seed，并把变体 `projected` 覆盖成实际核名 `matern52`。summary 已按状态目录恢复 data seed，并用 `trainable_projection` 恢复变体，另存 `model_random_seed`/`actual_kernel`。本次审计复现了该现象；没有更改旧数据，也未发现它改变旧汇总数值。新代码应分开命名 data seed 与 initialization。

L1005 的 fit 所属目录仍是 sequential study，Raw acquisition cache 来自 LCMD 延长目录；只要同 checkpoint/ordered L/U/contract/hash 匹配就可复用。早期 `historical_cw_available=False` 已由旧 `CW_CACHE_AUDIT` 更正：共同 L333 存在同态 Free CW 缓存，后期不同轨迹仍不得替代。不能用旧字段否定现有缓存，也不能用共同初始化为后期混轨迹辩护。

## train/cal/check 与 forward evidence 的硬边界

1. Frozen Qwen 的语言 encoder 不使用本任务响应。若其后 projection、PCA/标准化和 GP 只在 fit 集训练、温度只在 calibration 集学习、check 不参与选择，可得到该冻结语言表示下的分离诊断；还应报告分子分组、样本相关性及已有开发曝光。
2. **历史 GNN checkpoint 已用整个 L_t 的响应训练。** 将 L_t 再拆 GP fit/cal/check，不能让 GNN“忘记”check 标签。GNN/Fusion 的此类结果只能是固定监督表示条件下的 GP 分离诊断；不能称完整管线无泄漏或真正训练外确认。
3. 416 行 validation 参与过 GNN checkpoint selection 和既往表示开发。无论新的 GP 是否显式用其标签训练，所有此处 validation 分数都必须标注开发证据，不能反复调至覆盖满意后称独立确认。
4. SMILES 分组的内部 check 与旧 row-distribution validation 不同。前者包含分子组 holdout，后者是既定 row 任务；二者不得混为同一评价总体。
5. 真正 forward 核验要冻结采集前 checkpoint/L_t，使用之后已经合法揭示的观测及其 immutable acquisition/feedback receipts；特征必须由该旧 checkpoint 生成。不能用后来模型的特征重构早期 forward。当前 prepared cache 仅覆盖 L_t+固定256 U covariates+416 validation，并不支持任意未来 block；缺行需要合法、feature-only 的旧模型提取及新身份回执。
6. 后期 Free/NoCW/CW-Free 的 L/U、checkpoint、反馈与 LCMD 已分叉；同预算不等于同状态。其他轨迹未来标签不能直接补成 LCMD 的后续已揭示响应。当前分支的严格 forward 或 residual fusion 若缺冻结证据，应标记尚不能检验。

## 已有主动学习研究提供的上下文

项目级入口是 `docs/NEXT_STAGE_DECISION.md`，其正文更新日期为 10-29，需结合具体报告日期。Free/LCMD 五 seed 在 L1485 全部完成：Free endpoint NRMSE 0.334085，LCMD 0.347582；全程 AULC 分别 0.439159、0.417049，说明端点和整段效率方向不同。历史 Free 157/6101 保留 Astra/high 前缀，不是严格同配置的 CW 因果消融。

Sequential B32 的五 seed 结果是 Random/Hybrid/LCMD AULC 0.624260/0.479354/0.479539。旧判定 `NO_CLEAR_SEQUENTIAL_AL_GAIN` 不能改写为“对 Random 无收益”。Kernel-IVR AULC 0.500070，比 LCMD 差 4.28%，仅 1/5 配对获胜；机制审计的 Top32 稳定门槛未过，判定 `STOP_BEFORE_GATE_C`。高方差 rank 相关不保证最终批次稳定，更不保证下游 QGeoGNN 学习收益。

NoCW 去掉显式 CW 字段，仍保留点预测、分位数、宽度、结构、五条件和既测反馈，不是“无模型信息”。历史行为诊断显示前期覆盖与后期批次分子数存在差异，但模型状态、执行时期及前缀混杂，不能从理由文本或这些行为直接推断机制因果。

本次读取的 CW-Free 报告快照完成 L 为 813/1261/1197/1197/1101，NoCW 停止快照为 1069/1005/1133/1165/1229；各轨迹长度不同，不能直接比较这些不同积分区间的原始 AULC。这里记录的是报告内容，不是重新检查控制器存活或宣布终局。本审计不恢复、停止或修改任何既有运行。

Phase0 的 52 次低成本线性 GP 拟合无数值失败，却在固定 r00 模型的随后192观测点上只有 61.5%–74.5% 的90%覆盖，已经说明“线性优化成功”和“校准可靠”不同。其风险 GP 训练在前96个后来观测后才可用，不能回送历史 r00 或当成后期当前 GNN 的风险模型。

## 新受控训练器的独立复核

在不修改已冻结 `gp_training.py` 的前提下，另实现 [test_controlled_gradients.py](test_controlled_gradients.py)。6 项测试全部通过，直接截获 original/converged/regularized/noise_floor/staged 的实际 SciPy 闭包并逐参数差分；staged 检查两个阶段。另确认正则 penalty 与 NLL 分开、noise floor 进入真实 bounds、二阶段只拟合三个 GP 超参数且保持 W/b、最终 Torch/NumPy covariance/原尺度 NLL 一致，并直接比较 new original 与 unchanged legacy 的 objective/gradient。结果见 [controlled_gradient_results.json](controlled_gradient_results.json)。这些测试不消耗正式实验数据，也不构成训练收敛或概率放行。

新代码读取后暂未发现监督标签从 check/calibration 回流到 GP objective；Fusion 的历史 GNN 监督曝光仍然存在。同 fit-side PCA 预处理的 Frozen Fusion 对照由独立冻结 `supplement_frozen_fusion.py` 补充，回执为 8 次拟合完成，不改变主冻结训练器，不能用旧全 L_t Frozen Fusion 冒充相同训练集合。

独立运行本目录的旧/新真实闭包、Phase1、加性核四个测试文件，**31 项全部通过（5.44 秒，不含进程启动）**。测试含 cache 篡改拒绝、test identity 拒绝、group disjointness、train-only PCA、lambda=0 退化、PSD、噪声分离、梯度及后验恢复。真实状态权限测试只复用已授权 L_t/validation 缓存；无 test 或 U 响应请求。Toy/合成测试只构成工程证据，不产生 T5、LoRA 或预测性能结论。

追加 [test_data_permissions.py](test_data_permissions.py) 的 5 项对抗性测试全部通过：prepared cache hash、checkpoint state、源文件 hash 在反序列化前被拒绝；LLM cache drift 在读取该语言缓存前被拒绝；将 test ID 注入候选元数据会触发权限拒绝。修改均仅在内存中 mock，没有改动历史文件；后两项最多加载已有合法 L_t/validation 缓存，不读取禁止标签。回执见 [data_permission_results.json](data_permission_results.json)。

代码复核也发现一个报告门槛的健壮性条件：若主矩阵未完成或存在失败，不能只在有限成功产物上计算通过率；固定每个表示×方法的完整 16 次主拟合必须进入分母，并核对两个 seed、两个预算和两初始化均在场。正式结论需同时检查 completion/失败记录，不得用 partial 子集放行。Phase3 每拟合绑定配置与输入摘要；最终 artifact source hash 只能证明交付源码身份，若缺评分前源码冻结回执，不应称为独立的评分前代码冻结证明。

所有拟合、模板与采集结果只能支持离线开发诊断；没有新增 AULC 或正式 AL 证据。
