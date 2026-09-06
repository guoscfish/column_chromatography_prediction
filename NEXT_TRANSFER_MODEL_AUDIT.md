# Next transfer model audit

基点：`study/cross-column-transfer-validation` / `61f20c9d10e1e5cfbd523ae0cac448f55de74eb2`；远端已 fetch/pull。新分支：`codex/study-transfer-residual-diagnostics`。本文件在新实验拟合及评估前冻结设计。

## 已覆盖的思想与重复候选

| 实际实现 / 历史证据 | 已回答的问题；不能换名重做的方案 |
| --- | --- |
| `transfer/calibration.py`；cross-column 的 scale-only / affine | source q50 的线性尺度、偏移；再做 linear correction / linear output adapter 属于重复。 |
| S1、T1 condition Ridge；cross-column affine + condition Ridge（内层按 compound 重拟合 affine） | 条件线性 residual 已测；“GAM”若只有线性 condition 项也重复。当前 25g/40g 增益约 0.4–2.7%，未达 5%。 |
| T1 head / last1 / last2；final-V2 transfer head / last2 / full | 冻结表示后的头适配、浅层到全网络解冻已测。历史 T1 使用旧 source/monotonic head，不能当作当前 V2 精确排名；当前 V2 full 已测且不支持继续普通 fine-tuning。 |
| `model.py:ResidualGraphAdapterHead`；T1b-1 r8/r16/r32 | `head(h + up(relu(down(h))))`，h 已经过固定 sum pooling。属于 pooled representation 的非线性容量探针；扩大 width、改称 graph readout adapter 都是重复。 |
| G0-4 paper-style column RBF + last2 + 新头 | 已做过单柱 column-input adaptation，而且同时改多个因素；不是多目标柱标签联合拟合，不能重新包装为新 controlled column-context 实验。 |
| `models/qgeognn_v2.py` + final qualification | 五层 node、四层 edge、global_add_pool、typed condition residual、Linear/ReLU 六输出；458,952 参数，qualified baseline 不动。真正 adaptive pooling 未测，但未测不等于值得立即投入。 |
| cross-column 8g/25g/40g × row/compound × 5 seeds × 30/50/70/100 | 复用原始 sample IDs、roles、预算、source checkpoint 与尺度。target-compound holdout 不是 source-unseen OOD。 |

审阅入口：`scripts/studies/run_cross_column_transfer.py`、`run_final_v2_transfer.py`、T1/T1b/S1 runners，`src/qgeognn_al/{models,training,transfer,evaluation}`，historical `model.py`，相关 README/报告、protocol、decision、split/data audits、method register。旧 register 中的阶段授权/排名不能覆盖后续 V2 qualification 和本次明确授权。

## 实验 1：单调、正则化的一维非线性校准

每个输出分别拟合 source q50 → target volume。用 train source q50 的 1/3、2/3 分位数两个内结点，连续三段折线，三段斜率非负，尾部按边界斜率线性外推。目标为 train MSE 加相邻斜率差惩罚；每个输出四个系数，只比 affine 多两个自由度。仅检查两个固定正则强度 0.1 / 1；以原 validation 的 source-normalized score 选一个全输出共同强度。另报告 validation 在 scale / affine / nonlinear 中选择的实际策略，避免只比较较弱 affine。所有结点、标准化与系数仅在 gradient_train 拟合。

复用 120 个 frozen contexts，与现有 scale-only / affine / target-head-only 做逐 seed、逐预算配对。选择完成、预测落盘后才能读对应 test truth。此实验最直接检验映射曲率，完全不改变表示与条件分支；负结果仅排除这种低容量单调曲率，不能证明一切非线性无效。

## 实验 2：共享柱校准系数，匹配总标签预算

联合形式为 `y_c / (mass_c/4) = a_c * (source_q50/source_std) + b_c`（拟合时 y 同时除 source_std）。mass-ratio 只作固定数值参数化。三个柱各有自己的 slope/intercept；惩罚它们偏离三个柱的平均系数，检验数据驱动 partial pooling。只比较 λ=0 / 0.1 / 1；λ=0 为完全独立 affine，validation 为每个 focal column 选择 λ。增加同样 λ 网格、向固定 identity 系数 (1,0) 收缩的 local control，以区分普通 shrinkage 和其他柱标签提供的信息。不得添加其他特征或继续调参。

比较单位是三柱 portfolio：每柱原预算 B，总计划预算 3B，实际预算为三柱 actual_budget 之和，计入所有 validation。独立 portfolio（scale / affine / head）和共享 portfolio 使用完全相同的已购买 label IDs；不得把共享模型的 3B 成本标成单柱 B。同时保留各柱条件指标，但它们不是“只花 B 标签”的共享性能。按原 B 轴报告逐柱 AULC；另报告总计划/实际 portfolio 预算轴，避免把 compound 近似预算当成精确数量。

为原样保留 focal frozen split，各 focal 模型分别拟合：供体只用其原 gradient_train；compound 下进一步排除 focal validation/test 化合物的全部供体标签；不使用供体 validation 选参。row 保留原 row 语义（允许同 compound 的不同记录），不宣称 compound isolation。每个 focal 拟合独立执行并记录允许/排除 IDs，模型之间不传递参数。λ=0 必须数值还原该柱独立 affine。如果任一供体不足两个 compound 或设计秩不足，停止该共享 context 并记录，不补充标签或重划 split。

当前共有 574/490/529 行，88/78/80 个 compound；柱间交集 48/43/58。可检验已观测三柱间的统计共享，但只有三个规格，flow 各自恒定为 10/15/30，mass 与 flow 完全按规格混杂。不拟合物理定律，不声称可外推到新规格。全局合并独立 splits 会泄漏 compound 标签，因此禁止直接拼接训练。

## 决策与数据边界

预先固定：稳定收益 = 配对 AULC 均值和中位数改善、至少 4/5 seeds 获益；material = 均值相对改善 ≥5%。nonlinear 主候选为 validation 选择的 `nonlinear_policy`，独立 spline 同时报告。保留 nonlinear 需要相对 scale、affine 和仅在二者间选择的 `linear_policy` 在至少两个不同柱的 compound contexts 同时稳定/material，且这些柱 row 平均不恶化 >5%。共享需相对 scale、affine 和 local identity-shrinkage control 同样通过上述门槛，并且相对这三个参考的 portfolio compound 均值改善均 ≥5%；head-only 配对完整报告，25g/40g 必须稳定/material 超过 head，8g 不能隐藏其强 baseline。这是一条审慎的“值得继续研究”门槛，不是部署选择或统计显著性结论。

同时报告 V1/V2 R²、RMSE、MAE、arithmetic NRMSE（沿用 cross-column AULC）、RMS NRMSE（沿用 validation metric），各 seed/budget、mean/std/median/range、配对 wins 和 actual budgets。已反复使用的 test 是开发性冻结评估，不是全新确认集；不据 test 调 λ、结点或追加模型。若都不通过，结论为 `NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA`，并不证明 residual 全来自噪声，也不自动支持 readout。最多这两个实验；不启动 AL、Clean、HPO、adapter sweep 或新 readout 训练。

现有数据足以执行上述有限诊断，不需先收集新数据。确认候选或识别机制需要独立 compound/column 批次、同 mass 多 flow / 同 flow 多 mass 的交叉设计、重复实验与高 source-q50 尾部覆盖；评估 source-unseen OOD 需要真正未见的更多 compound。
