# E0-3b：4g→8g 多种子稳健性与控制变量实验

## 本阶段回答的问题

本阶段不是主动学习实验，也不使用 test 选择模型。它在 E0-3 单种子迁移矩阵之后，检查以下问题：

1. 4g 预训练对 8g 是否稳定优于 8g scratch；
2. 仅预测头、末两层+预测头、全模型微调的相对排序是否跨随机种子和划分方式保持；
3. 先前很低的 row-level test 指标是否由特定样本或特定 split 主导；
4. V2 损失权重、目标尺度归一化和重新初始化预测头是否值得进入正式多种子对照。

## 冻结协议

- 数据：`experiments/e0_8g_transfer/canonical_8g.csv`，按原代码启用 V1≤60 mL、V2≤120 mL 阈值，共 552 行；重复实验保留。
- 图：87 个 8g 结构复用 4g 图缓存，1 个结构使用 8g 新建缓存。
- 随机种子：42、525、1101。
- 划分：row-level 和 compound-group；每个 seed 内所有策略共用同一 split。
- 核心策略：8g scratch、仅预测头微调、末两层+预测头微调、全模型微调，共 24 组。
- 因素预实验：仅在 row/seed=42 上增加预测头重置、末两层+头重置、V2 等权、按训练集目标标准差归一化损失，共 4 组。
- 训练：最多 500 epoch，patience=100，batch size=2048；checkpoint 只按 validation 的训练集方差归一化点预测分数选择。
- test 只在训练和 checkpoint 选择结束后评估。

共完成 28 组训练。

## 核心结果

以下为 6 个配对运行（2 种 split × 3 seeds）的汇总。`normalized score` 越低越好；R² 越高越好。

| 策略 | validation normalized score | test normalized score | test R² V1 | test R² V2 | test MAE V1 | test MAE V2 |
|---|---:|---:|---:|---:|---:|---:|
| 8g scratch | 0.975 ± 0.532 | 0.960 ± 0.630 | 0.435 | 0.581 | 3.336 | 5.886 |
| 仅预测头 | 0.750 ± 0.290 | 0.541 ± 0.349 | 0.643 | 0.750 | 2.908 | 4.355 |
| 末两层+预测头 | **0.619 ± 0.301** | **0.502 ± 0.473** | **0.687** | **0.765** | **2.366** | **4.347** |
| 全模型 | 0.683 ± 0.364 | 0.525 ± 0.418 | 0.685 | 0.710 | 2.497 | 5.028 |

在每个 seed/split 内与 scratch 配对后，三种迁移策略的平均 test normalized score 均更低。末两层+预测头的跨运行 validation 均值最低，因此仍是当前候选；样本量只有每种 split 3 个 seeds，尚不能表述为统计显著或永久冻结。

按 split 分开看：

| split | 策略 | test R² V1 | test R² V2 | test MAE V1 | test MAE V2 |
|---|---|---:|---:|---:|---:|
| compound-group | scratch | 0.594 | 0.657 | 3.198 | 5.649 |
| compound-group | 末两层+预测头 | **0.901** | **0.892** | **1.803** | **3.676** |
| row-level | scratch | 0.277 | 0.505 | 3.473 | 6.123 |
| row-level | 末两层+预测头 | **0.472** | **0.637** | **2.929** | **5.019** |

这里不能据此断言 compound-group 天生更容易。8g 只有 88 个化合物，3 个 seeds 的 test 组成差异很大；compound 结果说明模型可以达到较好的跨化合物预测，但还不足以估计稳定的总体性能。

## 源行 224 的敏感性

该记录只进入 row/seed=42 的 test。末两层+预测头在保留该行时 test R² 为 V1=0.009、V2=0.377；仅在敏感性计算中排除该行后为 V1=0.804、V2=0.874。

正式结果继续保留该记录。敏感性实验只能证明小 test 可被一个极端点支配，不能证明该记录无效；是否更正或删除必须回查原始 UV/实验记录。

## 四个因素预实验

这些结果只有 row/seed=42，作用是筛选下一轮因素，不是最终结论。

- `V2 weight=1`：相对 legacy 的 0.5，validation 分数从 0.231 略变差到 0.235，test 分数从 1.118 改善到 1.065，V2 coverage 从 0.589 提高到 0.643。信号较小，需要多种子验证。
- 按训练目标标准差归一化损失：validation 分数从 0.231 改善到 0.228，V2 coverage 提高到 0.696，但含源行224的 test 分数变差到 1.168。它改善了任务平衡/区间覆盖的迹象，但未证明点预测更好。
- 仅预测头重置：validation 分数恶化到 0.782，test V2 MAE=7.854，crossing rate=50%。
- 末两层+头重置：validation 分数为 0.254，crossing rate=96.4%。随机新头破坏了已学到的分位数顺序，不能直接视作论文方法复现。

## 当前结论与下一步

1. 4g 预训练对 8g 有稳定的相对收益；此前“所有方法都很差”的印象主要来自单个 row split。
2. `last2_head_lr1e-4` 保留为下一阶段的 provisional candidate，但在 P0 控制项完成前不冻结。
3. 下一轮优先把等权损失和目标尺度归一化扩展到全部 seeds/splits；比较时同时报告点误差、V1/V2 平衡、coverage 和 crossing，而不是只看一个 R²。
4. 论文式“柱规格输入适配 + 新输出层 + 迁移隐藏层”必须作为新架构单独实现。当前随机重置同构输出头的失败不能否定论文方案，但说明新头需要合理初始化、单调分位数参数化或专门训练策略。
5. 在正式主动学习前，仍需完成最低能构象、阈值开/关、源行224原始记录复核、分位数单调性和 validation-only 区间校准。

## 文件索引

- `../../scripts/run_e0_8g_controls.py`：E0-3b/E0-3c共用的配对控制实验入口。
- `comparison.csv`：28 组运行的逐配置 validation/test 指标。
- `paired_summary.csv`：核心四策略按 split 的 3-seed 均值和标准差。
- `paired_effects_vs_scratch.csv`：每个 seed/split 内相对 scratch 的配对差值。
- `sensitivity_excluding_source_row_224.csv`：只用于诊断的排除敏感性结果。
- `test_predictions.csv.gz`：全部 test 逐行真值、分位数预测和原始行号。
- `training_histories.csv.gz`：全部 epoch 的训练/validation 轨迹。
- `splits/`：6 个冻结 split；`scalers/`：相应 8g-train scaler。
- `config.json`、`environment.json`、`artifact_manifest.json`：协议、运行环境和文件校验信息。
