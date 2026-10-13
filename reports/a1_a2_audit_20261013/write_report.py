"""Render the Chinese audit from this directory's reporting-only evidence."""
from pathlib import Path
import json
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def link(path, title=None, line=None):
    target = str(ROOT / path) + (f':{line}' if line else '')
    return f'[{title or Path(path).name}]({target})'

def evidence(name, title=None):
    return link('reports/a1_a2_audit_20261013/' + name, title)

def table(frame):
    def cell(v):
        return f'{v:.4f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])

counts = pd.read_csv(OUT / 'outer_counts.csv')
count_table = []
for (protocol, seed), frame in counts.groupby(['protocol', 'seed']):
    row = {'协议': protocol, 'seed': seed, '4g 训练/验证/测试': '3330 / 416 / 417'}
    for col in ('25g', '40g'):
        a = frame.loc[frame.column.eq(col)].iloc[0]
        row[col + ' 训练/验证/测试'] = f'{a.gradient_train} / {a.validation} / {a.test}'
    count_table.append(row)
epochs = pd.read_csv(OUT / 'epoch_bridge.csv')
epoch_table = []
for (protocol, seed), frame in epochs.groupby(['protocol', 'seed']):
    a, b = frame.set_index('arm').loc['A1'], frame.set_index('arm').loc['A2']
    epoch_table.append({'协议': protocol, 'seed': seed, 'A1 五折 best_epoch': a.fold_best_epochs,
                        'A1 中位数=最终步数': a.median_epoch, 'A2 五折 best_epoch': b.fold_best_epochs,
                        'A2 中位数=最终步数': b.median_epoch})
pairs = pd.read_csv(OUT / 'paired_errors.csv')
row40 = pairs.loc[pairs.protocol.eq('row') & pairs.column.eq('40g')].sort_values('seed')
pair_table = []
for a in row40.itertuples():
    pair_table.append({'seed': a.seed, '内层 NRMSE 改善%': a.inner_mean_nrmse_gain_pct,
                       '外层 NRMSE A1→A2': f'{a.combined_normalized_rmse_A1:.4f} → {a.combined_normalized_rmse_A2:.4f}',
                       '外层 NRMSE 退化%': a.combined_normalized_rmse_deterioration_pct,
                       'V1 RMSE A1→A2 (mL)': f'{a.V1_rmse_A1:.4f} → {a.V1_rmse_A2:.4f}', 'ΔV1 (mL)': a.V1_rmse_delta,
                       'V2 RMSE A1→A2 (mL)': f'{a.V2_rmse_A1:.4f} → {a.V2_rmse_A2:.4f}', 'ΔV2 (mL)': a.V2_rmse_delta})
tails = pd.read_csv(OUT / 'tail_decomposition.csv')
rowtail = tails.loc[tails.protocol.eq('row') & tails.column.eq('40g')].sort_values(['seed', 'endpoint'])
tail_table = []
for a in rowtail.itertuples():
    tail_table.append({'seed': a.seed, '终点': a.endpoint, '阈值 mL': a.threshold, '尾部 n/测试 n': f'{a.tail_count}/{a.n}',
                       '尾部 RMSE A1→A2': f'{a.A1_tail_rmse:.3f} → {a.A2_tail_rmse:.3f}', '尾部变化%': a.tail_change_pct,
                       '非尾部 RMSE A1→A2': f'{a.A1_non_tail_rmse:.3f} → {a.A2_non_tail_rmse:.3f}'})
overlap = pd.read_csv(OUT / 'outer_identity_overlap.csv')
overlap_table = []
for seed in sorted(row40.seed):
    sub = overlap.loc[overlap.seed.eq(seed) & overlap.role.eq('test')]
    row = {'seed': seed}
    for protocol in ('compound', 'row'):
        for col in ('25g', '40g'):
            a = sub.loc[sub.protocol.eq(protocol) & sub.column.eq(col) & sub.donor.eq('other_target')].iloc[0]
            row[protocol + ' ' + col + ' 跨柱分子/行'] = f'{a.overlapping_molecules} / {a.overlapping_rows}'
    own = sub.loc[sub.protocol.eq('row') & sub.column.eq('40g') & sub.donor.eq('same_column')].iloc[0]
    row['Row 40g 同柱见过行/测试行'] = f'{own.overlapping_rows} / {own.heldout_rows}'
    overlap_table.append(row)
loss = pd.read_csv(OUT / 'final_train_loss.csv')
loss = loss.loc[loss.protocol.eq('row')].sort_values('seed')
loss_table = loss[['seed', 'A1_40g_last_train_loss', 'A2_40g_last_train_loss']].rename(columns={
    'A1_40g_last_train_loss': 'A1 最后一批 40g raw loss', 'A2_40g_last_train_loss': 'A2 最后一批 40g raw loss'})
dist = pd.read_csv(OUT / 'aggregate_distribution_diagnostics.csv')
dist = dist.loc[dist.protocol.eq('row') & dist.column.eq('40g') & dist.arm.eq('A1')].sort_values('seed')
dist_table = dist[['seed','outer_scale_V1','outer_scale_V2','outer_test_sd_V1','outer_test_sd_V2','inner_val_sd_V1_mean','inner_val_sd_V2_mean']].rename(columns={
    'outer_scale_V1':'外层 train σV1','outer_scale_V2':'外层 train σV2','outer_test_sd_V1':'test σV1','outer_test_sd_V2':'test σV2',
    'inner_val_sd_V1_mean':'内层 val σV1 均值','inner_val_sd_V2_mean':'内层 val σV2 均值'})

report = f'''# A1/A2 Column-conditioned FiLM 批判性审计

审计日期：2026-10-13（Asia/Shanghai）。对象：`guoscfish/column_chromatography_prediction` 的本地工作副本，HEAD `42ab126`。范围为 A1/A2 与后续 PCGrad 的冻结证据；未拉取、提交、重训或执行评分 action。仓库原有未提交改动未修改。

**核心结论：没有发现能解释 Row 40g 退化的 checkpoint 加载、epoch 计数、任务路由或 cache 错误。最强证据指向内层分子分组选择与外层 Row 插值目标不一致，并与 A2 显著更早的固定重训停止时点共同作用。退化的共同模式是非尾部误差和 Center 误差增大，不能概括为五个 seed 都是高体积尾部失败。该解释仍是待受控验证的假设。**

边界与披露：没有调用任何训练函数、`score`、`prepare` 或 `_read_authorized_truth`；没有读取 `filtered_canonical_*`、4g 原始标签表，也没有新增标签评分。只复核保存的指标、身份、历史、权重和盲预测。**本次测试初版整体解析了 `filtered_features_*`；这些文件意外含有 `t1/t2`、`verified t1/t2`，因此初版读取范围并未完全遵守无标签意图。冗余列没有输出或参与任何计算，已改为显式 `usecols=FEATURES` 后重跑。此疏漏如实记录，不将本次过程描述成从未解析过原始时间列。**

严格白名单版本的 {evidence('test_unlabeled_contracts.py', '无标签测试')} 为 **8 passed**；20 个最终 checkpoint、40 个柱/模型上下文重放盲预测的最大绝对差为 **4.5776×10⁻⁵ mL**。原始输出见 {evidence('unlabeled_test_results.txt')}。145 个 completed fit 的文件链有效，逐 fit 的 156 个冻结前缀张量一致；A1/A2 与 PCGrad study 共 1315 个文件在证据采集前后哈希一致。证据采集程序本身不解析原始标签或时间列，见 {evidence('collect_evidence.py')} 与 {evidence('evidence_summary.json')}。

## 1. 已确认没有问题的实现

**FULL-data 真实范围。** 25g 为 408 行、78 个 canonical 分子；40g 为 456 行、80 个分子。使用预先固定的 `V1≤60, V2≤120 mL`（25g）及 `V1≤150, V2≤200 mL`（40g），同时满足两个条件才保留。过滤前分别为 490/529 行，删除 82/73 行；数目来源于冻结 filter audit，未重新核算原始标签。4g 合格源集为 4163 行、217 分子，沿用 `V1≤60, V2≤120 mL`。过滤函数见 {link('src/qgeognn_al/transfer/full_data.py', 'full_data.py', 68)}；4g 定义见 {link('experiments/e0_4g_baseline/README.md', '4g README', 8)}。

目标 FULL-data 是过滤后所有不属于冻结 validation/test 的行：历史 `pool` 也成为 `gradient_train`。`planned_budget=100` 是继承身份的元数据，不能解释为只训练 100 行。目标的外层 validation 在 A1/A2 中只生成盲预测，不用于选择。真实记录数逐 seed 如下：

{table(pd.DataFrame(count_table))}

4g 每个上下文都固定使用合格源的 **row-seed-42 train 3330 行**；416 行源 validation 每轮仅记录诊断、不能进入 joint selector；417 行源 test 不进入本实验的新拟合或选择。这里 4g 的“验证/测试”是源 qualification 角色，不能与目标 outer 角色混为一谈。历史 4163 行 source replay 是预测一致性验证，不能解读为 4163 行再监督。代码见 {link('scripts/studies/run_column_conditioned_multitask.py', 'task_data / fold_schedules', 141)}。

记录数与父 benchmark 的 `label_accounting.csv` 全部一致；同一柱、协议、seed 内 sample_id 唯一，train/validation/test 角色没有重复。A1/A2 的逐折训练/验证身份 hash、计数、归一化尺度、batch size、步数完全相同。两个目标柱的 inner train/validation canonical 分子交集为零；4g 重叠不在此断言范围。完整记录见 {evidence('outer_counts.csv')}、{evidence('epoch_bridge.csv')}、{link('studies/transfer/column_conditioned_multitask/ROW_INNER_RESULTS.csv')}。

**架构、状态和测试覆盖。** A1/A2 的来源权重及记录在 execution provenance 中的全部代码 hash 与当前代码一致，见 {evidence('provenance.csv')}。A1 总参数 460500、可训练 130615；A2 总参数 469252、可训练 139367；增加的 8752 个参数恰好为 3×16 embedding 和两个 Linear(16,256)。两者保留 128 维 backbone、sum pooling、condition completion 和六分位数输出。

| 检查 | 结论与代码依据 | 原有覆盖 / 本次补充 |
| --- | --- | --- |
| 初始化 | 三个 head 是独立的 source head 精确复制；FiLM 两层 weight/bias 为零，调制为 `h×(1+γ)+β`。{link('src/qgeognn_al/transfer/column_conditioned_multitask.py', '构造与 FiLM', 112)} | 原有 epoch-zero/source 对照；本次用目标 feature 白名单图复核所有 task id、head 存储独立性 |
| 身份路由 | `4g/25g/40g→0/1/2`，node 使用所属 graph 的 task，head 按 graph task 选取。{link('src/qgeognn_al/transfer/column_conditioned_multitask.py', 'FiLM 与 head 路由', 171)} | 原有 head bias 路由；本次加入非零 FiLM 的逐图身份检查、非法 task id 和六输出次序 |
| 调制位置 | node GIN 输出后、block 3 的 ReLU 前；block 4 本身没有 node ReLU，与源模型一致。不是在 condition branch 或输出头上做调制。{link('src/qgeognn_al/transfer/column_conditioned_multitask.py', 'forward', 177)} | 原有零调制等价性；本次非零调制 cache/full 梯度与更新对照 |
| 输出映射 | `V1_q10,q50,q90,V2_q10,q50,q90`；点指标取索引 1、4。{link('src/qgeognn_al/evaluation/point.py', 'point_metrics', 6)} | 本次用不同 head/channel 常数验证，未使用真实终点 |
| 可训练范围与 LR | node 3/4、最后 edge/angle update、condition branch、所有 heads；4g head 与共享参数 1e-4，目标 heads/FiLM/embedding 3e-4，Adam wd=1e-5。{link('src/qgeognn_al/transfer/column_conditioned_multitask.py', 'scope / optimizer', 129)} | 静态核对与 checkpoint 张量核对；未发现额外解冻 |
| BatchNorm | 每个 epoch 先 train；无可训练 affine 的早期 BN 切回 eval，后期 BN 在单次混合 forward 中更新。{link('src/qgeognn_al/transfer/baseline.py', 'set_training_mode', 22)} | 原有 cache 状态对照及 Compound 审计；本次补核 Row、final checkpoint 的冻结 buffers，late BN 计数与实际步数一致 |
| frozen-prefix cache | 源 eval 的 blocks 0–2 输出，进入 block 3；生成过程不使用 `y`。{link('src/qgeognn_al/transfer/column_conditioned_multitask.py', 'cache_frozen_prefix', 260)} | 原有单任务、零 FiLM 的梯度/Adam update 测试；本次多任务、非零 FiLM，prediction/gradient/update/BN 等价 |
| 最终模型保存/预测 | 保存完整 state_dict，包括 BN；保存模型重放与冻结盲预测一致 | 本次重放两个协议×五 seed×两 arm，未评分 |

原有测试分别见 {link('tests/test_column_conditioned_multitask.py')}、{link('tests/test_column_conditioned_pcgrad.py')}。本次没有直接运行其中会装载真实标签的 fixtures，而是独立构造零占位标签的图和算术合成损失。单元测试中的有限次 Adam 更新只验证等价性，不保存实验模型、不构成重新训练。首次测试中的 module deepcopy 非叶张量失败属于新增测试夹具问题，已改为重新构造模型并加载同一 state_dict；没有改动实验实现。

**内层选择到最终重训。** 一个 fold 只有一个 joint best_epoch，25g/40g 记录一致。每轮 selector 是两个目标柱的 NRMSE 等权均值，4g validation 不参与；严格 `<` 保留最早最优 epoch；内层恢复 best 的完整 state_dict。最终重新从同一源初始化，seed 为 outer seed，不加载内层 checkpoint/Adam 状态；按五折中位数固定训练，不运行 validation selector。见 {link('scripts/studies/run_column_conditioned_multitask.py', 'fit_model', 277)} 和 {link('scripts/studies/run_column_conditioned_multitask.py', 'final_context', 479)}。

{table(pd.DataFrame(epoch_table))}

本数据所有 inner/final context 均为 **1 step/epoch**：内层 per-task size 为最大目标训练人口（各折 274–312 左右；完整明细见 CSV），最终为 348–367，均小于 cap 682。因此最终中位数 epoch 就是实际 optimizer.step 次数，没有固定步数换算遗漏。40g 通常每轮遍历一次，25g 较小时 permutation cycle 补齐；4g 每轮只抽同样数量，不能将“4g 3330 可用标签”理解为每轮都遍历。各 epoch 独立重排，A1/A2 在同一 epoch 的抽样配对。

100 个 A1/A2 inner fit 的 best_epoch 与 history 的 joint argmin 一致，记录的最优指标与相应 history 行一致；20 个 final fit 的 `fit.json`、checkpoint best_epoch、history 长度与中位数一致。late BN 的 `num_batches_tracked` 相对源增量等于训练步数。final 没有恢复中途 optimizer 的路径；未完成 run 从确定 seed 重启、完成 run 用 hash 复用，不是训练状态续接。见 {evidence('checkpoint_checks.csv')}、{evidence('epoch_bridge.csv')}。这证明执行符合协议，**不证明中位数规则对 FULL 重训统计上合适**。

**PCGrad 实现。** 对同一次混合 forward 的三个 task loss，使用 `autograd.grad(..., retain_graph=True)` 读取任务梯度；不用 `.grad` 作为任务梯度来源。逐 task 对其他 task 的**原始**梯度按确定性随机顺序投影，只在 dot<0 时执行 `g_i ← g_i − dot(g_i,g_j)/(||g_j||²+1e-12)·g_j`；最后对投影后的 task 梯度求均值。先对 mean loss 正常 backward，再只替换 `shared_gradient_parameters()` 的 **44 个参数张量**，不等于 44 个标量。heads、embedding、FiLM、condition branch 等其余可训练参数保留 ordinary mean-loss 梯度及相同 Adam/weight decay。见 {link('src/qgeognn_al/transfer/pcgrad.py', '任务梯度', 32)}、{link('src/qgeognn_al/transfer/pcgrad.py', '顺序投影', 74)}、{link('src/qgeognn_al/transfer/pcgrad.py', '局部替换', 122)}。本次用三任务手算顺序参考和合成损失验证，包括非共享参数的实际 Adam update。

**99.28% 的定义复算正确。** 在 epoch 1 及每 10 epoch 的 483 个 sampled step 上，展平三个 pair，共 1449 个 cosine；`burden = mean(max(−cos,0))`。raw burden=0.1776443491，projected=0.0012731668，`1−projected/raw=0.9928330577`。这不是 99.28% 的负冲突次数减少，也不是误差减少。负 cosine 比例实际为 66.05%→4.14%，投影触发率 66.32%，平均 mean-gradient 范数比为 1.1718。定义见 {link('scripts/studies/run_column_conditioned_pcgrad.py', 'mechanism_summary', 353)}；raw/projected 的 seed/fold/epoch/step/permutation_seed 坐标全部配对。

PCGrad Compound mean paired-fold NRMSE 改善仅 **25g −0.37%、40g +0.50%**；25g V2 tail 退化 2.99%，未通过预测 gate。没有运行 Row 或新 outer 评分。因此它说明已测共享梯度机制受到干预，不能说明修复了 Row 40g，更不能把冲突视为退化主因。混合 BN 使某个 task loss 的梯度包含同批其他任务输入对 BN 的影响；这些是实际混合目标的梯度，并非单柱独立 forward 的梯度。投影后的梯度几何也不是 Adam 二阶预条件后的实际参数位移几何。

## 2. 明确发现的错误

**E1：命名为 features 的文件仍携带可关联/反推终点的时间信息。**

文件/行号：{link('scripts/studies/run_filtered_full_data_benchmark.py', 'run_filtered_full_data_benchmark.py', 268)}，尤其第 269 行仅 `drop(columns=["V1_ml", "V2_ml"])`，没有以输入白名单导出。两份文件 header 均保留 `t1,t2,verified t1,verified t2`。这是数据发布边界错误；它使整体读取“features”的后续工具可以接触 held-out outcome 信息。

可复现证据：只读取两份文件第一行即可看到上述四列；本次 {evidence('evidence_summary.json')} 的 `redundant_outcome_column_headers` 记录了 header 证据，不需要读取任何单元格。正式 `task_data` 的第 157 行 `usecols=FEATURES` 排除了它们，因而**没有发现这些列进入 A1/A2 拟合、cache、选择或预测的证据**。本次初版审计夹具的误读也在开头披露，不能据此倒推历史模型发生标签泄漏。

修复建议与预期影响：以后另建明确版本的、只含获准身份和输入列的 feature export，并加“禁含 t1/t2/verified times/端点”的 schema 测试。不能覆盖现有冻结 CSV 或直接重写其 hash；否则会破坏 protocol 的人口/特征证据链。预期改善读取隔离，不应自动改变现有预测，更不构成重训理由。**本阶段只修改新增审计夹具，未修复正式 exporter。**

**E2：README 要求的两个复现入口缺失。**

文件/行号：{link('studies/transfer/column_conditioned_multitask/README.md', 'multitask README', 36)} 引用 `scripts/studies/audit_column_conditioned_multitask_evidence.py`；{link('studies/transfer/column_conditioned_pcgrad/README.md', 'PCGrad README', 25)} 引用 `scripts/studies/summarize_column_conditioned_pcgrad.py`，第 28 行还要求失败后运行该 summarizer。两文件在当前工作副本和 `git ls-files` 中均不存在。

可复现（只检查存在性）：

```python
from pathlib import Path
for name in ["audit_column_conditioned_multitask_evidence.py",
             "summarize_column_conditioned_pcgrad.py"]:
    print(name, (Path("scripts/studies") / name).is_file())
# 两项均为 False
```

这阻断 README 所写的完整独立审计/报告复现链，但不是训练数值错误；原始冻结报告与 completed 文件仍存在，当前证据采集也复核了相应状态。建议先从历史版本恢复原脚本及其出处，或修订为可用的 reporting-only 入口，单独确认后实施；不运行旧脚本的标签读取或覆盖冻结报告。

**未确认的训练错误不得包装成发现。** 没有证据支持错 head、FiLM 柱 id 混用、BN 未恢复、cache 过期、重复使用 inner checkpoint、epoch off-by-one 或实际步骤少跑。A2 较早停止是协议效果，暂不应作为实现 bug 修复。最终报告第一页只写 `REPRESENTATION_GATE_PASSED` 容易让人把 continuation gate 当成最终成功，但其后明确给出 `REPRESENTATION_SIGNAL=False`、`PROJECT_TRANSFER_GAIN=False`；这是应改善的结果呈现问题，不能称为冻结数值错误。

## 3. 方法设计或评价协议的局限

**内层与外层的独立性边界不同。** 两个协议的内层均在目标 outer gradient_train 中，跨 25g/40g 全局 canonical-SMILES GroupKFold：目标内层验证分子在两个目标训练集都不可见。Row 外层按行划分，同分子其他条件/重复测量可以在同柱训练中。4g source_train 分子则不 purge，40g 的所有 outer test 分子在源 train 都见过；Row 内层 40g 验证分子也全部在 4g source_train 中。因此内层是“目标柱未见该分子、但可能有源柱监督”，Row 外层主要是“已见分子的目标条件/记录预测”；两者都不是 source-unseen 化学 OOD。

外层 Compound 划分继承每柱自己的 roles，没有跨柱全局 purge：

{table(pd.DataFrame(overlap_table))}

此处“跨柱分子/行”指 outer test 在另一目标柱 gradient_train 见过的 canonical 分子数、受影响的 held-out 行数。Compound 各柱同柱 train/test 分子交集为零；但 25g 的 7–11 个 test 分子、40g 的 7–10 个 test 分子获得 donor 柱训练监督。Row 40g 有 99/100 或全部 test 行的分子在同柱训练中。以上重叠符合继承协议并已披露，不是意外 sample_id 串入角色；它限制所有“独立泛化”论断。包括 outer validation 的完整重叠矩阵见 {evidence('outer_identity_overlap.csv')}。

**Row 存在相同可见输入跨角色。** 25g 的 408 行只有 405 种模型可见输入；40g 的 456 行只有 449 种。按 canonical_SMILES、PE/EA、loading solvent、density、loading volume、loading-solvent volume 比较，Row 40g 五 seed 分别有 **3、3、2、5、2 个 test 行**（按 seed 536279090、769539383、1362771960、1425370602、2767143051）与训练行输入完全相同；Compound 为零。这些不是同 sample_id，未读取标签来裁定它们是真重复实验还是记录复制，故应归为评价依赖性限制，不能武断称为数据错误。见 {evidence('same_input_overlap.csv')}。

**数值尺度和汇总口径不能混淆。**

- 各 fold 的 NRMSE 为 `sqrt(((RMSE_V1/σ_train,V1)²+(RMSE_V2/σ_train,V2)²)/2)`，σ 是各任务 inner_train 的总体标准差（ddof=0）；outer 用整个 outer gradient_train 的 σ。A1/A2 同 context 尺度相同，差异不是把 A2 的分母偷偷换大。绝对 mL RMSE 同样退化，不能用归一化差异解释掉结果。
- gate 的 “mean fold gain” 是 25 个 `1−NRMSE_A2/NRMSE_A1` 的均值；不等于“均值之比”，也不等于所有 OOF 行 pooled RMSE。外层报告对五个 seed 的 RMSE/NRMSE 求均值。不同分母/行权重意味着内外层 headline 大小不可直接比较。
- 同一 validation 数据选 best checkpoint 后又报告其指标，存在模型选择乐观偏差；500 次查看不是 500 个独立试验。25 folds 跨五个 outer seed 高度共享数据，5 seeds 也不是五次独立外部实验，不能用朴素独立二项概率给 5/5 退化做显著性声明。
- 内层一个 joint selector 等权平均两柱 NRMSE，但训练优化的是三个任务的 raw quantile loss 均值。raw q50 MSE、pinball 和 crossing penalty 混合，较大终点的梯度作用与 NRMSE 选择目标不一致；此设计两 arm 相同，属于机制局限。

**样本/条件分布。** Row 40g 的 test 仅 53–60 个分子、87–100 行；内层验证会整体留出目标分子的所有记录。安全特征白名单统计显示：test/train 的 EA 比例差异方向并不统一，例如 seed 769539383 为 0.302/0.391，而 536279090 为 0.419/0.357；40g flow 均为 30。现有条件统计不支持所有 seeds 共享单一溶剂漂移原因。全部柱/角色统计见 {evidence('feature_distributions.csv')}。

下表只由冻结 aggregate 指标推导：`σ_test = RMSE/sqrt(1−R²)`；outer train σ 则从多个已存方法的 `2×NRMSE² = RMSE_V1²/σ1² + RMSE_V2²/σ2²` 联立复核。没有读取新的 test 单元格，且只能恢复方差，不能恢复 test 标签均值或分位数。

{table(dist_table)}

test 方差相对 inner val 有升有降，没有统一尺度异常。完整 inner train σ 范围、绝对 Center/Width 平均误差大小等见 {evidence('aggregate_distribution_diagnostics.csv')}。缺少批次/重复实验独立性信息、不可在本阶段重读 test 分布，是结论的实质边界。

**过滤与历史曝光。** 按终点阈值过滤定义了受限运行域；不能外推至被剔除的高体积记录，也不能把阈值当作物理定律。本次没有证明阈值的仪器或测量学依据。outer 身份曾用于历史研究，现有结果只能是 DEVELOPMENTAL CONFIRMATION；新架构候选越多，历史外层越不能充当未曝光确认集。固定源 checkpoint 也意味着五 seeds 只覆盖 target split/初始化等变化，没有覆盖独立源训练的不确定性。

**provenance 和恢复的剩余边界。** 当前记录的代码 hash 都吻合，145 个文件链通过；但 provenance 没有枚举所有间接依赖（例如定义 BN 模式的 `transfer/baseline.py`），也不包含所有库的完整环境锁。没有保存每个 optimizer.step 的 Adam 动量，不能对历史优化轨迹做 bitwise 重放；对最终权重与冻结盲预测的一致性已有验证。不能把未记录的轨迹当成已测等价，也没有迹象提示它是当前退化原因。

## 4. 最可能解释 Row 40g 退化的假设及证据

**先固定事实。** Compound 内层 mean paired-fold gain 为 25g +5.05%、40g +8.18%；Row 内层为 25g +5.10%、40g +6.51%。Row 25g 是 **4/5 seed wins、19/25 fold wins**，40g 是 **5/5、21/25**，不能把所有内层柱都称为 5/5。下面 “内层改善%” 按该 seed 的五折 NRMSE 均值之比，口径与上述 mean paired-fold gate 稍有区别。外层正变化表示退化：

{table(pd.DataFrame(pair_table))}

五 seed 外层均值从 **NRMSE 0.4214→0.4609（+9.37%）**，V1 **11.9158→13.0467 mL（+9.49%）**，V2 **14.7453→16.1155 mL（+9.29%）**。逐 seed 退化百分比的算术平均为 9.83%，不要与均值之比 9.37% 混用。V1 5/5 变差，V2 4/5 变差（1362771960 微改善）；两端点 MAE 均 5/5 变差。原表和完整 Center/Width 明细见 {evidence('paired_errors.csv')}。

**高体积尾部并非共同失败区。** 阈值直接沿用冻结 outer gradient_train 的终点 80th percentile；以下均为已存 tail SSE/count 推导，未重新按标签分组：

{table(pd.DataFrame(tail_table))}

非尾部 RMSE 的复算为 `sqrt((total_sse−tail_sse)/(test_n−tail_n))`。结论：**两个终点的非尾部 RMSE 全部 5/5 上升**。V1 tail 为 3/5 变差，V2 tail 仅 2/5 变差；769539383 两个 tail 都改善，V2 改善 22.83%，整体却退化 10.36%。五 seed tail RMSE 均值为 V1 **19.952→21.401（+7.26%）**、V2 **24.411→24.294（−0.48%）**。以五个重复评估的 SSE 作描述性相加，V1 新增 SSE 约 64.3% 来自非尾部；V2 非尾部增加 20029.88，tail 净变化 −29.75。这个加总不是独立样本统计。详见 {evidence('tail_decomposition.csv')}。

**共同变化更像区间位置/逐样本共同扰动。** Center RMSE 5/5 增大，均值 **12.4789→13.8383（+10.89%）**；Width RMSE 4/5 改善，均值 **9.8644→9.7387（−1.27%）**。只看 width 会掩盖两终点的退化。仅比较冻结预测，`A2−A1` 的 V1/V2 变化相关系数为 **0.863–0.954**；两端点变化多为同向逐样本扰动。平均预测位移方向及绝对平均 Center error 并不一致，因此不能认定为一个统一“上偏/下偏”的 calibration 错误，更不能赋予其物理意义。见 {evidence('prediction_only_shifts.csv')}。

| 假设（按当前解释优先级） | 支持证据 | 反证/尚不能确认 |
| --- | --- | --- |
| H1：分子分组内层选择不适合外层 Row 的已见分子条件插值 | 内层禁止目标分子重叠，外层 Row 40g 的同柱分子覆盖为 99–100%；内层稳胜、外层稳败；Compound 外层仅小幅 A2 优势 | 两种外层训练人口、验证集都不同；没有受控 crossed validation 实验，不能证明单一因果 |
| H2：H1 与 A2 较早停止/中位数 FULL 重训衔接共同造成插值不足 | Row A2 固定 69–112 步，均值 89.2；A1 170–422，均值 280.8；A2 最后一批 40g raw loss 五 seed 都更高；非尾部和 Center 退化最一致 | 步数完全符合协议；原始 loss 包含六分位数和 crossing，不等于 q50 train RMSE；长训可能过拟合，不能直接把 A2 epoch 改大 |
| H3：FiLM 与 mixed-task BN、FULL 人口变化产生域相关预测扰动 | 非零调制涉及 node 表示；终点预测变化高度相关；内层→FULL 的 mixed population 和 source 每步抽样规模变化；Center 主导 | cache 和 BN 状态操作均通过，未做独立 BN/FiLM 消融；没有证据证明是 BN 实现失误 |
| H4：小样本、尾部稀疏、分子/条件混杂使效果不稳定 | tail 每终点仅 12–24 行；同数据重复划分、每柱约 80 分子；Compound donor/source 重叠；阈值域与 batch 独立性限制 | “所有 seed 都是 tail 恶化”已被数据否定；不能将 tail scarcity 作为整个 Row 失败的充分解释 |
| H5：共享梯度冲突是主要原因 | 既有 A2 4g/40g cosine 常为负 | PCGrad 减少已测负 burden 99.28% 却几乎无预测增益，并未测试 Row；证据不足以把此假设置于 H1/H2 之前 |

H2 的直接历史证据如下，单位为原协议 raw loss 的混合量纲；每行都是固定 refit 最后 epoch 的更新前训练 batch loss，不是新评分：

{table(loss_table)}

这些 A2 loss 约为 A1 的 **2.2–3.3 倍**。40g 在 final 每个 epoch 被完整遍历，因此不是单次随机漏掉大部分 40g 所致；mixed BN 仍受其余任务共同影响。A1/A2 不同的最终 epoch 数也改变累计源 replay 曝光量。不能从该表推断“只要多训就能修好”，因为 Group 选择更早停止可能确实有利于目标分子泛化。

对照关系同样限制结论：Compound A2 相对 A1 外层仅 +1.47%（25g）、+2.65%（40g）的 NRMSE 均值改善，远小于内层 headline；A2 仍未超过 corrected HIER。Row 25g A2 也轻微退化 +1.10%，但 Row 40g 更明显。A1 在 Row 40g 优于 P0，A2 的 Row NRMSE 也仍低于 P0；故本次否定的是 **FiLM 对 A1 的稳定增益/推广**，不是否定所有多任务监督。最终 promotion=False 的冻结判决与证据一致。

## 5. 最少成本的后续受控验证方案

**本阶段已完成的零新标签验证。** 身份/计数、joint argmin 与中位数、checkpoint/BN/冻结前缀、非零 FiLM cache 等价、PCGrad 梯度范围及非共享 Adam update、全部最终盲预测重放均已完成；无需为这些检查重训。报告和脚本只新增在本目录，正式训练代码、协议、结果没有修复或覆盖。

**修复必须先确认。** E1 建议建立新版安全 feature schema，E2 建议恢复/更正文档中的 reporting 入口；预期影响仅为数据读取隔离和可复现性。先确认具体修复范围、版本与 hash 迁移办法，再实施。保留全部历史冻结证据，不以“修复”名义触发新 outer 评分或覆写已冻结结果。

**如另行授权计算验证，先验证 H1/H2，而不是搜索更好结果。** 最小试点可预先固定冻结 seed 顺序第一个 **769539383**，只使用其获准 outer gradient_train 标签，创建独立 audit study；目标 outer validation/test 继续不可读取。用相同样本人口构建两个开发验证视角：全局分子 Group 留出与已见分子的 Row 留出；每个视角对 A1/A2 配对初始化、batch draws、BN 和 loss，保持当前架构/优化器。结果报告完整 epoch 曲线及两种开发视角的 q50 mL RMSE/NRMSE；不要根据旧 Row test 的退化来选择新 epoch。

为隔离“较早停止”与“表示本身”的效应，在预注册中明确比较：①原 joint 选择规则；②一个预先固定、两 arm 相同的更新预算（可用已有设计的固定上限 500，不能事后挑选峰值）。关注 equal-step 下 A2 是否仍在 Group 胜、Row 败，及其非尾部/Center 误差。这个对照属于**新受控实验**，不能悄悄替换旧协议；只有内层开发评价，不能宣称修复了原 outer test。单 seed 可证伪明显机制，不能确认跨 seed 效应；若信号支持，再事先锁定全五 seed 复核。

只有 H1/H2 不能解释且预先批准时，才考虑单因素 FiLM 或 BN 干预；不要同时更改 loss、归一化、head、LR 和训练时长。无标签冻结模型消融可以量化预测敏感性，但无法证明预测改善。真正 promotion 仍需独立 compound/batch 数据或事先锁定的全局目标分子隔离评估，并按科学问题明确是否 purge 4g source；它不是当前 study 的合法再评分动作。

在确认前，**停止于报告、修复建议和后续验证设计，不实施正式修复或重训**。
'''

(OUT / 'AUDIT_REPORT.md').write_text(report)
print(OUT / 'AUDIT_REPORT.md')
