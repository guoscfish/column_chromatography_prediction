# 2026 年 10 月组会主动学习图表（reporting-only）

核查对象：[guoscfish/column_chromatography_prediction](https://github.com/guoscfish/column_chromatography_prediction)，GitHub main commit **42ab12600a353bba4a063ef5d1b623ee2f7b976a**（2026-10-11 23:29:25，Asia/Shanghai）。2026-10-12 执行 `git fetch origin main`，确认当时 HEAD 与 origin/main 一致；冻结快照 captured_at 为 2026-10-11T23:28:59+08:00。

工作副本存在用户其他未提交修改，尤其最新 REPORT.md；本次直接用 **固定 commit 的 Git 对象**读取来源，未覆盖那些修改。`input_sources.csv` 给出源路径、SHA-256、字节数、固定提交、工作副本是否一致；所有读取的 main 输入保留于 `frozen_inputs/`，代码也有冻结副本。

## PPT 素材

所有图使用白底、北大红 `#8C0000`，辅以蓝灰/暖灰；16:9（16×9 inch），**PNG=4800×2700 / 300 dpi**；SVG 为向量、文字转路径以避免跨平台中文字形丢失。

| 页面 | PNG / 同名 SVG | 内容 |
|---|---|---|
| 10 | `fig10_overview.png` / `.svg` | 六种完整方法的 validation 333–1005 曲线，粗线均值、浅色细线各 seed；右侧独立显示 AULC / 端点排序 |
| 10 表格 | `fig10_summary_table.png` / `.svg` | Method、两 seed 单独 AULC/端点与算术均值 |
| 11 曲线 | `fig11_factorial_curves.png` / `.svg` | LCMD / IVR / MaxDet 三面板 Raw/CW 曲线，标出 525 阶段边界 |
| 11 柱图 | `fig11_paired_aulc.png` / `.svg` | 333–1005 validation AULC Raw/CW 配对；柱为均值，圆/方点为各 seed |
| 12 | `fig12_same_state_branching.png` / `.svg` | 两历史状态到三分支、两轮 +32 的机制图，展示 seed157/429 的 winner reversal |
| 方法说明 | `methods_two_layer.png` / `.svg` | 样本表示 → 选点方法二层示意，不暗示所有组合均已验证 |

图 11 正文核心标题为“输出梯度表示的收益依赖具体选点准则”。图 12 明确标注 **历史冻结后 test**；前两页仅为 **validation**，两者没有混合排名。所有图片已按最终 PNG 做视觉检查。

## 数值摘要（仅同预算、同 split）

以下为 seed157/6101 算术均值，完整精度见 CSV：

| Method | validation AULC333–1005 | validation NRMSE@1005 |
|---|---:|---:|
| Random32 | 0.726538 | 0.626034 |
| LCMD32 | 0.518672 | 0.360565 |
| Hybrid traditional | 0.501517 | 0.373684 |
| Gradient MaxDet | 0.554292 | 0.420059 |
| CW32 (LCMD) | 0.500029 | 0.368524 |
| CW-IVR | 0.519921 | 0.385253 |

Random 明显落后。全程 AULC 以 CW-LCMD 最低，1005 端点以 Raw-LCMD 最低；CW-LCMD 与 Hybrid 的 AULC 均值很接近，不能宣称稳健显著优势。这里只描述匹配训练标签预算的性能，**Hybrid 需要额外 K=3 ensemble 训练，不能宣称等计算预算优势**。各 predictor 的最大训练配置一致，实际早停 epoch 不同。

| selector | Raw validation AULC333–1005 | CW validation AULC333–1005 |
|---|---:|---:|
| LCMD | 0.518672 | 0.500029 |
| IVR | 0.542433 | 0.519921 |
| MaxDet | 0.554292 | 0.574833 |

CW 对 LCMD/IVR 的全区间 AULC 在两个 seed 均有改善；MaxDet 的 seed157 变差、seed6101 改善，均值变差。不同 selector 还有不同 RMS 规则，不能把 selector 间差异完全归因于 objective。Raw 是输出梯度，不是 Latent；CW 不是一个新的模型头。审计映射见 `method_mapping_audit.csv`，真实公式见 `METHODS_APPENDIX.md`。

图 12 的 seed157/429 历史 test short-AULC：

| 起始状态 | LCMD | IVR | MaxDet | 同状态胜者 |
|---|---:|---:|---:|---|
| LCMD 历史 A | 0.676142 | 0.628679 | 0.669908 | IVR |
| IVR 历史 B | 0.594013 | 0.602321 | 0.602957 | LCMD |

只比较每行内部三条分支，不将 A/B 绝对误差排名解释为策略能力。8 起始状态 / 24 分支 / LCMD 与 IVR 各胜 4 次。结论保持为：“策略偏好可能依赖当前 state/history，而非固定标签数量；当前数据不足以训练可靠 controller”。

## 来源与完成状态核查

- 读取了指定 snapshot、最新 REPORT、旧 MASTER_REPORT、factorial/extension RESULTS、相应 validation curves/summary、branch FINAL_REPORT/branch_results/winner_matrix。
- **旧 master_full_333_1005.png 不是本次数据依据**：旧 MASTER_REPORT 的主比较是 test，数值和方法集合都与当前 validation 不同（含 Raw-IVR；本次六方法要求含 CW-IVR）。本次从 snapshot/validation CSV 重绘。
- 快照中的 Random32、LCMD32、Hybrid、Gradient MaxDet、CW-LCMD、CW-IVR 均为 seed157/6101 完整 22 点（333+32k，k=0…21）。其历史来源不是彼此独立的新重复实验；使用真实代码映射、冻结 fit receipts 与 reuse compatibility 确认 matched 标签预算、validation identity 和模型配置。没有将旧 master/test 指标拼入。
- Factorial 的原开发区间为333–525，36/36 原 fit 已完成。1005 extension 完成14条完整曲线（7 arms×2 seeds）、525之后60次额外 fit。Whitening 按用户要求停止于525，在CSV中仅7点，完整区间图中排除。
- Extension `COMPLETION_RECHECK.json` 报告60个完成fit、63次尝试含3次中断；全部数据hash与原delivery audit一致。它也披露后续共享工作区代码变更导致**完整实验审计重跑受阻**；本次没有绕过该限制或重跑实验审计，仅核对固定main表格、delivery hash与已有fit receipts。`RUN_STATE.json`的44 fit计数是最后执行阶段，不替代525后60 fit总数。
- Same-state branching 历史实验已完成8状态/24分支/48次fit，统一冻结后test揭示。branch表与winner表的SHA-256匹配 postfreeze manifest；本次不再读取真值。
- 原333–525与后续525–1005只能作为开发/探索连续轨迹，不能合并为独立确认；Whitening、截断Fusion、portfolio、LLM不进入本次六方法排名。当前两seed比较不能代表新seed确认或外部验证。

## 文件与审计

- `plot_reporting_only.py`：只使用 NumPy / pandas / Matplotlib 和标准库；**不 import 仓库训练/选点模块**。
- `fig10_validation_curves.csv`、`fig10_summary.csv`、`fig10_summary_wide.csv`：验证曲线与18行 per-seed/mean摘要。
- `fig11_validation_curves.csv`、`fig11_summary.csv`、`fig11_paired_effects.csv`：六arm验证曲线、18行摘要、9行配对效应。
- `fig12_branch_summary.csv`、`fig12_reversal_values.csv`、`fig12_winner_audit.csv`：历史test汇总与6条示例准确原值（以17位有效数字保存）。
- `experiment_status.csv`、`method_mapping_audit.csv`：完成状态与方法映射。
- `overview_fit_receipts.csv`：每个主图点对应的冻结fit摘要；`retained_fit_hash_audit.csv`：本机已存在fit JSON与snapshot SHA校验，缺失不会读checkpoint或启动重算。
- `input_sources.csv`、`numerical_audit.csv`、`AUDIT.json`：输入SHA、逐项数值一致性、边界记录。`OUTPUT_SHA256.csv`记录本次最终成果hash（不含它自身与只读输入副本）。
- `METHODS_APPENDIX.md`：Latent、Raw/CW、CountSketch、LCMD、conditional MaxDet、Kernel-IVR、Hybrid、Fusion定义与近似假设。

数值审计覆盖：完整预算网格与有限值、每点已完成fit来源、无test参与fit、同seed validation identity/416行、458952参数/同初始化规则、所有14条完整extension曲线AULC的333/525/653三个区间及端点、factorial前缀一致性、snapshot与CSV重叠方法逐点一致性、24条short-AULC复算、每个state胜者与矩阵一致性、4/4胜者计数、原delivery及postfreeze hash。

本次仅读固定Git对象和必要已存在fit JSON的hash，未读原始dataset标签、未开checkpoint、未做预测，未重新训练、未调用LLM、未执行主动选点、未新增test访问；全部新增文件位于本目录。原实验目录和工作副本其他修改均保留。

## 复现绘图

在仓库根目录，使用已有 `fish` 环境：

```bash
conda run --no-capture-output -n fish python reports/ppt_202610_al/plot_reporting_only.py
```

默认读取上述固定commit，不自动fetch/切换分支、不自动采用日后更新的main。可显式指定 `--source-rev`，但换commit后需重新核查方法定义、来源和阶段状态；此选项不是授权新实验。绘图会只更新本目录的素材、CSV与审计。PNG字体使用本机 Arial Unicode，SVG文字已转路径；其他平台需要可用中文字体。
