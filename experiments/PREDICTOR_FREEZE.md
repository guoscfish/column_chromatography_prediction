# Gate 0 scientific Predictor freeze

冻结日期：2026-08-16。状态：G0-1～G0-4科学对照完成；D28工程接口检查尚未完成，因此这里冻结的是科学配置，不代表整个Gate 0工程验收已经通过。

## 冻结配置

| 项目 | 冻结值 |
|---|---|
| 数据源 | 当前仓库CSV为唯一事实源；不以论文行数覆盖本地记录 |
| 4g anchor | `experiments/e0_4g_baseline/checkpoints/best.pt`，SHA256 `7b9e3d0d4c8036c738ef220802e7ee46bc6ab8261cc541fb7d194e8c17044323` |
| 8g target | `experiments/g0_3_threshold_sensitivity/canonical_8g_no_threshold.csv`，574行，SHA256 `4cc533b00c4528053c7f033645c50a7d03b35bfff3fcfe4c3f684376ac136203` |
| Threshold | no-threshold；保留原60/120 mL规则会删除的22条tail记录 |
| 重复实验 | 保留；compound泛化时同一canonical SMILES不跨split |
| 构象 | `first_embedded`，保持原代码口径 |
| 主干 | QGeoGNN，5层GINConv、128维、sum pooling；不启用新增柱规格输入 |
| 输出 | 单调softplus分位数：`q50=m, q10=m-softplus(d_low), q90=m+softplus(d_high)` |
| 迁移范围 | 4g anchor → 8g，训练末两层GNN与prediction head |
| 优化 | Adam，`lr=1e-4`，weight decay `1e-5`，batch 2048，最多500轮，patience 100 |
| Loss | V1/V2等权；每个目标为q10 pinball + q50 MSE + q90 pinball |
| Scaler | 只使用4g source-train scaler；SHA256 `43f99c0244fd3e60f19dd3cee6029fd3831c5e90242e6f7bd669984a97774cbf` |
| Checkpoint | 只按validation的`MSE_V1/Var_train_V1 + MSE_V2/Var_train_V2`选择 |
| Calibration | 每个run仅用validation、按V1/V2分别估计split-conformal inflation；test只应用并报告 |
| Seeds/splits | 42、525、1101；row与compound两套paired协议 |
| Test角色 | 只作最终报告，不参与scaler、early stopping、checkpoint、calibration或结构选择 |

## 冻结依据

- G0-1：单调头使validation/test crossing从13.6%/17.7%降为0，点预测退化低于预注册5%上限。
- G0-2：validation-only校准提高平均test coverage，但存在极端inflation；因此校准协议冻结，Quantile Width仍需E1资格测试。
- G0-3：validation-only选择no-threshold；tail在高width样本中富集约8倍，但小tail test误差不稳定，必须单列失败切片。
- G0-4：last2、full、paper-style的平均validation normalized score为0.256/0.333/0.295。Paper-style只赢1/6，未达替换门槛；full也没有更好，因此冻结last2。Test上paper-style为0.445、last2为0.451，排序不一致不用于反向选择。

G0-4 validation-only决定文件SHA256为`3ee01b93abe1279b2874a74be9223a384e8b73213fdab84308e3b99060e4aeb6`。正式逐样本结果、校准与full/common/tail切片位于`experiments/g0_4_paper_style_transfer/`；随机新增适配器破坏source function的诊断轮位于`experiments/g0_4_paper_style_transfer_random_init_diagnostic/`，不得与最终修正版合并汇报。

## 不得反向调整

进入E1/E2/E4后，不得依据主动学习test RMSE、AULC或某一采样策略表现重新选择threshold、构象、迁移范围、loss权重、分位数参数化或校准方式。若出现新的数据质量证据或模型接口缺陷，应登记新decision ID，并以新实验ID进行前向修正，保留原结果。

## 进入E1前剩余条件

- 10k+候选分块推理通过内存与顺序检查；
- 预测输出与`sample_id/canonical_index`一一对应且跨batch稳定；
- AL round状态可中断恢复，已标注集合、pool、checkpoint与RNG状态不漂移；
- 抽出统一`fit/predict`接口并保存固定split/L0。
