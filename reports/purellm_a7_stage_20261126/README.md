# PureLLM-A7/M2 与 Hybrid 五 seed 阶段对比

验证数据快照的核验时间为北京时间 2026 年 11 月 27 日 04:41:56。PureLLM 已完成 45 轮 CUDA 训练，记录了 1,440 条冻结选点及其理由。Hybrid 为正式 CUDA Uncertainty + Diversity Hybrid32，五条轨迹均已到 L1485。

两种方法在每个 seed 上共用同一份 L333 CUDA checkpoint、验证划分和 QGeoGNN 训练协议，每轮新增 32 个实验。图中数值来自各预算选定 checkpoint 的验证记录。左侧放大截至 L973 的阶段性曲线，右侧展示完整预算范围。实线止于各 seed 最后一次已完成的 PureLLM 训练。颜色和标记配对同一个 seed。 [L333 配对核验](paired_l333_anchor_receipts.json) · [Hybrid 正式归档](../eit_cuda_three_methods_20261124/README.md)

![PureLLM 与 Hybrid 的五 seed 验证曲线](purellm_vs_hybrid_stage.png)

## 同预算配对

| Seed | PureLLM 已完成轮次 | 最新预算 | PureLLM 验证 NRMSE | 同预算 Hybrid NRMSE | 差值 |
|---:|---:|---:|---:|---:|---:|
| 157 | 7 | L557 | 0.4944 | 0.4759 | +0.0185 |
| 887 | 20 | L973 | 0.2831 | 0.3291 | -0.0460 |
| 2357 | 5 | L493 | 0.4282 | 0.4412 | -0.0130 |
| 6101 | 8 | L589 | 0.5108 | 0.4954 | +0.0154 |
| 12203 | 5 | L493 | 0.4620 | 0.5130 | -0.0510 |

差值为 PureLLM 减 Hybrid，NRMSE 越低越好。五个 seed 当前均覆盖到 L493，该预算处的平均验证 NRMSE 为 PureLLM **0.4893**、Hybrid **0.4942**。

6101 到 L493 的这一轮来自已归档的完整模型回答。系统核验并移除重复引用别名，保留原有 32 个选点的顺序和理由，随后完成 CUDA 训练。这一步沿用归档回答的使用量记录，无新增 API 请求。

887 的 R13 原答因引用字段和关系错误被校验拒绝。随后完成了一次付费局部纠错，系统从该响应的显式最终答案恢复，保留 32 个选点的顺序并在通过校验后训练至 L781。该归档恢复新增 0 次 API 请求。选点记录分别保留原答与纠错的供应商回执与用量，纠错响应报告 353,339 个输入 token 和 285 个输出 token，其中包含 29 个推理 token。

887 的 R14 和 R15 首答均通过正式 A7 校验，各冻结了 32 个选点并完成 CUDA 训练。R14 至 L813 的验证 NRMSE 为 **0.3070**，R15 至 L845 为 **0.2927**。两轮的服务商终态均报告 `gpt-6.2-astra / xhigh`。

887 的 R16 经一次付费局部纠错修正了第 26 个选点的对照关系和理由，保留原来的 32 个候选及顺序。R17、R18 首答均通过正式 A7 校验。三轮均已完成训练，最近的 L941 验证 NRMSE 为 **0.3064**。

887 的 R19 原付费请求已收到服务商的完整响应。旧解析器因一条只有一个空格的未标注消息拒绝了该响应。系统从归档回执提取显式最终答案，核验原有 32 个选点、顺序及理由后完成训练至 L973。该轮原请求报告 358,211 个输入 token 和 14,039 个输出 token，其中推理 5,435 个。归档恢复新增 0 次 API 请求，L973 的验证 NRMSE 为 **0.2831**。

## 数据和选点

- [PureLLM 验证指标](purellm_validation_curves.csv) 包含 5 个初始检查点和 45 个后续训练点。`round=0` 对应共享 L333，`acquisition_round=0` 对应首次选点和 L365 训练。 CSV 保留每个指标文件和 checkpoint 的摘要。 [PureLLM 逐文件清单](purellm_export_manifest.json)
- [Hybrid 验证指标](hybrid_validation_curves.csv) 共 185 个检查点。 [来源和校验记录](hybrid_validation_receipts.json)
- 每个 seed 的 JSONL 每行对应一个已冻结并完成训练的轮次，保留模型给出的理由、引用和选点顺序。 [157](selections/seed_157_accepted_rounds.jsonl) · [887](selections/seed_887_accepted_rounds.jsonl) · [2357](selections/seed_2357_accepted_rounds.jsonl) · [6101](selections/seed_6101_accepted_rounds.jsonl) · [12203](selections/seed_12203_accepted_rounds.jsonl)

## 复现图

使用 Python 和 Matplotlib 3.10.8，在本目录运行以下命令。

```bash
python plot_stage.py --snapshot '27 Nov 2026, 04:41:56 UTC+8'
```
