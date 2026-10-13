# 最新主动学习结果

快照时间：2026-10-13T17:09:17+08:00（Asia/Shanghai）。
本页统一汇总已完成的 validation 拟合；运行中的选样或未完成训练不计作结果。

![LLM comparison](llm_validation.png)

## LLM 与匹配对照

| Seed | 方法 | 已完成标签数 | 端点 validation NRMSE |
| --- | --- | ---: | ---: |
| 157 | CW16 + LLM16 | 845 | 0.344879 |
| 157 | Free LLM32 | 845 | 0.437181 |
| 157 | Dual expert | 621 | 0.597021 |
| 157 | CW32 (LCMD) | 1005 | 0.335051 |
| 157 | Random32 | 1005 | 0.672336 |
| 6101 | CW16 + LLM16 | 813 | 0.380292 |
| 6101 | Free LLM32 | 845 | 0.391570 |
| 6101 | CW32 (LCMD) | 1005 | 0.401997 |
| 6101 | Random32 | 1005 | 0.579732 |

端点预算不同，不用上表直接排名。双专家状态为COMPLETE。目标为621标签；完成点以表中冻结拟合为准。快照只报告已完成点；双专家333→397使用astra/high，后续改用sol61/xhigh。Free在seed157/6101分别从525/653之后切换模型，Hybrid从813之后切换；历史格式／证据修复记录保留，模型贡献不能由混合轨迹单独识别。

## 相同预算比较

| Seed | 共同区间 | 方法 | 端点 NRMSE | Validation AULC |
| --- | --- | --- | ---: | ---: |
| 157 | 333–621 | CW16 + LLM16 | 0.398095 | 0.582171 |
| 157 | 333–621 | Free LLM32 | 0.625707 | 0.722777 |
| 157 | 333–621 | Dual expert | 0.597021 | 0.699514 |
| 157 | 333–621 | CW32 (LCMD) | 0.429871 | 0.567951 |
| 157 | 333–621 | Random32 | 0.696536 | 0.817237 |
| 6101 | 333–813 | CW16 + LLM16 | 0.380292 | 0.581291 |
| 6101 | 333–813 | Free LLM32 | 0.422698 | 0.698584 |
| 6101 | 333–813 | CW32 (LCMD) | 0.422949 | 0.591969 |
| 6101 | 333–813 | Random32 | 0.598080 | 0.771843 |

AULC为梯形积分除以该共同区间长度，越低越好。不同seed／区间之间不可直接排名。

## 其他传统策略

![All strategies](all_strategies_validation.png)

CW-MaxDet/CW-IVR两seed均完成1005；Whitening按要求停525。原333–525确认门槛与1005探索性扩展分开解释。

## 证据与复现

[snapshot.json](snapshot.json)保留完整曲线、拟合摘要与原始文件SHA-256。checkpoint、完整请求和训练缓存仍在本地忽略的runtime目录；新克隆可直接读取或重绘快照。

```bash
python scripts/report_active_learning_snapshot.py --from-snapshot reports/active_learning_latest/snapshot.json
```

报告不打开测试真值、不训练、不调用LLM。两seed及历史修复后的曲线属于探索性开发证据，不能识别化学知识或额外模型调用的独立因果贡献。
