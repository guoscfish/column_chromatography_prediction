# 科学复盘位置与重构边界

按用户最新要求，分位数/损失机制复盘完整保留在 `main`，不放进重构稳定 API。提交：`fcb51852c8bb6e38403fe95a302287a0a1779d47`；主报告路径：`studies/diagnostics/quantile_loss_audit_20261109/SCIENCE_AUDIT.md`。本地文件位于 `/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/diagnostics/quantile_loss_audit_20261109/SCIENCE_AUDIT.md`。

可在原仓库任一工作树查看已提交报告，无需切换正在运行的 main：

```sh
git show fcb51852c8bb6e38403fe95a302287a0a1779d47:studies/diagnostics/quantile_loss_audit_20261109/SCIENCE_AUDIT.md
```

该报告及核心 CSV、两张图、输入/代码哈希区分 `HISTORICAL_MEASURED`、`NEW_OFFLINE_MEASURED`、`NOT_MEASURED` 和 `HYPOTHESIS`。既有 3-seed row/compound 资格结论复用；新增离线测量覆盖五 seed 的冻结 L333 训练/row validation，未训练、未调用 LLM、未转换 U0/test 真值。后续预算、当前 compound AL、真实同 seed K3、实际 Adam 更新竞争及 AL 效率提升均未冒充实测。

重构迁移保持原 target_loss、训练、模型和评分数学；checkpoint/cache、表示及选样的实测范围与限制见 [TEST_REPORT.md](TEST_REPORT.md)。q50 用 MSE 学习，不能直接解释为校准中位数；分位数宽度没有得到校准 epistemic UQ 资格，跨目标 `V2_q90-V1_q10` 也不是 `V2-V1` 的分位数。此工程重构不改变原方法定义或以诊断结果选择新策略。
