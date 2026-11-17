# 2026-11-17 研究 Git 发布

本次发布包含两项已完成研究的代码、测试、报告、图表、固定配置、拟合回执和科学结果：

- [历史决策机制后续诊断](../../studies/active_learning/decision_mechanism_followup_20261117/README.md)
- [体积响应与纠错策略审计](../../studies/active_learning/volume_response_retry_audit_20261117/README.md)

原研究文件及封存 hash 保持原字节。按[仓库产物保留规则](../../docs/README.md#产物保留规则)，新 checkpoint、逐次训练历史、进程状态、控制台日志和重复数据缓存保留本地，不新增到 Git；拟合回执和固定预测结果作为本轮科学证据保留。原有两项实时监控状态修改不属于本次提交。

体积研究的三个大体积审计文件在 `archives/` 无损 XZ 压缩保存：`retry_audit_inputs.json`、`retry_transport_attempts.csv`、`VERIFICATION.json`。`MANIFEST.json`记录原始及压缩文件 SHA-256、字节数、已发布文件和本地运行产物清单。压缩后均已解压逐字节 hash 核验，没有改写原封存或历史数值。

在仓库根目录核验发布包：

```sh
python -B reports/research_publication_20261117/publication.py
```

在新克隆中恢复这三个被忽略的大文件：

```sh
python -B reports/research_publication_20261117/publication.py --restore
```

恢复程序只创建缺失文件；已有文件必须匹配原 SHA，否则停止，不覆盖。研究的完整 `verify.py --check` 还依赖其封存列出的本地源日志、图缓存及模型运行文件；发布包核验不冒充完整科学环境复现。本次发布没有训练、模型推理、LLM/API请求或新标签查询。
