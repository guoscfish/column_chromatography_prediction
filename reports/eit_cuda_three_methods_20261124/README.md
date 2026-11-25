# EIT 4g 三方法 CUDA 实验归档

Random、Uncertainty + Diversity Hybrid32 和 CW-LCMD 各有五个 seed 从 L333 跑到 L1485。15 条正式轨迹共封存 555 个预算点，最后一条于北京时间 2026 年 11 月 25 日 07:47:50 完成。测试指标在全部轨迹封存后计算。

- [最终报告](analysis_final/REPORT.md)包含五 seed 配对比较、AULC、标签效率和选样诊断。
- [学习曲线](analysis_final/learning_curves.png)、[逐预算验证与测试指标](analysis_final/learning_curves.csv)和[计算成本](analysis_final/trajectory_cost.csv)可直接查看。
- [运行协议与远端验证](REPORT.md)记录冻结输入、任务调度和恢复检查。
- [归档清单](archives/MANIFEST.json)列出 15 个正式轨迹归档和一份资格验证与运行证据归档。

正式轨迹归档保留每个预算的训练历史、预测、选样、标签集、校验凭据和运行日志，以及每条轨迹的 L1485 最佳 checkpoint。每个压缩包都已逐文件解压核对 SHA-256，覆盖同步回 Mac 的全部原始结果及运行证据。完整远端运行目录保存在原 GPFS 路径。

## 解包与核对

在本报告目录执行以下命令。文件恢复到原有的 `raw_results/` 和 `evidence/` 相对路径。

```bash
(cd archives && shasum -a 256 -c SHA256SUMS)
for archive in archives/*.tar.gz; do
  tar -xzf "$archive"
done
gzip -dc archives/files.sha256.gz | shasum -a 256 -c
```
