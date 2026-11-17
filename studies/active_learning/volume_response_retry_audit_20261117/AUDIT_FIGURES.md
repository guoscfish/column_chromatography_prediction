# A2与纠错审计的静态科学图

这三张图仅使用已保存的A2/C统计表，**不读取任何15fit结果**，不增加推理、训练或API调用，不改变数据与历史报告。英文标注，PNG＋PDF保存于 [audit_figures/](audit_figures/)，输入/输出SHA见 [figure_manifest.json](audit_figures/figure_manifest.json)。

复现：

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/volume_response_retry_audit_20261117/make_audit_figures.py
```

1. [A2：条件响应幅度](audit_figures/a2_frozen_condition_amplitudes.png)：原冻结L525/L1005 checkpoint在当时已知训练历史中的真实/预测单因素变化。先在每seed内求pair均值，再五seed等权；小点是seed均值，细线是min–max，不能解释为pair层面置信区间。体积面板使用log轴，PE面板使用线性轴。它不是本轮新fit的隔离检查成绩。
2. [C2：首错类型与最终prefix](audit_figures/c2_first_failure_and_prefix.png)：左图为全540个已接受Free32轨迹batch的prefix/非prefix数量，右图为方法×首错类型条件频率，每格明确标记prefix/总batch；没有案例的格子单列，不填为0%。Historical seed157/r5是手工接受nonprefix，仍在candidate-ID首错61例分母中；匹配原接受receipt为539/540。不能由这个观察性图宣称错误类型造成prefix或服务商降级。
3. [C4：原选择保留与缺额](audit_figures/c4_local_repair_preservation_and_gaps.png)：只含127个prefix案例，其原可解析方案各32项，总4064个选择事件。合法原ID共3634，局部原型全部保留，历史最终保留440；不同batch可能重复同一ID，因此不是全仓库不同分子/候选数。430个空额明确保留，未用前缀自动补位。两个没有ID空额的案例仍有科学证据问题；总体0/127草稿通过完整gate，所以不能把100%合法ID保留画成100%自动修复成功。

图中均为描述性数据，没有新增显著性检验、因果估计或学习性能反事实。仅解释有明确定义的数据分母与可观察行为。
