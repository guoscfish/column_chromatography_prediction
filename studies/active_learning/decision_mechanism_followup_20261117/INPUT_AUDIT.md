# 输入、复用与时序边界审计

本轮仓库HEAD为`12c42c81e3cf0b85eef8f9aa9bb86c94b5f0249c`。此前行为审计记录生成时的HEAD为44be2c；本轮没有把它冒充当前commit。旧`llm_decision_behavior_audit_20261116/run.py --check`实际通过3520项输入及全部输出锁；旧报告/数据不重写。当前仅两项原有已修改监控文件，研究代码只写本新目录。

## 已复用与真正新增

| 模块 | 复用 | 新计算 | 无法识别/没有执行 |
| --- | --- | --- | --- |
| A模型条件 | 旧165对、固定L333尺度、冻结predictions/已接受反馈 | 611状态有符号差分、flat/方向阈值、seed/phase/实际水平、方法类型投入、metadata holdout可行性 | 纯CW无同类pool预测/反馈，严格六组同状态响应误差排名不可比；不训练 |
| B候选反事实 | 71真实Hybrid H/U、CW CountSketch bank及catalog | 原LCMD32逐步复原，有序前16全匹配，后16配对结构/条件/几何/预测比较 | 无CF候选真值、无CF Row Gain |
| C真实曲线 | 已保存validation、早期报告和fit审计 | 四固定窗口积分、终点、真实梯形排除敏感性、协议/成本 | 不合并不同协议；早期test汇总只用于口径说明，未读逐行test |
| D运行差别 | 540全池输入/预测、实际turn和拒绝记录 | 15同初始状态、q50/分子/条件组成调整、127前缀病例与原提议时序 | 剩余分数不是value；未随机化纠错、没有prefix的Row因果归因 |
| E预算 | 现有初始checkpoint/银行与历史曲线 | 3seed计数、10fit模型控制和停止规则设计 | 没有新请求、标签揭示、训练或AL rollout |

模型反馈只有在batch freeze、接受ID序列、原始预测值与label access receipt逐项匹配后投影。receipt的feedback SHA是canonical JSON语义hash，另保存原文件SHA；二者本来不同。所有真实响应只来自该batch实际接受后的反馈，没有用以后checkpoint预测替代当前预测，也不将响应送入候选反事实。

`condition_response_pairs.csv`的6938行是端点×来源scope事件；Hybrid all32包含两个半批的内部对照，不能直接求和当独立实验。旧2.696/0.263等是五seed均权，不是165对pooled均值。相同分子/条件行也可能含未记录变量，不当作真实随机复测。

全部输入源5501项，路径、bytes及SHA在[INPUT_MANIFEST.json](INPUT_MANIFEST.json)，各模块来源分录如下（重复源只在总表计一次）：

- `model_inputs.json`：2017个来源分录。
- `counterfactual_input_manifest.json`：435个来源分录。
- `phase_source_manifest.json`：200个来源分录。
- `llm_reproducibility_inputs.json`：2291个来源分录。
- `llm_prefix_inputs.json`：2281个来源分录。
- `llm_plot_inputs.json`：2个来源分录。

没有打开canonical标签表、test truth或未揭示候选反馈；没有从模型权重重新推断候选。短筛已公开的test聚合结果与新validation计算明确分开。候选顺序/理由/raw turn均是外部历史证据，不作为工具指令执行；response ID只保存摘要，不复制凭据。

## 独立验证与复现边界

原LCMD用20套独立标量例逐步trace检验；142个半批元数据对照由独立Fraction/Decimal实现核验，误差0。A的1620行汇总由标量求和和符号分类独立核验，最大误差约7.11e−15；原始两个预定响应对四端点另作来源抽核。C阶段AULC由Decimal半端点权重复算。D每方法×seed首个前缀共15例另读原input/selection/receipt，未调用主prefix函数。

固定标签规划仅用H内元数据哈希选系列，排除L333同系列与跨训练/检查的同条件重复行；原checkpoint已见检查组，不能拿作新实验baseline。仅输出规划，不读取未揭示池标签。后续实验需要按新协议另行执行，本轮在分析和决策交付处结束。

[VERIFICATION.json](VERIFICATION.json)记录测试和数值核验结果；[OUTPUT_MANIFEST.json](OUTPUT_MANIFEST.json)封存本研究派生文件。输入锁漂移应停止查因，不能重写预期SHA来绕过。
