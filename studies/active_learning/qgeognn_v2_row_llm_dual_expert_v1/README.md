# 双专家主动学习

2026-10-10 按用户要求补齐多轮执行；2026-10-11 完成 seed 157 的两轮真实试跑。`free_llm32_dual_expert_v1` 由独立的 Chemistry Scientist 和 ML Scientist 提建议，Scientific Planner 选择完整的 32 个样本。与 free 的主要方法差异是决策角色；不设 CW 预选名额、专家配额或隐藏排序。

选样与反馈采用当前 llm+cw/free 的完整候选池流程：每个角色一次收到全部合法候选、全部已观测记录和无损列式编码；planner 额外收到当前两个专家的 memo。下一轮三个角色均收到之前的选样方案、假设、理由、专家 memo、最近 32 个实测结果及预测误差、历史最大误差记录和所有已选 ID。当前两个 scientist 的对话互相独立。

状态重建、CW 诊断特征、预测、冻结、标签揭示、同初始化重训和 checkpoint 规则复用 `scientist_lifecycle` 与原 Scientist 训练/预测组件。每批全部 32 个 ID 和测量前预测冻结之后才揭示标签；验证标签仅供训练选 checkpoint，测试标签不进入 LLM 或试跑报告。每个角色最多三次格式纠正，网络重试和成本记录沿用原 transport。

已完成运行保存于 `revisions/full_pool_feedback_schema_20261011/`，原 round-0 草案与冻结文件保留。首次试跑因另一个任务恢复共享文件而中断，未获得模型答案；其记录保留于 `revisions/full_pool_feedback_20261010/interruption.json`。独立生命周期模块避免修改正在使用的旧 Scientist 文件。代码变更后需使用新 revision；不重写历史协议 hash。运行中 checkpoint 和训练历史存于忽略的 `runtime/`，输入、角色对话、冻结及反馈保留用于审计。

## 当前扩展（2026-10-11）

两轮astra/high试跑是历史起点。seed157随后改用`gpt-6.1-sol`、`xhigh`，完成397→429，validation NRMSE为0.704016。当前使用`sol61_xhigh_from429_evidence_repair_20261011`继续429→493，已完成461标签（validation NRMSE 0.687444）；已完成与仍在运行的预算见[最新快照](../../../reports/active_learning_latest/REPORT.md)，未完成轮次不计作结果。

`sol61_xhigh_from397_20261011`记录memo格式失败；`sol61_xhigh_from397_repair_20261011`保留一次额外格式修复及429拟合；当前revision保留planner引用未观测证据的修复来源与有界额外回答。继承的批次、标签和checkpoint保持不变，所有原始失败回答和成本仍保留。不能把astra/high→sol61/xhigh的混合轨迹解释成单一模型确认。

继续当前已授权运行的命令（相同命令复用冻结前缀）：

```sh
/Users/fish/miniforge3/envs/fish/bin/python -u scripts/studies/run_qgeognn_v2_row_llm_dual_expert.py run --seed 157 --rounds 5 --start-round 3 --revision sol61_xhigh_from429_evidence_repair_20261011 --model gpt-6.1-sol --reasoning-effort xhigh
```

## 历史两轮试跑与运行说明

```sh
/Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_dual_expert.py run --seed 157 --rounds 2 --revision full_pool_feedback_schema_20261011 --model gpt-6-astra --reasoning-effort high
```

当前CLI默认是`responses`、`gpt-6.1-sol`、`xhigh`；复现历史两轮协议必须显式传入`--model gpt-6-astra --reasoning-effort high`，沿用原provider配置。`--rounds 2` 完成 333→365→397 两次新增选样和再训练；再次执行会重放冻结选样并复用已完成拟合。支持 `prepare`、`validate`、`check-transport`、`stage`、`select`、`advance`、`report` 和自定义 `--revision`。`--round` 是单步操作的零基轮号，`--rounds` 是 run/report 的累计轮数，最多六轮。

## 历史两轮结果（333→397）

两轮已完成：333→365→397，新增64条标签、两次同初始化再训练，无新test评价。双专家验证集NRMSE为0.853887→0.841029→0.762186；397端点优于free的0.786118与Random32的0.851246，仍弱于CW16+LLM16的0.704643和CW32的0.713259。两轮validation AULC为0.824533，未优于free的0.802251；因此尚不支持双专家整体优势。

[完整结果、对比图和成本](revisions/full_pool_feedback_schema_20261011/reports/seed_157/free_llm32_dual_expert_v1/rounds_02/README.md)。对照均复用历史同seed、同分割和同训练规则的fit audit，无新增对照拟合。

首轮planner的共识/分歧返回文字列表，旧validator只接受字符串，导致两次额外格式纠正。揭示标签前修复为兼容非空文字列表，并冻结独立revision；重放最早合法的原始回答，内容与32个ID不改，不按效果挑选回答。输入/响应与代码迁移来源记录在 [schema_migration.json](revisions/full_pool_feedback_schema_20261011/schema_migration.json)。有效角色回答6次，含两次额外格式回答共8次provider回答、3,704,920 tokens；失败格式成本未丢弃。2026-10-10的429记录保留为历史状态。

72项相关测试通过，包括格式兼容、合成两轮freeze→reveal→fit→feedback、历史方案与测量反馈传递、独立专家上下文、重放、篡改检测和标签访问边界。旧冻结Scientist代码保持不变。

局限：单 seed、两轮、探索性验证集比较；双专家增加模型调用和上下文，结果不单独识别化学知识的因果贡献。

- [设计说明](../../../docs/research/4G_LLM_DUAL_EXPERT_ACTIVE_LEARNING_V1_2026-09-27.md)
- 原草案未提交文件 hash：`draft_provenance/source_manifest.json`
