# Free LLM32-NoCW paired ablation

研究预计算 Center/Width 梯度覆盖数值是否改善自主LLM选点和标签效率。两个新Arm分别为 `llm32_cwaware_control` 与 `llm32_nocw`；保留预测、化学知识、数值特征及各自实验反馈。这里交付代码、冻结协议、离线测试和mock dry-run，未授权/未执行正式实验。

见 [PROTOCOL.md](PROTOCOL.md)、[SCHEMA_DIFF.md](SCHEMA_DIFF.md)、[CODE_AUDIT.md](CODE_AUDIT.md)、[PREFLIGHT_REPORT.md](PREFLIGHT_REPORT.md)。`config.json` 冻结共同模型/参数，`protocol.lock.json` 和 `dry_run_audit.json` 提供机器证据。完整输入、请求和mock冻结位于本地 `dry_run/`，可用以下命令重建；历史源码、冻结数据和用户已有改动不受影响。

在仓库根目录使用已有 fish 环境（base Python 没有 torch）：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
PY=/Users/fish/miniforge3/envs/fish/bin/python
SCRIPT=scripts/studies/run_qgeognn_v2_row_llm_nocw.py
$PY "$SCRIPT" validate
$PY "$SCRIPT" dry-run
# 无 action 也只执行 dry-run；不会训练、调用LLM或揭示新标签。
$PY "$SCRIPT"
```

正式运行前必须另获明确授权，确认该endpoint支持 `gpt-6.1-sol/xhigh`、32768输出限制及完整输入所需的上下文容量（L333约87–90万字符，后续还含记忆）。当前未验证远端支持、价格或凭据，不调用接口来试探。以下仅为未来运行指令，本次未执行：

```bash
# 一次性的预注册/授权记录，绑定已经冻结的协议。
$PY "$SCRIPT" preregister --authorize-formal-run --confirm-provider-support
# SCIENTIST_API_KEY 或 SCIENTIST_API_KEY_FILE 由操作者安全配置。
# 只准备一个输入，不请求LLM，不reveal，不fit。
$PY "$SCRIPT" stage --seed 157 --arm llm32_cwaware_control --round 0
# 只选点并freeze，不reveal、不fit。
$PY "$SCRIPT" select --seed 157 --arm llm32_cwaware_control --round 0 --allow-paid-llm
# 只推进已经freeze的一个批次：reveal+fit下一个预算，不自动发下一轮请求。
$PY "$SCRIPT" advance --seed 157 --arm llm32_cwaware_control --round 0 --allow-label-reveal --allow-training
# 后续round的全量推理/梯度计算也需显式允许。
$PY "$SCRIPT" select --seed 157 --arm llm32_cwaware_control --round 1 --allow-paid-llm --allow-feature-compute
# 每次对另一个Arm和seed分别操作，轮次0..20；不提供默认全实验run入口。
# 完整配对矩阵达到525/1005后，独立生成validation报告。
$PY "$SCRIPT" report --budget 525
$PY "$SCRIPT" report --budget 1005
```

调度建议：seed157偶数轮A先、seed6101偶数轮B先，奇数轮反转，交错运行以减少时间漂移；不要按L525分数淘汰任何Arm。mock目录永远不能用于正式advance。修改冻结源码、文档、config、环境或初始文件会使validate失败；应另建新版本重新预注册，禁止覆盖旧lock。

复用仅限每seed的 `qgeognn_v2_row_sequential_b32/runtime/seed_*/lcmd/round_00/model/best.pt` 及其L333审计/特征，不复用L365+历史模型。L525全四轨迹需24个主采集+24次拟合；L1005需84+84。格式纠错和传输失败另计，上界见协议。远端usage不可用的失败账单列为未知，不能视为免费。
