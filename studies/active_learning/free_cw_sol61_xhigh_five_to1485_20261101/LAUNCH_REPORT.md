> 当前执行已按后续用户要求调整为五 key、五 seed 并行选点；重训/特征计算仍为单并发。当前入口及运行状态见 PARALLEL_RUN_REPORT.md 与 progress/parallel_supervisor.json。下文保留初次启动历史。

# 带 CW 的 Free LLM32 五 seed 正式重跑启动记录

核验时间：2026-11-01T17:48:34.126602+08:00。

正式实验已启动；并非等待确认或付费试点。

- 独立目录：`/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101`
- 新轨迹名：`free_cw_sol61_xhigh_rerun32`。历史 Free 目录与曲线未覆盖。
- 请求模型固定 `gpt-6.1-sol`，推理参数 `xhigh`；独立既有 seed key，未记录凭据值。
- L333 → L1485，每轮 32 个，36 轮/seed，计划接受 180 批。实际 API 请求数另计。
- 实际新并发上限 1（一次执行一个 seed 的一轮），CPU 两线程；按已完成轮数与固定 seed 顺序轮转排队。NoCW 训练阶段优先。
- 已确认后台 supervisor PID 83387，seed 157 worker PID 83861。

## 实际启动命令

```sh
env KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /Users/fish/miniforge3/envs/fish/bin/python /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/frozen_source/scripts/studies/run_free_cw_sol61_five_20261101.py launch --allow-paid-llm --allow-training --allow-label-reveal
```

以上启动已经执行，不要再次运行 launch。查看状态：

```sh
env KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/frozen_source/scripts/studies/run_free_cw_sol61_five_20261101.py status
```

实际后台监督命令与 PID 另见 `progress/launch.json`；该监督进程按本次冻结协议持续执行至终点。

## 首次运行状态

| seed | 当前 L | 状态 |
|---:|---:|---|
| 157 | 333 | 第 1 轮正式选点，HTTP 200 / IN_PROGRESS |
| 887 | 333 | QUEUED，等待单并发轮转 |
| 2357 | 333 | QUEUED，等待单并发轮转 |
| 6101 | 333 | QUEUED，等待单并发轮转 |
| 12203 | 333 | QUEUED，等待单并发轮转 |

首个响应 ID：`resp_02bb5a082888b8df016ae70ad7cf34819b9e2c978ba2a69797`。请求 ID：`25c3d4a8-e797-483d-b76d-bac845a64c90`。
首个正式请求已被服务端接收，完整选点响应尚未返回；因此尚不声称实际 returned model 已验证。响应完成后严格检查并锁定返回模型标识；变化或配置不受支持会停止受影响轨迹，不降级、不截断、不跳过预算。

## 校验与恢复

- 五 seed 的实际 NoCW split、L333、U0、初始 checkpoint、预测、预处理与 normalization 已对齐；初始 checkpoint/预测只读复用。
- CW 初始真实分位从哈希、checkpoint、ID 顺序绑定的 L333 特征缓存独立重算，遵循原 `_card` 的 `round(value, 5)` 序列化。后续特征、选择、反馈、模型均为本次新轨迹。
- 五份 packet_diff 列出 5 × 2997 个 CW 字段增量、必要说明与绑定哈希差异；其他首轮输入、顺序和数值精度与实际 NoCW 相同。
- 首轮科学记忆为空；候选与已测证据分别验证；重复、已测当候选、候选当已测证据均被拒绝；合法 CW 文字通过。
- 9 项离线测试通过。离线阶段真实 API 调用、新标签揭示、模型拟合均为 0。
- 成功响应在关闭流前保存；恢复优先复用本地完整结果。未完成且无完整响应凭据的请求不自动重发。训练完成凭据先落盘，预测导出失败后可复用已完成 scratch 训练。
- 有界规则：每轮最多 8 个回答（初次 + 7 次纠错），每个回答最多 4 次传输尝试，传输退避 2/10/60 秒；最多 2 次可恢复本地 worker 重启。180 是接受批次目标，不是总 API 调用上限。
- 当前 NoCW 的历史无上限重试与本次有限重试存在接口差异；见 SCHEMA_DIFF.md。训练从头重训、初始化、RNG、split、训练参数和 checkpoint 口径保持 NoCW 实际协议。
- 原 NoCW 五个 worker PID 1581/4386/4387/64135/86832 在启动核验时全部存活，配置和协议哈希未变；无 pull/checkout/reset，无共享源码或旧轨迹覆盖。

## 产物位置

- 冻结源码与清单：`frozen_source/`、`source_snapshot.json`、`protocol.json`、`protocol.lock.json`。
- 离线总审计与机器差异：`offline_checks/audit.json`、`offline_checks/packet_diff_seed_*.json`、`offline_checks/initial_state_seed_*.json`。
- 正式请求、响应、冻结与标签收据：`selections/seed_<seed>/free_cw_sol61_xhigh_rerun32/round_XX/`。
- 每轮 checkpoint、history、预测与训练完成凭据：`runtime/seed_<seed>/free_cw_sol61_xhigh_rerun32/round_XX/`。
- 日志、锁与实时进度：`progress/seed_<seed>.log`、`progress/supervisor.log`、`progress/*.json`、`locks/`。
- 启动证据：`startup_verification.json`、`launch_readiness.json`、`progress/launch.json`。
- 自动报告：`reports/REPORT.md`、`reports/validation_curves.csv`、`reports/completion.csv`、`reports/paired_differences.csv`、`reports/comparison_summary.json`、`reports/costs.json`。
- 每轮完成自动更新；五 seed 共同完成 L525/L813/L1005/L1389/L1485 时冻结对应 milestone 汇总。
- 比较保留原始非单调曲线，使用归一化梯形 AULC；与历史 Free、NoCW、数值 Hybrid32、Gradient-LCMD32 分别使用五 seed 公平共同预算，并报告逐 seed、配对差值、均值与样本标准差（ddof=1）。
- token usage、请求 ID、返回模型与时间保存；没有验证的货币单价不臆造成本。
- 报告、学习曲线和验证指标不进入 LLM 选点输入或科学记忆；未读取测试标签。

统一声明模型与推理配置不能消除 API 输出随机性、运行时段或不可见后端变化；本次不直接宣称 CW 的独立因果机制。

准备阶段旧版本与零 API 的失败核对记录保存在 `setup_attempts/`；正式入口只使用当前 `frozen_source/` 和本目录正式轨迹，不从这些准备记录回退。
