# 4g 行级三方法 CUDA 正式实验

## 统一协议

实验使用冻结的 4g 柱色谱行级任务和 seed 157、887、2357、6101、12203。每个 seed 的三方法共享有序 L333、2997 行候选 U0、416 行验证集、417 行测试集，外层训练域为 3330 行。由 L333 起完成 36 批 B32，终点 L1485，共 37 个评估预算。

Mac 原实验是在各方法自己的历史 L1005 标签集上续跑，首个新增 L1037。五个 seed 的三个历史 L1005 标签集中，Random 与 Hybrid32 只重合 473–488 行，Random 与 CW-LCMD 重合 471–495 行，Hybrid32 与 CW-LCMD 重合 633–664 行。新 CUDA 实验从真正共同的 L333 重新起跑，历史 CPU 续跑路线上保留原结果。

复用 QGeoGNN-V2 458,952 参数模型和固定 seed/member 初始化，逐轮从头训练。Adam 学习率 0.001、weight decay 0，无 scheduler，batch 2048，最大 1000 epoch、patience 100。训练目标为等权 V1/V2 原始 quantile 损失，原确定性逐 epoch 排序，checkpoint 按验证综合 NRMSE 选择。图及协变量预处理沿用冻结训练域，监督与目标归一化的固定尺度来自 L0，新增训练标签仅在对应 B32 选样冻结后揭示。训练与选样只见 test 图的清零 label 组件，15 条完整 CUDA 预测轨迹全部冻结后才开放测试真值作离线评分。

CUDA PyTorch 2.10.0+cu128，CUDA 12.8，Python 3.12.3。完全保留历史科研包发行版本，形式上将镜像 PyTorch 本地版本后缀和 Python 平台记录为本次独立实验的运行时。启用 deterministic algorithms，CUDA matmul 和 cuDNN 使用 IEEE FP32。继承已验证的 checkpoint CPU 映射和旧 RBF 固定值迁移。每张 GPU 同时运行一条任务。

| 指纹 | SHA-256 |
| --- | --- |
| 4g canonical CSV | `6cbe81311177ae178d10ff75a8f07c41c99a18648ec6104d2ffedca6e912d51c` |
| 4g 图缓存 | `a48f8ea1ce4f89c5a1ea6d17466e7c7ffa0439edb8ec79bd8b765bad3a97c3ee` |

工作树 HEAD `b6b03b9eb62ebc75b7e60a10fbe078492d7efd92`，参考预注册 commit `a419ea75e7894004aef393cb3734a7ae5e1356a1`。实际源文件哈希保存在服务器 `source_receipt.json` 和每条任务的 `run_contract.json`。首轮只传送三个变化文件，已校验的冻结输入仍由原 GPFS 工作树提供。

## 方法定义

| 方法 | 冻结实现 |
| --- | --- |
| Random | `sequential_acquisition.deterministic_random_permutation` 对原 U0 以 `default_rng(seed*1_000_003+900001)` 生成一次排列，每轮切出下一个 32 行。选择器与模型预测无关。 |
| Uncertainty + Diversity Hybrid32 | `sequential_acquisition.v2_hybrid_select_current` 在当前标签集同协议训练 K=3 QGeoGNN。对候选两个 q50 预测分别计算成员间总体标准差，按固定 L0 对应输出尺度归一化并求 L2 范数。取前 `ceil(.25×U_t)`，再用成员 0 的当前 128 维 QGeoGNN 表征进行欧氏最远优先，起始中心为全部 L_t，每选一行即添为中心。两级选择，无线性权重。 |
| CW-LCMD | 最新五 seed CW 轨迹调用 `cw_lcmd_extension_runner.select_current_cw_lcmd`，继承 `gradient_transforms.center_width_transform`、`gradient_features.extract_linear_output_gradient_sketches` 和校正过的 `lcmd.lcmd_tp_select`。对当前模型 q50 构造 `C=(V1+V2)/(2s_C)` 与 `W=(V2−V1)/s_W`，其中尺度为 L0 的总体标准差。两路输出的全网络梯度用 seed+4,000,037 的 512 维 CountSketch 表示。候选各分配到最近中心，选择平方距离和最大的簇里最远样本，加为中心并更新，直到选满 32。按原输入顺序稳定处理平局。 |

CW-LCMD 是历史 Center/Width 变换后的梯度特征方法，与单独排除的普通 q50 Gradient-LCMD 仍是不同方法。正式适配为独立入口，模型、训练和选择规则复用原函数。

## 运行管理

GPFS 工作区 `/1/column_chromatography_prediction/worktree`，用原环境 `/1/column_chromatography_prediction/environment/venv`。正式实验根目录 `/1/column_chromatography_prediction/formal_cuda/row4g_b32_20261124`。

- `queue.json` 保存资格状态、15 条正式计划、每个尝试的命令、PID 和 GPU 分配。
- `trajectories/<method>-<seed>/rounds/<method>/round_NN` 保存已标注/候选索引、验证和训练历史、每轮选样、模型与校验收据。
- `qualification/` 保存独立完整资格轮，`jobs/<run-id>/console.log` 和 `gpu.csv` 记录作业运行和 GPU 采样。远程日志在 SSH 断开后继续写入。
- 已完成轮次通过连续前缀及文件哈希恢复。中断的未封存轮次独立归档，重跑该轮。新实例沿用源版本、路径和环境，启动新的 dispatcher 名称。

```bash
root=/1/column_chromatography_prediction
suite="$root/formal_cuda/row4g_b32_20261124"
python3 scripts/remote_compute.py --run-directory formal_cuda status row4g_dispatch_20261124_a --tail 12
ssh -o BatchMode=yes eit-hpc "cat '$suite/queue.json'"
ssh -o BatchMode=yes eit-hpc "tail '$suite/jobs/random-157/console.log'"

# 重挂同一 GPFS 后，在旧 worker 已退出时重新打开两个调度槽
python3 scripts/remote_compute.py --run-directory formal_cuda launch \
  --name row4g_dispatch_20261124_b --seeds 0,1 --gpus 0,1 --threads 2 -- \
  -m scripts.studies.queue_qgeognn_cuda --root "$suite" worker --slot '{seed}'
```

## 计算成本初估

独立运行 555 次主 fit、360 次额外 Hybrid 成员 fit、180 次 CW 全池特征提取。历史早/中/晚阶段实际 epoch 数外推约 24 万 epoch。上一轮同配置完整 CUDA fit 在 L1005 暖态约 0.20 秒/epoch，对应训练约 13.3 GPU·小时或双卡约 6.7 小时。CW 特征速度、预算阶段差异和调度时间尚需本轮校正。首轮估计双卡实例 8–11 小时，约 48–66 元。建议续租后剩余 12 小时，约 72 元，给资格验证、收尾和同步留余量。

CW-LCMD 首轮真实提取在完成 1200/3330 行时累计 127.1 秒，线性外推单轮约 353 秒。若后续接近此速率，180 次特征提取需要约 17.7 GPU·小时。与训练合计约 31 GPU·小时，双卡理想下限约 15.5 小时。考虑两卡排程尾部和资格验证，更新为建议剩余 18–20 小时，约 108–120 元。此处为初步外推，后续以完整特征实测校正。

## 首次启动回报

13:58:24 北京时间，GPU 0 正式启动 Random/157（PID 5407），GPU 1 启动 CW-LCMD/157 独立资格轮（PID 5408）。至 14:02 已回收 Random 的 L333、L365、L397、L429 验收轮次。四个正式主 fit 分别为 189、167、180、144 epoch，代码内耗时 32.71、29.30、32.16、26.17 秒。两张卡各有一个独立研究进程。其他 14 条正式轨迹登记在队列，两条 Hybrid32 完整资格复跑仍待运行。新 Random 领取暂缓到固定 checkpoint CUDA 重放给 GPU 1 留出窗口，当前 Random/157 继续训练。

同一 Mac 审计时刻正在运行 Hybrid157 与 Hybrid2357，Random157、Random887、Hybrid887 已到 L1485，另五条臂等待本地 supervisor。正式 Mac 与 CUDA 路径各自保存，继续运行的 CPU 工件只用于冻结协议审核。

13:53 复用的容器自 11:26:14 启动，GPFS 挂载、现有 CUDA venv 和两张卡均在线；当时两卡空闲，无正式运行。SSH 环境无平台计费倒计时字段，具体租期以平台续租为准。

## 已取得的中间证据

CW-LCMD/157 的完整 CUDA 采集从 L333 生成 32 条，3330 行 512 维特征提取 352.844 秒，LCMD 选择 0.263 秒。正式 Random 前 10 个预算点的验证、选样诊断及成本快照已回收，测试真值读取次数为零。远端最终分析等待 15 条正式任务全部成功，由全局收据校验通过后开放测试真值。

## 资格验证与远端独立运行

Hybrid32/157 两次完整 K=3 资格轮的三个成员分别训练 189、174、169 epoch。三个 checkpoint 状态、逐 epoch 验证 NRMSE 和 32 项选样顺序完全一致。固定同一组 CUDA checkpoint 重放两次，短名单和选样均复现，表征最大差为 0。
CW-LCMD/157 完整通过 L333 选样和 L365 复训。全部 3330×512 特征有限，3330 行均为非零。用原 LCMD 复算的 32 项与候选位置、canonical 索引和下一轮 L365 状态一一对应。

先关闭 SSH，再禁用连接复用，以新连接复查。服务端记录 2026-11-24T14:17:49+08:00 至 2026-11-24T14:18:26+08:00。以下进程的 PID 与内核启动标识均不变，GPU 0 继续推进 epoch，GPU 1 新封存一轮。

| GPU | 正式轨迹 | 调度器 PID | 训练 PID | 最新封存预算 |
| --- | --- | --- | --- | --- |
| 0 | random-157 | 5405 | 5407 | L973 |
| 1 | hybrid-157 | 5406 | 6288 | L397 |

守护进程 PID 6569，结果分析等待器 PID 6021。两者同样通过远端 nohup 独立运行。
排队顺序为 cw_lcmd-157, random-887, hybrid-887, cw_lcmd-887, random-2357, hybrid-2357, cw_lcmd-2357, random-6101, hybrid-6101, cw_lcmd-6101, random-12203, hybrid-12203, cw_lcmd-12203。

守护进程通过 GPU 排他锁发现 worker 消失。暂时性或信号退出每条轨迹最多自动重试一次，每个 GPU 槽位最多自动重拉两次。科研合同或数值校验异常保留为失败，其他任务继续领取。训练进程继承 GPU 与任务锁，worker 意外消失时不会启动重复子进程。已用独立目录验证封存轮保留、未封存轮归档并定位下一轮。完整任务队列耗尽后，训练 worker 结束，最终分析才运行。

租期由平台计量，SSH 不暴露倒计时。提供确切截止时间后可以使用队列的 deadline 参数，提前 45 秒停止新领取并在设定时刻停止仍在运行的任务。即使平台直接关机，最后已封存的轮次和队列可从原 GPFS 恢复。Mac 关机期间结果继续在 GPFS 写入，重新连线后可再同步回本地。

远端状态为实验根目录的 `queue.json` 与 `watchdog.json`。守护进程日志为 `watchdog.log`，每条轨迹的日志和 GPU 采样分别为 `jobs/<run-id>/console.log`、`jobs/<run-id>/gpu.csv`，最终分析日志为 `finalize_when_complete.log`。

[断线前状态](evidence/ssh_disconnect_before.json) · [重连后状态](evidence/ssh_reconnect_after.json) · [恢复测试证据](evidence/recovery_probe.txt)
