当前 supervisor PID：56694。controller_guard.py 只补充有界恢复检查；五个已启动 worker、在途请求及 frozen_runner.py 均未改变。核验证据见 parallel_startup_verification.json。

# 五个独立 key 并行选点：执行修订

2026-11-01 用户追加授权：“五个seed用五个不同的key跑一下”。本机已再次核验五个既有 key 非空且互不相同，不输出凭据值或指纹。

当前设置为五个 seed 选点 worker 并行，每 seed 始终使用自己的既有独立 key；昂贵的特征计算和重训共用一个独立计算锁，且继续让 NoCW 训练优先。模型仍固定 gpt-6.1-sol / xhigh，终点仍为五 seed 各 L1485（36 轮/seed、180 个接受批次）。

只替换旧 Free supervisor，不打断其子 worker。887 的现有在途请求由新 supervisor 直接接管，不重复发送；其余 seed 从本次新轨迹的冻结记录恢复。原 NoCW 进程、配置、源码与历史产物均不修改。

157 此前在首轮第 4 个传输尝试返回 HTTP 522 后停止。该状态应按服务端网关/传输失败处理，不能视为模型/凭据不支持。本修订保留原失败记录和请求哈希，另存分类修复收据；首轮首次回答的总传输上限冻结为 8 次（包含已发生的尝试），其余每个回答仍最多 4 次，逻辑回答仍最多 8 个。退避固定为 2/10/60/120/120/300/600 秒。没有无限重试。模型变化、配置不支持、凭据错误和无完整回执的模糊中断仍停止。

原冻结 config.json、protocol.json、源码和所有请求消息均保持不变。执行修订由 parallel_execution_amendment.json 固定并核验，成功请求和已完成训练继续本地复用。6 项新增离线测试已通过，新增离线测试真实 API 调用为 0。

当前状态入口为 progress/parallel_supervisor.json；原 progress/launch.json 和 supervisor.json 保留为初次单并发启动历史，不代表当前并发。

实际执行入口：

```sh
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /Users/fish/miniforge3/envs/fish/bin/python "/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/concurrency_v2/frozen_runner.py" supervise --allow-paid-llm --allow-training --allow-label-reveal
```

启动已经执行，记录见 progress/parallel_launch.json。状态查询使用同一入口的 status 子命令。

- 修订与 key 匿名审计：parallel_execution_amendment.json。
- 五 seed 最新进度：progress/seed_<seed>.json；流状态：progress/stream_seed_<seed>.json。
- 新 worker 日志：progress/seed_<seed>_parallel.log；被接管的 887 沿用原 seed_887.log 至本轮结束。
- 控制器日志：progress/parallel_supervisor.log。
- 模型、反馈、冻结、报告目录保持原独立新轨迹路径；不会复制历史 Free 的后续选择或 checkpoint。
