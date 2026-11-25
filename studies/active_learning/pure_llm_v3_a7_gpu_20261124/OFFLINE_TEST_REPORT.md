# 部署资格与测试结果

本地运行旧版 34 项离线测试及新增正式运行测试，共 **70 项通过**。合成多轮测试验证一次授权、冻结后反馈、按请求重试与恢复、CUDA 后端契约、5 个 seed 的独立 A7 初始请求和动态 Key 池。测试中的模型输出、标签以及训练产物均由测试夹具或已有公共初始状态提供，未消耗 API 或训练正式模型。

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -B -m pytest -q \
  studies/active_learning/pure_llm_v3_20261124/tests \
  studies/active_learning/pure_llm_v3_a7_gpu_20261124/tests
```

上述组合在本机耗时 63.84 秒，现有依赖发出 118 条弃用或可选文本检测依赖警告。CPU 环境明确拒绝 CUDA 执行。远端实机资格结果另见本地保存的部署回执。

部署包在全新目录已通过 SHA 安装验证、两个 CLI 的导入与参数解析，以及 scheduler 的五 seed 合成预检。发布包没有历史 LLM 运行产物和数值 CUDA runner。初始请求均为 333 条观测、2997 条未测，M2 记忆为空。每 seed 的 416/417 个 validation/test 角色保持隔离。

| seed | 合成预检请求 JSON 字节 |
| --- | ---: |
| 157 | 506191 |
| 887 | 506223 |
| 2357 | 506323 |
| 6101 | 506950 |
| 12203 | 506611 |

真实代理、模型、推理级别、SSE 能力和用量留给正式请求验证。五进程 mock 验证了全局最大 3、每把 Key 最大 1、FIFO、异常退出释放槽、凭据与共享冷却、预算跨 Key 不重置及运行工件无测试密钥。 [执行与恢复入口](README.md) · [部署说明](DEPLOYMENT.md)

## 新增文件

- `a7_gpu/backend.py`、`cuda_runtime.py`、`seed_runner.py`、`scheduler.py` 和 `metrics.py` 提供训练资源、设备适配、单 seed 闭环、多 seed 协调及评价统计；同目录另有 package marker。
- `a7_gpu/formal_contract.py`、`credentials.py`、`key_pool.py` 和 `provider.py` 实现单次正式授权、安全 Key 读取、动态槽限额与供应商传输。
- `deployment.py`、`requirements_provider.txt`、正式配置、授权模板及 Key 池例子提供部署来源和配置。
- `tests/test_backend.py`、`test_deployment.py`、`test_formal_contract.py`、`test_key_pool.py`、`test_key_pool_processes.py`、`test_metrics.py` 和 `test_seed_runner.py` 覆盖新增正式路径。
- `README.md`、`DEPLOYMENT.md` 和本报告提供正式运行、恢复、部署和验证说明。原 V3 仅更新 `pure_llm_v3/transport_manager.py` 的物理租约与有界人工对账接入，并重新生成原离线工件校验清单。既有数值方法文件与任务没有改动。
