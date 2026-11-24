# 离线验证报告

执行环境为当前工作区，Python 3.12 系统环境和 Conda fish 的既有依赖按需使用。无新付费 API 请求、新标签揭示或正式模型训练。

```sh
/Users/fish/miniforge3/envs/fish/bin/python -m pytest -q studies/active_learning/pure_llm_v3_20261124/tests
python3 -m compileall -q studies/active_learning/pure_llm_v3_20261124/pure_llm_v3
python3 -m studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.offline_examples \
  --out /tmp/pure_llm_v3_release_examples
PYTHONDONTWRITEBYTECODE=1 /Users/fish/miniforge3/envs/fish/bin/python -B \
  studies/active_learning/pure_llm_v3_20261124/audit_history.py
python3 -m studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner \
  stage --seed 157 --round 0 --arm A7 --out /tmp/pure_llm_v3_release_stage0
python3 -m studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner \
  stage --seed 157 --round 1 --arm A7 --out /tmp/pure_llm_v3_release_stage1
python3 -m studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner \
  stage --seed 157 --round 6 --arm A6 --out /tmp/pure_llm_v3_release_stage6
```

| 验证 | 结果 |
| --- | --- |
| Pytest | 34 通过（18.55 s）。依赖发出 118 条第三方兼容性或弃用警告。 |
| Python 编译 | 通过。 |
| 冻结状态 | seed 157 L333/L365/L525 均完整构造，候选数 2997/2965/2805。拟议的全部 30 个条件生成 30 个唯一请求身份，每个 L/U 合计 3330 行。 |
| 模拟完整请求 | 初轮和后轮使用已可见的测量。一次 31 合法加 1 非法的提案由模型模拟补丁修复并冻结 32 个 ID。再次回放复用冻结结果。 |
| 历史审计 | 611 个完整目录批次和 6 个另列早期 shortlist Hybrid 决策，1,213 份完整回答，528,855 条归档记录 SHA 重算。 |
| 未来轨迹预检 | CPU 图与 split 合同在 label-store 构造前通过带拦截的独立复核。合成 B32 反馈可构造 2965 C、365 O、32 条近期反馈；mock 冻结不能进入未来标签步骤。 |
| CUDA 与 provider | 代码路径和既有资格收据完成静态核对。新真实代理能力、GPU 实际拟合和 CPU/CUDA 输出一致性待下一阶段。 |

测试覆盖角色迁移、数据精度、同结构多 CAS、缺失及等级纯度、完整条件复测、预测/反馈字段隔离、候选重复、未提供 ID、错绑、不同物理条件、正反证、版本身份和锁定保真。传输测试覆盖 429、403、timeout、SSE 缺完成事件、空响应、模型不符、started/terminal 恢复、重复调用和逻辑计数。测试重放的每个 provider 对象均由本地 fixture 提供。

按实际样例的紧凑 Responses payload 计算，初轮 A0/A2/A3/A4/A5/A7 为 904966/1390957/1276867/873925/844225/846971 字节，后轮 A7 为 828435 字节。同字段 A3→A4 的初轮压缩约 31.6%。完整数据见 [序列对比](examples/serialization_comparison.json)。 `lexical_proxy_units` 是可复验的词法代理计数，数字按最多三位、ASCII 字母按最多四个、其余非空字符逐个计数。它没有模型 tokenizer 校准，不用于上限或价格判断。本机的 tiktoken 包存在，o200k_base 词表未缓存且 DNS 不通，精确 tokenizer 估算列保留 null。词表可用时示例命令支持 `--tokenize`。历史 provider usage 单独见历史审计。

模拟验证证明软件可拒绝不合格输出并保留有效选择。真实首答合法率、科学收益、成本和速率仍需实验估计。
