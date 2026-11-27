# Mac 运行状态和控制

`purellm-a7-status` 默认查看 v3 的两端进程、五个 seed、三把 Key 的编号、两张 GPU 和 API 回执。使用 `--version v1` 或 `--version v2` 查看已暂停的历史运行，使用 `--json` 获取结构化结果。

将两个脚本安装到 `~/.config/purellm/`，确保 Python 3.11 或更新版本可通过 `python3` 找到。在该目录创建权限为 `0600` 的 `purellm-a7-v3.json`，填写实机核验的合同 SHA 和发布包 SHA。

```json
{
  "contract_sha256": "填写已核验的 v3 合同 SHA256",
  "release_sha256": "填写已安装的 v3 发布包 SHA256",
  "run_basename": "a7-gpt62-max-ssh-v3",
  "reasoning_effort": "max"
}
```

`purellm-a7-control` 提供 `status`、`gpu-pause`、`gpu-resume`、`mac-stop` 和 `mac-resume`。控制仅适用于已获授权的 v3。首次正式启动使用 GPU 正式入口。Mac 开始派发前会检查匹配的正式授权、Coordinator 进程及暂停状态。 [分布式运行手册](../DISPATCHER.md#v3-正式合同)

## 正式发布包的运行时补丁

在独立发布工作树、且合同 `source_sha256` 明确绑定目标哈希时，才应用相应补丁。先核对下表的基础 SHA-256，在仓库根目录运行 `git apply --unidiff-zero --check <补丁路径>` 和 `git apply --unidiff-zero <补丁路径>`，再用 `shasum -a 256` 核对产物、以 `git diff --check` 检查改动。历史发布保留原源码。归档核验使用原合同对应的不可变发布包。

- [887 R20 消息阶段解析](overlays/887-r20-final-answer-whitespace.patch) 从有且仅有一个显式 `final_answer` 的回复中忽略仅含 ASCII 空白的无阶段占位消息，保留最终答案及归档原文。
- [2357 W19 响应头截止](overlays/2357-w19-response-headers-deadline.patch) 仅在签署参数 `response_headers_seconds=300` 的范围内，在请求体上传完成后开始计时。超时记为执行状态未知，按同一合同的对账及恢复规则处理。
- [887 R33 续跑首轮预算](overlays/887-r33-normal-boundary-budget.patch) 以 W11 的签署源码为底，从已核验的 L1389 接回普通 A7 校验与纠错流程。只在签署的三个收尾轮次使用每轮最多三次逻辑请求和每个逻辑最多两次提交。
- [6101 R9 续跑首轮预算](overlays/6101-r9-normal-boundary-budget.patch) 以 W28 的签署源码为底，从核验后的 L621 接回普通 A7 校验与纠错流程。R9 至 R35 共 27 轮，合同上限为 81 次逻辑请求和 162 次物理提交。
- [6101 R11 续跑首轮预算](overlays/6101-r11-normal-boundary-budget.patch) 以已签署的 R9 和 R10 归档恢复发布包为底，从核验后的 L685 继续普通 A7 校验与纠错。R11 至 R35 共 25 轮，合同上限为 75 次逻辑请求和 150 次物理提交。解析器复用上面的 887 R20 补丁。

| 补丁 | 文件 | 基础 SHA-256 | 签署产物 SHA-256 |
|---|---|---|---|
| `887-r20-final-answer-whitespace.patch` | `transport_manager.py` | `25bb9cd757b1d14153bf1f758427d195a4a2299a9a9f840dd7619c783e76c739` | `2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c` |
| `2357-w19-response-headers-deadline.patch` | `provider.py` | `aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f` | `4b3f0f181401f03f3b54bc8a35861fa91a40a497c93a0162fbb7f65dd2ffa443` |
| `2357-w19-response-headers-deadline.patch` | `dispatcher.py` | `cde7ba080ba728b7600764d733cfab8a87bbe8e537840929bc1f47cd6c2cb3fc` | `073fe84d02496a4e6ca8187922df58399d87e40b062d0521a925716fe84d576b` |
| `887-r33-normal-boundary-budget.patch` | `seed_runner.py` | `19122c372294d5dd0f08d2ba4b1388b027ac1bacf65d84004c5eacf5b498e9f6` | `417f41255d53d7794abcdc98c0a9258141d01c076df845bfa877013c84de1792` |
| `6101-r9-normal-boundary-budget.patch` | `seed_runner.py` | `d872883ca1d987d36b9610ab59f1a2957dc25a697750879f41839783dda147ff` | `193269dc7dab89e4e9c529338a130f224d331625b6646ed32e085ec97ca5005b` |
| `6101-r11-normal-boundary-budget.patch` | `seed_runner.py` | `193269dc7dab89e4e9c529338a130f224d331625b6646ed32e085ec97ca5005b` | `8368fc31377202dd4fb451a069cdfd28bcc88e08ca5ba44963cf187e18fc93e0` |

仓库共享的 `seed_runner.py` 支持 887 R33、6101 R9 和 6101 R11 的签署入口。各不可变发布包的文件摘要以其签署产物为准。复现 6101 R9 发布包时，使用 W28 原发布包中的基础文件，或从 Git 提取 `1793de14d76a57e5b1913e31204353f564835d33` 中对应的 `seed_runner.py`。在临时目录应用 6101 补丁并校验上表的签署产物。6101 的 R8/R9 入口继续核对各自发布绑定的源码摘要。


复现 6101 R11 时，从已签署的 R10 归档恢复发布包出发，应用 6101 R11 首轮预算补丁和 887 R20 消息阶段补丁，核对上表产物摘要，再加入两个 `v13_6101_r11_continuation.py` 入口。按照 `run_v13_6101_r11_continuation inspect-handoff` 的参数指定 R9 与 R10 的历史发布包及运行目录，重放 L685 的只读核验并生成完整 R11 输入摘要。将复核后的 JSON 交给同一入口的 `contract --handoff-pins`，以合同和授权校验结果启动对应的 Mac Dispatcher 与 GPU Coordinator。R10 历史归档仍由原发布包的解析器验证。

## Astra/xhigh 实时派发与已核验失败的恢复

Mac 统一查看命令

```bash
~/.config/purellm/purellm-a7-v5-status --live-dispatch --watch 30
```

科学进度取已验证的 CUDA 检查点。实时派发另行匹配 Mac 的 launchd 参数和 GPFS 上的正式合同、授权及请求状态。已经核实归属本次请求的 `response.failed` 会显示服务商错误码。报告记录了请求并发及失败时间分布。[传输诊断](transport_evidence_20261127.md)

对于已经发送、两端回执一致且响应别名匹配的 `upstream_error`，`run_v13_retry_successor` 与 `dispatch_v13_retry_successor` 都接受 `--reviewed-record` 和 `--reviewed-record-sha256`。记录使用项目的规范 JSON 摘要。Mac 副本放在 `~/.config/purellm/reviewed_recovery/<seed>-<两位运行段号>-<记录SHA>.json`，权限为 `0600`。远端合同前序字段绑定同一记录摘要，并在首次运行时将记录保存在正式运行目录。

先在远端用 `contract` 核对发布、父合同、原始请求和检查点。启动或恢复时，远端同时校验 `--authorization` 和 `--authorization-sha256`。Mac 开始派发前先运行 `--verify-only`，核对当前合同、授权和原始失败回执。每份新合同只为当前恢复轮保留一次物理提交，需满足原有 180 秒冷却。固定发布包可供后续逐份签署的记录复用，每条记录链最多追加 16 个运行段。状态未知的超时和匿名连通性检查失败继续使用各自的恢复流程。
