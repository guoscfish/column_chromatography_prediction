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

| 补丁 | 文件 | 基础 SHA-256 | 签署产物 SHA-256 |
|---|---|---|---|
| `887-r20-final-answer-whitespace.patch` | `transport_manager.py` | `25bb9cd757b1d14153bf1f758427d195a4a2299a9a9f840dd7619c783e76c739` | `2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c` |
| `2357-w19-response-headers-deadline.patch` | `provider.py` | `aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f` | `4b3f0f181401f03f3b54bc8a35861fa91a40a497c93a0162fbb7f65dd2ffa443` |
| `2357-w19-response-headers-deadline.patch` | `dispatcher.py` | `cde7ba080ba728b7600764d733cfab8a87bbe8e537840929bc1f47cd6c2cb3fc` | `073fe84d02496a4e6ca8187922df58399d87e40b062d0521a925716fe84d576b` |
| `887-r33-normal-boundary-budget.patch` | `seed_runner.py` | `19122c372294d5dd0f08d2ba4b1388b027ac1bacf65d84004c5eacf5b498e9f6` | `417f41255d53d7794abcdc98c0a9258141d01c076df845bfa877013c84de1792` |
