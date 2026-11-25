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
