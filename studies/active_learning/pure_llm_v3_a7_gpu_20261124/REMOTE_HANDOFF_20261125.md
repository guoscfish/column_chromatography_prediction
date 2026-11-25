# 2026-11-25 A7 正式运行交接

## 实际部署

11 月 25 日的资格部署将 SHA256 为 `6235a3df8f3c447111c927552b1e1833c877431ee12a5bba725f82e87f109525` 的 release 安装到 GPU 平台独立目录。此 release 来自离线基线 `b295939a2f6ec0bd90ed14e5c9a9b92aae0909d7` 的训练代码、公用资源和 V3 科学协议，加上正式 A7 代码与一份固定 SHA 的传输模块。原数值工作区与 GPU 队列未改动。重新构建应以新包打印的 SHA 为准。 [部署方法与输入清单](DEPLOYMENT.md)

11 月 25 日 09:40 的现场探针显示 Random、Uncertainty+Diversity Hybrid、CW-LCMD 分别五条正式任务成功，两张 RTX 3090 各有 24126 MiB 可用显存，当时无 GPU 计算进程。五份 Random L333 CUDA 锚点均在持久盘。SDK 2.28.0 已安装于独立 overlay，Python 3.12.3 导入与 Responses 接口检查成功。共享数值环境仍使用原包。运行前重新检查资源，因为占用和实例寿命会变动。

## 启动资格

实机资格使用五份通过合同、split、checkpoint 和张量摘要核验的既有 CUDA L333 模型。重新准备公共 scrubbed graph、固定 L333 标度并在 RTX 3090 上计算 U2997 与 validation 推理。五个 seed 都得到 333 条已有观测、2997 条候选、416/417 条隔离的 validation/test 角色及空 M2 科学记忆。模型卡保持 Flow/Purity/CAS/name。正式入口从独立 L333 开始。

| seed | 已揭示 L / 当前 U | 完整 A7 请求字节 |
| --- | --- | ---: |
| 157 | 333 / 2997 | 843825 |
| 887 | 333 / 2997 | 807705 |
| 2357 | 333 / 2997 | 850519 |
| 6101 | 333 / 2997 | 850867 |
| 12203 | 333 / 2997 | 850767 |

70 项本地测试通过，覆盖旧版离线协议、真实资源结构、mock 多轮、模型锁、恢复和动态 Key 池。五进程测试实证同一运行最多三笔物理请求、每把 Key 至多一笔，进程中断释放槽并保留待对账状态。CUDA 和模型流式协议分别完成实机推理与 SDK 导入检查。第三方实际响应身份、ultra 和全上下文兼容性须由正式第一个 round 0 验证。 [本地测试与命令](OFFLINE_TEST_REPORT.md)

## 在平台直接使用的命令

先从本机进入平台。

```bash
ssh -t eit-hpc
```

以下变量针对当时安装的 release，在远端 Bash 会话设置。这套部署和运行参数生成的合同 SHA256 为 `3adec146134e47b237c40ea3b97e51ef4b9486e3007997be67e11bbadf16d5a2`。更改部署、运行目录、GPU、锚点或正式配置后，按 `contract` 的实际输出核对并授权。三把 Key 可在暂停后调整，始终受批准并发上限约束。

```bash
remote=/1/column_chromatography_prediction
release="$remote/purellm_a7/releases/6235a3df8f3c447111c927552b1e1833c877431ee12a5bba725f82e87f109525"
python="$remote/environment/venv/bin/python"
export PYTHONPATH="$remote/purellm_a7/provider/openai-2.28.0"
study=studies/active_learning/pure_llm_v3_a7_gpu_20261124
module=studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.scheduler
suite="$remote/formal_cuda/row4g_b32_20261124"
run="$remote/purellm_a7/runs/a7-gpt62-ultra-v1"
keydir="$HOME/.config/purellm"
keyconf="$keydir/key_pool.json"
cd "$release" || exit
common=(--run-dir "$run" --gpus 0,1 --gpu-lease-root "$suite" \
  --anchor-root "$suite/trajectories" --key-pool-config "$keyconf")

# 这两个检查不发送模型请求。
"$python" -B -m "$module" contract "${common[@]}"
"$python" -B -m "$module" key-status --key-pool-config "$keyconf"
```

在同一远端 Bash 会话录入三把 Key。终端静默输入，已有文件保留。若服务商批准的共享并发低于 3，将私密 JSON 内的全局上限调低。

```bash
umask 077
install -d -m 700 "$keydir" "$keydir/keys"
if [ ! -e "$keyconf" ]; then
  install -m 600 "$study/key_pool.example.json" "$keyconf"
fi
for id in key_01 key_02 key_03; do
  if [ ! -e "$keydir/keys/$id" ]; then
    IFS= read -r -s -p "Provider key for $id: " purellm_value
    printf '\n'
    if [ -n "$purellm_value" ]; then
      (set -C; printf '%s\n' "$purellm_value" > "$keydir/keys/$id")
    fi
    unset purellm_value
  fi
done
"$python" -B -m "$module" key-status --key-pool-config "$keyconf"
```

首次正式运行需由用户明确批准。使用独立私密授权 JSON，按 [空白模板](formal_authorization.template.json) 填入上述准确合同 SHA、批准人和授权依据，分别允许付费 API、冻结后揭示和 CUDA 训练。获批后执行下列启动命令。

```bash
: "${A7_AUTHORIZATION:?须指向已批准的私密授权 JSON}"
umask 077
mkdir -p "$run"
nohup "$python" -B -u -m "$module" run "${common[@]}" \
  --authorization "$A7_AUTHORIZATION" >> "$run/coordinator.log" 2>&1 </dev/null &
```

首个真实请求属于 seed 157 正式 round 0。接受并冻结后只用该批 32 点做第一次 GPU 拟合。通过 L365 状态与记忆检查后，其余四个 seed 自动放行。每条轨迹到 L1485 停止。实际代理的模型 ID、`ultra`、完整上下文和账号限额须在正式请求中确认。更换某把 Key 不扩张单轮预算。五 seed 的总预算上限为 180 次首请求、540 次逻辑请求、1080 次物理尝试。

```bash
"$python" -B -m "$module" status --run-dir "$run"
tail -n 40 "$run/coordinator.log"
"$python" -B -m "$module" pause --run-dir "$run"

# 等旧协调器与 worker 结束后，使用同一合同恢复。
nohup "$python" -B -u -m "$module" resume "${common[@]}" \
  --authorization "$A7_AUTHORIZATION" >> "$run/coordinator.log" 2>&1 </dev/null &
```

状态或账单未知的请求需先按原 request ID 和 attempt ID 对账。身份漂移及真实能力不匹配直接停止。恢复文件、补充证据格式、耗时和指标口径见 [完整运行说明](README.md)。

## 授权前检查记录

三把 Key 已在正式任务环境通过检查，私密文件权限为 0600。运行入口默认读取 `~/.config/purellm/key_pool.json`。使用该配置的无联网测试中，三个 worker 各占不同 Key，另外两个等待，随后五个 seed 全部获调度且队列清空。资格检查阶段的付费 API 请求、新候选标签揭示、正式模型训练和数值作业变更均为 0。正式进度可用上方 `status` 命令查看。
