# A7 的 Mac 请求派发和 GPFS 训练

A7/M2 的实验协议继续使用五个 seed、B32 和 L333→L1485。计算平台读取 4g 数据、构造完整请求、校验 LLM 科学计划和局部修复、冻结合法批次，再揭示该批实验并用 CUDA 更新 QGeoGNN。Mac 用带授权绑定的完整请求调用 Token4Research，并把完整响应交回平台。模型固定为 `gpt-6.2-sol`，推理等级由各运行的批准合同固定。Mac 需要持续连通 GPU 平台的 SSH 和 Token4Research。

## v3 正式合同

V3 使用独立发布包和 GPFS 运行目录。启动所用的模型、推理等级和输出上限以 [v3 执行配置](execution_gpt62_a7_v3.json) 为准。旧 v1 和 v2 继续暂停并保留原合同和错误回执。V2 的首笔真实请求在推理参数验证阶段被拒绝，尚未产生完整回答、冻结批次、标签揭示或后续训练。V3 合同绑定旧失败证据和五份已复核的公共 CUDA L333 检查点。

2026-11-25 的独立流式短请求收到 HTTP 200 和完成响应。Token4Research 报告模型 `gpt-6.2-sol` 及 `reasoning.effort=max`。正式运行的长上下文和选择结果在 seed 157 的 round 0 核验。服务商请求 ID 和用量留在 Mac 私密能力验证回执。

在 GPU 新 release 根目录使用下列命令生成合同。`A7_GPU_RUN` 为新独立运行的绝对路径，`A7_AUTHORIZATION` 指向经用户批准的文件。将下面的 `common` 参数同时用于 `contract`、`run` 和 `resume`。Mac 派发器也传入其私密运行目录中同一份 `execution_gpt62_a7_v3.json` 的绝对路径作为 `--config`。

```bash
remote=/1/column_chromatography_prediction
suite="$remote/formal_cuda/row4g_b32_20261124"
study=studies/active_learning/pure_llm_v3_a7_gpu_20261124
module=studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.scheduler
gpu_python="$remote/environment/venv/bin/python"
common=(--run-dir "$A7_GPU_RUN" --gpus 0,1 --gpu-lease-root "$suite"
  --anchor-root "$suite/trajectories" --api-execution mac_ssh_dispatch_v1
  --predecessor-run "$remote/purellm_a7/runs/a7-gpt62-ultra-ssh-v2"
  --config "$PWD/$study/execution_gpt62_a7_v3.json")
"$gpu_python" -B -m "$module" contract "${common[@]}"
```

新 SHA 获得明确批准后按 [授权模板](formal_authorization.template.json) 写入私密授权，启动 GPU Coordinator 和 Mac Dispatcher。短请求校验保存在 Mac 的独立私密目录。首个完整科学请求仍计入 seed 157 的正式 round 0，L365 核验通过后开放其余四个 seed。

## v2 启动边界与通用恢复机制

旧云端直连运行 `a7-gpt62-ultra-v1` 仍保留为独立轨迹。Seed 157 从 L333 发起的两次物理尝试均记录连接失败，现有记录没有完整模型回答、冻结批次或后续拟合，服务商账单未独立核验。切换到 Mac 中转需要新发布包、新运行目录和新的准确合同授权。新的轨迹从同样的五份已核验 CUDA L333 初始检查点开始。首个 seed 157 的真实 round 0 直接计入该轨迹。以同一批 32 点完成 L365 并核验下一轮状态后，其余 seed 放行。 [已有运行和模型入口](REMOTE_HANDOFF_20261125.md)

## 凭据和派发

Mac 的 provider 凭据保存在 `~/.config/purellm`。使用当前 Mac 用户拥有的目录权限 `0700` 与文件权限 `0600`。查看或制作配置时只显示逻辑 Key 标识、权限和可用性。Mac 同时持有三把获准使用的凭据。每把 Key 最多占用一个物理请求槽，整体并发与退避仍受同一 provider 配额约束。跨主机运输使用已验证主机密钥的 SSH，收据通过 SHA256 绑定数据和授权身份。

从 Mac 的 checkout 根目录运行以下命令，已有配置和凭据文件保留。保持终端静默输入。

```bash
study=studies/active_learning/pure_llm_v3_a7_gpu_20261124
keydir="$HOME/.config/purellm"
umask 077
install -d -m 700 "$keydir" "$keydir/keys"
if [ ! -e "$keydir/key_pool.json" ]; then
  install -m 600 "$study/key_pool.example.json" "$keydir/key_pool.json"
fi
for id in key_01 key_02 key_03; do
  file="$keydir/keys/$id"
  if [ ! -e "$file" ]; then
    printf 'Provider key for %s: ' "$id" >&2
    IFS= read -r -s purellm_value
    printf '\n'
    if [ -n "$purellm_value" ]; then
      (set -C; printf '%s\n' "$purellm_value" > "$file")
    fi
    unset purellm_value
  fi
done
```

Key 池路径可用 `PURE_LLM_KEY_POOL_CONFIG` 指定。并发上限取决于 provider 实际批准的共享额度。更换失效 Key 时先暂停派发，待运行中的请求到达已记录的终态，再更新凭据并恢复。 [Key 池模板](key_pool.example.json)

SSH 传输携带 GPU 侧生成的原请求及尝试身份。GPFS 将该运行和每个尝试绑定到同一个 Mac 私有 spool 标识。Mac 将完整的服务商终态 JSON、原始回答文本和用量落盘后才回传 GPFS。回传中断时，使用同一个 spool 恢复，派发器转发原完成收据。若完整终态文件已落盘但正式回执还没封存，派发器会从同一份原始返回重建回执。其他 spool 无法重新领取或重发该尝试。若 provider 请求已开始而终态未知，暂停该尝试并按原请求和尝试 ID 对账。GPU 在等候 Mac 时保留当前轮 L/U 和尝试身份，并释放 GPU 槽给训练工作。发生 seed 或 checkpoint 错绑、消息篡改、模型身份不符、认证或权限故障时进入对应的失败或等待状态。

## 服务商流中途结束

Mac 侧在收到 HTTP 回执和流式生命周期事件时记录进度，包括可用的服务商请求 ID 和响应 ID。进度记录留在当前尝试的私密 spool。收到有效的 `response.completed` 后立刻封存终态，已经完整的结果可从 Mac 收据恢复。完整终态之前发生断流时，保留诊断记录和尝试身份，按执行状态未知对账。流事件中的部分答案只用于传输，科学选择仍以完整回执和 GPU 校验为准。

使用已有响应 ID 查询属于读取原状态。只有服务商支持且能够访问原结果时，才能用取回的完整响应走 `import_completed`。2026-11-25 的合成长流诊断复现了 HTTP 200 后响应体协议错误，随后读同一个响应 ID 返回 404。按服务商日志确认原请求状态时，优先提供已保存的请求 ID、响应 ID 和出错时间。服务商切换、启用后台模式或改动数据保留需要对应的新合同。 [OpenAI 响应查询接口](https://developers.openai.com/api/reference/resources/responses/methods/retrieve) · [后台模式与存储](https://developers.openai.com/api/docs/guides/background)

## 运行检查

- 对照旧云端运行记录，核实最后一个合法状态和批次边界。新运行位于独立 GPFS 路径，与数值基线共用同一物理 GPU 锁。
- 核对实际部署的发布包、数据和源代码 SHA。确认五份固定 L333 锚点、完整 U2997 预测、空初始科学记忆及模型协议。
- 核查 Mac 私有 Key 池的所有者及文件权限。凭据值留在私有读写路径。
- 先用明确标记的 mock 工作和响应核对双向传输、摘要校验与断点重投递。完成测试后，正式运行仍取决于新合同授权。
- 同时监看三处进度。Mac 显示待派发、正在执行、待回传与需对账。GPU 侧显示准确的轮次和是否在等候响应。正式输出中单列 GPU 物理尝试预留、Mac 实际发出、逻辑答复、合法冻结和训练完成。仅有 GPU 预留的等待项不代表已经调用服务商。

## Mac 派发入口

在 Mac 上从经核对的同版 checkout 根目录运行。Mac Python 需要安装锁定的 OpenAI SDK。选择一个单独的虚拟环境，并用该 release 的依赖文件安装。保留 spool 路径，使进程、SSH 或 Mac 重启后能读取同一份完成收据。

```bash
study=studies/active_learning/pure_llm_v3_a7_gpu_20261124
base=studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu
remote=/1/column_chromatography_prediction
gpu_run="$remote/purellm_a7/runs/a7-gpt62-ultra-ssh-v2"
keyconf="$HOME/.config/purellm/key_pool.json"
mac_python="${A7_MAC_PYTHON:?指定已安装固定 SDK 的 Mac Python}"
contract="${A7_CONTRACT_SHA256:?填写已核验并批准的新合同 SHA256}"
spool="$HOME/.config/purellm/spool/$contract"
umask 077
install -d -m 700 "$spool"

"$mac_python" -m pip install -r "$study/requirements_provider.txt"
"$mac_python" -B -m "$base.scheduler" key-status --key-pool-config "$keyconf"
dispatch=(--spool-dir "$spool" --ssh-host eit-hpc --remote-run-dir "$gpu_run"
  --remote-root "${A7_REMOTE_RELEASE:?填写 GPU 侧新 release 的绝对路径}"
  --contract-sha "$contract")

# GPU 协调器已发布请求时，仅核查并取回可用的工作。
"$mac_python" -B -m "$base.dispatcher" stage "${dispatch[@]}"
"$mac_python" -B -m "$base.dispatcher" status --spool-dir "$spool"
```

以下命令仅在用户批准这一新合同之后执行。保持 Mac 网络和 SSH 可达。 `caffeinate -i` 在程序存活时抑制空闲睡眠。

```bash
nohup /usr/bin/caffeinate -i "$mac_python" -B -u -m "$base.dispatcher" run \
  "${dispatch[@]}" --key-pool-config "$keyconf" --execute-paid --watch \
  >> "$spool/dispatcher.log" 2>&1 </dev/null &
dispatcher_pid=$!
ps -p "$dispatcher_pid" -o pid=,state=
"$mac_python" -B -m "$base.dispatcher" status --spool-dir "$spool"
```

派发器报告任务、已封存结果、已回传结果和终态待确认计数。`status` 中的 `key_pool` 提供每个逻辑 Key 的可用、占用、冷却或禁用状态，以及请求、完成、失败和已知用量。Mac 到 GPU 失联时保留其状态。修复连接后用完全相同的 spool 路径和合同重新启动。先查 GPU `status` 和 Mac `status`，确认是否有本地已封存待回传响应或上游执行状态未知的尝试。需要暂停时先在 GPU 执行 `pause`，等待已发出的调用留下响应或可对账的终态，再结束本地进程。所有 seed 达到 L1485 后可结束本地监听。保留本地 spool 与 GPU 运行记录。

## 在 GPU 平台核对新合同

从独立 release 根目录运行。以下路径中的 `A7_RELEASE` 应来自安装收据，`A7_AUTHORIZATION` 指向用户另行批准的私密授权。保留旧轨迹的暂停标记与收据。不同版本的合同 SHA 不可互相授权。

从本说明所在提交构建并将发布包安装到 GPFS 的 `purellm_a7/releases/<发布包 SHA256>`。按部署说明的构建与安装步骤执行，核对安装器返回的 SHA。Mac checkout 使用同一提交。 [构建与安装命令](DEPLOYMENT.md#构建与安装)

```bash
remote=/1/column_chromatography_prediction
suite="$remote/formal_cuda/row4g_b32_20261124"
old="$remote/purellm_a7/runs/a7-gpt62-ultra-v1"
run="$remote/purellm_a7/runs/a7-gpt62-ultra-ssh-v2"
gpu_python="$remote/environment/venv/bin/python"
base=studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu
cd "${A7_RELEASE:?填写新 release 的绝对路径}" || exit
common=(--run-dir "$run" --gpus 0,1 --gpu-lease-root "$suite"
  --anchor-root "$suite/trajectories" --api-execution mac_ssh_dispatch_v1
  --predecessor-run "$old")

"$gpu_python" -B -m "$base.scheduler" contract "${common[@]}"
findmnt -T "$remote" -n -o TARGET,FSTYPE
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
```

对照新合同中的 GPU、锚点、当前运行目录、旧运行证据和源码清单。前序运行的摘要覆盖两次失败尝试和暂停标记。用户明确批准新 SHA 后，在源码外的私密路径复制空白模板，填写该 SHA、批准人与授权依据，并按批准范围开启 API、冻结后揭示和 CUDA 训练。核查通过后，用下列命令启动。GPU 协调器在发布首个请求后等待 Mac。 [空白授权模板](formal_authorization.template.json)

```bash
: "${A7_AUTHORIZATION:?填写正式批准的私密授权文件}"
umask 077
mkdir -p "$run"
nohup "$gpu_python" -B -u -m "$base.scheduler" run "${common[@]}" \
  --authorization "$A7_AUTHORIZATION" >> "$run/coordinator.log" 2>&1 </dev/null &
```

GPFS 保留已发布工作、已接收响应、GPU worker 进度、选择冻结、验证指标与模型。GPU worker 等待同一 Mac 尝试的时限为 30 分钟，超时会显示 `bridge_pending`，保留原尝试供恢复。新容器挂载同一持久盘后，以相同的 release 和同一个批准合同恢复。Mac 私有 spool 保持原位置和身份，不复制为第二个活跃派发器。先用 `status` 核对状态，暂停时等现有 worker 自行停在合法边界，再用 `resume`。保留共享 GPU 锁文件及原工作目录。

新容器使用相同基础镜像、GPU 驱动和绝对 GPFS 挂载路径。先核对原 venv 及两张显卡可用，再以已有参数重新生成合同并与原批准 SHA 比较。分布式模式的 GPU 端只需要训练环境。SDK 和 Key 池均由 Mac 管理。容器如更新了 SSH 主机密钥，先通过平台确认新指纹并更新本机 `known_hosts`。

```bash
"$gpu_python" -B -m "$base.scheduler" status --run-dir "$run"
"$gpu_python" -B -m "$base.scheduler" pause --run-dir "$run"

# 协调器和各 worker 均退出后，再恢复同一正式版本。
nohup "$gpu_python" -B -u -m "$base.scheduler" resume "${common[@]}" \
  --authorization "$A7_AUTHORIZATION" >> "$run/coordinator.log" 2>&1 </dev/null &
```

若 Mac 显示已发出请求而终态待确认，暂停并确认 GPU worker 和 Mac 派发器均已退出，保留两侧的原始请求和尝试。根据服务商的独立证据，操作者只在确认未执行或取得可核验完整响应后提交对账。对账 JSON 绑定原 `request_id`、`attempt_id`、`authorized_by`、`evidence` 和 `decision`。 `confirmed_not_processed` 仍受原逻辑请求的剩余尝试预算约束。 `import_completed` 使用原返回的完整响应。 [对账字段与响应文件](README.md#恢复和评价)

```bash
"$gpu_python" -B -m "$base.scheduler" reconcile "${common[@]}" \
  --authorization "$A7_AUTHORIZATION" --seed "$RECONCILE_SEED" --round "$ROUND_INDEX" \
  --logical-index "$LOGICAL_INDEX" --attempt "$ATTEMPT_INDEX" --reconciliation "$RECONCILIATION_JSON"
# 导入完整响应时，同一命令再加 --recovered-response "$VERIFIED_RESPONSE_JSON"。
```

### 三 seed pilot 的单次补跑授权

`a7-gpt62-astra-xhigh-ssh-v4-pilot-3x10` 的补跑附录仅适用于 seed 157 的 round 0 logical 0、seed 887 的 round 3 logical 1 和 seed 2357 的 round 0 logical 0。附录锁定原合同、独立发布的执行源码，以及三笔原请求的终态和 GPFS 回执。原执行和计费状态继续记为未知。每笔只开放原预算内尚未使用的最后一个物理尝试。Mac 继续使用原 Dispatcher、发布包和 spool。

恢复时先确认旧协调器已退出，再从原运行目录执行 `recovery/resume-authorized-replay.sh`。该入口校验原发布包、附录 SHA 和补跑源码。每个 `attempt_01.replay_link.json` 保留新旧物理尝试的关联。中断时继续同一个已预留尝试。补跑耗尽后按原故障停止规则处理。

### 改进提示后的 pilot 延续

原版三 seed pilot 保留暂停状态及完整回执。seed 887 第四轮从已验证的 L429 生成一条独立新请求。新系统提示只追加 `condition_anchor` 格式约束。首轮请求的用户数据逐字节复用原始候选包，消息中没有旧第四轮首答或纠错内容。旧提示已取得并通过验证的首轮回执可直接用于冻结和训练。

新版后续轨迹从各 seed 已验证的持久化 checkpoint 独立启动，在总计第十轮停止。每条轨迹有独立运行目录、授权和 Mac spool，共用原三 Key 状态数据库和单卡 GPU lease。操作入口位于各运行目录的 `operator/launch.sh`。恢复时等待此前协调器退出，执行该脚本的 `resume` 完成原尝试的断点恢复。Mac 使用该 spool 的原 launchd 服务恢复传输。

### L1485 接续与有界传输补跑

现有三个 seed 达到各自的 L653 后独立启动 L1485 子运行。后继合同核验旧轨迹从首轮到 L653 的已封存反馈、训练工件及科学记忆，并记录剩余 26 轮的独立预算。系统提示沿用修订版，单卡训练继续使用共享 lease。Mac 在远端授权和真实 checkpoint 均通过核验后独立加载该 seed 的派发器，并使用原共享 Key Pool。

服务商的原响应查询不可用时，明确接受可能重复计费的授权可以为已封存的歧义传输失败开放剩余的一次物理尝试。保留原始请求、终态、回执及新旧尝试关联。正在等待的物理尝试保持身份，已保存的完整响应直接回传。共享限流、认证错误、模型异常、科学校验失败和尝试额度耗尽按原规则停止或等待。监督进程及接续器都尊重持久化暂停标记。

### 物理预算耗尽后的单次直连

经单独批准后，直连版本从已封存的 checkpoint 开始新合同。该合同绑定父运行的两次失败回执，逐字节复用系统提示及用户数据，保留模型、推理等级和完整科学校验。每个 seed 只允许一个新物理请求，硬截止为 3600 秒。独立 Mac 运行时禁用客户端代理读取，原共享 Key Pool 和 GPU lease 继续生效。已有完整本地回执时沿用原结果，启动失败的旧暂存记录保留。

### 已归档完整回答的定向引用修复

887 的独立定向修复合同绑定完整原回答、GPFS 回执、原 L461 checkpoint 和 32 个候选的顺序。只删除与规范 `reference_id` 完全相等且角色一致的冗余字段。模型只返回第 16 项的一个引用和一句理由，对照单位与方向由原始记录验证。原始修复回执与投影后的计划分别保留摘要。其余选点、锚点、引用及 M2 更新保留原内容。完整计划经原 A7 校验后才冻结标签并启动共享 GPU 训练。合同仅包含一次新物理请求。

### 从已封存断点继续到 L1485

三个独立续跑合同分别验证 157 和 2357 的 L365 与 887 的 L493。后续选择使用共同的新版科学提示。887 的定向纠错只作为完成轮次的训练与记忆来源。Mac 继续使用共享三 Key，GPU 使用原单卡 lease。执行采用直连，读取空闲上限为 1800 秒，硬截止为 3600 秒，每个逻辑请求最多两次物理尝试。等待、失败或拒绝的旧尝试保留在各自归档内。

## 离线复测

```bash
python -m pytest -q "$study/tests" studies/active_learning/pure_llm_v3_20261124/tests
```

合成测试核验请求原样往返、外部 seed 响应拒收、Mac 节点响应回传失败后原收据可重投递，以及 GPU 节点等待时不揭示标签和不占用 GPU。Mac Key 池复用已有五进程测试，覆盖三 Key 动态分配、等待队列、公平退出与中断释放。实机的 provider 模型版本、推理等级及超长上下文能力留给正式 seed 157 的首轮。

### Five-seed recovery with background Responses

The prepared cohort continues 157 and 2357 at L365, 887 at L493, and 6101 and 12203 at the original CUDA L333 anchors. Use `prepare_five_seed_boundaries.py` with the immutable V11 release, its runs parent, and a separate private GPFS preparation directory. It verifies old receipts, retains their failures, and runs only CUDA inference from the two original anchors. It writes `five_seed_boundary_draft.json` and the per-seed state and memory files. The remaining cohort starts with 173 selection rounds.

Use `run_prepared_five_seed.py contract` against that draft and a route policy. The policy binds the SDK, endpoint, model, timeouts, retrieval evidence, and `dispatch_prepared_background` as the Mac entry point. A policy with `qualification: null` only generates a draft. Before `run` or `resume`, the route needs a positive background acknowledgement and completed retrieval of the same ID, plus the exact formal authorization. A new long scientific request counts toward the cohort. Store provider-specific credentials only in the private Mac Key Pool configuration for that service. For the five-seed unified status command, use a private spool named by the contract SHA and pass its `key_pool` subdirectory as `--shared-key-pool-dir`.

`dispatch_prepared_background.py` reserves one POST per physical attempt. It saves the accepted response ID before polling, retrieves with the original Key, and can continue GETs while SSH is unavailable. It only uploads a validated terminal response. An unknown initial submission pauses new POSTs while preserving existing retrievals. A 429 follows the shared or credential scope and Retry-After from its receipt.

On November 25, the short Token4Research Astra/xhigh test returned a correct synchronous answer with `background:false`; a separate invocation using the same Key and response ID returned 404. This is a failed qualification for the tested background route. Its private ledger retains the actual response and sanitized GET receipt. A compatible provider credential and a successful same-ID retrieval are needed before launching the cohort.

### Serial IPv4 recovery on the existing Token4Research route

The opt-in `responses_stream_ipv4` mode signs the direct connection policy while preserving the original Responses SSE payload, Astra/xhigh and `store:false`. It keeps certificate and hostname verification, uses HTTP/1.1, and verifies the actual peer address family before accepting streamed output. An unauthenticated same-client check of the models endpoint runs before the POST. A route check that fails never submits model input.

The IPv4 cohort uses the prepared five-seed draft and the authenticated prior full direct Responses receipt. That receipt is historical evidence about the same provider and wire protocol. The new route has its own live gate. The coordinator initially starts 157 alone from L365 and verifies the accepted selection and sealed CUDA L397 transition before releasing the other four workers. The Mac dispatcher submits one physical request at a time through the original shared Key Pool. The GPFS bridge wait covers a serial queue and the existing hard request deadline. Ambiguous transport failures retain their receipts and stop without an automatic new paid POST.

Use the same `run_prepared_five_seed.py` contract and authorization entry point and the signed `dispatch_prepared_ipv4.py` Mac module. The cohort run name for the unified status command is `a7-gpt62-astra-xhigh-ipv4-v13`.

### Restore parallel dispatch after the IPv4 gate

The optional `dispatch_prepared_ipv4_parallel.py` launcher accepts a separate scheduling approval. It binds the existing run and authorization SHA, launcher SHA, and Mac dispatcher identity. The approved ceiling is three API workers under the original shared Key Pool. The launcher uses the same Mac spool, GPFS outbox, provider model lock, and request attempts. A schedule change grants zero additional attempts. The GPU coordinator still runs from the original release and uses its existing GPU lease.

Run the launcher with `--verify-only` to validate the approval, original release, and shared pool. Pause new bridge claims through the existing GPU scheduler. Keep the old Mac process working until every started request has a sealed result and delivery acknowledgement. Once the old workers have stopped, replace the Mac launchd entry with the addendum launcher using `--execute-paid`, then resume the same GPU run. Install the new launcher outside the original release and use that release on `PYTHONPATH`. Preserve the original dispatcher identity and spool when restarting either process.

The unified status command shows queue time, API elapsed time, first answer text, the last SSE event interval, and completion state. An HTTP 200 or an output text event can precede a missing response terminal. The original timeout and reconciliation rules still apply.

### One-attempt transport extension for seed 2357

The V13 seed 2357 L365 round exhausted both physical attempts without a complete response. The sibling run `a7-gpt62-astra-xhigh-ipv4-v13-2357-extension-01` verifies the unchanged L365 state and repeats the exact archived model input at most once under a separate explicit authorization. It binds both V13 timeout receipts and preserves their unresolved execution and billing status. Pause the V13 seed 2357 lane before using `run_v13_seed2357_extension.py run`. Check the separate contract with its `contract` action first.

Start `dispatch_prepared_ipv4_recovery_2357.py` with the extension contract SHA and its authorization SHA. It uses its own Mac spool and the original shared Key Pool. Recover using the same spool, authorization and `run_v13_seed2357_extension.py resume`. A complete response must pass the A7 validator before the normal training lease is acquired. The single extension ends at verified L397 and emits a boundary receipt for the remaining 34 planned rounds.

The sibling `a7-gpt62-astra-xhigh-ipv4-v13-2357-continuation-01` verifies that boundary receipt, the archived complete provider result, and the exact checkpoint and memory. Use `run_v13_seed2357_continuation.py` for the GPU entry and `dispatch_prepared_ipv4_continue_2357.py` for the Mac entry. It uses the original budget for rounds 2–35 and the existing shared Key Pool and GPU lease. Retain the pause marker on the original seed 2357 lane.

### Shorter one-attempt extensions for 6101 and 887

Both original turns have an unauthenticated preflight failure followed by one ambiguous model POST. Verify the exact archived request and both old receipts with `run_v13_lane_extension.py contract --seed 6101` or `--seed 887`, using the corresponding sibling `a7-gpt62-astra-xhigh-ipv4-v13-<seed>-extension-01` run. Each approved extension reconstructs the full original model input, permits one physical attempt, and uses a 900-second read-idle timeout and an 1800-second hard deadline. The 6000-second GPU bridge wait includes time spent waiting for a Key. A new response goes through the normal A7 validator and the one-GPU training lease.

Set `PYTHONHASHSEED=0` for contract generation, authorization checks, and execution. This reproduces the original 887 host serialization as well as its full outgoing payload. The 887 verifier can also bind both intermediate digests if an archive needs that compatibility check. Transport failure records remain in the attempt ledger. The Mac command is `dispatch_v13_lane_extension.py --seed <seed>` with the exact child contract and authorization hashes. Restart with the existing spool and use the GPU `resume` action for the same child.

After a verified extension fit, `run_v13_lane_continuation.py` and `dispatch_v13_lane_continuation.py` continue 6101 from L397 or 887 from L557. Both entries require the exact extension contract and authorization SHA. The continuation takes 34 or 29 unspent rounds respectively, retains the shorter API timeout policy, and shares the original Key Pool and single GPU lease.

### Detect inactivity at the decoded SSE layer

For new attempts with the signed SSE inactivity policy, the Mac starts a 900-second inactivity timer when response headers arrive. Initial lifecycle events and new decoded reasoning or output progress renew that timer. Heartbeat comments, repeated sequence frames, and empty deltas do not. A completed response is archived as soon as its valid terminal event arrives. Keep the original total deadline as a second limit.

`dispatch_with_sse_inactivity.py` binds the policy to the original approved V13 or 2357 continuation, existing Mac dispatcher identity, and exact approved provider and dispatcher file hashes. The GPU release, request payload and attempt budget remain bound to the original contract. Run its `--verify-only` mode first and drain the selected Mac spool before replacing the launchd entry. An event-idle timeout seals an ambiguous transport failure. Recover an existing complete receipt if one is available, or use only a remaining physical slot through the existing authorization procedure.

Unified status shows both the time since the latest decoded SSE and the time since the last event that renewed the inactivity timer. For an inconsistent response identity, the private progress receipt records both sanitized IDs and the request remains ineligible for selection.

The separate 157 one-attempt extension verifies the original L429 checkpoint, exact payload, and two previous physical slots. Its contract binds the 900-second SSE inactivity policy, a 3600-second total safety deadline and a 7800-second bridge wait. If A7 accepts the response and CUDA training seals L461, 32 originally planned rounds remain.

The 2357 R2 extension is limited to the L397 request whose first physical attempt timed out after POST and whose authorized retry stopped in the unauthenticated preflight. The original continuation seed remains paused. A separate approval binds those sealed receipts, the exact model input, and one new physical slot under the same SSE inactivity policy. Its completion receipt links the accepted response and L429 CUDA state to the original remaining 33 rounds.

### Token4Research response retrieval check

On November 26, a separately approved synthetic Astra/xhigh request sent both `store:true` and `background:true`. The completed response reported both as false. A fresh client using the original credential returned HTTP 404 for the same response ID shortly afterwards. This route and request configuration did not provide a retrievable result. The private qualification ledger holds the request and sanitized receipts. Formal runs retain their existing storage configuration and archive the complete terminal response on the Mac before SSH delivery.

### Single whole-response recovery for seed 887

The 887 R6 JSON extension binds one separately approved Token4Research request from the exact L525 checkpoint. Its new wire payload changes only `stream` to `false`. The original A7 messages, model, effort, output cap and `store:false` are preserved. The contract binds both earlier response-ID failures and their original archives.

The Mac accepts a single complete Responses JSON object with matching identity and usage, within a 30-minute hard deadline. It saves the original object and receipt before returning it to the existing GPU-side A7 validator. The one-attempt limit remains in force after an unknown execution outcome. Any automatic continuation past L557 requires a separately bound continuation contract.

The approved 887 whole-response attempt received HTTP 524 after 138.3 seconds. Its request body had already been sent and the provider execution state remains unknown. The failure was delivered to GPFS.

### Single WebSocket recovery for seed 6101

The 6101 R1 WebSocket extension pins the original L365 input and both earlier streaming failures. It uses one authenticated IPv4 Responses WebSocket per physical attempt with `websockets==15.0.1`. The request archive retains the exact original input and explicit HTTP stream flag. The provider frame uses the WebSocket protocol's implicit streaming with one `response.create` event and fresh connection.

Effective response progress has a 900-second inactivity threshold and a 3,600-second hard deadline. WebSocket control frames do not renew the application timer. A complete response must pass identity, status, usage, and original A7 validation before it can authorize a frozen batch. The provider's authenticated upgrade and ping/pong succeeded in a credential-scoped connectivity check without creating a model response.

### Bounded SSE restart of exhausted lanes

The shared recovery entry prepares individual contracts for 157 R3, 887 R6, 2357 R2, and 6101 R1. It verifies the original SSE jobs, latest failed recovery, authorized CUDA boundary, and unchanged scientific prompt and memory. The new contracts permit one original-input direct SSE attempt per seed and share the original three-Key scheduler plus GPU lease.

Each attempt preserves the 900-second meaningful-event inactivity threshold and the 3,600-second hard deadline. A legal completed response can freeze and train exactly one batch. Future rounds are available only after the new verified checkpoint is carried into a continuation contract.

### Resilient SSE continuations to L1485

The `run_v13_resilient_sse` and `dispatch_v13_resilient_sse` entries use a separately approved contract for each seed. The recovery round starts from its verified CUDA checkpoint and scientific memory, and permits only one model submission of the original logical request. Later rounds retain the original limit of three logical requests and two physical slots per logical request. The signed later-round replay rule can open an unused second slot for that same request after a verified ambiguous transport failure. It preserves the original receipt and checks for a saved completion first. A signed local validation rule may resolve an exact alias of an already observed row before freezing, with the original answer and normalization receipt retained.

Before the model submission, the Mac can use up to three IPv4 readiness windows, with up to two credential-free probes per window. When a transient probe failure occurs before POST, the Mac releases the Key lease and retains the same staged attempt for the next window. The SSE effective-event idle limit is 900 seconds and the hard deadline is 3,600 seconds. All five runs share the existing Key Pool and GPU lease.

From the staged Mac release, use the unified read-only view to see the signed run, current label count, IPv4 readiness window, actual dispatch, and delivery status.

```bash
python3 -B -m studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.prompt_pilot_status --watch 30
```

The per-seed Mac service uses `com.purellm.a7-ipv4-v13-<seed>-resilient-sse-01`, with its private spool named by the approved contract SHA.
