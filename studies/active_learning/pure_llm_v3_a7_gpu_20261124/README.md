# PureLLM-V3 A7 正式 CUDA baseline

本研究入口固定 A7/M2、Token4Research `gpt-6.2-sol` / `ultra`、每轮 32 个 LLM 自主选择，五个 seed 从 L333 到 L1485。正式选择使用已发布的独立英文 Prompt、完整 4g 元数据、按分子分组的无损上下文、C/O/M 身份、宿主请求绑定、结构化证据、局部纠错和持久 M2 记忆。首轮记忆为空，无固定 Top-8 极端误差。每轮由 LLM 提出全部候选，宿主只做身份、证据、条件、授权、预算和状态核验。

CUDA 新入口可使用同 seed 的合格公共 L333 checkpoint，把原已冻结 L333 数据协议带入独立轨迹。若签署的合同选择它，也可从同一初始化重新拟合 L333。初始和后续模型都用数值方法同一训练参数、顺序、标度、损失和验证 checkpoint 规则。模型及标签数据的用途见 [CUDA 与部署核查](DEPLOYMENT.md)。旧历史入口仍可用作已保留的研究资源，A7 正式运行只需本研究入口。

| 项目 | 本次结果 |
| --- | --- |
| 历史运行工件依赖 | 独立部署路径只需 4g 原始与 canonical 数据、图缓存、E0 元数据及 scaler、冻结 Row splits、模型与已发布 V3 代码。合格公共数值 L333 模型是可选初始化来源。 |
| A7/M2 与预算 | 直接运行完整 A7。历史版本和消融不进入正式启动路径。五 seed 合计 180 个首请求，最多 540 个逻辑请求、1080 个物理尝试。该总上限来自每轮 3 个逻辑回答，每回答 2 次物理尝试。共用账户限制可提前停止。首个真实 round 0 的上限为 3 个逻辑回复、6 次物理尝试，全部计入正式轨迹。 |
| Provider 与 SDK | 已配置指定 URL、Sol 6.2、ultra、32768 输出、600 秒上限、Responses streaming、store=false，锁定 OpenAI SDK 2.28.0。真实模型标识、版本字段、ultra 和完整请求兼容性须用正式首轮核实。 |
| GPU 部署 | 独立部署包使用离线基线 `b295939a2` 的固定训练代码。数值远端源码收据记录 `b6b03b9e` 加 CUDA overlay，166 个共同模型源码文件逐字节一致。 |
| 11 月 25 日资格记录 | 数值三方法各 5 个 seed 已成功结束，当时两张 RTX 3090 空闲。五份 CUDA L333 锚点通过核验，独立 SDK overlay 与三槽凭据池通过无联网检查。实时状态通过下方 `status` 命令读取。 |
| 运行方式 | 一次正式授权覆盖五条轨迹。157 首轮实际选择、冻结、揭示、GPU 训练和下一轮构造完成后自动放行，其余各 seed 独立推进至 L1485 或停在可核查的失败位置。 |
| 离线部署资格阶段 | 付费 API、新候选标签揭示、正式训练和既有数值任务变更均为 0。 |

[正式配置](execution_gpt62_ultra_a7.json) · [发布与资格检查](DEPLOYMENT.md) · [离线测试](OFFLINE_TEST_REPORT.md)

## 在 GPU 平台配置 Key 池

下面在远端登录用户自己的 Bash 会话执行。进入 [已安装的 release](DEPLOYMENT.md) 根目录，复制模板路径及保守并发配置。Key 通过终端静默输入，每个文件只含一把 Key。现有文件保持原样。可以把实际 Key 文件放在平台的私密 Secret 路径，然后修改私密配置中的相对或绝对路径。

```bash
study=studies/active_learning/pure_llm_v3_a7_gpu_20261124
keydir="$HOME/.config/purellm"
umask 077
install -d -m 700 "$keydir" "$keydir/keys"
if [ ! -e "$keydir/key_pool.json" ]; then
  install -m 600 "$study/key_pool.example.json" "$keydir/key_pool.json"
fi
for slot in key_01 key_02 key_03; do
  file="$keydir/keys/$slot"
  if [ ! -e "$file" ]; then
    IFS= read -r -s -p "Provider key for $slot: " purellm_value
    printf '\n'
    if [ -n "$purellm_value" ]; then
      (set -C; printf '%s\n' "$purellm_value" > "$file")
    fi
    unset purellm_value
  fi
done
```

正式入口默认从 `~/.config/purellm/key_pool.json` 读取池，也支持 `--key-pool-config` 或 `PURE_LLM_KEY_POOL_CONFIG`。父进程传递配置和 Key 的逻辑标识；每个发布授权请求领取物理槽后，worker 才从相应私密文件向当次 SDK client 注入内容。Key 路径和模式核验可以在不打开 Key 文件时完成。动态分配不按 seed 固定，默认最多 3 个同时物理请求，每把 Key 至多 1 个。未知限流范围按服务商共享冷却处理。只有服务商明确指明 credential 范围时才单独冷却该 Key。HTTP401/403 隔离对应 Key 并停止该逻辑请求，其他 seed 可用剩余合法凭据继续。共享额度耗尽、版本漂移或已确认的模型/推理/上下文配置不支持会停止全池新请求。按账号和项目实际允许的并发设置池上限；未核验时可将配置降为 1。需要增减 Key 时先暂停并等 worker 退出，再编辑私密池配置和恢复；已有失败记录保留，替换失效凭据使用新的非敏感 ID。单 Key 兼容接口支持 `OPENAI_API_KEY`、`PURE_LLM_API_KEY_FILE` 和默认文件，但正式多 seed 入口通过池文件登记。 [Key 池模板](key_pool.example.json)

```bash
python=/1/column_chromatography_prediction/environment/venv/bin/python
module=studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.scheduler
keyconf="$HOME/.config/purellm/key_pool.json"
"$python" -B -m "$module" key-status --key-pool-config "$keyconf"
```

## 远端资格和正式命令

先依据 [只读平台探针](DEPLOYMENT.md) 的当前证据选择可用卡，在已安装 release 根目录执行。CUDA 预检会领取该卡的共享锁，核对合格 L333 anchor 后仅运行 CUDA 推理。没有合格 anchor 时，当前阶段可运行 `backend` 的公共资源准备或 `preflight --synthetic`，正式 L333 重训留到明确授权后。合成预检使用同一组真实列和角色，预测值为 mock。

```bash
remote=/1/column_chromatography_prediction
python="$remote/environment/venv/bin/python"
suite="$remote/formal_cuda/row4g_b32_20261124"
module=studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.scheduler
run="$remote/purellm_a7/runs/a7-gpt62-ultra-v1"
keyconf="$HOME/.config/purellm/key_pool.json"
: "${GPU_SLOTS:?先从平台探针选择可用物理卡编号，例如 0 或 0,1}"
common=(--run-dir "$run" --asset-root "$PWD" --gpus "$GPU_SLOTS"
  --gpu-lease-root "$suite" --anchor-root "$suite/trajectories" --key-pool-config "$keyconf")

"$python" -B -m "$module" key-status --key-pool-config "$keyconf"
"$python" -B -m "$module" preflight "${common[@]}"
"$python" -B -m "$module" contract "${common[@]}"
```

若采用授权后的从头训练 L333，删去 `common` 中的 `--anchor-root` 及其值，使用改变后的合同 SHA，并在正式授权前改用合成或资源预检。首次真实请求计入 seed 157 的正式 round 0。同次合格选点用于第一次正式 GPU 拟合，所学状态验证后自动继续，失败则暂停并保留证据。

复制 [一次正式授权模板](formal_authorization.template.json) 到源码外的私密路径。由授权人核对 `contract` 输出和 scope，填入完整合同 SHA、授权人和授权凭证。SHA 同时绑定唯一运行目录、GPU 资源、完整代码与数据；分别授权付费调用、冻结后标签揭示和 CUDA 训练。模板默认均关闭。这一步需要用户的正式明确授权。

以下命令仅在配置及正式授权完成后执行。使用 GPFS 保留日志，`nohup` 可让任务在 SSH 断开后继续。

```bash
: "${A7_AUTHORIZATION:?设置为用户实际批准的私密授权 JSON 路径}"
umask 077
mkdir -p "$run"
nohup "$python" -B -u -m "$module" run "${common[@]}" \
  --authorization "$A7_AUTHORIZATION" >> "$run/coordinator.log" 2>&1 </dev/null &

"$python" -B -m "$module" status --run-dir "$run"
tail -n 40 "$run/coordinator.log"
"$python" -B -m "$module" pause --run-dir "$run"

# 旧协调器退出后，按原 release、anchor、物理 GPU 和授权恢复。
nohup "$python" -B -u -m "$module" resume "${common[@]}" \
  --authorization "$A7_AUTHORIZATION" >> "$run/coordinator.log" 2>&1 </dev/null &
```

请求已开始但服务端执行未知时，保留原始请求身份并暂停。确认未处理或已取得可核对的完整返回后，由操作者准备绑定 `request_id` 和 `attempt_id` 的对账文件。选择 `confirmed_not_processed` 或 `import_completed`，同时记录 `authorized_by`、`evidence`，后者再提供已知完整返回。记录会校验对应的 seed、轮、逻辑序号和物理尝试序号。对已确认未执行的 401、403、429 或本地凭据错误也可显式对账，但必须还留有物理尝试预算。至少有一把合法可用 Key 才能继续。对账须在暂停并确认旧 worker 退出后执行。完整恢复响应可附独立核实的 `model_version` 与 `reported_reasoning_effort` 字段，仍按同一服务端身份锁校验。

```bash
"$python" -B -m "$module" reconcile "${common[@]}" --authorization "$A7_AUTHORIZATION" \
  --seed "$RECONCILE_SEED" --round "$ROUND_INDEX" --logical-index "$LOGICAL_INDEX" \
  --attempt "$ATTEMPT_INDEX" --reconciliation "$RECONCILIATION_JSON"
# import_completed 的同一命令另加 --recovered-response "$VERIFIED_RESPONSE_JSON"
```

## 恢复和评价

完成的响应、接受选择、标签反馈、优化器完成以及下一状态有分层原子收据。每个 seed 对自己的选择 lineage、图缓存、checkpoint 和记忆重新核验。已冻结批继续原批次；已完成训练进入预测或恢复下一轮，标签预算只随完整同轮回执推进。科学非法响应由同一 LLM 修复。有界传输预算、未知状态对账和第三方模型标识漂移都可以使进度停止。GPU lease 和 API slot 分阶段取得，已有数值作业优先保有其当前锁。

`status` 和 `a7_gpu.metrics --run-dir` 从收据统计每轮验证 NRMSE、V1/V2 RMSE/MAE/R²、原始 normalized label-AULC、L525/L813/L1005/L1485、逐轮分子数/证据关系、首答合法率、局部修复与锁定条目保留、逻辑和物理计数、已记录的用量和耗时、失败与暂停。五 seed 的分母和样本标准差随有据结果更新。物理尝试与已完成 provider 响应分别计数。已曝光的 test 集不用于 A7 配置变更。

[部署与局部资格结果](DEPLOYMENT.md)
