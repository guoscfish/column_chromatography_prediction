# PureLLM-A7 Astra/xhigh 正式运行

当前启动范围是独立的 `a7-gpt62-astra-xhigh-ssh-v4-pilot-3x10`。三个 seed 157、887、2357 同时进入调度，每条轨迹从公共 CUDA L333 锚点开始，每轮 32 点，最多 10 轮，到 L653 自动停下。Mac 的三个 Key 全局最多同时处理三笔请求。GPU 0 为唯一训练卡，三个 worker 共用原物理 GPU lease，API 等待时释放 GPU。该阶段按三条轨迹各自的合法回执推进。全五 seed 到 L1485 的完整 v4 合同是独立范围。

| 项目 | 旧 v3 | 当前阶段 |
| --- | --- | --- |
| 请求及严格匹配的返回模型 | `gpt-6.2-sol` | `gpt-6.2-astra` |
| 推理等级 | `max` | `xhigh` |
| 输入 | 原始 layout B | 分子分组、显式区间宽度、三位预测显示及分位数交叉保护 |
| 传输超时 | 600 秒旧截止 | 连接 30 秒、写入 120 秒、读取空闲 900 秒、独立硬截止 1,800 秒 |
| GPU 等待 Mac | 1,800 秒 | 4,200 秒，超限后恢复同一尝试 |
| 初始调度 | seed 157 达 L365 后放行 | 三个 seed 一起排队，GPU 0 串行训练 |

两版均使用 Token4Research、OpenAI SDK 2.28.0、Responses SSE、`store:false` 和 32,768 Token 输出上限。Mac 沿用成功资格测试的默认代理路由，无需变更系统网络。A7 科学 Prompt 原样保留，系统消息同时提供已验证的紧凑表解码说明。完整候选、全部已测记录、原实验条件、M2 记忆和局部修复规则保持原样。正式请求没有合成测试探针。原精度数据及真实选点校验留在 GPU 平台。原始包和实际传给 Mac 的紧凑包各有独立摘要。 [长上下文验证](LONG_CONTEXT_QUALIFICATION.md) · [32 点语义审计](ASTRA_XHIGH_SEMANTIC_AUDIT.md)

本阶段最多 **30 次首请求、90 次逻辑请求、180 次物理尝试**，每轮延续原有三次逻辑、每逻辑两次物理上限。输出 Token 的累计上限为 5,898,240。输入用量随轮次变化，现金费用按服务商实际费率和用量结算。状态未知的请求保留原队列和对账规则。v3 超时请求的执行和计费保持未决，旧合同、日志和诊断均留在各自目录。 [v3 证据](LONG_CONTEXT_QUALIFICATION.md#8-正式-v3-超时对账)

## 部署并生成准确合同

先按 [发布流程](DEPLOYMENT.md#构建与安装) 安装与本地相同的独立 release。在 GPU release 根目录执行以下参数。合同生成验证全部五份旧锚点，当前阶段只选用其中三份。

~~~bash
root=/1/column_chromatography_prediction
suite="$root/formal_cuda/row4g_b32_20261124"
runs="$root/purellm_a7/runs"
run="$runs/a7-gpt62-astra-xhigh-ssh-v4-pilot-3x10"
study=studies/active_learning/pure_llm_v3_a7_gpu_20261124
module=studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.scheduler
python="$root/environment/venv/bin/python"
common=(--config "$study/execution_gpt62_astra_xhigh_a7_v4_pilot.json"
  --run-dir "$run" --gpus 0 --gpu-lease-root "$suite" --anchor-root "$suite/trajectories"
  --api-execution mac_ssh_dispatch_v1 --predecessor-run "$runs/a7-gpt62-max-ssh-v3")

"$python" -B -m "$module" contract "${common[@]}"
~~~

验证合同 SHA、GPU、锚点和实际预算后，按已获得的授权范围从 [空白模板](formal_authorization.template.json) 生成对应 SHA 的私密授权。旧合同和凭据留在旧目录或本机私密目录。使用独立进程启动 GPU 协调器。

~~~bash
umask 077
mkdir -p "$run"
nohup "$python" -B -u -m "$module" run "${common[@]}" \
  --authorization "${A7_AUTHORIZATION:?指定新合同的私密授权}" >> "$run/coordinator.log" 2>&1 </dev/null &
"$python" -B -m "$module" status --run-dir "$run"
~~~

在 Mac 切换至同一校验版 release，按 [Mac 派发入口](DISPATCHER.md#mac-派发入口) 设置目标 run、release 和合同 SHA，并额外传入 `--config "$study/execution_gpt62_astra_xhigh_a7_v4_pilot.json"`。Mac 使用原 Key 池和新合同专属 spool，经 `stage` 收取任务，再以 `caffeinate -i` 启动已授权的 `run --execute-paid --watch`。暂停时先暂停 GPU 协调器并等 API 回执封存。恢复时保留原 release、合同、运行目录、spool 和授权，GPU 使用 `resume`，Mac 重启原监听。

Mac 统一状态命令支持 `purellm-a7-status --version v4-pilot`。私密设置 `purellm-a7-v4-pilot.json` 的四个字段为 `contract_sha256`、`release_sha256`、`run_basename` 和 `reasoning_effort`。

## 研究记录和停点

在 L653 停止后逐 seed 分析 10 轮选择。每轮保留首次回复合法性、局部修复、用量和物理调用耗时。冻结时记录与批次 SHA 绑定的 `acquisition_diagnostics.json`，包括不同结构、新结构比例、条件覆盖增量、批内和历史单因素及多因素配对，以及假设更新和引用。模型对代表性、不确定性和科学先验的原文理由、引用和记忆变更保留在 `selection.json` 和 `memory.json`，用于后验编码。验证 NRMSE、端点 R² 和 L333→L653 的 AULC 从受控训练回执提取。以上统计均不进入选择请求。
