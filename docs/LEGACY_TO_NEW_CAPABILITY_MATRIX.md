# Main 方法 → 统一框架来源矩阵

来源M=`b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28`，不是live进度。远端 `archive/framework-v3-main-source-20261112` 固定M。Git不备份ignored资源/环境；清理前重构快照见验收报告。

实现/资格分开：见 [CAPABILITY_MATRIX](CAPABILITY_MATRIX.md)、[EQUIVALENCE_REPORT](EQUIVALENCE_REPORT.md)。缺证据写`UNKNOWN`，不宣称FULL READY。

## 共享科学身份 R

各profile复用R：QGeoGNN-V2/458,952参数/六输出不排序；4,163行，outer train 3,330=L0 333+U0 2,997，validation416/test417。G=`157/887/2357/6101/12203`；D=`157/6101`。cohort/预算见各行，L333必须匹配M的partition ID。

- 只解析7个X列，图缓存hash绑定；outer-train X拟合covariate scaler，原L0标签拟合ddof=0尺度，跨轮固定。validation选checkpoint、test仅最终冻结评价，不进selector；32 IDs+预测值先冻结再授权reveal。
- Adam(.001,wd0)、batch2048/max1000/patience100；原pinball+q50 MSE+crossing loss。strict validation Combined Row NRMSE保存best.pt；CPU2，GPU/训练轨迹`NOT_VERIFIED`。
- 每轮同初始化重训：init=`s+2,000,003+m*1,000,033`；epoch seed=`s+3,000,017+m*1,000,033`，排列`default_rng(seed*10000+epoch)`；不再用epoch seed重置模型。sketch=`s+4,000,037`/512D；L按获得顺序、U按canonical顺序，stable tie。
- 先前外部fixture仅G各自member0/L333；本轮进一步按原freeze链受控快照同seed/current-L成员0/1/2，五seed r0/r1真实K3回归已通过。不能跨outer凑K3；legacy/public state hash序列化不同，禁止混比。

迁入predictor的 [Row API](../src/qgeognn_al/predictor/row.py)/row_fit及workflow/row_backend，保留application依赖；fit需授权，partial/linked收据fail closed。

## 固定源码索引

所有行在M；`A/`=`src/qgeognn_al/active_learning_v2/`，`S/`=`scripts/studies/`；blob定义原字节。

| Key | M路径 | Git blob |
| --- | --- | --- |
| R1 | `A/protocol.py` | `e258b7022c890a95734f6cff91a6c7846c043d32` |
| R2 | `A/benchmark_protocol.py` | `8688680d67e4d8b5cb95150974ad1c66c9c832d4` |
| R3 | `A/runner.py` | `840b5b10000dc6990631e2e59a10d2fed2e0b2f9` |
| L1 | `S/run_free_cw_sol61_five_20261101.py` | `deb86570f8d867e273dbd0c2d9ceb5c931e35e58` |
| L2 | `S/run_qgeognn_v2_row_llm_nocw_five.py` | `f6ef9cfcc0d30dbe4143c5c76459bb52cbe3d0eb` |
| L3 | `A/llm_screen.py` | `5bb55938da4e2259a586e66753d43f0df04d064f` |
| L4 | `A/scientist_study.py` | `d600d68eeef30900c0ee239697fcd1c25e1e5d36` |
| L5 | `S/plot_five_methods_cw_gradient_20261103.py` | `63279a01f22cffcd3ff8a37f2afd3ce6cd19fb73` |
| P | `A/nocw_inputs.py` | `4da0ca103e176ce183156b6e2f3e730f137d2a66` |
| N1 | `A/sequential_acquisition.py` | `7bdf763c6aabc7dfea4910bf6635b5b39b422abe` |
| N2 | `A/maxdet_study.py` | `2de2fbb69c2a65f73b042fe87b5bb4e60a9f2b4f` |
| N3 | `A/ivr_study.py` | `d1748492d782851a279aac911410c6a3a3567610` |
| C | `A/short_sequential_runner.py` | `3f8376a2200513c2d38c9d524c3b98a6cc04b2b0` |
| K | `A/sequential_runner.py` | `bffec8b5f7c91b86ed1a08344e1a00dc9d92de28` |

## 数值profile

数值profile无Prompt/provider/retry，训练/标签/checkpoint用R；[路由](../src/qgeognn_al/al/numerical_profiles.py)复用原核。

| CLI profile / method version | 源Key；原cohort/预算 | 当前U、数值信息与RNG |
| --- | --- | --- |
| `random32` / fixed-U0-permutation-b32 | N1；G，333→1005 | 原U0一次排列，seed=`s*1,000,003+900,001`；按轮取32切片，非每轮重新抽样 |
| `gradient_lcmd32` / q50-countsketch512-current-L | N1；G，333→1005 | 完整当前U；scaled q50全网络梯度，当前L作已有centers |
| `cw_gradient_lcmd32` / linear-autograd-CW-L0-ddof0 | C；D，333→429，后续延长各自冻结 | L0 center/width变换+512D线性autograd+LCMD；非whitening/fused，factorial仅作机制参考 |
| `gradient_maxdet32` / all-current-U-ridge1 | N2；D，333→1005 | 完整当前U；当前L RMS尺度和conditional inverse，ridge1 |
| `kernel_ivr32` / fixed-reference-universe | N3；G，333→1005 | 当前候选，固定原L0+U0 reference universe/RMS；不随round换reference |
| `numerical_hybrid32` / current-K3-top25-latent-kcenter | N1+K；G，333→1005 | 同s、同当前L的K3 q50总体std/L0尺度，top max(32,ceil(.25U))；member0 latent128 k-center；与任何LLM半批次无关 |
| `uncertainty_gated_maxdet32` / 旧`u50_gradient_maxdet` | N2；D，333→1005 | K3 top ceil(.50U)后canonical-order MaxDet。当前需要性`UNKNOWN`，`NOT_IMPLEMENTED`，不能用普通MaxDet替代 |

旧账本：各study的`runtime/seed_s/<method>/round_rr/{model/fit_audit.json,round_freeze.json}`及selections/最终freeze。数值正式test评价已暴露；源CW延长段按自己的test/validation口径，禁止拼接。

## LLM profile：各自身份与可见信息

训练同R。full-pool按历史salt/ID hash展示完整U，不缩成128。memory仅本seed/method的L、近期误差/解释/假设、旧高误差/IDs；L333无out-of-sample误差。quantile宽度非ensemble不确定性。screen无个体实测值/旧packet/reason。

| CLI profile / version | 源Key；原身份/预算 | 选样、Prompt/反馈边界 |
| --- | --- | --- |
| `free32_cwaware` / Nov1 CW-visible rerun | L1；`free_cw_sol61_xhigh_rerun32`；G，333→1485 | 完整U自由选32，pending=0；CW percentile可见；历史Free salt/memory；DIRECT Prompt删除pending-CW outcome句。不是下述paired_v1 |
| `free32_nocw` / five-seed historical-control-v1 | L2；`llm32_nocw`；G，配置333→1005 | 完整U选32，pending=0；去掉所有CW字段/定义/历史与纠错泄漏；保留五seed override的共同正向schema和历史Free salt/memory。后续unbounded amendments未由该名字涵盖 |
| `cw16_llm16_screen_v1` / conversation screen | L3；`cw16_llm16`；D，333→429 | CW16先选；40CW+40IVR+40MaxDet+8Random去重、补至128，LLM16。5个独立SeedSequence streams `[s,r,stream,1616]`；单答保留合法项、按固定fallback补齐；无个体结果/历史reason反馈 |
| `cw16_llm16_scientist_v2` / astra-high-full-pool-20261001 | L4；`cw16_llm16_scientist_v2`；D，333→525 | CW16 pending+完整剩余U供LLM补16；DIRECT Prompt，完整observed/预测表和逐轨迹scientific feedback；并非screen |
| `free16_cwgradient_lcmd16` / 展示别名 | L5→L4；旧图`free16_lcmd16` | 图实际读取Scientist V2的CW16+LLM16路径，不是独立第三实验。当前复用该selector身份；图中的延长段按各revision冻结，不能推断全段服务版本相同 |

另注册 `free32_cwaware_paired_v1`/`free32_nocw_paired_v1`（P，`llm32_nocw_paired_v1`）：COMMON_PROMPT共同修改与typed evidence/prose-ID限制。缺paired-CW真实回放，不能把这对源码身份与实际五seed override混同。

L1/L2目标是固定L333尺度的validation归一化梯形AULC，分别333..1485/1005；screen/Scientist原开发比较为已暴露test NRMSE/AULC333..429/525；L5图是validation混合修订来源，非新正式统一对照。

### Prompt、服务与纠错

[Prompt](../src/qgeognn_al/al/agents/prompts.py) UTF-8 SHA256如下；alias同Scientist V2，paired构造见P及 [profiles](../src/qgeognn_al/al/agents/profiles.py)。

| Profile | Prompt SHA256 | 历史服务 / bounded selection；network |
| --- | --- | --- |
| `free32_cwaware` | `1e1b54ca81497f70c7f3a57bd2ec7a6884293feb07e639a84d41518048e603ad` | gpt-6.1-sol / xhigh；8×4 |
| `free32_nocw` | `e0c32645b73f70c7d7099e0166ff66abaae43bbffcb276e15e64c15a6d40ff76` | gpt-6.1-sol / xhigh；3×3 |
| `cw16_llm16_screen_v1` | `bd956b5c7a323bd62ed4ac9d1477afbfda97a1fe382943f4ab659e3b11820e1b` | UNKNOWN / UNKNOWN；1答+fallback |
| `cw16_llm16_scientist_v2` | `f304b667ec2e1d97f0726e24bced2535a9e386e5ec206ea979cab0b5426ff441` | gpt-6-astra / high；3答 / UNKNOWN |

历史Responses：`token4research.cn/v1`、timeout600s、tools=[]、store=false、SDK retries=0。served model/兼容性须receipt核验。Free backoff2/10/60，NoCW backoff2；codex_cli/h2传输`NOT_IMPLEMENTED`。

[AgentSelector](../src/qgeognn_al/al/agents/service.py)+自动packet builder接唯一workflow；Responses实现但付费需明确授权，保留request/response provenance与schema/ID/NoCW审计。unknown-dispatch拒绝自动重发，比旧retry更严格，非失败路径等价。在线均`ONLINE_SCIENCE_NOT_MEASURED`。

## 回放、账本与缺口

原LLM账本：`studies/active_learning/<study>/selections/seed_s/<arm>/round_rr/`，full-pool记catalog/input/request/turn/attempt/selection/freeze；screen记packet/proposal/response/validated_response/freeze。L1 study=`free_cw_sol61_xhigh_five_to1485_20261101`，L2=`qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029`，L3=`qgeognn_v2_row_llm16_screen`，L4=`qgeognn_v2_row_llm_scientist_v2/revisions/astra_high_full_pool_20261001`。freeze内hash不等于ignored原文已备份。

已回放Screen两seed×3轮、Scientist157/6101各首轮+纠错轮及CW/NoCW五seed各首/后两轮，共30个独立真实case。Screen outgoing transcript `UNKNOWN`，其余full-pool原request hash可核；私有字节在独立外部只读snapshot，不是Git已备份。未执行paired_v1另一个计划控制的真实回放，不与实际five-seed override混同。selection/freeze通过不等于重新训练或完整在线轨迹通过，mock packet不证明真实全等。

本轮无真实训练/付费/新真值，未改main。

## 2026-11-12 物理资源核验补充

此前仅Git追踪审查不足以判定物理缺失：CW/NoCW五seed首轮/后续轮的20组ignored原input/catalog/request/turn均存在，已按M的tracked freeze artifact hashes制成独立readonly snapshot并全量精确回放；Scientist6101 r0/r1同样通过。Numerical Hybrid的同seed/current-L成员1/2也实际存在，五seed r0/r1的30个真实checkpoint推理、完整latent/ranking/32-ID精确匹配；L333/L365 validation NRMSE及两点AULC差0。这些是外部授权冻结资源，不是Git可自动恢复的私有备份，也不是重新训练/在线资格。旧表中针对“尚未配置外部fixture”的BLOCKED仅指使用者环境，不能解释为原资源不存在。详见当前能力矩阵及等价报告。
