# 统一运行、回放与模块合同

```text
scripts/run_al.py
  └─ al/workflow/entry.py → RoundWorkflow（唯一轮次状态机）
       ├─ predictor/row.py + row_fit.py → 原模型/训练数学 → application
       ├─ representations → acquisitions → NumericalSelector
       ├─ RowAgentPacketBuilder → AgentSelector → recorded / Responses transport
       └─ LabelBroker → durable freeze → 明确授权的逐ID标签收据
```

`predictor`负责图/数据/归一化、六输出、固定初始化、Adam/早停/最佳checkpoint；真实拟合只走Row fit事务，已退出早期generic backend的另一套拟合scaffold；`representations/acquisitions`只消费授权模型/特征，不接secret或真值loader；`agents`负责各profile的完整packet、原Prompt、schema/NoCW、quota、纠错与来源；`workflow`唯一拥有L/U转换、fit/reveal事务和恢复；`infra`负责身份、快照、运行目录与授权。

## 命令与能力边界

| 命令 | 可运行用途 |
| --- | --- |
| validate | 检查schema、profile、预算、显式资源描述，返回protocol hash；不打开标签或secret |
| dry_run | 同上并输出prepare→fit→predict/represent→select→freeze→reveal→feedback→next-fit→evaluate计划 |
| offline_replay --fixture | 运行仓库内真实LLM录制选择回放，原始字节/hash、合法ID顺序和纠错匹配；不等于重训轨迹 |
| offline_replay --config | 通过ReplayBackend和同一RoundWorkflow消费完整外部fit/feature/reveal录制；缺资源或K3身份失败关闭 |
| authorized_real_run | 以独立授权启动真实Row数值或LLM profile；本轮未执行 |
| resume | 合成演示或匹配manifest的真实运行；复用已有合法成功结果 |
| report | 从本run ledger读validation曲线与观察预算内normalized AULC，不读取test真值 |

合成快速开始见 [README](../README.md)。各profile状态见 [CAPABILITY_MATRIX](CAPABILITY_MATRIX.md)，旧版本差异见 [来源映射](LEGACY_TO_NEW_CAPABILITY_MATRIX.md)。

## 真实Row准备

仅使用独立、普通、只读、无symlink/hardlink且SHA256匹配的SnapshotFile。不可从live main借fixture。资源source只解析七个公开X列；partition只解析身份/角色；graph cache须为明确授权的可信PyTorch快照。L0与validation标签分别以仅含该角色ID的JSON映射提供。Row标签按原float32消费语义归一化。

schema3配置形状如下；路径与SHA必须替换为自己的受控资源。此示意不构成执行授权：

```json
{
  "schema_version": 3,
  "execution_mode": "qgeognn_row",
  "profile": "gradient_lcmd32",
  "outer_seed": 157,
  "target_budget": 365,
  "resources": {
    "source": {"path": "/snapshot/canonical.csv", "sha256": "<64 hex>", "root": "/snapshot"},
    "partition": {"path": "/snapshot/partition.csv", "sha256": "<64 hex>", "root": "/snapshot"},
    "graph_cache": {"path": "/snapshot/graphs.pt", "sha256": "<64 hex>", "root": "/snapshot"},
    "l0_labels": {"path": "/snapshot/l0.json", "sha256": "<64 hex>", "root": "/snapshot"},
    "validation_labels": {"path": "/snapshot/validation.json", "sha256": "<64 hex>", "root": "/snapshot"},
    "reveal_directory": "/authorized_receipts"
  }
}
```

4,163行Row划分为333 L0、2,997 U0、416 validation、417 test；target为333+32k。outer-train公开X拟合covariate normalization，L0标签固定target scales/CW，后续不重拟合尺度。各member每轮从同一初始化重新拟合；初始化种子 `outer+2000003+member*1000033`，epoch-order种子 `outer+3000017+member*1000033`，CountSketch种子 `outer+4000037`。原训练以CPU两线程、Adam1e-3、B2048、最多1000epoch、patience100、validation Combined NRMSE选best；GPU/跨环境轨迹未资格化。

LLM profile另加显式 `provider`，包含 `model`、`reasoning_effort`、HTTPS `base_url`、`key_env`，必要时 `max_output_tokens`。不自动选模型/endpoint、不读全局key文件或CLI登录。当前CW-Free初始Nov1配置为gpt-6.1-sol/xhigh、32768 token；后续transport修订不冒称等价。provider依赖：`python -m pip install -r requirements-provider.txt`。

授权JSON仅接受 `schema_version:1`、`protocol_hash`（validate输出）、精确 `output`、`actions`。数值运行需要`training,reveal`，LLM还需`provider`。例如 `{"schema_version":1,"protocol_hash":"<validate hash>","output":"/new/runtime","actions":["training","reveal"]}`；LLM另加`provider`。它是调用者提供的独立执行许可，不是Git或hash自动产生的权限，也不是签名认证。本任务没有生成真实运行授权。

每个新选中ID的收据为 `/authorized_receipts/<ID>.json`，只包含 `sample_id`、`freeze_hash`、`values:[V1,V2]`。必须匹配本run持久freeze。无收据时返回 `AWAITING_AUTHORIZED_LABELS`，不先写reveal-started或读取整份U真值。

## Profile、信息与组合

NumericalSelector调用唯一kernels：Random固定原U0 permutation；Gradient/CW-LCMD、MaxDet使用各自512D；IVR固定原L0+U0 reference成员顺序；Numerical Hybrid以真实同seed/同L的member0/1/2 q50 disagreement取top ceil(.25U)，再用member0 latent128 k-center。K3不是CW16+LLM16，也不能拿三个不同outer seed拼成K3。

RowAgentPacketBuilder从当前fit和本run ledger构建packet：Free32全授权池、无128预选；Scientist16保留CW16的有序pending，LLM从完整剩余池补16；Screen保持40CW/40IVR/40MaxDet/8Random和独立RNG去重补128、一答fallback、无个体反馈；NoCW从字段、Prompt、memory、错误/纠错消息剥离CW，含Unicode转义检查。模型预测、已揭示L及实测反馈可见；validation/test标签、provider handle和secret不进入packet。

纯表示与acquisition的组合仍可直接调用 `FeatureSnapshot → acquisitions.select`，示例与维度契约见模块docstring；B32历史identity dispatcher见 [numerical_profiles](../src/qgeognn_al/al/numerical_profiles.py)。不新增第二套近似算法。

## 恢复与来源

真实入口拒绝未提交或未跟踪的科学源码；run manifest在prepare写入前绑定config/code/科学源码摘要/output，运行中逐轮复核。重算scaler先写独立临时目录，只比对原scaler，不覆盖不匹配旧文件。backend/selector/图缓存/实际GraphRow/有序L与validation均绑定；freeze返回字段不能覆盖scope或checkpoint。工件与`.tmp`拒绝链接，写入采用exclusive create/fsync/atomic replace。

成功响应receipt存在时直接复用；started-only请求没有durable answer时 `AmbiguousRequest`，不自动再次付费。首次served model锁定后不允许版本漂移，response ID必须非空。真实fit的checkpoint/predictions/audit三件套完整、输入输出合同匹配时可无训练复用；缺件或模糊状态不自动重训。原direct-SSE/CLI/H2恢复及无界重试尚未迁移，不能把新的fail-closed说成旧恢复等价。

SDK迁移保留完整显式messages、`tools=[]`、`store=False`和完成响应检查；依据 [官方Streaming文档](https://developers.openai.com/api/docs/guides/streaming-responses) 与 [Responses参数](https://developers.openai.com/api/reference/python/resources/responses/methods/create)，本机SDK2.28.0仅mock集成，无在线测量。源码/固定响应等价与在线采样非确定性分开验收。

## 外部私有冻结回归

内置schema1 golden只含tracked历史字节。`offline_replay --fixture /snapshot/manifest.json`另支持schema2 `private_snapshot`：manifest及文件必须独立readonly、无链接、SHA匹配；仅tracked freeze/Prompt使用Git blob锚，私有原文用freeze artifact hashes，不能伪造已入Git身份。不读stream/header/secret/raw reasoning，不自动借main。

`QGEOGNN_PRIVATE_AGENT_REPLAY_MANIFEST`支持以路径分隔符提供多个显式manifest；`QGEOGNN_K3_REGRESSION_MANIFEST`提供同seed/current-L三checkpoint、原本zero-sentinel graph、bank/合同及validation授权的collection。未配置时frozen tests明确BLOCKED_EXTERNAL_NOT_CONFIGURED。原始唯一资源保持main原位，临时readonly snapshot并非长期私有备份。
