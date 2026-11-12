# 等价验证、可用性与最终精简

2026-11-12；仅在现有 `codex/repository-architecture-20261109` linked worktree实施，保护起点33f0ab460，没有reset本地未推送成果、新分支或main合并。**FULL READY = false**。当前状态逐项见 [CAPABILITY_MATRIX](CAPABILITY_MATRIX.md)，版本/来源见 [LEGACY_TO_NEW_CAPABILITY_MATRIX](LEGACY_TO_NEW_CAPABILITY_MATRIX.md)，可执行接口见 [README](../README.md) 和 [运行指南](ARCHITECTURE.md)。

## 实际迁移与验收口径

真实Row数据/图/固定归一化、初始化/同初始化重训事务、完整候选/观测/预测、LLM各profile Prompt与反馈、Responses transport、schema/NoCW/纠错、真实K3 member编排已接入同一个RoundWorkflow。修复旧QGeoGNNBackend对CW同时传`scales`导致失败的问题；同时移除早期generic backend未资格化的另一拟合scaffold，仅保留Row source recipe真实拟合事务。没有改模型头、loss、六输出、选点公式或既有正式结果。源码恢复不在运行时导入旧study模块，也没有旧目录镜像。

- `NUMERIC_EQUIVALENCE_PASS`仅覆盖明确输入下的partition/初始化/已有模型与数值输出。
- `EXACT_REPLAY_PASS`仅覆盖有原始字节记录的选择层，不能外推到fit→reveal→指标整条轨迹。
- `INTEGRATION_PASS`包含明确标注的自有合成输入/mock transport/fit测试；不作为真实AL pass。
- 所有在线科学实验 `ONLINE_SCIENCE_NOT_MEASURED`；重新训练及完整新实验NRMSE/AULC轨迹 `NOT_VERIFIED`。已测既有冻结权重的K3、选择与匹配预算validation另列，不混为新训练。本轮没有新optimizer、付费请求或真实新标签揭示。

## A：确定性输入、模型及数值

源main `b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28`。14个模型/数据/训练核心函数/类AST相同：模型、checkpoint、图/条件、原loss、Adam/epoch shuffle/early stop/selection、Row partition与AL preprocessing。`training.atomic_json`仅加强输出文件链接保护，训练数学函数未改。

五个outer seed 157/887/2357/6101/12203，4,163行真实Row partition按每role及有序ID比对，差为0；公开X只解析7列。15组outer×member0/1/2初始化与源纯AST参考逐tensor最大差0，CPU两线程。初始化seed、epoch-order、sketch相互独立，不额外把epoch-order seed用于重置torch训练RNG。旧state hash用带空格shape JSON，新API用compact JSON；row fit审计保留legacy/public双hash，不把格式差称权重差。

五个授权L333 checkpoint各16行L0/validation六输出与hidden：

| outer seed | checkpoint SHA256 | 六输出 / hidden最大绝对差 |
| --- | --- | --- |
| 157 | `87e92ccdabc1dd64ecf89a3061624450ce125906c4cbb988d7263b190991b1c0` | 0.0 / 0.0 |
| 887 | `8a9c26efc0e8fd25420e56177f8becc8d12ddec204e564c6037fe58c12587059` | 0.0 / 0.0 |
| 2357 | `ac34c895d573510f86f9aa1dd0b7b0c565dbf3fd80fa3763acab92c787de6731` | 0.0 / 0.0 |
| 6101 | `bcfffa247c3d24c8c74993ddb66cb59741ae14b1e67e8ed9543a8d1af7c6c494` | 0.0 / 0.0 |
| 12203 | `114ba06d0fde3bce6c05f7ec2c642ed2b1c9b50a0a235c9bce06530d7bc9be5a` | 0.0 / 0.0 |

图重建bitwise一致，权重精确；raw q50 gradient/CW/latent/fused在单独授权的label-free特征fixture验证。既定预测容差rtol=0/atol=1e-6，实测上述差为0；保存LCMD trace容差rtol=1e-12/atol=1e-8，ID及顺序必须完全相同。CPU安装环境固定，未证明GPU或跨版本逐位轨迹。

新named `gradient_lcmd32` dispatcher实际重放三批历史完整3,330-row银行：

| outer / budget | 保存feature NPZ SHA256 | evaluation checkpoint SHA256 | ID diff |
| --- | --- | --- | --- |
| 157 / L333→365 | `80c9a397f0c2d1b7caf8377929b046f24ed324718508ec35f6c6043672854147` | `87e92ccdabc1dd64ecf89a3061624450ce125906c4cbb988d7263b190991b1c0` | 0 |
| 157 / L525→557 | `abe446d3b61a82f314f1219ab7f8748e7812e7ad172ed2266aeb00ba3f6f8983` | `fe86aa20290cf8c2d0898237d17e8a0500ebbc55fbf12677d4d7c2024863df73` | 0 |
| 157 / L973→1005 | `8dea3454e6e310786621c30894d0e938365db16ef2467706a8fd31f7624a8433` | `3e6885cb697804881819af5f30f49fdf26085e0f58ba0526924596b138390c1e` | 0 |

每批32个ID原顺序完全一致；L333/L525/L973不是三轮连续训练，**不能据此声称多轮fit等价**。其它核心Random/MaxDet/IVR/Hybrid有独立参考、seed/tie/reference/排序和失败保护；已有冻结模型上的反事实数值选择不称为其历史正式AL批次重现。

### 已解锁的全量Row及真实K3

进一步核对物理资源，原ignored工件并未缺失。按tracked global pre-test freeze→round→acquisition→fit audit/hash链核对后，从原位复制独立readonly snapshot；原main不写、不训练。不能把“Git未跟踪/尚未配置”解释成“原资源不存在”。

五seed各L333的RowDataset.prepare重新生成全部4,163图：各字段bitwise一致，normalization/preprocessing/scaler完全一致；2,997个U的原始六输出与CW、NoCW冻结r0 catalog全部bitwise一致（共14,985候选/89,910输出，max diff0）。目标只取该冻结observed-L333；源CSV仅7X，不解析U/test真值。

真实K3为五seed×r0/r1（L333/L365），30个已训练checkpoint，member1/2全U六输出float32 bitwise一致，member0完整3,330行latent最大差0；uncertainty ranking、top25 shortlist、每轮32-ID有序全相同。图来自原本zero-sentinel的scrubbed_graphs.pt，固定源码证明构图前V1/V2为常量0，加载后再次检查全0，只推理current L/U，未加载真实U/test y后再删除。

当前接口 `infer_frozen_graph_ensemble` 可重复完整权重/graph→K3+latent→named selector路径；不是只比较保存的分数。外部manifest `/private/tmp/framework_v3_20261112/private_k3_snapshots/MANIFEST.json` 及其PROVENANCE记录30个checkpoint SHA、旧/新state hash格式映射、图/源合同/ID/feature hash。真实重新训练后最佳epoch/checkpoint及完整训练轨迹仍未测。

证据：`row_predictor/ROW_EQUIVALENCE.json`、`row_predictor/PARTITION_GOLDEN.json`、`numerical/PROFILE_REPLAY.json`、`MODEL_EQUIVALENCE.json`、`ALGORITHM_EQUIVALENCE.json`，均在 `/private/tmp/framework_v3_20261112/`。fixture manifests固定于 `/private/tmp/column_phase2_science_snapshots/{MANIFEST,REGRESSION_MANIFEST}.json` 与 `/private/tmp/column_phase2_frozen_acquisition_inputs/MANIFEST.json`，没有从live main借工件。

## B：LLM真实固定响应回放

源main同上。内置golden `tests/fixtures/llm_recorded_replay.json.gz` 为8个tracked原始case的最小独立字节包，SHA256 `88acf5925945f59c5938391c864ffb3c43aea73a6d5da6e51dbb05c205aa1c11`；逐源文件SHA/Git blob校验。另22个原ignored私有case经原tracked freeze artifact hashes制作独立readonly snapshot，保持Git之外，原始字节未改造；不是把私有文件伪称Git可恢复。

```sh
python -m scripts.run_al offline_replay --fixture tests/fixtures/llm_recorded_replay.json.gz
python -m scripts.run_al offline_replay --fixture /private/tmp/framework_v3_20261112/private_llm_snapshots/manifest.json
python -m scripts.run_al offline_replay --fixture /private/tmp/framework_v3_20261112/private_scientist_snapshots/manifest.json
```

| Profile | seed / round / 当前L | 原input文件SHA256 | 选择ID diff |
| --- | --- | --- | --- |
| cw16_llm16_screen_v1 | 157 / r0 / L333 | `3b493be39f6541bcbe6824c8e730ed2a0c5d4c6f9ec926864f2327a6ade6042f` | 0 / 原顺序相同 |
| cw16_llm16_screen_v1 | 157 / r1 / L365 | `b5a93fb9f4d270a19fb1b87740208e0a96112ea73bd7e7ad283d724d6d16b318` | 0 / 原顺序相同 |
| cw16_llm16_screen_v1 | 157 / r2 / L397 | `df9c63e2647c01450bbc4d110b625387c01827b3717a77c5d16d36a500434d35` | 0 / 原顺序相同 |
| cw16_llm16_screen_v1 | 6101 / r0 / L333 | `f590846c81e731f194409e951f7fcbdc181f33aea0adbc9b520d43907f19a5bf` | 0 / 原顺序相同 |
| cw16_llm16_screen_v1 | 6101 / r1 / L365 | `7bb7d84d3dd690792805dd45fdabaf59a588df2c0faad068f61c5d9de3a8184a` | 0 / 原顺序相同 |
| cw16_llm16_screen_v1 | 6101 / r2 / L397 | `3f606bc9f884da997bd28e3cc4960cdccbf383cb75b4a49fb2476056e8f01c3e` | 0 / 原顺序相同 |
| cw16_llm16_scientist_v2 | 157 / r0 / L333 | `de844cfc57a836930cb309c5987f1b46d8971beeb5ba30204e771eaf8f790b84` | 0 / 原顺序相同 |
| cw16_llm16_scientist_v2 | 157 / r1 / L365 | `8179e209ffa8fd92cc44dfa54163eec653d1d1ddfccb2c5c08f0e56e533e5778` | 0 / 原顺序相同 |
| free32_cwaware | 157 / r0 / L333 | `edde485866b2bdab2ca77869b49fa0b25d0001da7a794d097502b8a92707a7d7` | 0 / 原顺序相同 |
| free32_cwaware | 157 / r1 / L365 | `2ee0254f94b7104c0596406877724b76ad2dc0546c0723ee6c49b2fb8123d494` | 0 / 原顺序相同 |
| free32_cwaware | 887 / r0 / L333 | `63f9ed9e18b4c2e9ea8c73a11f4eda96432e4bf4d9283c1fbdfcd6b27f77e64a` | 0 / 原顺序相同 |
| free32_cwaware | 887 / r1 / L365 | `1e88a6072d19992fa526d75ec6b3d210b3ad8300697c62b99c5c09013110f337` | 0 / 原顺序相同 |
| free32_cwaware | 2357 / r0 / L333 | `ba78c02e9aaea0ef3f8173e631f15223d38c1959e52de6f8ed09cf1ebcd73f1c` | 0 / 原顺序相同 |
| free32_cwaware | 2357 / r1 / L365 | `ce8684bee5157cabb350c307f4479e7bf2a10ccadde080b943a8785acb526b02` | 0 / 原顺序相同 |
| free32_cwaware | 6101 / r0 / L333 | `9b2646d36280741b24ee75426b1546463af42319641d5316a811c746e8c293fb` | 0 / 原顺序相同 |
| free32_cwaware | 6101 / r1 / L365 | `782a50c8786b76b384c2cbc0ea46340c8b7ee6b115e37054ccfc15f697026420` | 0 / 原顺序相同 |
| free32_cwaware | 12203 / r0 / L333 | `fa37112f36b6943e38b07efa588f3633d28af25c2080cde513261893b56e6d32` | 0 / 原顺序相同 |
| free32_cwaware | 12203 / r1 / L365 | `34f871e156dacd66953ddc32ca0b73fccdbe156aa6de154b6f50d597764ea797` | 0 / 原顺序相同 |
| free32_nocw | 157 / r0 / L333 | `982a9868c6296109912c268c20b37ac1257a966dc60b1de2fa3ec693feb11382` | 0 / 原顺序相同 |
| free32_nocw | 157 / r1 / L365 | `f68db368b1479f5df887d3e75a25503791756bfa19ac166d626085919ea11c08` | 0 / 原顺序相同 |
| free32_nocw | 887 / r0 / L333 | `48588597abedb9608617c802c5c6a48a680cafc3fd94d0ff17a619ad5bae04ed` | 0 / 原顺序相同 |
| free32_nocw | 887 / r1 / L365 | `dc192191f68f17725621039fff7a89a6b46bdabb13beb72648dd957b46c2593e` | 0 / 原顺序相同 |
| free32_nocw | 2357 / r0 / L333 | `eef90e0023e59836c5c1104b515b54c15a42d50f1725e33fc396565095475e81` | 0 / 原顺序相同 |
| free32_nocw | 2357 / r1 / L365 | `77b7ea04a89178124cb36d3fa6d92626eed08aef0d19f1771d432ccd21a27418` | 0 / 原顺序相同 |
| free32_nocw | 6101 / r0 / L333 | `b9bea362a5a138eacfb1621dddc67c3cc3b50ffe72de188781950dc4e8c8d3b5` | 0 / 原顺序相同 |
| free32_nocw | 6101 / r1 / L365 | `3059b71a84b4d30df0cda72412aa642701261e31f08f689f693f14bbfdeeea90` | 0 / 原顺序相同 |
| free32_nocw | 12203 / r0 / L333 | `94a3ede3dc4622bf0c66faa97abead25a144e692964750f46d146c9aba28de95` | 0 / 原顺序相同 |
| free32_nocw | 12203 / r1 / L365 | `d097db8c18bdf6ca76bf15eacac716204bc41ce7c2ed76ee48f7592afcd3f50f` | 0 / 原顺序相同 |
| cw16_llm16_scientist_v2 | 6101 / r0 / L333 | `ec132cf6d698e384d5a0336d6c8bfba3da7b548d1895b134f799827c5a5b1579` | 0 / 原顺序相同 |
| cw16_llm16_scientist_v2 | 6101 / r1 / L365 | `f9bf615ffb5ed4cceb8905043900d75b5841de727dd16ba6a9f16285a6a35839` | 0 / 原顺序相同 |

Screen：原128候选packet、CW16+valid-retain/fallback、ordered32及freeze相同；5路RNG与packet纯构建器结构/序列化hash和固定原函数一致。原outgoing messages/模型checkpoint未在这些文件中独立冻结，写 `UNKNOWN`，不借重建文本假证。

Scientist V2：原full catalog→packet、Prompt/messages、每turn原request SHA、后续轮非法evidence纠错、合法ordered32/freeze均相同；seed157的r0 checkpoint为上表157 L333，r1为 `ecae96913244cf3df5a6234f6bea937cc1c788a0938976e8dddd008eb8f51dc9`。原batch freeze/hash与原request hash完整保存在golden/执行JSON，不在报告复制全部ID。

`free16_cwgradient_lcmd16`只按原图例作为Scientist谱系别名，不重复计数。合计 **30个独立真实录制选择回合**：Screen6、Scientist157/6101各2，共4；CW/NoCW各五seed×r0/r1，共20。原完整packet/Prompt/messages/config/request SHA、真实非法回答纠错、accepted JSON、ordered32/freeze全部一致；从原catalog与可恢复首轮完整memory重建packet/messages也精确。

private loader强制manifest/文件readonly、无链接、实际SHA与固定Git freeze artifact hashes匹配，不能只信可被同时改写的manifest摘要。少量Scientist rejected_turn侧车未被原freeze直接哈希，明确标为not_directly_anchored；其纠错内容由后续已锚定turn的request hash约束。源构造器只持久化messages/config及answer/provenance；未复制headers、credential、stream或raw reasoning。

所取五seed实际CW/NoCW初始L333输入只差CW列、对应Prompt定义及派生hash；限定这些初始source条件，不扩大到早期Free32、后续不同模型/已选历史、COMMON_PROMPT另一版本或整研究因果结论。原paired_v1独立身份仍保留。

Scientist6101额外验证原已揭示64行的两次freeze→label access receipt→feedback lineage，L333→365→397有序身份、预测与signed error一致；原fit audit/history的已发布validation标量[1.0208867471225853,0.9003519700114196,0.7644487426226468]及normalized AULC333–397=0.8965098574420178可核。这是原记录回放，不是candidate重新训练的等价证明。

完整外部fit/feature/reveal录制可通过ReplayBackend及唯一workflow消费；全量从头重训练的连续轨迹仍NOT_VERIFIED。缺某个显式fixture时报告只指该环境/该录制集合，不再推断原物理资源缺失。

## C：真实接口的无付费集成与在线边界

RowAgentPacketBuilder已从当前fit及本run已冻结/揭示ledger生成完整U目录、原CW百分位/pending、observed反馈/记忆；host-only raw P50保存于freeze供后续实测误差，未把validation/test送给LLM。六个full-pool身份有首轮+反馈后续轮的owned-input测试，Screen原packet构造器diff0；NoCW覆盖历史/纠错/Unicode转义。它们属于INTEGRATION_PASS；全量真实Row graph/六输出、原full catalog→packet重建及K3权重回归另已实际验证，新fit优化轨迹仍不由此证明。

Responses SDK保留显式完整messages、tools空、store=false、max_retries=0、按profile可选token limit、completed/model/native-tool检查、非空response ID与首次served-model锁。录制/mock/真实transport隔离；quota按profile明确，secret仅从显式env取，授权前不读。依据 [官方Streaming文档](https://developers.openai.com/api/docs/guides/streaming-responses)，本机SDK2.28.0只做注入mock，未在线调用。

尚未迁移原Codex CLI、Nov1 direct-SSE response/completed journal恢复、H2/403/upstream错误分类/调度、后续unbounded retry。新unknown-dispatch failclosed是更严格的恢复边界，**不是历史transport EXACT**；新LLM请求也不能承诺旧ID/AULC。具有合法完整receipt的响应/fit保留并复用，partial/模糊状态不重复请求或优化。

## 测试与fresh clone

标准自动发现 `python -m pytest -q`：**261 passed / 0 skipped**，0 failures/errors；本机显式配置所有独立只读外部fixture，缺资源的环境另以BLOCKED验证。最终完整证据 `FINAL_ALL_TESTS.xml/log`。

长期测试覆盖模型/预处理/Row数据、各acquisition与K3身份、Agent profiles/真实recorded responses、packet反馈/NoCW、schema/授权/越权、唯一freeze/reveal/fit状态机、CLI/报告、资源和历史恢复。新增mockfit测试验证真实Row参数/当前L/validation/member seed以及成功恢复，未运行optimizer。原核心断言保留，旧registry“不支持provider”断言改为“已实现但必须授权、轨迹未资格化”，没有删除行为测试掩饰差异。

独立file-URL shallow clone checkout实际精简提交，`--no-local --no-hardlinks --single-branch --no-tags`。无alternates、共享对象、sparse或原工作树文件；清空全部私有fixture/PYTHONPATH，禁用user site，历史fixture指向不存在目录。**unit 187 passed/0 skip**；无私有冻结fixture **36 BLOCKED skips**。当前源码全部从clone导入，JSON合同可加载；8个真实LLM录制选择回合通过；合成fresh/resume全部工件hash相同，validate/dry_run/report通过。

测试绑定除本报告外的路径/模式/blob摘要 `640455ff0db02d5804c0d5c894faf3456779a689bb4ea8043d7ef18c3296d2a8`；补充报告不改变代码/资源/测试，最终提交复核见 `CLEAN_CLONE_RECHECK.json`。这是新clone在已准备依赖环境中的可用性证明，不是空白环境依赖安装或生产资格。

可重复完整外部回归命令（仅已有授权只读fixture）：

```sh
QGEOGNN_READONLY_REGRESSION_MANIFEST=/private/tmp/column_phase2_science_snapshots/MANIFEST.json \
QGEOGNN_REGRESSION_MANIFEST=/private/tmp/column_phase2_science_snapshots/REGRESSION_MANIFEST.json \
QGEOGNN_FROZEN_ACQUISITION_MANIFEST=/private/tmp/column_phase2_frozen_acquisition_inputs/MANIFEST.json \
QGEOGNN_K3_REGRESSION_MANIFEST=/private/tmp/framework_v3_20261112/private_k3_snapshots/MANIFEST.json QGEOGNN_PRIVATE_AGENT_REPLAY_MANIFEST=/private/tmp/framework_v3_20261112/private_llm_snapshots/manifest.json:/private/tmp/framework_v3_20261112/private_scientist_snapshots/manifest.json QGEOGNN_FULL_ROW_REGRESSION_MANIFEST=/private/tmp/framework_v3_20261112/private_row_validation/manifest.json QGEOGNN_HISTORY_REPOSITORY="$PWD" python -m pytest -q
```

本轮实际CPU环境另固定 `KMP_DUPLICATE_LIB_OK=TRUE`、OMP/OPENBLAS/MKL线程1、PYTHONDONTWRITEBYTECODE=1；Row初始化内部按原source设CPU2。未显式配置外部fixture时冻结类明确BLOCKED_EXTERNAL_NOT_CONFIGURED skip，不自动fetch或借main；配置后本机真实回归通过。没有GPU确定性资格。

## 匹配预算的指标与明确缺口

实际冻结K3 member0权重重新推理已有授权validation：每seed先确认仅749行=333 L0+416 validation及ID hash，然后才解析已授权标签。五seed L333/L365 Combined Row validation NRMSE与原fit_audit完全一致，两点normalized AULC333–365差0：

| seed | L333 NRMSE old=new | L365 NRMSE old=new | AULC333–365 old=new |
| --- | ---: | ---: | ---: |
| 157 | 0.8538865655827187 | 0.7828789033656687 | 0.8183827344741936 |
| 887 | 0.5846035924384305 | 0.5122659748814721 | 0.5484347836599512 |
| 2357 | 0.6419540403146382 | 0.5742062392371617 | 0.6080801397758999 |
| 6101 | 1.0208867471225853 | 0.9856056266553619 | 1.0032461868889735 |
| 12203 | 0.6556567752043931 | 0.6292459238425397 | 0.6424513495234665 |

这是既有权重的重放，区间仅333–365，不外推1005/1485；没有newfit、test评价或新标签。checkpoint/oldfit_audit/validation ID、数据角色和数组hash、dtype/容差在 `private_k3_snapshots/VALIDATION_MANIFEST.json`；聚合frozen test可直接重复。

原 `research/` 的最终qualification和5seed Free/LCMD validation统计保持原身份。下表仅抄录已公开历史aggregate，非candidate新fit或新评价，也不能跨test/validation混比：

| 历史数值profile，5seed test | L333 Combined Row NRMSE | L1005 Combined Row NRMSE | 同网格normalized AULC |
| --- | ---: | ---: | ---: |
| Random | 0.706752581 | 0.543301006 | 0.624260329 |
| Numerical Hybrid | 0.706752581 | 0.409709183 | 0.479353509 |
| Gradient-LCMD | 0.706752581 | 0.408613890 | 0.479538594 |

Free/LCMD另一个validation L1485对照：endpoint0.334085/0.347582、AULC0.439159/0.417049，保留mixed-model前缀与授权例外，见 [原说明](../research/FIVE_SEED_AL.md)。NoCW与CW只使用确切共同预算的历史状态，未拼接成统一最终结果。

**所有profile从头重新训练的最佳epoch/checkpoint、完整NRMSE/AULC及重训prediction diff仍NOT_MEASURED。** 已测的冻结checkpoint两预算validation与原历史表分别标注，均不冒充重新优化或完整训练轨迹。恢复原始私有完整recordings或另获训练/在线授权，才可进行下一科学门槛；本轮没有自行生成该授权。

## 远端历史、实际精简与main保护

清理前两个annotated tag均push并ls-remote匹配，独立HTTPS partial clone全tree路径/模式/blob身份相同并真实恢复17份源码逐SHA比对：

| 标签 | annotated对象 | peeled commit |
| --- | --- | --- |
| archive/framework-v3-main-source-20261112 | `f0acbaa9a4281a54ef2b64db564936c8234b095a` | `b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28` |
| archive/framework-v3-pre-migration-20261112 | `3a6405c68f4824a34ec3b8f4ebe3edbc0322b2c7` | `33f0ab46020c9175b23aff9741d143cb4a869e9a` |

完整22,294-file旧研究仍由973ccaf8的V2归档恢复。仅tracked字节可取回，ignored/private checkpoints、请求、凭据、运行状态和环境未由Git证明备份；本轮不物理删除它们。

先完成实际功能/回放，再从当前树退出唯一旧施工验收 `docs/FRAMEWORK_V2_ACCEPTANCE.md`（15,875B），原字节在33f0ab460归档。其它当前文件为必要职责模块/长期测试/一个真实golden，未堆回studies/reports/archive/legacy。精简基线已是V2的148文件，功能新增有实际用途，不能为机械降低数量删掉真实能力。

| 顶层 | tracked files before→after | tracked MB before→after | 子目录 before→after |
| --- | ---: | ---: | ---: |
| 根文件 | 7 → 9 | 0.008443 → 0.010188 | 0 → 0 |
| application/ | 4 → 4 | 0.287213 → 0.287213 | 0 → 0 |
| configs/ | 6 → 6 | 0.188211 → 0.202684 | 0 → 0 |
| dataset/ | 10 → 10 | 1.066811 → 1.066811 | 0 → 0 |
| docs/ | 6 → 8 | 0.030410 → 0.058041 | 1 → 1 |
| experiments/ | 47 → 47 | 28.366568 → 28.366568 | 15 → 15 |
| reports/ | 0 → 0 | 0.000000 → 0.000000 | 0 → 0 |
| research/ | 7 → 7 | 0.065510 → 0.065510 | 0 → 0 |
| scripts/ | 2 → 2 | 0.003821 → 0.006761 | 0 → 0 |
| src/ | 38 → 57 | 0.157659 → 0.339551 | 9 → 9 |
| studies/ | 0 → 0 | 0.000000 → 0.000000 | 0 → 0 |
| tests/ | 17 → 28 | 0.098851 → 1.769004 | 1 → 2 |
| tools/ | 4 → 4 | 0.033882 → 0.034441 | 0 → 0 |
| 总计 | 148 → 182 | 30.307379 → 32.206772 | 36 → 37 |

表以33f0ab460与最终Git树统计；MB为tracked完整checkout字节/1,000,000，目录为顶层以下子目录（总计为全仓库内部目录，不含仓库根）。完整精确bytes与删除path在临时 `FINAL_TREE_STATS.json`，没有向docs复制inventory。当前studies/reports仍0，experiments保留47项/28,366,568B最小受保护资源，9份原始CSV、application、模型/选点数学及三份原机器合同原字节不变。没有其它物理数据删除或新平行branch/worktree。

执行中main由独立结果归档推进到 `44be2c0070219d71b2560604f8234a390cc14304`（`results: archive completed five-seed CW Free32 experiment`），39个reports/scripts路径变化；本任务未向main发起commit/checkout/reset，保留该进展。最终只读核对原623份源码、13份固定控制/来源身份的SHA+inode仍精确不变，main暂存为空，index与观察到该外部提交时一致；原两项未提交状态不stage/覆盖。回放来源继续固定b1a6ed7a，不把外部结果归档混入本次等价试验。原26个ignored路径保留。没有向main写入、发信号、停止/重启控制器；不把PID/进度自然变化视为本任务写入。

仅保留main、迁移学习archive-transfer、原重构分支三条项目branch。新代码仅提交原重构分支，本轮push的是上述历史tag；未自动push代码分支或merge main。

私有snapshot仅为本轮独立只读验证副本，原文件保持原位；临时目录并非长期受控备份，未将私有bytes加入Git。大量命令日志/JSON/XML都在 `/private/tmp/framework_v3_20261112/`（临时目录可能被系统清理）：`REMOTE_SOURCE_RESTORE.json`、`BASELINE.json`、`FINAL_MAIN_CHECK.json`、`EXTERNAL_MAIN_ADVANCE.json`、`CLI_LLM_REPLAY.json`、`FINAL_ALL_TESTS.*`、`CLEAN_CLONE*.json`、`FINAL_TREE_STATS.json`以及各能力子目录。可长期复用的最小golden、source metadata与科学合同保留在Git；报告不承诺原环境/私有资源完整重建。
