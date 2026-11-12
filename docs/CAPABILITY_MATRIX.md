# 当前能力矩阵

2026-11-12；**FULL READY = false**。源为main `b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28`，精确版本、候选/可见信息、训练与transport差异见 [来源映射](LEGACY_TO_NEW_CAPABILITY_MATRIX.md)，执行命令/hash及阻塞见 [等价报告](EQUIVALENCE_REPORT.md)。

`入口`均为 `python -m scripts.run_al`；`I`表示代码实现及无真实动作集成通过，`G`表示需要独立授权、没有本轮真实测量，`B`表示本项尚未验证或所需外部fixture未配置，不推断原物理工件缺失。实现状态不等于科学等价或生产资格。

| Profile | CLI entry | dry-run | recorded-response replay | real fit | real provider | result equivalence / blocker |
| --- | --- | --- | --- | --- | --- | --- |
| free32_cwaware | schema3 config | I | EXACT：5seed×r0/1（外部） | G | Responses G | EXACT_REPLAY_PASS：独立私有snapshot五seed r0/1；完整pool/Prompt/messages/纠错/ID/freeze精确，重新训练NOT_VERIFIED |
| free32_nocw | schema3 config | I | EXACT：5seed×r0/1（外部） | G | Responses G | 五seed初始profile与paired_v1分开；原catalog/Prompt/完整消息/纠错/选择精确，后续无界transport修订未迁移 |
| cw16_llm16_screen_v1 | schema3 config | I | EXACT_REPLAY_PASS：157/6101各r0/1/2 | G | Responses G | 原packet/validator/fallback/ordered32/freeze匹配；原outgoing messages UNKNOWN；fit/reveal/metrics B |
| cw16_llm16_scientist_v2 | schema3 config | I | EXACT_REPLAY_PASS：157/6101各r0/1及原纠错 | G | Responses G | 全池packet/Prompt/messages/request SHA/ID/freeze一致；6101外部私有snapshot同样通过；完整重新训练NOT_VERIFIED |
| free16_cwgradient_lcmd16 | schema3 config | I | 复用上行的明确谱系别名 | G | Responses G | 原图例读取同Scientist路径；没有虚构新的独立实验 |
| random32 | schema3 config | I | 数值录制接口I | G | 不适用 | 固定U0 permutation/seed/tie核心通过；完整真实fit轨迹及新指标NOT_VERIFIED |
| gradient_lcmd32 | schema3 config | I | 数值录制接口I | G | 不适用 | NUMERIC_EQUIVALENCE_PASS：157 L333/L525/L973三个保存批次32-ID顺序完全一致；不是连续重训 |
| cw_gradient_lcmd32 | schema3 config | I | 数值录制接口I | G | 不适用 | CW提取/dispatch通过；原短期实验157/6101，真实完整CW bank/多轮fixture B |
| gradient_maxdet32 | schema3 config | I | 数值录制接口I | G | 不适用 | 原MaxDet核心/reference/tie通过；完整真实轨迹B；不可替代下面的K3门控版 |
| kernel_ivr32 | schema3 config | I | 数值录制接口I | G | 不适用 | 固定L0+U0 reference及核数学通过；完整真实fit/metric轨迹B |
| numerical_hybrid32 | schema3 config | I | 5seed×r0/1真实三权重/完整bank通过 | G，member0/1/2路径I | 不适用 | 同seed/同L三checkpoint供应与真实fit编排已实现；30个真实checkpoint已测，完整U/3330-row latent/ranking/32ID精确；L333/L365 validation NRMSE及两点AULC差0；未重训练 |
| free32_cwaware_paired_v1 / free32_nocw_paired_v1 | 各自schema3 config | I | B | G | Responses G | 保留原COMMON_PROMPT共同修改，不冒称当前CW-Free与实际five-seed NoCW是单字段因果对照 |
| uncertainty_gated_maxdet32 | NOT_IMPLEMENTED | 不接受 | B | NOT_IMPLEMENTED | 不适用 | 另一个top50% K3门控MaxDet；UNKNOWN_CURRENT_NEED，不能混同gradient_maxdet32 |

## 通过层次与缺口

- **NUMERIC_EQUIVALENCE_PASS**：五个真实Row partition、15组初始化tensor、已有五seed L333六输出/hidden/gradient/CW、三批真实LCMD、五seed两轮真实K3及匹配预算validation，以及纯kernels的独立参考与排序。每项范围分别列证据，不能合并成五seed完整AL轨迹。
- **EXACT_REPLAY_PASS**：30个独立真实录制选择回合（内置8+外部22），以及可恢复的packet/Prompt/纠错/选择/freeze层。Screen没有原始outgoing transcript；不能用重建messages替代缺失证据。
- **INTEGRATION_PASS**：唯一多轮workflow、Row fit adapter、完整packet/反馈、NoCW边界、Responses mock、K3 member编排、成功结果恢复。使用自有合成输入或mock的项目明确标明。
- **NOT_VERIFIED**：相同历史checkpoint选择/最佳epoch的重新训练、五seed真实多轮、完整重新训练的K3及所有profile新训练的Combined Row NRMSE/AULC。已测K3的L333/L365冻结模型validation区间不外推完整轨迹。
- **ONLINE_SCIENCE_NOT_MEASURED**：所有在线profile；没有新付费调用、真实optimizer或真实新标签揭示。

Responses transport已实现且必须授权；原Codex CLI、CW-Free direct-SSE持久化完成恢复、H2/403/upstream错误调度与后续无界重试 **NOT_IMPLEMENTED**。未知dispatch结果使用新的fail-closed边界，不标历史transport/recovery EXACT。原V3/V3.2及旧简单Free/CW/NoCW标签继续作为固定历史协议，不自动映射到新profile。

已证实原ignored私有工件实际存在，并经freeze哈希链受控快照验收；未入Git、不是永久备份。当前只能在另获明确授权后测量真实训练/在线smoke；本次未生成授权文件，未接管main实验。用户可先使用无付费 [快速开始](../README.md) 和 [回放指南](ARCHITECTURE.md)。
