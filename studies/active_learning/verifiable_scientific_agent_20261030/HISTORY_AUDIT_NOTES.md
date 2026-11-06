# 历史可靠性审计说明
固定本地截止：2026-10-30T16:17:05.764164+08:00。仅分析 capture 时 batch_freeze 已存在且 SHA 通过的批次。

| 方法 | 冻结批次 | 完整LLM回答 | 首次合法 | 需纠错轮 | 纠错请求 | 非法完整回答 | 传输失败记录 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Free | 180 | 270 | 112/180 (62.22%) | 68 | 90 | 91 | 305 |
| Hybrid | 71 | 93 | 51/71 (71.83%) | 20 | 22 | 22 | 99 |
| NoCW | 59 | 99 | 33/59 (55.93%) | 26 | 40 | 40 | 330 |

分母与限制：Free每seed36批（333→1485）；Hybrid157有36批、6101有35批（333→1485/1453），完整权威冻结尾端均保留；NoCW按本地快照冻结批次数逐seed见JSON。冻结合法率只说明已经冻结的批次满足执行约束，不能推断进行中/未冻结轮的最终成功率。

Free157 round05历史授权ID抄写修正使selection通过但无原始turn与selection完全一致；必须与LLM最终原始合法率分开。原始无唯一匹配的修正不在本次重做。

V2 transport retries未必纳入旧batch_freeze，已逐文件锁定为completed_snapshot_sidecar；若为CLI recovery/parser失败不保证产生一个新的付费请求。NoCW成功attempt与turn为同一次回答，绝不重复计数；started/stream中间内容不作为答案。

token以已保存provenance.usage为实测子计；失败网络请求可能缺token，费用均留空。旧V2普遍无耗时，NoCW可用elapsed_seconds；不估算计费或墙钟成本。

纠错前后只对严格JSON且choices为字符串ID对象列表的完整回答比较。合法候选子集与最终计划的分子/条件指标保留各自分母，不自动判定科学意图改变或科学质量优劣；几何冗余未在该脚本重算。

首错分类来自原始冻结validator纯函数AST的离线执行；候选错误细分扫描raw choices中明确的ID字段，允许同一回答同时有重复、已测和未知ID；schema不合法时该扫描不恢复实验批次。NoCW依照实际five-seed runner覆盖版：字段白名单和CW内容约束；未运行paired协议的额外prose ID禁令不适用于该历史。

本次新增LLM调用、模型训练、未观测真值表读取、test读取均为0。现有observed/catalog里的已观测信息仅用于合法证据和条件关系；未读取预测真值文件。

复现：`python studies/active_learning/verifiable_scientific_agent_20261030/audit_history.py`。首次快照：`--snapshot`，已有manifest禁止静默刷新。

补充核验：462份完整回答全部为原json.loads可接受JSON，原validator首错153/153与历史rejected.error一致。严格JSON歧义检查（重复键、NaN/Infinity、浮点溢出）无历史触发。只有460份满足可可靠恢复choice实验对象的schema；另外2份仍保留原始字段扫描，不恢复实验批次。

实际重复ID为0/462；历史"distinct legal IDs"报错是合并错误信息，不构成发现重复的证据。原validator首错：candidate_identity114、unobserved_evidence28、hypothesis_reference4、packet_binding2、count_or_choice_structure5。Raw字段扫描117份含池外引用、63份含已测ID、66份含未知引用，分类可重叠；严格choice-schema子集对应116/63/65，全部重复0。

152个追加纠错请求已在correction_request_causes.csv逐条绑定上一失败回答：113个由candidate identity、5个由count/schema、34个由证据/其他约束触发。对应完整回答额外输入tokens合计70,029,763，输出1,381,906；费用缺失。网络失败token未知，不包含在上述完整回答子计。

四个NoCW授权例外887r3、6101r8、2357r10、6101r10均在第4份完整回答后冻结，实际各用1个超出原三答案上限的回答，未用到第五/第六份。

2357r10：turn00有32个结构完整choices、30个唯一合法候选、2个未知ID，无重复；turn01仍有1个未知ID；turn02的choices数组37元素混入5个hypothesis形状对象，不可作为确定实验批次，也不自动摘取前32；turn03合法接受32。其最终32与最初30个合法ID无交集，说明完整重新输出可能显著改变选择，不能由此认定科学质量变差。

114个纠错轮的首次完整提案均可按严格choice结构分析；Free/Hybrid/NoCW分别12/1/8轮初始合法候选与最终无交集。分子集中和loading对照在部分轮增加、更多轮减少，均为不同合法子集分母的描述；不证明科学意图保持或损失。

按输入规模的描述分箱见input_scale_reliability.csv；候选U随L增长而下降，输入packet字节在约1.7MB的窄区间，版本/预算/历史共变，不能把错误因果归于长上下文或比较移除CW的因果效果。

NoCW完成回答99与传输失败330均有elapsed，记录请求耗时合计132,159.710684秒=36.711小时，59个已冻结批次均摊37.333分钟；是跨并发worker累加的请求时间，不含backoff/人工等待，不是墙钟历时。Free/Hybrid elapsed全部缺失，留NA。
