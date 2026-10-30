# Loading机制：支持、反证与未识别部分

阶段A两seed333→813的结论是暂停阶段B实现。不是认定loading无用，而是loading-only尚不能解释Hybrid表现。

| 假设 | 支持的观察 | 反证/局限 | 当前判断 |
| --- | --- | --- | --- |
| H1 Free过度loading导致早期覆盖不足 | Free全批严格单轴补全更多、同分子集中更多，整体raw/CW覆盖较弱 | 宽loading为Hybrid稍高；6101 PE差更大；157早期loading差很小；标签价值未知 | 行为部分相容，过度与因果未识别 |
| H2 CW16覆盖、LLM16补充 | CW16同状态几何增益为大头；LLM仍有额外约1.1%覆盖，直接分子重叠低 | 新分子不是CW配额；覆盖不是validation因果收益；LLM后段也很局部 | 与分工相容，不能认定机制有效性 |
| H3 pending CW可能改变内部LLM loading偏好（不限定方向） | 看到seed和阶段间的偏好差异，6101最早阶段LLM严格loading少 | 157早期LLM比Free更高；全段6101LLM高于Free；pending与16/32名额、路径等混杂 | 异质变化不能归因pending；统一降低的更强子命题不成立 |
| H4 限loading改善Free效率 | 极集中批可由固定cap改变 | 没有同版本同状态实验；cap不保证补足覆盖；早期binding异质 | 不足以启动原假设下的闭环 |

必须分开的三个问题：严格loading是否较多（可从冻结ID回答）；这些标签是否浪费（未回答）；控制它是否提高AULC（需要新的公平实验）。两个配对seed不支持统计“显著”，轮次和4800条来源视图不是独立重复。

Free与Hybrid系统Prompt主版本字节一致。历史Astra/Sol与reasoning切换、pending16和selection_count16/32、不同L_t/checkpoint/记忆、授权纠错与seed均仍可混杂。旧test曝光属于证据历史，不能因本次未读而抹去。

可审计数值与逐seed、逐阶段表在REPORT.md及相应CSV。当前不冻结主K；8/12仅为行为强度候选，未用validation择优。更合理的下一步是同版本同状态单轮配对选择，确认干预改变的到底是loading还是覆盖/重复，以及纠错是否造成额外差异。单轮不能代替长期AL效果。
