# Free LLM32与LCMD32：五seed完整历史基线

状态：PRIMARY_FROZEN。原最终报告于2026-10-29采集，10/10条轨迹COMPLETE。outer seeds157/887/2357/6101/12203；4g row划分、每seed outer-train3330、固定validation416；L0=333、batch32、36次采集至L1485，37个冻结拟合点/轨迹。初始状态与已完成前缀可复用，因此这些逻辑点数不表示本轮新拟合成本。

## 同预算结果

| seed | Free validation NRMSE@1485 | LCMD validation NRMSE@1485 |
| --- | ---: | ---: |
| 157 | 0.331755 | 0.330408 |
| 887 | 0.274699 | 0.304093 |
| 2357 | 0.339940 | 0.366228 |
| 6101 | 0.375767 | 0.380761 |
| 12203 | 0.348263 | 0.356421 |

| 指标（越低越好） | Free | LCMD |
| --- | ---: | ---: |
| validation终点均值±seed样本SD | 0.334085 ± 0.037097 | 0.347582 ± 0.030467 |
| L333–1485归一化AULC均值 | 0.439159 | 0.417049 |

Free终点均值低0.013497（相对3.88%），但整个标签预算区间AULC较高。端点、全程标签效率和计算成本是不同问题，不能只拿终点宣布全程更优。3330标签全训练参考均值0.325433；LCMD后段趋缓不等于已达到全量性能。

## 科学身份与混杂

LCMD是冻结raw q50 gradient CountSketch512 + corrected LCMD-TP，使用当前模型特征并沿用原初始化、scratch训练、validation与batch32；不是后来CW变换的LCMD。Free的157/6101保留Astra/high历史前缀，其余三seed为gpt-6.1-sol/xhigh。多处单轮纠错/ID抄写例外有显式授权，原失败/成本回执保留。因此这是已完成的历史开发比较，不是模型/Prompt完全一致的新因果消融或独立确认。

指标是已保存validation开发轨迹；报告不新增test评价，validation指标不进入selector，标签在batch freeze后揭示。不是未使用过的确认性cohort。

本地 [free_lcmd_validation_curves.json](free_lcmd_validation_curves.json) 是原冻结 `figures/validation_curves.json` 的原字节副本，SHA256 `ad789470ca76b6a98ab5eac191054a088c5f97a7bfd0800a0e4ec1eb63b97115`。它还包含Random/Hybrid参考，但两者只有L333–1005的22点，不能外推到1485或拼接为五seed长程对照。L1485的Free/LCMD各37点均完整保留。

## 原始来源

[权威最终报告](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md) · [LCMD扩展身份](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/reports/free_lcmd_five_seeds_to1485_20261026/lcmd/extension_manifest.json) · [Free amendment](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/reports/free_lcmd_five_seeds_to1485_20261026/free/continuation_amendment.json) · [完整冻结核验](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/reports/free_lcmd_five_seeds_to1485_20261026/final_audit.json) · [历史授权/混杂](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/reports/llm_eight_to1549_20261018/historical_protocol_deviations.json)。

原选择序列、每轮ID/顺序、请求/失败谱系与已有tracked回执均在archive；Git不保证ignored模型、逐fit预测和私有runtime存在。当前稳定Agent未声称可端到端重跑该历史LLM协议。
