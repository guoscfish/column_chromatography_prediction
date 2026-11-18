# 受控体积曲线与案例

| case_id | al_seed | full_max_endpoint_mean_span_ml | full_min_endpoint_mean_span_ml | formal_response_span_spread_ml | al_L1005_endpoint_mean_span_ml | recovery_in_at_least_one_formal | all_three_formal_flat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 03668451123a | 6101 | 0.8486 | 0.0197 | 0.8289 | 0.0366 | False | False |
| 31b6feb11194 | 887 | 0.7115 | 0.0122 | 0.6994 | 0.0019 | False | False |
| 322dbd2dcede | 6101 | 0.4912 | 0.0077 | 0.4835 | 0.0120 | False | False |
| 326fc4a36721 | 12203 | 0.2421 | 0.0056 | 0.2365 | 0.0109 | False | False |
| 3e6ba54c34fb | 887 | 0.8448 | 0.0139 | 0.8309 | 0.0032 | False | False |
| 41c41ae63cc7 | 2357 | 0.9504 | 0.0157 | 0.9347 | 0.2988 | False | False |
| 5b16b5f667c6 | 12203 | 0.9256 | 0.0213 | 0.9043 | 0.0387 | False | False |
| 6e0017524ddf | 157 | 1.0832 | 0.0262 | 1.0570 | 0.7558 | False | False |
| 6ef7df527768 | 157 | 0.2680 | 0.0062 | 0.2618 | 0.1817 | False | False |
| 73795122affd | 887 | 0.7190 | 0.0166 | 0.7024 | 0.0019 | False | False |
| a1ec3f387018 | 2357 | 0.2998 | 0.0049 | 0.2948 | 0.0942 | False | False |
| af59e5070744 | 157 | 0.7263 | 0.0168 | 0.7095 | 0.4925 | False | False |
| d996614cb9c7 | 2357 | 0.9029 | 0.0150 | 0.8879 | 0.2838 | False | False |
| f0226164aec1 | 12203 | 0.9474 | 0.0198 | 0.9276 | 0.0421 | False | False |
| fb2a0fcbd6a3 | 6101 | 0.8073 | 0.0187 | 0.7886 | 0.0346 | False | False |

15个series在新推理前按metadata hash固定，每AL seed最多3个，全部原观察端点对三个正式模型均非test且不触及隔离ID；没有按预测好坏扩展案例池。源条件只改变V/ul，通过原图构建器同步更新Density×V；没有修改加载溶剂体积。请求网格30/60/90/120/150µL在全域合法范围内，超出该series实际最小/最大体积者另标局部外推，仅作敏感度图，不计算真实误差。

图中从固定池按预定规则选择部分恢复、全三seed平坦、seed响应差异案例，具体选择见`CURVE_EXAMPLE_SELECTION.json`。恢复只指至少一个正式seed，不冒充三seed都恢复；若没有全三seed平坦或其他类型，明确标criterion absent，使用固定池剩余最小hash作展示，不补选外部案例。

| category | case_id | criterion_met | rule |
| --- | --- | --- | --- |
| Partial recovery | 03668451123a | False | First hash in the frozen pool satisfying the declared threshold; if absent use first remaining metadata hash and explicitly label absent |
| Flat responses | 31b6feb11194 | False | First hash in the frozen pool satisfying the declared threshold; if absent use first remaining metadata hash and explicitly label absent |
| Seed disagreement | 6e0017524ddf | True | First hash in the frozen pool satisfying the declared threshold; if absent use first remaining metadata hash and explicitly label absent |

`controlled_volume_curves.csv`保留全部15个案例，不只保留画图的三个。span统计只用局部支持范围内的共同网格点，且区分响应span差异和绝对预测水平差异。真实值仅叠加已授权、并已在对应AL H525中的记录；模型与AL的曝光并不相同，曲线不能证明物理因果、LLM增量价值或泛化误差。
