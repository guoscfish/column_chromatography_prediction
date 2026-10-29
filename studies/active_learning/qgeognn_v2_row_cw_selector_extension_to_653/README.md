# 研究记录：qgeognn_v2_row_cw_selector_extension_to_653

本页合并该研究原先分散的说明、方法、审计或资源说明；不新增训练、标签访问或研究结论。

以下按原文件分节合并，日期、方法与数值保留其历史语境；其中的后续建议不代表当前待办。当前进度以[研究状态](../../../docs/NEXT_STAGE_DECISION.md)为准。

<a id="记录-1"></a>

## 记录 1：`README.md`

# CW selector factorial: exploratory continuation to653

Authorized after inspecting the completed333–525 development results. Three new arms × two seeds × four additional batches =24 new fits maximum. Own exact L525 histories continue; all retraining remains scratch from the original shared initialization. Forty historical control budget points were audited and reused. No LLM calls or new test labels.

Primary descriptive extension metric: normalized validation AULC525–653. Secondary: AULC333–653, NRMSE653, V1/V2 RMSE/MAE/R². Original AULC333–525 conclusions and failed new-selector confirmation gates remain unchanged. This extension is post-development exploration, not confirmation. Hard stop at653 regardless of effect.

Protocol/code/input hashes: protocol.json. Running/completed state: RUN_STATE.json and fit_ledger.json. Runtime remains local and ignored. A partial fit retains its quota reservation rather than silently retrying.

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_extension.py run
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_extension.py report
```

## User stop amendment (current scope)

Whitening-LCMD stopped at525 for both seeds before any extension fit; preserve its historical result. Continue CW-MaxDet and CW-IVR only. Effective additional fit limits:16 through653 and44 thereafter,60 total. Seven arms × two seeds reach1005; Whitening's two curves end525. No late-window Whitening comparison is available. The original code/protocol is retained; the explicit stop amendment overrides execution membership and quota only.

Use `python scripts/studies/continue_cw_without_whitening.py` in the fish environment with `KMP_DUPLICATE_LIB_OK=TRUE`. Do not resume the old unamended supervisor.

## Verified delivery

Fourteen curves completed through1005; Whitening remains at525 for both seeds by user instruction.60 additional fits (16 to653 plus44 to1005);150 control points reused. Zero paid LLM calls or new test reveals. Original333–525 conclusions unchanged; extension exploratory. See the1005 RESULTS.md and DELIVERY_AUDIT.json for final effects and verification.

<a id="记录-2"></a>

## 记录 2：`RESULTS.md`

# Four-round CW extension results

Completed16/16 new fits: 2 arms × two seeds × four batches (525→653). Forty historical control points reused. Whitening stopped at525 by user request.

This extension was authorized after inspecting333–525. Its frozen descriptive primary metric is validation AULC525–653; it is exploratory and cannot reverse the original confirmation gate or establish significance with two seeds.

| Seed | Arm | AULC525–653 | AULC333–653 | NRMSE653 |
|---|---|---:|---:|---:|
| 157 | cw_ivr | 0.495249 | 0.583045 | 0.435080 |
| 157 | cw_lcmd | 0.458227 | 0.553107 | 0.409144 |
| 157 | cw_maxdet | 0.526313 | 0.647940 | 0.502597 |
| 157 | random | 0.753380 | 0.805733 | 0.707850 |
| 157 | raw_ivr | 0.504889 | 0.596398 | 0.450012 |
| 157 | raw_lcmd | 0.461078 | 0.587176 | 0.403946 |
| 157 | raw_maxdet | 0.499351 | 0.585931 | 0.475949 |
| 6101 | cw_ivr | 0.519476 | 0.693898 | 0.487683 |
| 6101 | cw_lcmd | 0.499207 | 0.666688 | 0.461569 |
| 6101 | cw_maxdet | 0.573371 | 0.712537 | 0.535570 |
| 6101 | random | 0.751679 | 0.831658 | 0.696075 |
| 6101 | raw_ivr | 0.586244 | 0.717143 | 0.559875 |
| 6101 | raw_lcmd | 0.575534 | 0.726576 | 0.462123 |
| 6101 | raw_maxdet | 0.617688 | 0.738855 | 0.582805 |

## Paired late-window effects

Positive relative improvement favors the method; negative AULC/endpoint deltas favor the method. Selector-specific normalization and finite CountSketch effects remain part of these trajectories.

| Seed | Method / reference | Late AULC delta | Relative improvement | Endpoint delta |
|---|---|---:|---:|---:|
| 157 | cw_lcmd / raw_lcmd | -0.002852 | +0.62% | +0.005198 |
| 157 | cw_maxdet / raw_maxdet | +0.026962 | -5.40% | +0.026647 |
| 157 | cw_ivr / raw_ivr | -0.009640 | +1.91% | -0.014932 |
| 6101 | cw_lcmd / raw_lcmd | -0.076327 | +13.26% | -0.000554 |
| 6101 | cw_maxdet / raw_maxdet | -0.044317 | +7.17% | -0.047235 |
| 6101 | cw_ivr / raw_ivr | -0.066768 | +11.39% | -0.072192 |

Per-target RMSE/MAE/R² are in endpoint_metrics.csv; all eleven-point curves in validation_curves.csv and validation_curves.png. Raw values and per-seed effects take precedence over pooled rankings.

Training wall time: 3101.5s; acquisition: 718.8s. Exact costs and epochs: resource_costs.csv. Delivery audit verifies all authorized fits, unchanged initialization, own ordered label trajectories,32 unique IDs per batch and premeasurement prediction hashes.

## Reproduction and stopping point

No further training is needed. Verification/report commands:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_extension.py report
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/audit_cw_factorial_extension.py
```

The original no-new-selector-candidate gate remains unchanged. Any future new-seed study needs its own rationale, fixed candidate/control set and untouched-seed audit; P2 live LLM ablations remain unrun with zero paid budget. Historical test exposure must not be described as a new blind test.
