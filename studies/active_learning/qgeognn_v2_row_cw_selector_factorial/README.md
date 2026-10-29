# 研究记录：qgeognn_v2_row_cw_selector_factorial

本页合并该研究原先分散的说明、方法、审计或资源说明；不新增训练、标签访问或研究结论。

以下按原文件分节合并，日期、方法与数值保留其历史语境；其中的后续建议不代表当前待办。当前进度以[研究状态](../../../docs/NEXT_STAGE_DECISION.md)为准。

<a id="记录-1"></a>

## 记录 1：`README.md`

# CW selector factorial and offline feedback ablations

See PROTOCOL.md, RUN_STATE.json, MECHANISM_AUDIT.md and RESULTS.md. Primary development window is validation AULC333–525. Runtime stays local and ignored.

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_geometry_feedback.py run
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_geometry_feedback.py report
```

Completed fit hashes are verified before reuse. An interrupted fit without fit_audit.json stops explicitly: its reservation is retained and another attempt must be counted within the 36-fit cap. No automatic LLM transport is provided or enabled.

Completed: P0/P1a/P1b/P2 offline; exactly36 new fits,70 reused control points,31 regression tests passed. Neither new selector met the confirmation gate. No paid LLM requests or new test-label reveals. See RESULTS.md and GLOBAL_MATRIX_FREEZE.json.

<a id="记录-2"></a>

## 记录 2：`MECHANISM_AUDIT.md`

# Geometry and compatibility audit

Completed on the actual two seed-specific L333 checkpoints. Eight rows per seed (first four L, first four canonical U), 458,952 parameters, two output gradients; only these small in-memory Jacobians were formed, never saved. All transform estimates use L333 targets. Full-dataset gradient extraction continues to use the unchanged streaming production extractor.

Uncompressed relative Frobenius Gram discrepancy is at most 1.39e-15 for Raw versus 45° Rotation and 1.33e-15 for CW versus H=sqrt(TcwᵀTcw). Shared task projection with two half-sized blocks preserves these isometries to at most 1.40e-15. The task blocks are a diagnostic only.

Production concatenated CountSketch was diagnosed with seeds 17,101,1009 and total dimensions 256,512,1024. Across this small sample, Raw/Rotation pair discrepancy ranges 2.60%–20.02%; CW/H discrepancy ranges 2.40%–12.49%. These are substantial finite-projection effects, not a historical implementation bug. Overall Gram errors range 1.62%–21.66%. No production mapping or dimension was changed after examining these results. Do not attribute all measured CW benefit to the exact output metric independently of sketch orientation. Follow-up, if warranted, must preregister a projection-robust confirmation rather than alter the current curves midstream.

`geometry_audit.csv` contains all 166 measurements; it reports relative Frobenius errors, not an estimate of population-wide distortion. Eight rows cannot establish global sketch quality. Formal acquisition uses the existing 512-dimensional sketch and outer_seed+4000037.

`compatibility_audit.json` records exact first-batch Raw-LCMD/CW-LCMD replay for both seeds and independent re-extraction checks on the small subset. State hashes were unchanged. `reuse_compatibility.json` verifies 70 control points: identical checkpoint training config (including split hash), preprocessing, condition normalization, initialization, own-label ordered trajectory, L/U partition, canonical U order, no repeated acquisition, and saved checkpoint/prediction hashes. CW's 429→461 batch is in the extension study, while its L429 checkpoint belongs to the short-sequential study; this lineage was explicitly joined, not refitted.

29 tests passed, including the existing direct logdet and explicit IVR covariance reinversion oracles, deterministic ties/no duplicates, gradient extraction and new representation/selector dispatch and ablation boundaries. MaxDet normalizes by current-L feature RMS; IVR by the fixed L+U universe RMS. Their differences are not solely objective-function effects. CW-IVR remains a scalar row-kernel proxy rather than an exact multi-output V1/V2 posterior-risk derivation.

Supplemental diagnostics: `historical_sketch_audit.json` verifies unchanged 512D mapping provenance in historical feature audits; `small_row_selection_audit.csv` compares all five representations and all three selectors against uncompressed small-row Gram factors over the registered projection grid (270 selection checks). These are eight-row selection diagnostics, not additional training trajectories.

Final delivery suite on2026-10-09:31 tests passed (including two real-packet adapter boundary tests). All36 formal fit/checkpoint/batch-freeze audits passed; the all-matrix prediction receipt is GLOBAL_MATRIX_FREEZE.json.

<a id="记录-3"></a>

## 记录 3：`P2_ABLATION_PLAN.md`

# LLM feedback ablations — offline implementation, live execution deferred

Historical Full code, prompts, pending CW16 baseline and trajectories are unchanged. `feedback_ablation.py` implements four variants of a typed selector surface and fail-closed field validation. Candidate/observed/pending must be disjoint, selection must contain 16 eligible unique IDs, and a frozen packet cannot be changed or rebound to another batch. A stable private trajectory HMAC key maps row and compound IDs consistently across variants and rounds; neither key nor original identity enters the anonymized packet.

No-narrative preserves structured numeric responses, premeasurement predictions/errors and acquisition timing, removing all free text history. No-direct-response additionally strips endpoint observations, historical errors, premeasurement feedback, before/after predictions and response-derived summaries. Current model predictions still reflect training labels. No-chemistry removes SMILES/names/functional-group strings, descriptors and all potentially leaking historical narrative; it retains conditions, stable anonymous groups, predictions, numeric feedback and explicitly enumerated geometric/novelty signals. These numeric features retain chemical origins. Unknown fields and text smuggled into numeric fields are rejected rather than recursively copied.

The new offline schema is an input boundary for future runs, not a modification of the historical Full packet format. The read-only adapter now maps existing Full column tables and structured memory into this schema; 16 variant audits cover both seeds at rounds0/1. Future live integration must freeze an adapted common prompt/response schema for all new arms and test the final serialized requests; the existing scientific prompt discusses SMILES and molecular dictionaries and cannot be reused verbatim for the no-chemistry surface. The offline fixture is synthetic and makes no scientific efficacy claim. Full historical request bytes stay frozen; new trial arms must have independently acquired histories.

Minimum next matrix: CW32, CW16+Random16, Full, No-narrative, No-direct-response, No-chemistry, seeds157/6101 from L333 to525. Each paid arm has 12 selection events, six per seed, with explicit finite transport/correction ceiling inherited from the frozen Full revision. No automatic fallback, manual ID correction, or replay of Full choices. Positive dollar and call budgets require a later authorized config; current config hard-sets both to zero and contains no transport. Save request time, parameters and returned model identifier; service drift is uncontrolled and must be disclosed.

The historical Full revision's protocol/prompt hashes and verified completed checkpoints are in LLM_HISTORY_AUDIT.json. The dated source report records quota exhaustion; this round made no provider probe. Historical deviations include seed157 Free round6 extra correction plus transcription repair and seed6101 Free round1 one extra correction; neither authorizes new repairs. Existing run_status.json is an old interrupted Free snapshot and is not the authoritative completed-budget inventory. The all-target global test barrier remains incomplete; no test reveal was performed.

If Full does not consistently beat CW32 on the preregistered overall validation metric, report the ablation effects without claiming improved total label efficiency. Later confirmation needs frozen new measurements/test regions or a second system; selected-row error reduction is not evidence of generalization.

Offline rerun:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_geometry_feedback.py offline
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q tests/active_learning_v2/test_cw_factorial.py
```

The adapter retains PE/EA and solvent categories via a fixed PE=0, EA=1, DCM=2 dictionary. No-direct-response removes high-error history membership/order in addition to its numerical values; recent selection IDs are separately retained without response ranking. P2_REAL_PACKET_AUDIT.json stores hashes/counts only, never raw historical request contents.

Handoff update (2026-10-09): an independent historical-LLM task advanced seed6101 hybrid to621. LLM_HISTORY_HANDOFF_SNAPSHOT.json records the current checkpoint-verified budgets read-only. These external calls/fits are outside this factorial study and were neither initiated nor committed here. The original LLM_HISTORY_AUDIT.json remains the initial2026-10-08 snapshot. The incomplete Free6101 trajectory still prevents the historical full global test barrier.

<a id="记录-4"></a>

## 记录 4：`PROTOCOL.md`

# Protocol

The authoritative preregistration is [the research plan](../../../docs/research/CW_GEOMETRY_AND_LLM_FEEDBACK_PLAN.md). Machine-readable parameters, L333 transforms, package versions and data hashes are frozen in protocol.json; execution_code_freeze.json binds the implementation. REUSE_MANIFEST.json and reuse_compatibility.json audit 70 historical control points. New fits use the historical CPU two-thread training protocol, with unmodified production CountSketch and independent trajectories. Historical test exposure is acknowledged; this runner contains no test-label reveal operation.
