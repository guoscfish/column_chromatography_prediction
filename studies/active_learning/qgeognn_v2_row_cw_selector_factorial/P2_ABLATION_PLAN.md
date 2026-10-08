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
