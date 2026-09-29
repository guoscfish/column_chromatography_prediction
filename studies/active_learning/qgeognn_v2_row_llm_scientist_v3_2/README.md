# Scientist V3.2 — seed157, one acquisition only

Registered scope: `free_llm32_scientist_v3_2`, clean Row L333 → select/freeze 32 → reveal → retrain from the same initialization → L365 → STOP. No L397 or hybrid execution exists in this version's registered runner. V3.1 remains a frozen pilot at `archive/scientist-v3.1-pilot-2026-09-29`.

The independent `active_learning_v3_2` protocol layer reuses stable V3 catalog/evidence/audit and V2 training/split/gradient primitives. The predictor Context and label-array construction are unchanged. Extra post-fit checks enforce identical initialization and validation IDs and zero test truth access. Validation scores are reported only after selection/training and never enter selector context.

Queries use machine-readable `query_api_schema`, cross-role `get_rows`, flat AND filters (`field`, `op`, `value`), explicit role restrictions and pagination. Only the unambiguous `get` alias is normalized, with an audit flag. Scientific budget: 24 accepted queries; separate validation repair allowance: four; at most 30 logical calls, with a finalization slot reserved. FINALIZE_ONLY exposes previously viewed IDs/SMILES plus working state, denies new queries and requests selection only. Full history stays in local audit. Diagnostics never impose quotas.

Provider capability was tested with no experiment data: both a deliberately conflicting minimal-schema probe and the complete response schema succeeded on token4research / gpt-6-sol. Live Responses requests use `text.format` JSON Schema with strict=true; the wire envelope is `{response: ...}`. Because strict Responses requires every property, optional query arguments are nullable on the wire; null means omitted and is decoded before the documented query validator. This is a registered wire convention, not scientific-content repair. See [official Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs).

Strict JSON recovery is retained defensively: accept pure JSON or exactly one complete top-level object surrounded by prose; reject multiple objects, arrays, unbalanced syntax, duplicate keys, nonfinite constants. Recovery never invents fields and is audited as `noncanonical_json_recovered=true`. Host working-state and scientific selection validation remain mandatory.

Hypotheses with latest directly linked observations or two rounds without updates require explicit retain/revise/support/weaken/reject review. Full belief history remains in the ledger. The breadth/depth prompt preserves chemistry priors, matched controls, condition curves and hypothesis discovery; acquisition-time residuals are explicitly historical measurement-time errors.

## Execution

Use the established fish Python environment and the dedicated runner. `prepare --source-root /path/to/original/repository` pins clean L333 catalogs, split, checkpoint, scrubbed graphs and preprocessing from the existing baseline. `stage` is label-safe; `select` performs only selection/freeze; `advance` performs exactly one reveal/retrain. No automatic loop is used. Source and input hashes are frozen before live selection.

Keep protocol, provider preflight, tests, compact outcomes, report and hash manifest in Git. Large initial catalogs, request/response streams, full audit and checkpoints remain in ignored local storage. Do not remove these local audit files; manifest hashes allow later integrity verification. The Git archive alone does not contain the new full runtime.

A second model batch-composition review is intentionally omitted: it would add an extra decision call; deterministic final diagnostics provide transparent analysis without changing acceptance criteria.
