# Full-pool response-feedback LLM16: frozen development protocol

## Research question

For 4g Row-split QGeoGNN-V2 prediction, does a stateless LLM choosing 16
experiments in addition to 16 Center/Width-LCMD (CW) experiments improve
generalization and label efficiency? The target is V1/V2 prediction accuracy,
not the best chromatography condition. This is a two-seed development study,
not a statistical confirmation.

## Arms, labels, and training

- Seeds: 157 and 6101. Each uses its existing Row split and identical L333 IDs,
  initial checkpoint, fixed L333 target scales, validation checkpoint rule, and
  scratch retraining from the same initialization.
- Budgets: 333, 365, 397, 429, 461, 493, 525. Six batches of 32.
- A: historical pure CW32. Reuse only after split, configuration, lineage,
  checkpoint, and prediction hashes are audited. No A refit.
- B: CW16 then 16 uniform random IDs from its own entire current U_t minus CW16.
- C: CW16 then 16 LLM IDs from its own entire current U_t minus CW16.
- The existing 40 CW + 40 IVR + 40 MaxDet + 8 random, deduplicated and filled
  to 128, is a numeric reference set only. It cannot restrict C or B.
- No forced molecule diversity, new-molecule quota, warm start, or LLM fine tuning.

## Selector information and fixed budget

Model: `gpt-5.4`, Responses API, reasoning effort `medium`, default
temperature, stateless `store=false`. No automatic model switch. The exact
system prompt is `selector_prompt.txt`; `protocol.json` hashes the code and
source artifacts. Each decision can make at most 16 Responses API calls,
12 catalog queries, view 240 distinct candidate cards, and receive at most
24 results per query. Twenty numeric reference cards are shown initially.
The full candidate pool is searchable. An empty search returns an empty list.
Exhausting a budget raises a visible failure or requests a final answer; it
never creates a random LLM substitute. Retrying an incomplete call requires
`--retry-selector` and creates a separately numbered transcript.

The model receives a fresh request for each seed/round with only this
trajectory's own measured L_t rows, earlier choices, hypotheses, reasons,
measurement feedback, current model predictions, pending CW features, and
candidate overview. It can query current candidate details, exact conditions,
descriptors, same-molecule alternatives, fingerprint-near molecules, and the
original measured records. Catalog code opens no target or repository file.
There is no arbitrary shell, file, web, or code tool. The selector never
receives validation/test rows, another seed/method's response, prior test
metrics, or this development chat. The packet contains 24 illustrative
measured records plus IDs and tool access to every measured L_t record.

V1 and V2 predictions and truths are in mL. Model q50 outputs (columns 1 and
4 of the six-output checkpoint) are already in this physical scale. The
fixed L333 standard deviations normalize the final metric, not selector
responses. V2 minus V1 is elution interval width, not uncertainty. There is
no sample-out-of-sample error assigned to initial L333 records.

## Per-round barrier and recovery

The current checkpoint selects CW16 first. Its labels remain hidden. The
selector searches the remaining pool and returns 16 valid unique IDs with
reasons and structured hypotheses. The program validates membership, count,
uniqueness, evidence citations, packet identity, and budget. The 32 IDs and
their exact measurement-time q50 predictions are frozen together before any
new truth is revealed. One feedback record per ID then stores truth, signed
prediction error, source, reason, and hypothesis link. Subsequent rounds load
this trajectory's earlier raw feedback, not only a compressed summary.
Completed fits, selection freezes, and feedback are reused on resume after
hash checks. An absent or invalid independent LLM response pauses arm C.

## Reporting and conclusion limits

Primary: fixed-denominator combined NRMSE at each budget and trapezoidal
label-AULC over 333–525. Secondary: V1/V2 RMSE, MAE, R²; L525; each seed and
two-seed descriptive means; training seconds; API calls/tokens; candidate
views; outside-128 choices; new molecule and existing-molecule new-condition
counts; condition contrasts; hypothesis evidence history. The test truth
barrier opens only after all new trajectories and predictions freeze.

This version jointly changes full-pool access, real measured responses, and
context management. A performance change cannot be attributed to one factor.
The prior 128-candidate result and its failure criterion remain unchanged.
The next single validation, if promising, is a same-interface and same-budget
ablation without measured-response feedback. Identity masking (which also
changes input information) and more independent seeds are later suggestions.
