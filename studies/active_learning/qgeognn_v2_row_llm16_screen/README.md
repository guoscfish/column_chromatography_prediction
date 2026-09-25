# CW16 + LLM16 development screen

Test whether an LLM selecting half of a 32-record batch improves active learning
with the existing QGeoGNN and training procedure unchanged.

Completed: 12 new fits, six LLM decisions, no failed responses or fallback picks.
Mean AULC is 0.804957 for CW32, 0.811373 for CW16 + LLM16, and 0.839394 for
CW16 + random16. The frozen extension gate failed. See [FINAL_REPORT.md](FINAL_REPORT.md)
and [selection_integrity_audit.json](selection_integrity_audit.json).

## Frozen design

- Row split, development seeds 157 and 6101.
- Budgets 333, 365, 397, 429; three acquisition rounds per trajectory.
- Controls: historical CW-LCMD32 and new CW16 + random16.
- Treatment: CW16 + LLM16; no handcrafted-rule arm.
- Select CW16 first without revealing labels; exclude those rows from proposals.
- Propose 40 CW, 40 IVR, 40 MaxDet and 8 random rows, deduplicate, then use
  independent seeded random backfill to reach exactly 128 candidates.
- Both supplemental arms use this same rule, schema and information boundary;
  their later pools and fitted models evolve independently.
- Freeze all 32 choices before revealing any of their labels.
- Reuse compatible historical round-zero models and all pure-CW predictions.
- Inherit `ShortSequentialContext.fit` without predictor or training changes.

`protocol.json` records code/input hashes, RNG streams, the prompt, normalization,
and the decision gate. `prompt.txt` is the exact selector prompt. Each decision
has its packet, proposal, raw response, validated response and batch freeze.

## Information boundary and limitations

Packets contain IDs, SMILES, experimental conditions, five molecular descriptors,
predicted V1/V2, CW coverage-distance percentile, maximum Tanimoto similarity to
the labeled set and same-molecule labeled-row counts. They also show pending CW16.
They contain neither candidate truths nor validation/test records.

This existing conversation-mode implementation omits individual observed outcomes
to reduce cross-seed outcome leakage through shared conversation context. It does
not train an ensemble and does not describe geometric distance as uncertainty.
The same fixed prompt is used for six decisions. The exact conversation-model
snapshot is not exposed, so this does **not** meet strict model-version replay or
independent-context requirements. Response files preserve the actual decisions.
Historical test results have already been exposed in earlier work. New test
metrics are evaluated only after all four new trajectories and their 16 prediction
points freeze. Treat this as a development screen, not independent confirmation.

The LLM may use numerical signals as well as chemical structure. A gain over the
random supplement would support this selector package, not isolate chemical
reasoning. All conclusions are conditional on the 128-row shortlist.

## Failure behavior

One response attempt is accepted per frozen batch. Valid distinct IDs with a
nonempty reason among the first 16 entries are retained; an independently seeded
permutation fills missing slots. Malformed JSON and nonobject JSON use all 16
fallback picks. Failed rounds and fallback picks are reported. A parsed response
bound to a different packet hash stops execution to prevent stale-packet use.
An absent response pauses execution and is not silently counted as a model call.

The initial preparation, made before any responses or new fits, is preserved in
`revisions/before_json_failure_fix`. Preparation was re-frozen after repairing
malformed-JSON handling, before any first-round choice or new-label reveal.

## Evaluation

Primary: normalized trapezoidal AULC over 333-429, namely
`(E333 + 2*E365 + 2*E397 + E429) / 6`, where
`E = sqrt(((RMSE_V1/sV1)^2 + (RMSE_V2/sV2)^2) / 2)` and scales come from L0.
Also report V1/V2 RMSE and R2 at every budget and final combined NRMSE.

Extend only if LLM has lower mean AULC than **both** controls and wins against
each control on both development seeds. This stricter seed-consistency gate was
already frozen in the initial implementation. A win against CW alone is not
evidence for incremental LLM value over random exploration.

## Execution

Use the existing `fish` environment:

```sh
python scripts/studies/run_qgeognn_v2_row_llm16_screen.py validate
python scripts/studies/run_qgeognn_v2_row_llm16_screen.py execute --seed 157 --method cw16_llm16
python scripts/studies/run_qgeognn_v2_row_llm16_screen.py execute --seed 6101 --method cw16_llm16
python scripts/studies/run_qgeognn_v2_row_llm16_screen.py execute --seed 157 --method cw16_random16
python scripts/studies/run_qgeognn_v2_row_llm16_screen.py execute --seed 6101 --method cw16_random16
python scripts/studies/audit_qgeognn_v2_row_llm16_screen.py
python scripts/studies/run_qgeognn_v2_row_llm16_screen.py report
python scripts/studies/plot_qgeognn_v2_row_llm16_screen.py
```

LLM execution returns `AWAITING_LLM` until its matching `response.json` exists;
rerunning resumes completed fits after verifying their contracts and hashes.
Reporting refuses incomplete trajectories before accessing test truths.

Preflight: 21 tests passed across `test_llm_screen.py`,
`test_short_sequential_b32.py`, and `test_row_protocol.py`. Both seeds' first-round
random/LLM packets were verified byte-identical, with 16 pending + 128 distinct
eligible rows and zero LLM fallback picks.
