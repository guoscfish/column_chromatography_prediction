# Handoff: reviewed and prepared, no formal run

Date: 2026-09-21. Base: main at 03d93cef39c38565718ae055d1898c99b540f10b.

## 1. Critical judgment

Worth a single exploratory proof of concept, not independent confirmation.
653 was motivated by already inspected LCMD test curves. Pure IVR's later
advantage does not establish an advantage after LCMD's different L653 state.
Existing IVR AULC is 4.28% worse than LCMD, with only 1/5 paired AULC wins;
its better endpoint and sustained cohort N90 support a hypothesis, not a
general early-LCMD/late-IVR law. Choose C, then consider separately designed
validation-only adaptive switching after this result. No adaptive tuning now.

## 2. Final design

Five original seeds, 157/887/2357/6101/12203. Exact frozen Gradient-LCMD prefix
through 653; switch acquisition to Kernel-IVR, B32 through 1005. Primary contrast
is late normalized AULC and endpoint versus continuing LCMD from the same state.
Full AULC, early/middle/late partial AULC, endpoint metrics, target crossings and
gap closure are secondary descriptive outcomes. One five-method summary includes
Random, LCMD, Hybrid, pure IVR and the switch; no two-seed MaxDet/Fusion pooling.

Keep the existing predictor, scratch initialization, L0 scales, fixed validation,
512D full-network q50 kernel, uniform original outer reference, unit prior/noise
and conditional greedy IVR. The scalar surrogate is not a two-output posterior.

## 3. Added files

- scripts/studies/run_qgeognn_v2_row_lcmd_to_ivr_b32.py: explicit CLI actions.
- src/qgeognn_al/active_learning_v2/strategy_policy.py: policy/state interface.
- src/qgeognn_al/active_learning_v2/lcmd_to_ivr_study.py: source audit and seal.
- src/qgeognn_al/active_learning_v2/lcmd_to_ivr_runner.py: resume and continuation.
- src/qgeognn_al/active_learning_v2/lcmd_to_ivr_reporting.py: gated five-arm tables.
- tests/active_learning_v2/test_lcmd_to_ivr.py: 19 new checks including parameterizations.
- This study: PROTOCOL.md, README.md, HANDOFF.md, five lineage/seed_*.json records,
  seal.json, preflight_tests.xml and engineering_smoke.json.

No existing tracked code or historical artifact was edited. Earlier unrelated
untracked short_sequential files remain in the worktree.

## 4. Reuse decision

All five seeds use option 1: reuse L653 checkpoint and gradient bank. Verified
exact ordered training IDs and truth hash via reconstruction of the original
fit contract, initialization hash, preprocessing, scaler bytes, L0 target scales,
source/split/graph hashes, model architecture/config, checkpoint file/state hashes,
gradient cache receipt and acquisition freeze, and scientific package versions.

The gradient module has additive alternative extractors; its original AST prefix
is unchanged except typing.Mapping. The audit checks this against the recorded
historical file digest. Three recomputed gradient rows per seed match the cached
values exactly (max absolute difference 0). Source banks are reordered from
L653+U653 into fixed L0+U0 reference order by canonical IDs before IVR.

## 5. Switch provenance

Protocol schedule, source round input contracts, cached arrays and replayed
selections all confirm round 10 = 653 and round 11 = 685. L653 has 653 distinct
IDs, U653 has 2677, union has the same 3330 original outer rows. Source labels
restored are L0 plus exactly ten historical B32 batches. No source post-653
selection or label is used to define the continuation.

SHA256 prefixes below are only for navigation; full hashes and exact IDs are
in the sealed lineage records.

| Seed | Ordered L653 | Checkpoint | Gradient |
|---|---|---|---|
| 157 | 90ddbbdc1bb3 | dde10fa5a1ef | 06e596cf3af1 |
| 887 | 689df4b3fca4 | 82e60999e00b | 636dac69d08e |
| 2357 | 2122c7b3278e | 128076ac89a2 | b74a0be136a8 |
| 6101 | 3552c013a65d | e858389dfe01 | f3526afe5534 |
| 12203 | 898a61c4b89b | fb56a15e7c57 | 024c3e4c9282 |

Source dataset: 6cbe81311177ae178d10ff75a8f07c41c99a18648ec6104d2ffedca6e912d51c.
Graph cache: a48f8ea1ce4f89c5a1ea6d17466e7c7ffa0439edb8ec79bd8b765bad3a97c3ee.

## 6. Verification

55 tests passed: 19 new plus 36 existing IVR/sequential tests. No failures or
skips. Existing dependency/deprecation warnings remain. Checks include exact
real-data state reconstruction, 32-row transitions to 1005, duplicate/universe
guards, forbidden test labels, mock-fit interruption/resume versus uninterrupted
selection, nested artifact mutation, fallback planning, gradient row reindexing,
and synthetic reporting/crossings. Modules also pass Python compilation.

Five real-data selection-only smoke checks passed: each starts at LCMD-L653,
selects exactly 32 valid rows, and repeats with identical IDs and IVR trace.
No new labels were revealed by this smoke. No formal fits, complete real seed
trajectory or new test evaluation was executed. There are no runtime/seed_* dirs.

## 7. Git state

Fetched origin, including its branch refs. main and origin/main both point to
03d93cef39c38565718ae055d1898c99b540f10b. New files are untracked and uncommitted;
no new commit was made. Existing short_sequential additions were preserved.

## 8. Exact next commands, after authorization to train

```sh
cd /Users/fish/Documents/GitHub/column_chromatography_prediction
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_lcmd_to_ivr_b32.py --validate
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_lcmd_to_ivr_b32.py --run-all
```

--run-all executes seeds sequentially, freezes all five, and stops without test
evaluation. To resume just one seed, replace --run-all with --execute-seed 157
(or another registered seed). After completion and authorization to evaluate:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_lcmd_to_ivr_b32.py --report
```

Do not change sealed source/code files, delete source caches, reset the seal, or
reprepare to evade a mismatch. Resolve and document any mismatch before running.

## 9. Planned cost

55 new scratch fits = 11 per seed at 685..1005. Reuse all five L653 predictors;
50 new full-bank gradient extractions, 55 IVR batches. No 333..653 fits.
Validation adds 416 shared labels to every budget. Runtime audits distinguish
new fits/epochs/training, new gradients, IVR selection and reused prefix work.

## 10. Remaining risks

Test-informed switch-point selection, correlated row splits, no untouched
confirmation data, possible LCMD/IVR state interaction, and the scalar unit-noise
surrogate limit scientific conclusions. Sustained crossing means only through
the observed 1005 endpoint. Finite-difference validation trends can be noisy and
must not become post-hoc tuning knobs. Real closed-loop performance and actual
training wall time are deliberately untested in this preparation task.
