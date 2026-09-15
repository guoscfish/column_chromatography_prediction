# Execution Protocol

## Phase 1 Boundary

Run preparation, retrospective label-free audit, unit/integration tests and
isolated smoke. Commit with the preregistered subject, then stop. The Phase 1
manifest includes docs, protocol, ten splits, retrospective outputs and test
evidence. No formal result tables, decision, FINAL_REPORT or Commit B are
generated in this phase.

The authoritative scientific choices are in PREREGISTRATION.md and protocol.json.
Seed split hashes and L0/U0/validation/test identity hashes are in
`splits/split_manifest.json`. Existing B=333 study artifacts are read-only.

## Module Boundaries

| Module in active_learning_v2 | Responsibility |
| --- | --- |
| benchmark_protocol.py | Constants, seed schedules, budgets, preparation, Commit A guard |
| benchmark.py | Across-seed barriers, primary/secondary orchestration, isolated smoke |
| benchmark_seed.py | Source/X setup, scrubbed graphs, L0/ensemble fits, acquisition freeze, arm fits |
| acquisition.py | One four-method interface with exact batch validation |
| uncertainty.py | K=3 q50 disagreement ranking |
| coverage.py | Current representation extraction and TP k-center |
| gradient_features.py, lcmd.py | Existing frozen Jacobian sketch and LCMD implementations |
| cache.py | Array semantics and cache contract/content validation |
| runner.py | Shared preprocessing, authorized-label installation and same-initialization training |
| diagnostics.py | X/sketch duplicates and label-free mechanism audits |
| benchmark_reporting.py | Post-freeze metrics, paired comparison, decision and plotting |
| phase1.py | Validate Phase 1 evidence and seal the committed artifact manifest |

The new runner is a thin CLI. The old 1,000+ line pilot is neither extended nor
imported as the new orchestration layer. The old gradient and LCMD algorithms
remain untouched. Existing training cache receipts intentionally become
incompatible if they lack the new, stronger fit contract; old result artifacts
are not migrated or regenerated.

## Label Access and Ordering

1. Read canonical CSV with an explicit X-only `usecols` list: sample_id,
   canonical_smiles, density, sample volume, loading solvent, loading-solvent
   volume and PE/EA. Validate row identity and molecule order against the split.
2. `RestrictedLabelStore` parses only requested, authorized targets. Read L0 and
   validation targets for initial fitting; U0 and test targets remain hidden.
3. Fit graph-descriptor and eluent min/max scalers on outer-training X only.
   Loading amount/volume ranges also use outer-training X. Compute V1/V2
   population standard deviations on L0 truth alone. Freeze these scales.
4. Build model graphs with y=[0,0] for every row. Identity positions never become
   predictors. Only clones at authorized train/validation positions receive
   target values. Cached graphs must retain all-zero target sentinels.
5. Train baseline j=0. Extract gradients over canonical L0 followed by canonical
   U0; extract current-V2 representations in the same order. Train j=1 and j=2
   on identical L0/validation. Ensemble pool predictions must align by sample ID.
6. Freeze every within-seed acquisition batch and ordered selected IDs before
   revealing selected U0 targets. Diagnostic computations are label-free.
7. Train each evaluation arm from j=0 initialization on L0 plus its own selected
   labels. Enforce common L0/validation/test identity hashes and identical
   evaluation initialization hashes. Freeze checkpoints and test predictions.
8. Across all ten primary seeds, verify every receipt before writing the global
   pre-test freeze. Only then can a new restricted store unlock test targets.
   Verify source provenance and prediction identity before metric computation.
9. Aggregate confirmation and development separately. Freeze primary outputs.
   Secondary execution requires that complete primary freeze and matching
   Commit A SHA. No secondary acquisition function accepts test metrics.

The five Random controls and three non-Random methods therefore produce eight
after-batch arms at B=32. Baseline plus these arms share one evaluation
initialization. Two additional ensemble fits use distinct initializations solely
for acquisition. There are eleven fits per primary seed (baseline + 2 ensemble
+ 8 arms); each secondary batch adds six arm fits, reusing identical L0 caches.

## Cache and Freeze Contracts

- Raw graph cache is the canonical frozen 4g cache and is SHA256-verified.
- Scrubbed graph cache binds source/store SHA256, canonical ordered IDs,
  partition, code hashes, package versions, preprocessing, normalization and
  smoke/formal mode. Partial or incompatible graph/gradient caches are rejected.
- Completed fit caches bind context hash, arm, ordered train/validation IDs and
  indices, their authorized target array hashes, prediction IDs/order,
  initialization seed/state hash, training config, normalization and
  preprocessing. Verify checkpoint and prediction file SHA256 before reuse.
- Gradient caches additionally bind baseline checkpoint file hash, ordered outer
  IDs, target scales, sketch seed, dimension=512 and feature-file SHA256.
- Model `checkpoint_state_hash` identifies tensor names, types, shapes and
  values. Checkpoint file SHA256 also covers serialization/provenance. A hidden
  target-store mutation must change provenance/fit contracts but cannot change
  the trained semantic state, sketches or selected IDs.
- Files existing alone are not evidence of compatible completion. No silent
  cache reuse or fallback to a different model/representation is permitted.
- `assert_formal_authorized` requires Commit A subject, ancestry, committed
  manifest equality and frozen file hashes; current protocol, packages, Python,
  source data, graph cache and code must match the preregistered record.
- Primary execution refuses to overwrite an already frozen primary result.
  An interrupted run can resume compatible completed fits under the same code
  and protocol. Genuine incompatible changes require documented invalidation.

Execution is deliberately serial with CPU threads=2. This avoids resource races
while retaining deterministic cache reuse. No concurrency or training-semantics
change is required to perform this benchmark.

## Input-Equivalence Audit

The exact model-input key hashes typed atom x, atom/bond edge indices and edge
attributes, bond-angle edge indices and edge attributes (including scaled
descriptors). Labels, sample IDs and canonical positions are excluded. Separate
component hashes cover canonical molecular graph and raw descriptors; raw X
identities retain canonical_smiles, PE/EA, solvent, density, sample volume and
loading volume. Density and sample volume are recorded independently, although
the model consumes their product. PE/EA representations are the actual six
eluent descriptors and their fitted transforms. All nine intended condition
inputs, including categorical solvent and continuous completion-branch inputs,
are represented in the tensors.

`exact_X_excess_rows` counts rows beyond one representative per tensor identity.
`gradient_excess_rows` uses exact numerical equality of the 512D feature rows.
`selected_redundant_fraction` counts a selected row when its identity already
occurs in L0 or earlier in that ordered acquisition. A separate fraction counts
membership in *any* duplicate group, even when its peers were never selected.
These quantities must not be conflated. Mixed-X sketch groups trigger a
full-Jacobian check only for representatives of distinct input identities.

The recorded gradient norm is **the L2 norm of the 512D scaled CountSketch**, not
the norm of the unprojected network Jacobian. The old study used this convention;
it is retained and explicitly labeled. Nearest-L0 gradient distance uses the
same sketch geometry. Condition distance standardizes the frozen 9D condition
matrix on outer-training X with population mean/std (constant dimensions get
scale 1), then uses Euclidean distance to L0. It is an audit covariate, never an
acquisition input for these four methods.

## Commands

Use the existing `fish` environment. The local verified interpreter is
`/Users/fish/miniforge3/envs/fish/bin/python` (Python and package versions are
recorded in environment.json). Commands below run from the repository root.

Phase 1 only:

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --prepare
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --audit-existing
conda run --no-capture-output -n fish python -m pytest -q
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --smoke /tmp/qgeognn_v2_small_batch_phase1_smoke
```

The standalone smoke and integration tests use the same SeedContext,
acquisition, fit, freeze-validation and evaluator code as the formal runner.
The fixed fixture takes 160 evenly spaced canonical X rows and uses seed 29.
It replaces held-out target strings with synthetic values before float parsing;
formal test performance is never read. Full architecture, 512D sketches, K=3,
exact B=32/16/64 and five controls remain in force, with only training capped at
two epochs. Smoke metrics are plumbing output in an isolated directory, not
scientific evidence, and are not copied into formal result tables.

Future Phase 2 only, after explicit authorization (replace COMMIT_A_SHA):

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --execute-primary --preregistration-commit COMMIT_A_SHA
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --execute-secondary --preregistration-commit COMMIT_A_SHA
```

No formal per-seed shortcut or configurable seed/batch/K/epoch override is
exposed. A missing Commit A argument is rejected. The secondary command rejects
missing or changed primary freezes. Run primary to completion first.

## Outputs and Review

Phase 1 `results/` contains input_redundancy_audit.csv,
gradient_duplicate_audit.csv, gradient_mechanism_correlations.csv,
gradient_mechanism_selection_profiles.csv, lcmd_333_regression.csv, provenance,
label_budget_accounting.csv and validation evidence. These refer to the old
development L0 models or the explicitly identified fixture, not formal B=32.

Future primary metrics and diagnostics are written to `formal_results/` with
per-arm and per-seed identifiers and B32 suffixes. Checkpoint, initialization,
label-access and runtime audits are retained per seed and aggregated. Primary
ranking plots go to `figures/`. `formal_decision.json` explicitly separates the
confirmation primary decision from descriptive development outcomes. Secondary
metrics go to `secondary_results/`; sensitivity includes per-seed B16/B32/B64
Random means, LCMD NRMSE, relative improvements and median-based wins.

Later scientific reporting must explain endpoint tradeoffs, small-batch versus
old large-batch evidence, validation-inclusive label costs, uncertainty and
coverage comparisons, mechanism limitations and historical comparability. It
must not infer causal mechanisms from correlations, make a legacy numerical
superiority claim or generate a sequential recommendation before its gate.
