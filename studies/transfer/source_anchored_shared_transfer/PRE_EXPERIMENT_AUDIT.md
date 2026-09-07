# Pre-experiment implementation and evidence audit

Audit completed before implementation and new target-test evaluation. Base:
`9225cc28ae7e48b828888e58de2ab4dade638a2f`, fetched from
`origin/codex/study-scaling-failure-audit`; working branch:
`codex/study-source-anchored-shared-transfer`.

## Authoritative evidence read

Read in full: `docs/NEXT_STAGE_DECISION.md`,
`docs/research/CROSS_COLUMN_TRANSFER_STATUS.md`, and the scaling-failure
`SCALING_FAILURE_AUDIT.md`, `NEXT_MODEL_DECISION.md`, `MODEL_PREREGISTRATION.md`.
The empirical scale-only reference is strong, but its OLS weights are
proportional to source q50 squared. EA/V1 failure is reproducible. Conditional
scaling has local signals, without replicated material gains over strong
additive/shrinkage controls. Preserve
`NO_ADDITIONAL_COMPLEXITY_JUSTIFIED_FOR_TESTED_CALIBRATION_EXTENSIONS`.
This new stage tests representation transfer, not another scalar function.

## Exact current architecture and source

`src/qgeognn_al/models/qgeognn_v2.py` constructs `GeometryBackbone` directly:
128D atom/bond encoders, five node GIN blocks (`backbone.convs.0` through `.4`),
four effective bond-angle update paths. Node blocks include BatchNorm.
`extract_representation()` returns exactly
`global_add_pool(backbone(atom_bond, bond_angle), atom_bond.batch) + residual`.
The residual is the 128D output of `TypedConditionCompletionBranch`, not a
concatenation. Its 8 inputs comprise four continuous condition values and a
4D loading-solvent embedding; an 8->16->128 branch completes eluent acceptors,
logP, density-times-loading-volume, and loading-solvent volume. Four other
eluent descriptors already enter graph edge features. There is no explicit
column mass/flow context. The fixed six-output head is Linear(128,6)+ReLU,
ordered V1 q10/q50/q90 then V2 q10/q50/q90, in mL. Eval clamps at 1e8.

Source: `studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt`.
It is the preregistered final standalone V2 row-seed-42 source, qualified by
six source runs and exact R2-pruned equivalence. Best source epoch=95;
validation combined NRMSE=0.2791525916. Source split SHA256:
`a8a4a322cd9c75253b1a4ed8c5c91ededed74a933df2db2685015c8b3e008008`.
Source preprocessing uses 3330 train rows only (IDs hash
`5f0fbc143e13c77d15714729b0b2b4438de33b5c0ad776201ee6f4f0b5283d77`);
source SDs V1=7.8796590346, V2=16.0765095536. Source validation/test contain
416/417 rows. Checkpoint and all input hashes will be locked in protocol.json.

Parameter inventory: GeometryBackbone=455846; condition branch=2332;
head=774; total V2=458952. Last effective node block `.convs.4`=33281.
New shallow trainable count=33281+2332+774=36387; full=458952.
Each new wrapper also retains an independent frozen 774-parameter source
head, so all four wrappers have total=459726, with identical counts within
each matched pair. Source head cannot absorb backbone drift.

## Actual historical coverage

Current V2 8g row baseline (`run_final_v2_transfer.py`) tested head-only,
last-two effective node layers plus final edge update, and full fine-tuning.
The cross-column study tested head-only across all six column/protocols;
the 25g/40g last2 trigger did not fire and full was not run there.
Legacy T1/G0 last1/last2/full and pooled residual adapters are historical
predictors, not current V2 results. Historical shared affine shares
slope/intercept, not representations; pooled adapters do not test readout.
Our shallow definition additionally trains condition completion and only
node block 4; it is explicitly different from historical `last2`.

## Frozen target roles and costs

Reuse `studies/transfer/cross_column/splits/schedule_manifest.csv` verbatim,
including every sample ID, role and actual_budget. Target IDs are canonical
dataset row IDs (column + raw digest + original row); no resampling.
Five outer seeds: 769539383, 1425370602, 536279090, 2767143051, 1362771960.
Nested planned budgets: 30/50/70/100; 3 columns x 2 protocols x 5 x 4=120.
Row: fixed 20% test, eight purchased validation labels, budget-8 training.
Compound: approximately 20% compounds test, one fixed validation compound,
nested whole-compound train prefixes nearest each planned budget.

| column | valid rows | compounds | source-train overlap | compound actual budgets (30/50/70/100) | validation rows |
| --- | --- | --- | --- | --- | --- |
| 8g | 574 | 88 | 87/88 | 27-31 / 47-53 / 63-72 / 97-99 | 7 |
| 25g | 490 | 78 | 77/78 | 30-33 / 47-52 / 69-72 / 98-102 | 7-8 |
| 40g | 529 | 80 | 80/80 | 28-31 / 49-52 / 69-72 / 93-100 | 7 |

Target labels retain verified-time priority, raw fallback, no target threshold,
and V_mL=time*flow/1200. No new normalization is fitted. Source-unseen counts
are only 7/19/0 rows: **target-compound holdout != source-unseen molecular OOD**.
Source and target molecules may overlap by design. This is not donor-target
leakage: each learner consumes only 4g source-train and its own target ledger.
Source replay is historical source knowledge, outside target cost; report
replay draws and unique rows separately. No other target-column label enters
training, validation, normalization or model selection.

## Training, evaluation, and limitations

Existing V2 transfer uses Adam lr=1e-4, weight_decay=1e-5, maximum 500 epochs,
patience=100, validation RMS of source-SD-normalized RMSE, and the unchanged
two-output sum of `target_loss` (q10/q90 pinball, q50 MSE, crossing penalty).
All are batch means in mL-based units. Source/target residual magnitudes may
differ substantially, especially at zero-step; equal units do not imply
equal gradient strength. Freeze lambda=1 and report both losses, without
claiming balanced gradients or changing lambda after test.

Frozen backbone BN statistics stay fixed. Trainable BN statistics update
once on the target batch in every arm. Replay uses evaluation BN statistics
but keeps autograd through trainable BN affine/backbone parameters; it cannot
introduce a second running-stat update. This makes replay-loss attribution
more precise than simply calling model.train() for both tasks.

Source validation (all 416 rows) is a fixed diagnostic probe only, never
replayed or used to select a target checkpoint. All 120 sets of predictions
must be hash-frozen before target test truth is read by the new evaluation.
Historical tests have already been examined: this is developmental evidence,
not an independent confirmation. Seed partitions overlap and do not provide
five independent external datasets. Flow/mass/specification are confounded;
no physical mass or flow effect will be claimed.

Environment inspection: conda fish / torch 2.10.0, CPU-qualified graph/RBF
runtime. This environment requires the repository's historical
`KMP_DUPLICATE_LIB_OK=TRUE` workaround for duplicate OpenMP libraries; record
it explicitly and use deterministic single-thread workers and regression
checks. The initial read-only inspection without it aborted before fitting.
