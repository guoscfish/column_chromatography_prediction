# Column-conditioned multi-task implementation audit

Date: 2026-09-14
Status: completed before implementation or fitting

## Duplication decision

No implementation identical in scientific content to either proposed candidate
exists in the repository. `docs/research/CROSS_COLUMN_TRANSFER_STATUS.md`
explicitly lists column-conditioned shared representation, categorical task
embeddings, and globally grouped multi-column multi-task training as unlaunched
work. The existing source-anchored study shares a source and target backbone but
does not train 4g, 25g, and 40g simultaneously with separate six-output heads.
The existing conditioned-readout study changes pooling/readout after the
backbone and does not condition message passing on column identity. The proposed
A1 and A2 are therefore new and the study may proceed.

## Qualified source authority

The qualified transfer source is the active standalone `QGeoGNNV2` checkpoint:

- path: `studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt`
- SHA256: `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`
- model variant: `qgeognn_v2`
- implementation: `src/qgeognn_al/models/qgeognn_v2.py`
- output order: `V1_q10,V1_q50,V1_q90,V2_q10,V2_q50,V2_q90`

The path and digest agree with the final qualification decision, the converged
P0 authority, and the checkpoint on disk. Before implementation, the checkpoint
was replayed on all 4,163 canonical 4g rows. All outputs were finite and the
maximum absolute difference from the frozen row-seed-42 CSV was `3.8086e-6`,
consistent with CSV serialization precision.

## Active architecture and adaptation scope

The active predictor has five node/message-passing blocks named
`backbone.convs.0` through `backbone.convs.4`, four interleaved edge/angle
updates, fixed sum pooling, the additive `condition_branch`, and
`head = Linear(128, 6) -> ReLU`. The requested final-two-block scope maps to
`backbone.convs.3` and `backbone.convs.4`. The final edge/angle update feeds
block 4 and is therefore part of the controlled final-two-block computation;
its audited names end in `.3` for `convs_bond_angle`, `convs_bond_float`,
`convs_bond_embeding`, and `convs_angle_float`. Early encoders and blocks 0-2
remain frozen.

A1 will retain that shared molecular/condition representation and route each
sample to an exact source-head copy named for 4g, 25g, or 40g. A2 will differ
only by a fixed 16-dimensional categorical task embedding and zero-initialized
FiLM generators on node representations after blocks 3 and 4. No hidden width,
depth, pooling, attention, output semantics, or loss recipe changes are allowed.

## Frozen data and metric authority

The outer identity authority is
`studies/transfer/filtered_full_data_benchmark/split_manifest.csv`, SHA256
`33e079a2b7250a82b040605bddb372646ab424684268b5b0c63df6366194a0a3`.
It fixes 25g/40g, ROW/COMPOUND, five outer seeds, and every sample role. The
new study will copy and hash this schedule, never redraw it. COMPOUND is primary.
ROW is conditional on the preregistered COMPOUND inner gate.

All inner folds are built only from each outer `gradient_train` population.
Canonical-SMILES grouping is global across the joined 25g and 40g rows, so a
molecule shared across target columns cannot cross inner train/validation.
The 4g replay data is source supervision and is intentionally not purged by
target holdout identity. Target-compound holdout is not source-unseen OOD.

Metrics follow `src/qgeognn_al/evaluation/point.py`: q50 RMSE/MAE/R2 per
endpoint and the root-mean-square of endpoint RMSE divided by train-only
endpoint scales. Center and Width are algebraic diagnostics. Tail thresholds
are the per-endpoint 80th percentiles of the applicable outer-gradient-train
truth and never use held-out truth.

## Historical ideas excluded from this study

The following have already been tested and will not be renamed or repeated:

| Historical idea | Existing evidence and status |
| --- | --- |
| zero-shot | Current V2 frozen cross-column and full-data benchmarks. Reference only. |
| scale-only | Current V2 matched/full-data benchmarks. Strong historical calibration reference. |
| affine | Current V2 matched cross-column benchmark. Historical only. |
| affine/shrinkage variants | Local identity shrinkage and partial/shared affine pooling were tested. Historical only. |
| Conditional EA | Completed across frozen contexts; no replicated material gain over strong controls. Historical only. |
| additive condition residual | Affine plus condition Ridge residual was tested; incremental gain was small. Excluded. |
| Center/Width calibration and structured variants | M3, structured Center/Width, hierarchical Center/Width, and semantic repair are complete. Corrected HIER is the legitimate reference. No new Center/Width fitting. |
| nonlinear / monotone calibration | Train-only two-knot monotone q50 calibration was tested without stable material benefit. Excluded. |
| shared affine calibration | Shared-column affine and shrinkage controls were tested. Excluded. |
| target-head-only transfer | Current V2 tested across all six column/protocol contexts. Historical only. |
| historical last1 / last2 / full fine-tuning | Legacy and earlier current-V2 studies cover these scopes. Excluded. |
| current shallow/full target fine-tuning | Tested in all 120 contexts; no replicated material advantage. Excluded. |
| source-anchored shallow/full fine-tuning | Capacity-matched controls tested in all 120 contexts; source preservation alone was insufficient. Excluded. |
| latent Ridge | Direct and residual latent Ridge studies are complete. Excluded. |
| latent PLS | Direct latent PLS study is complete. Excluded. |
| tiny latent adapter | The fixed 128-to-16-to-2 adapter was tested and failed the promotion gate. Excluded. |
| pooled residual adapter | Historical post-sum-pooling r8/r16/r32 adapters had no stable benefit. Excluded. |
| adaptive readout | Global and condition-query adaptive readout candidates were completed. Excluded. |
| condition-query readout | Completed within `conditioned_source_readout`; it is not representation-level task conditioning. Excluded. |
| endpoint-normalized/weak-quantile loss screen | Completed; this study uses the authoritative raw quantile loss and performs no loss search. |
| physical mass scaling | Tested in physics-conditioned and Center/Width work; no physical claim is identified. Excluded. |
| dia/len/den legacy descriptors | Provenance/identifiability audit found no eligible new physical context. Excluded. |

`OLD_HIER_REFERENCE` is not a scientific authority because semantic repair
found the missing Center/Width source-scale restoration. Only
`HIER_CW_SHARED_LAMBDA_CORRECTED` may serve as the corrected hierarchical
reference. `M3_CENTER_WIDTH_FULL` is secondary historical context only.

## Study boundary

Only `MULTITASK_SEPARATE_HEADS` (A1) and `MULTITASK_COLUMN_FILM` (A2) are
authorized. The study excludes 8g, Active Learning, PCGrad, GradNorm,
domain-specific BatchNorm, new calibration or physical scaling, geometry
fields, arbitrary MLP or architecture sweeps, loss searches, embedding sweeps,
and post-result candidates. Gradient norms and cosine similarities are logged
for diagnosis only and cannot change training.

Outer validation/test labels are unavailable to fitting and model selection.
If the COMPOUND inner gate fails, outer prediction and scoring actions are
blocked and the negative report is the terminal scientific result. If it
passes, predictions must be frozen and SHA256-hashed before a separate scoring
action can load held-out truth; all scores are developmental confirmation.

## Execution addendum: outer cross-task molecular overlap

An additional identity-only audit during execution found that the inherited
COMPOUND schedules are per-column partitions, not one globally grouped
multi-target outer partition. Across the five seeds, 7-11 molecules in the
25g test set occur in 40g gradient_train, and 7-10 molecules in the 40g test
set occur in 25g gradient_train. No endpoint cells were read for this audit.

The requested exact outer role schedule and complete-gradient-train refit
therefore imply cross-task donor-label availability for some outer molecules.
Purging those donor rows would change the explicitly required training identity
schedule. The implementation preserves it and records this limitation. Inner
CV, which owns the continuation gate, is globally grouped across the target
tasks and has zero such cross-task train/validation molecular overlap.

Any eligible outer confirmation must be described as within-column compound
holdout with related-task donor supervision, not global target-molecule OOD.
This distinction is additional to the intentional overlap with 4g source
training molecules and the historical exposure of the outer identities.
