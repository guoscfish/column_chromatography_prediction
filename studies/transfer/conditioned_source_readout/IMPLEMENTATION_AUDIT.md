# Conditioned source-readout implementation audit

Date: 2026-09-13
Status: completed before implementing or fitting the new study

## Scope and corrections to the request

This audit covers the requested plan/decision documents, the current active
predictor, the legacy condition-completion wrapper, and the relevant filtered
FULL-data transfer artifacts. The requested path
`studies/transfer/formal_row_5seed/` does not exist. Its historical counterpart
is `studies/transfer/traditional_transfer_improvement/formal_row_5seed/`; it is
a 150-epoch, outer-validation-selected historical study and is not the
converged P0 authority.

Two prompt assumptions required correction before implementation:

1. The active P0 source model is `src/qgeognn_al/models/qgeognn_v2.py`, not
   the legacy `src/qgeognn_al/condition_complete_v2.py` wrapper.
2. The active P0 head does **not** hard-enforce `q10 <= q50 <= q90`. A stale
   table entry in `docs/QGEOGNN_IMPLEMENTATION_VARIANTS.md` was corrected in
   this change. No monotonic-head change is included in R1/R2 because it would
   confound a readout comparison and violate epoch-zero source identity.

## Active model and fixed-pooling evidence

`QGeoGNNV2.extract_representation` in
`src/qgeognn_al/models/qgeognn_v2.py:64-68` computes

```
h_base = global_add_pool(backbone(atom_bond, bond_angle), batch)
         + condition_branch(atom_bond)[0]
```

and `forward` applies `head: Sequential(Linear(128, 6), ReLU)` at lines
70-72. `global_add_pool` is fixed sum pooling. The older builder independently
hard-codes `graph_pooling="sum"` in `src/qgeognn_al/model.py:114-116`.

The new readout therefore keeps the complete active base representation and
adds only

```
h_final = h_base + gate * h_attention
```

where `gate` is exactly zero at construction. This preserves the loaded source
function at epoch zero while retaining sum pooling as the control. It does not
alter backbone depth, hidden width, FiLM, column identity, Center/Width, or the
prediction head.

## Condition-input audit

`src/qgeognn_al/data.py:205-211` constructs the continuous edge columns in
this semantic order after the three categorical bond fields:

1. bond length;
2. eluent ExactMolWt;
3. eluent TPSA;
4. eluent nRotB;
5. eluent HBD;
6. eluent HBA;
7. eluent LogP;
8. loading-solvent code;
9. density times loading volume;
10. loading-solvent volume.

The legacy `BondFloatRBF` consumes only positions 0-4
(`application/QGeoGNN.py:1015-1053`), hence the graph path gets bond length
and the first four eluent descriptors. HBA, LogP, loading solvent, loading
amount, and loading-solvent volume are not consumed by that legacy graph path.

The active V2 completion branch in
`src/qgeognn_al/schemas/conditions.py:150-191` (with an equivalent legacy
wrapper in `condition_complete_v2.py:174-214`) reads the five missing
conditions, embeds loading solvent, source-train-normalizes the two loading
values, and adds its residual **after** fixed sum pooling. It contains no
column geometry, column identity, packing mass, or recorded flow feature.

R2's query uses all nine audited semantic sample-level conditions rather than
only the completion branch: six source-scaled eluent descriptors plus a 4-D
loading-solvent embedding and two source-normalized loading values. HBA/LogP
are not duplicated. This uses no target endpoint, outer-role, column, or
physical-geometry value.

## Quantile-head audit

`src/qgeognn_al/model.py:13-32` contains a genuine monotonic softplus-offset
head. It is installed only when the legacy condition-complete wrapper loads a
legacy checkpoint (`condition_complete_v2.py:217-227`). It is not part of the
active QGeoGNNV2 source checkpoint used by P0.

For active P0, `src/qgeognn_al/transfer/adaptation.py:155-166` uses soft
adjacent-crossing penalties in the raw loss; it does not guarantee ordering.
The historical P0 inner summary includes non-zero crossing counts. Crossing is
kept as a diagnostic, and this stage deliberately makes no UQ/head claim.

## Converged P0 authority

The authoritative P0 record is
`studies/transfer/traditional_transfer_converged_baseline_v1/`, implemented by
`scripts/studies/run_traditional_transfer_converged_baseline_v1.py`.

| Contract item | Audited value |
| --- | --- |
| Source checkpoint | `studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt` |
| Source SHA256 | `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0` |
| Model | active `qgeognn_v2`, 458,952 parameters |
| Trainable P0 scope | `backbone.convs.4.*`, `condition_branch.*`, `head.*` |
| P0 effective scope size | 36,387 / 458,952 parameters |
| Optimizer | Adam, lr `1e-4`, weight decay `1e-5`, batch 2048 |
| Loss | raw q10 pinball + q50 MSE + q90 pinball + crossing penalty, summed over endpoints |
| Budget | max 500 epochs, patience 80 |
| BN policy | `current`: trainable final-layer BN updates; BNs with frozen affine parameters are returned to eval/source-stat behavior |
| Inner selector | five-fold canonical-SMILES `GroupKFold` within outer `gradient_train`; inner-train endpoint scales only |
| Final refit | round-half-up median of five selected epochs; fixed-epoch, gradient-train only |

The code evidence is
`run_traditional_transfer_converged_baseline_v1.py:35-80,191-290` and
`transfer/adaptation.py:79-179,444-706`. The previous plan's phrase “inner
compound GroupKFold” is clarified: the outer protocol is ROW; the *inner*
splitter is GroupKFold grouped by canonical SMILES.

## Split and baseline comparability

The frozen parent authority is
`studies/transfer/filtered_full_data_benchmark/split_manifest.csv`, SHA256
`33e079a2b7250a82b040605bddb372646ab424684268b5b0c63df6366194a0a3`.
It fixes 25g/40g, ROW/COMPOUND, and five outer seeds. Filtered populations are
408 rows (25g) and 456 rows (40g), with their existing threshold and split
definitions unchanged.

The ROW P0/P1 runtime, the finalization study, and the corrected HIER study
match the frozen 25g/40g role IDs. The latter two have a larger manifest because
they additionally include 8g contexts; their 25g/40g subsets match exactly.
The new runner asserts the exact `(column, protocol, seed, sample_id, role)`
schedule before fitting and writes role-ID hashes to `run_manifest.csv`.

Historical outer tests have already been scored in formal ROW, converged P0/P1,
full-data baseline finalization, architecture-headroom, and corrected HIER
studies. Every new outer score is therefore labelled **developmental
confirmation**, not a pristine blind test. Fit/selection code reads only
outer-gradient-train endpoints; final prediction files are hash-frozen before
the separate score action can read outer truth.

There is no completed converged-P0 COMPOUND artifact. If ROW passes its
continuation gate, the current runner rebuilds P0 and the frozen selected
candidate with the existing compound schedule. Historical paper-style,
`M3_CENTER_WIDTH_FULL`, and
`HIER_CW_SHARED_LAMBDA_CORRECTED` predictions are reference-only and must be
ID-checked and re-scored with the same outer-gradient-train denominator. The
old cross-method normalized rankings cannot be reused because some historical
tables used source-train scales rather than outer-train scales; see
`docs/NEXT_STAGE_DECISION.md` and
`studies/transfer/row_metric_harmonization/`.

## Physical-metadata decision

`studies/transfer/physical_metadata_identifiability_audit/FINAL_REPORT.md` and
`PHYSICS_MODEL_ELIGIBILITY.json` establish
`NO_NEW_IDENTIFIABLE_PHYSICAL_CONTEXT`. `column_dia`, `column_len`, and
`column_den` have unverified units/provenance; 25g and 40g share the same
legacy tuple `(2.15, 15.6, 0.5248)`; adding them has no rank increment in the
eligible context. This stage does not use them or derive column/bed/void
volume or linear velocity. Packing mass and recorded flow remain at most
potential predictive/diagnostic context and are highly confounded with column
identity; they are not causal evidence. The physics-scale branch remains
closed.

## Implementation consequence

The implemented study uses a new wrapper rather than changing the strict
active checkpoint schema:

- `src/qgeognn_al/transfer/adaptive_readout.py`: four-head segment-softmax,
  permutation-invariant residual attention; R1 global query and R2 full
  condition query.
- `src/qgeognn_al/transfer/source_augmentation.py`: frozen source q50/128-D
  representation cache and zero-initialized target fusion for R3/R4.
- `src/qgeognn_al/transfer/adaptation.py`: explicit L0/L1/L2 recipe dispatcher
  and opt-in trainable prefixes, leaving P0 defaults unchanged.
- `scripts/studies/run_conditioned_source_readout.py`: label-scrubbed context,
  inner CV, fixed-epoch refit, source-cache provenance, and global prediction
  freeze boundaries.

The accompanying unit tests cover source identity at gate zero, node-order
invariance, batch independence, attention gradients, finite/deterministic
inference, and zero-impact source fusion.
