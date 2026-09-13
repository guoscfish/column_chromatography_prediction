# Preregistration — conditioned source readout

This study is governed by the detailed plan in
[ROW_FIRST_CONDITIONED_SOURCE_TRANSFER_PLAN_2026-09-13](../../../docs/research/ROW_FIRST_CONDITIONED_SOURCE_TRANSFER_PLAN_2026-09-13.md)
and its implementation audit. The executable protocol is frozen in
`protocol.json` before fit actions begin.

## Question and hypotheses

Primary question: does condition-aware, residual attention readout improve
4g-to-25g/40g transfer ROW prediction over matched fixed-sum P0? Secondary:
does the same frozen candidate remain non-inferior or improve COMPOUND
prediction?

H1: R1 can improve fixed sum pooling. H2: R2 can add signal if experimental
conditions should choose molecular substructure readout. H3: R3 can add source
q50 signal beyond R2. H4: R4 can add frozen source representation beyond q50.
Each hypothesis is falsifiable under the inner-CV gates; no gain is an intended
scientific outcome.

## Exact arms and fixed settings

R0–R4, loss L0–L2, 4 heads, 128-wide five-layer backbone, Adam `1e-4`, weight
decay `1e-5`, max 500 epochs, patience 80, current BN policy, and
historical-shallow P0 scope are fixed in `protocol.json`. Attention gate and
source-fusion outputs initialize to zero. R3/R4 source features are frozen and
cache-provenanced. No column geometry/identity, mass/flow, FiLM, Center/Width,
MoE, domain adversarial method, MAML, larger model, new pretraining, 8g joint
training, physics calculation, or Active Learning is allowed.

## Data and selection boundary

The parent filtered FULL-data `split_manifest.csv` is fixed. Each 25g/40g ROW
outer gradient-train set is split using five-fold `GroupKFold` on canonical
SMILES. Inner-train endpoint scales are used for loss and inner selection. The
outer validation/test labels are not passed to fitting, epoch selection, loss
selection, or architecture selection. The identical frozen COMPOUND protocol
is used only after ROW continuation.

## Primary/secondary and gates

ROW is primary. L0/L1/L2 choose one global loss only by the two-column inner
rule in `protocol.json`. R3 requires R2's >=3% two-column inner signal; R4
requires R3's. If no arm passes, stop with
`NO_MATERIAL_ARCHITECTURE_GAIN`. If ROW confirmation fails its continuation
rule, stop before COMPOUND. COMPOUND cannot retune the candidate.

Strong promotion, promising-row-only, and no-material definitions are fixed in
the research plan. Endpoint regressions are checked alongside combined scores;
no threshold or population may be changed after scoring.

## Non-causal metadata policy

Only verified sample-level chromatographic conditions are used in R2. Legacy
geometry (`column_dia`, `column_len`, `column_den`) and derived physical scales
are prohibited. Any future mass/flow diagnostic is predictive and column-
confounded, never causal interpretation.

## Artifacts and analysis

The study records source/split/prediction SHA256 values, environment, device,
git SHA, seeds, trainable inventories, inner histories, selected epochs, and
blind prediction files. Score actions require a global freeze manifest.
Matched metrics use one outer-gradient-train scale authority. Five-seed summary,
paired differences, and canonical-SMILES clustered-bootstrap ΔRMSE/ΔMAE 95% CI
are reported. Historical tests are explicitly developmental/exposed.
