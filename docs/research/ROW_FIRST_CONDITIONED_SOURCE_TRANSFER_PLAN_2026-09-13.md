# ROW-first conditioned source transfer plan — 2026-09-13

## Status and decision boundary

This is a preregistered, staged developmental study of whether a qualified 4g
source model and audited sample-level chromatographic conditions can improve
transfer prediction to 25g and 40g. It supersedes no historical result. The
linked implementation audit is
[conditioned-source-readout audit](../../studies/transfer/conditioned_source_readout/IMPLEMENTATION_AUDIT.md).

The primary estimand is ROW prediction. COMPOUND is mandatory secondary
confirmation only after a ROW continuation gate; no test result may determine
which module is appended. All historical outer tests are exposed, so every
outer score is developmental confirmation rather than independent validation.

## Scientific question and motivation

Can a qualified 4g source model, together with target sample-level
chromatographic conditions, materially and stably improve 25g/40g transfer
prediction beyond the converged fixed-sum P0 baseline?

Molecular node embeddings are an unordered set at readout. Sum pooling is a
valid invariant baseline, but it enforces the same aggregation for every
condition. Set-function work establishes the need for permutation invariance
and permits metadata-conditioned set functions ([Zaheer et al., 2017,
*Deep Sets*](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)).
Attention pooling is a principled invariant alternative for modeling
element-level interactions ([Lee et al., 2019,
*Set Transformer*](https://proceedings.mlr.press/v97/lee19d.html)). These
papers motivate a small, controlled readout intervention; they do not predict
that it will work in this chromatography dataset.

The source feature arms test **source-domain augmentation**,
**source-prediction augmentation**, and **source-representation augmentation**.
They are not called multi-fidelity: 4g, 25g, and 40g have not been established
as a physical low/high-fidelity hierarchy.

## Current-code gap and design correction

Active P0 uses `global_add_pool(node_embeddings) + condition_residual` followed
by the six-output head. Its graph-side legacy path receives ExactMolWt, TPSA,
nRotB, and HBD; HBA, LogP, loading solvent, density-times-loading-volume, and
loading-solvent volume arrive as an after-pooling residual. P0's active head is
independent `Linear(128,6) -> ReLU`, with soft crossing penalties only; it is
not a hard-monotonic quantile head. The study preserves that head so R1/R2
remain exact readout comparisons.

R2's condition query makes all nine audited semantic sample-level conditions
visible to the query: six source-scaled eluent descriptors, categorical loading
solvent (4-D embedding), and two source-normalized loading values. It excludes
packing mass, recorded flow, column identity, and geometry.

## Fixed experimental arms

All neural arms retain the five-layer, 128-wide molecular backbone; 4 attention
heads are fixed before fitting. No hidden-width/depth change, FiLM, MoE, DANN,
CORAL, MAML, new molecular pretraining, 8g joint training, Center/Width head,
or source replay is allowed.

| Arm | Fixed intervention | Trainable additions beyond P0 | Purpose |
| --- | --- | --- | --- |
| R0 | converged P0 architecture | none | fixed-sum control |
| R1 | `h_base + gate * global-query-attention(nodes)` | readout only | test adaptive readout |
| R2 | R1 with query from full audited condition encoder | readout only | test whether conditions should determine readout |
| R3 | R2 plus frozen source V1/V2 q50 residual fusion | source-prediction fusion | test incremental source point signal |
| R4 | R3 plus frozen 128-D source pre-head representation residual fusion | representation fusion | test incremental source representation signal |

`gate=0` at initialization. R3/R4 fusion outputs are zero-initialized, so each
new arm can initially reduce exactly to its predecessor. Attention aggregation
uses segment softmax by graph batch index and is tested for node-permutation
invariance.

## Fixed loss screen

Before R1/R2, compare P0 architecture only within ROW outer-gradient-train
five-fold canonical-SMILES GroupKFold:

- L0: existing raw quantile target loss.
- L1: endpoint-normalized quantile loss, with V1/V2 scales fitted from each
  inner-train partition only.
- L2: standardized q50 MSE plus `0.1` times source-free, scale-normalized q10
  and q90 pinball losses.

L1/L2 are selected only if each improves mean inner combined NRMSE by at least
1% in both columns, wins at least 3/5 outer-seed means and 13/25 inner folds
in both columns. Otherwise L0 remains the global loss recipe. A loss cannot be
chosen separately for 25g versus 40g.

## Selection, confirmation, and stopping rules

For Phase A, run R0/R1/R2 on 25g and 40g × five ROW outer seeds. A readout arm
has reasonable signal only if, versus matched R0, it improves mean inner
combined NRMSE by at least 3% in *both* columns, wins at least 3/5 seed means,
and wins at least 13/25 folds in both columns.

- R3 is run only if R2 meets that rule.
- R4 is run only if R3 meets that rule.
- If no readout arm meets the rule, stop: report
  `NO_MATERIAL_ARCHITECTURE_GAIN`, do not read a new outer-test result to
  justify more modules, and do not run COMPOUND.
- If one or more eligible arms remain after gated screening, choose exactly one
  by its mean two-column inner improvement. That architecture and loss recipe
  are frozen before ROW confirmation.

ROW developmental confirmation uses the frozen parent ROW roles for 25g/40g ×
five seeds. It compares matched P0, `paper_style_current_v2`, and the selected
candidate. P0/candidate final epochs are medians of inner-fold selections and
final refits consume outer-gradient-train labels only. Candidate blind
predictions are SHA256-frozen across all contexts before any test truth read.

ROW continuation requires at least 3% stable combined improvement in both
columns without a meaningful endpoint reversal. If it fails, report the
negative result and stop. If it passes, run frozen COMPOUND confirmation with
matched P0, `M3_CENTER_WIDTH_FULL`,
`HIER_CW_SHARED_LAMBDA_CORRECTED`, and the same neural candidate. COMPOUND may
not change architecture, source fusion, loss, head count, or primary
hyperparameters.

## Promotion interpretation

Strong promotion requires ROW 25g and 40g combined train-normalized RMSE to
improve by approximately 5% each, at least 4/5 seed wins in each, and no mean
endpoint RMSE/MAE regression above 2%; COMPOUND must show positive gain in both
columns (target about 3–5%) without a one-column dependence. ROW >=3% stable
gain with COMPOUND non-inferior within ±2% is
`PROMISING_ROW_ONLY`, never universal improvement. Under 3% or strongly
seed-dependent results are `NO_MATERIAL_ARCHITECTURE_GAIN`.

Results include V1/V2 RMSE, MAE, R², common outer-gradient-train normalized
RMSE, paired seed differences and wins/5. A clustered bootstrap resamples
canonical SMILES rather than treating multiple rows of a compound as IID;
ΔRMSE and ΔMAE 95% intervals are reported, not over-interpreted as n=5
p-values.

## Metadata and causal interpretation

`column_dia`, `column_len`, and `column_den` are prohibited. Their units and
provenance are unverified, 25g/40g share their legacy tuple, and observational
data cannot independently identify their physical effects. This study does not
derive column volume, linear velocity, bed volume, or void volume, and does not
reopen the physics-scale branch. Packing mass and recorded flow, if ever used
in a later diagnostic, are predictive context highly confounded with column
identity—not causal discoveries.

## Leakage controls and reproducibility

- The frozen filtered population, split manifest, thresholds, source checkpoint
  SHA256, and five seed schedule are asserted before fitting.
- Only outer `gradient_train` endpoint cells are loaded for fitting. Inner
  validation is within that set. Outer validation/test labels are unavailable to
  epoch/loss/architecture selection.
- Source q50 and representation caches are generated in source eval mode,
  keyed by `sample_id`, source checkpoint SHA256, and feature-table SHA256;
  they contain no target truth.
- All final validation/test prediction files and checkpoints are hash-frozen
  before scoring. Matched historical baseline IDs are validated before use.
- Every run records git SHA, Python/Torch/device, seed, source checkpoint hash,
  split-manifest hash, population hash, cache hash, selected epochs, and
  prediction hashes.
- Historical artifacts are referenced, never overwritten. No real experiment
  data are added and Active Learning remains paused.
