# 4 g Uncertainty-Aware Active Learning

**Status:** diagnostic foundation and research plan; no new acquisition function or
expensive training has been authorized by this document.  
**Date:** 2026-09-19

## 1. Motivation

The repository already shows that Gradient-LCMD and Hybrid can reduce labels needed
to reach an early row-split error target. That does not answer why the predictor is
wrong, whether its interval width is trustworthy, or whether a high-uncertainty
experiment will improve the model. The research question is therefore

`prediction -> error landscape -> uncertainty attribution -> learnability -> acquisition`.

The analysis must distinguish:

- output sensitivity: what changes true retention volume;
- error sensitivity: where the fitted predictor fails;
- learnable uncertainty: which failures can plausibly be reduced by new labels.

No association in the observational dataset is a causal effect without a controlled
design.

## 2. Current baseline

The qualified predictor, splits, metrics, and active-learning state are summarized in
[CURRENT_STATE_AUDIT.md](../../CURRENT_STATE_AUDIT.md). The current uncertainty status
is intentionally conservative: q10/q90 are useful diagnostic outputs, but their
coverage is below nominal on both row and compound tests and they do not separate
epistemic from experimental uncertainty.

The scalar Kernel-IVR audit is in [KERNEL_IVR_AUDIT.md](../../KERNEL_IVR_AUDIT.md).
Its negative predictive result is evidence against immediately extending the same
surrogate family, not evidence that all information-based acquisition is impossible.

## 3. Error landscape: first deliverable

Before implementing a new acquisition rule, build one leakage-safe analysis table for
each fixed row and compound evaluation manifest. Each row should contain:

- true V1/V2 and q50 predictions;
- q10/q90, signed and clipped interval width;
- V1/V2 absolute and squared error, normalized errors, and combined error;
- compound identity and frequency;
- molecular descriptor/latent representation and train-nearest-neighbor distance;
- EA fraction/eluent representation, loading mass/solvent/volume, flow and column;
- predicted and true retention magnitude;
- train/pool density and condition distance;
- ensemble disagreement where a valid member set exists.

The first report should use binned curves, Spearman correlations, reliability/risk
coverage, and a shallow error model (with cross-fitting where a predictive model is
used). Pearson correlation alone is insufficient. Test labels may describe a frozen
result, but they must not choose a new strategy or hyperparameter.

The minimum outputs are: error by retention and EA bins, error versus chemical and
condition distance, top-error enrichment, calibration curves, and per-compound error
concentration. Tail slices must show sample counts; the current tail groups are too
small to support standalone conclusions.

The first read-only artifact is now available under
[`experiments/4g_error_landscape`](../../experiments/4g_error_landscape). It uses all
six qualified row/compound runs and records 24,978 prediction rows. The existing E1
uncertainty file was checked but rejected for joining because its source manifest has
zero sample-id overlap with the qualified predictions; its independent signal results
remain documented in E1. The artifact is descriptive and test results were not used
to select a strategy. The detailed table is compressed to keep the repository small;
the CSV summaries are human-readable.

Initial findings from that artifact are deliberately limited:

- q-width versus combined error Spearman is roughly 0.37--0.54 on the three row
  test runs and 0.46--0.65 on the three compound test runs; V1 and V2 widths have
  similar positive associations. This supports error ranking, not calibrated UQ.
- Row-split descriptor nearest-neighbor distance is degenerate at zero for nearly
  every row because the same compounds occur in train and test. It must not be
  interpreted as chemical OOD. The compound split gives the meaningful descriptor
  distance diagnostic.
- Row test 80% interval coverage ranges from 0.640--0.784 for V1 and 0.652--0.715
  for V2. Compound test coverage is lower and more variable, especially V2 at about
  0.456--0.476. This is consistent with the qualified quantile warning.
- Exact repeated-condition groups are present in the canonical 4 g data and are
  summarized separately. Their within-group standard deviation is an empirical
  repeat signal only where provenance supports independent measurements; it is not
  yet an aleatoric variance model.

## 4. Interaction analysis

Prioritize pre-specified interactions that are chemically plausible and statistically
identifiable in the available rows:

- EA fraction x molecular representation/property;
- EA fraction x predicted retention;
- loading mass x column specification;
- column x flow;
- chemical distance x condition distance;
- predicted retention x compound frequency/density.

Use two-dimensional binned surfaces or shallow interpretable trees first. GAM/SHAP
interaction terms are secondary and must be cross-fitted or treated as descriptive.
Every conclusion must label itself as correlation, statistical interaction, or causal
claim. The current mass/flow design cannot identify a causal column-physics effect.

## 5. Local chromatographic sensitivity

Only calculate local gradients or curvature where the data contain at least two
nearby conditions for the same compound and sufficiently controlled co-variables.
Use EA fraction as the primary continuous eluent coordinate when PE/EA is available.
Report the number of eligible neighborhoods, spacing, and condition mismatch.

For eligible neighborhoods define `G = |Delta V / Delta r|`, `C = |V_(i+1) -
2 V_i + V_(i-1)|`, and monotonicity violations. If coverage is too sparse or the
ordering of conditions is not reliable, record the analysis as not estimable rather
than interpolating unsupported gradients.

The first artifact finds 710 controlled compound/condition groups with at least two
EA levels, 693 with at least three, 3,385 adjacent gradients, and 2,675 curvature
triples. About 20.2% of V1 and 24.0% of V2 curvature triples change gradient sign;
these are observed monotonicity violations, not automatically physical anomalies.
On compound test splits, absolute local gradients have moderate positive association
with combined endpoint error (roughly 0.42--0.58 across seeds). The row-test
association is based on only 34--39 fully observed intervals per seed and ranges from
near zero to about 0.46, so it is not stable evidence. The full tables are
`local_gradient_summary.csv`, `local_curvature_summary.csv`, and
`local_sensitivity_error_relation.csv` in the artifact directory.

## 6. Uncertainty reliability

Evaluate q90-q10 and ensemble disagreement separately for V1 and V2:

- Spearman error association and top-k enrichment;
- risk-coverage/AUSE and hard-error AUROC;
- empirical 80% coverage overall and by compound, retention, eluent, chemical OOD,
  and condition OOD slices;
- interval width and sharpness together with coverage;
- calibration factors fit only on an explicitly declared calibration split.

The existing E1 signal qualification makes ensemble disagreement the main candidate
signal for the old E2 row AL protocol, while retaining quantile width as a secondary
diagnostic. That result has small per-slice sample sizes and does not make either
signal a calibrated posterior variance.

## 7. Epistemic versus aleatoric uncertainty

The preferred low-cost decomposition is:

1. a small independently initialized deep ensemble for between-model disagreement;
2. a separately calibrated conditional residual/interval model for within-observation
   variation;
3. repeated-condition variance estimates only where provenance confirms independent
   experimental repeats.

Deep ensemble variance is an epistemic proxy, not a proof of epistemic uncertainty.
Quantile width is not aleatoric variance by definition. Heteroscedastic regression
or conformal calibration should be considered only after the repeated-data audit and
with fixed cross-fitting; a method sweep is not justified by the current evidence.

## 8. Learnability/value-of-label analysis

The first learnability study should be post-hoc and label-efficient: partition
accessible candidate rows by a frozen signal (ensemble disagreement, q-width,
latent distance, and low-signal baseline), then compare revealed-error enrichment
and next-round improvement on an already frozen development protocol. It must report
the distinction between high predictive interval and high model disagreement.

A prospective confirmation should use the same cumulative label budget, same held-out
test set, at least five predeclared seeds, and learning-curve AULC plus fixed budgets.
The primary question is `performance gain per new label`, not only the final point.

## 9. Acquisition candidates after diagnosis

Do not implement all candidates listed in the original request. The current evidence
supports this shortlist, in order:

1. **Existing Gradient-LCMD-TP:** retain as the main geometry baseline and measure
   strata coverage/late-stage failure modes.
2. **Conditional design/MaxDet on the audited gradient representation:** a cheap,
   genuinely different design objective that can test whether label efficiency is
   limited by coverage rather than local gradient magnitude.
3. **Reliability-aware design:** only if cross-fitted disagreement and repeated-data
   diagnostics show that a signal predicts reducible error; combine epistemic score
   with diversity/density, not raw q-width alone.

Target-weighted multi-output IVR remains a later hypothesis. It requires a declared
deployment-risk distribution and an observation/noise model; the current scalar IVR
negative result makes an immediate parameter sweep low value.

## 10. Fair evaluation protocol

Every strategy comparison must freeze one test set, use equal cumulative active-label
budgets, and report V1/V2 separately plus combined NRMSE, RMSE, MAE, R2, AULC,
fixed-budget performance, labels-to-target, early/mid/late curve segments, and
training/selection cost. Five seeds are the minimum for an important claim; the
existing five-seed row curves are overlapping splits and therefore development
evidence rather than independent chemical replicates.

The 300-label question is a protocol comparison, not a slogan: one-shot 300 and
sequential batches summing to 300 must share the same initial state, final label set
definition, test set, and training configuration. Static and adaptive curves need
matched cumulative budgets; batch size is a separate factor.

## 11. Cross-column transfer gate

Transfer analysis starts only after the 4 g error/UQ landscape is sealed and a target
column uncertainty baseline is independently checked. The first transfer artifact is
the source-to-target residual table, stratified by source q50, source uncertainty,
embedding, EA, loading, source/target scale, chemical similarity, and condition
distance. The current physical metadata audit blocks causal interpretations of flow,
mass, and legacy geometry under the existing design.

An active-transfer experiment should first compare Random, fixed calibration-aware
coverage, and one uncertainty/design method at matched target-column budgets. Complex
neural transfer is not the default next step because the repository already has
negative or non-robust results for the tested Scale-only/Affine/Conditional,
Center-Width, readout, FiLM, shared-column, and PCGrad extensions.

## 12. Negative results and stop conditions

Stop a candidate if its proxy signal is not stable across seeds, if it only wins by
changing the label budget, if it needs test outcomes to choose a parameter, or if a
local sensitivity estimate is based on unsupported condition interpolation. A lower
surrogate variance is not enough; the candidate must improve held-out error per label.

## 13. Next steps

1. Produce the frozen row/compound error-landscape table from existing predictions.
2. Add calibration and repeated-condition provenance summaries without changing the
   predictor or test manifests.
3. Quantify interactions and chemical/condition coverage with explicit sample counts.
4. Run one small, pre-registered learnability pilot; only then select a new acquisition
   candidate for matched-budget training.
5. Revisit active transfer after target residual and uncertainty diagnostics pass.
