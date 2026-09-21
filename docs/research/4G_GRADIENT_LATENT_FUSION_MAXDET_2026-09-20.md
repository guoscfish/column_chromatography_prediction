# 4G Gradient + Latent Fusion MaxDet

Status: **complete developmental row-split evidence**. The method is an
ActiveFusion-inspired representation fusion adapted to conditional
molecular/chromatography regression; it is not a direct reproduction of
ActiveFusion.

## Motivation and existing evidence

The repository already contained Latent-LCMD, Gradient-LCMD, representation
audits, and the full Gradient-MaxDet trajectory. This study therefore asked
only whether the authoritative QGeoGNN gradient representation becomes more
label-efficient when its conditional D-optimal geometry is augmented by the
existing QGeoGNN-V2 pre-head latent representation. It did not rerun
Latent-LCMD, redesign the encoder, or repeat the earlier innovation screen.

## Method

For the current scratch-trained model and ordered universe `[L_t, U_t]`:

- `g_i` is the existing 512D
  `CountSketch(concat(grad(V1_q50/s_V1), grad(V2_q50/s_V2)))`;
- `h_i` is the existing 128D `extract_representation()` output;
- `s_g = sqrt(mean_{i in L_t} ||g_i||^2)`;
- `s_h = sqrt(mean_{i in L_t} ||h_i||^2)`;
- `phi_i = [sqrt(0.5) g_i/s_g, sqrt(0.5) h_i/s_h]` is 640D.

Only labeled feature rows determine `s_g` and `s_h`. The resulting linear
kernel is

`K_fusion = 0.5 K_gradient_normalized + 0.5 K_latent_normalized`.

The existing tested `conditional_gradient_maxdet()` selector was applied
unchanged to `phi`. Its secondary global scaling remained active and was close
to one after block normalization.

Block normalization is necessary because an unscaled concatenation would
conflate the 512D/128D dimension imbalance with the very different natural
norms of the gradient and latent blocks. The fixed 0.5 weighting makes the
experiment a kernel-level comparison rather than an implicit scale sweep.

## Protocol

- split: ROW;
- initial labeled set: 333 rows;
- batch size: 32;
- active-label budgets: 333 through 1005 (22 evaluation points and 21
  acquisitions);
- seeds: 157 and 6101;
- training: scratch QGeoGNN-V2 retraining with the matched historical
  initialization, validation set, target scales, and training configuration;
- acquisition: recompute current-model gradient and latent features each
  round, then run conditional MaxDet on the fused matrix;
- no alpha sweep, U50 fusion, uncertainty fusion, new predictor, compound
  split, or extra seed was run.

## Leakage and provenance audit

The independent study is
`studies/active_learning/qgeognn_v2_row_fused_maxdet_b32`.

Round zero legally reused the exact historical scrubbed graphs, model,
prediction, and 512D gradient cache after checking checkpoint, split, sample
identity, ordering, and feature contracts. Latent features were computed from
that same checkpoint. All later rounds followed the Fusion trajectory and used
new scratch fits plus new gradient/latent extraction; no later Gradient-MaxDet
model or acquisition was reused.

Every acquisition stored the gradient, latent, and fused arrays, contracts,
hashes, normalization scales, selector trace, selected batch, and round freeze.
The completed matrix contains 44 frozen prediction points. The global barrier
verified every round/checkpoint/prediction hash and `test_truth_access_count=0`
before test labels were opened.

During resume, two implementation-only defects were corrected before the test
barrier: cached feature hashes are now validated after loading rather than
incorrectly treated as pre-load inputs, and an interrupted acquisition can be
completed only when its deterministic IDs/order and numeric artifacts match.
The representation diagnostic was also changed from the sample-space spectrum
of `X X^T` to the identical non-zero feature-space spectrum of `X^T X`. This
removed an unnecessary cubic decomposition without changing the diagnostic or
selection. Six focused tests pass, including the spectrum equivalence and
cache-contract checks. The before/after hashes and zero-test-access state are
recorded in `pre_test_resume_amendment.json`.

## Primary results: Gradient-MaxDet vs Fusion-MaxDet

Lower NRMSE AULC and lower final error are better.

| Seed | Gradient AULC | Fusion AULC | Fusion - Gradient | Gradient final NRMSE | Fusion final NRMSE | Fusion - Gradient |
|---:|---:|---:|---:|---:|---:|---:|
| 157 | 0.538950 | 0.551877 | +0.012927 | 0.484456 | 0.509684 | +0.025228 |
| 6101 | 0.702644 | 0.703279 | +0.000635 | 0.534559 | 0.551528 | +0.016969 |
| mean | 0.620797 | 0.627578 | +0.006781 | 0.509508 | 0.530606 | +0.021099 |

Fusion is worse on whole-trajectory AULC and at 1005 labels in both matched
seeds. Mean final endpoint metrics also worsen:

| Metric at 1005 | Gradient mean | Fusion mean | Fusion - Gradient |
|---|---:|---:|---:|
| V1 RMSE (mL) | 3.566795 | 3.676780 | +0.109985 |
| V2 RMSE (mL) | 5.089433 | 5.428084 | +0.338651 |
| V1 R2 | 0.777865 | 0.763152 | -0.014713 |
| V2 R2 | 0.881067 | 0.864828 | -0.016239 |

The phase AULCs show why the whole-trajectory conclusion is negative. Fusion
is worse in both early phases and in both middle phases. Seed 6101 has a late
improvement (`-0.017193` AULC), but seed 157 has a late regression
(`+0.013607`), so the late signal is not stable.

| Seed | Early delta, 333-525 | Middle delta, 525-653 | Late delta, 653-1005 |
|---:|---:|---:|---:|
| 157 | +0.013036 | +0.010895 | +0.013607 |
| 6101 | +0.017210 | +0.024799 | -0.017193 |

## N80, N90, and N95

The initial generated `labels_to_target.csv` uses each method's own observed
initial-to-final span and is not suitable for a matched cross-method N80/N90
claim. The descriptive post-barrier correction in
`labels_to_full_reference_posthoc.csv` uses the already frozen, matched
full-data reference and the common initial model. It is explicitly post hoc.

On the two-seed mean curve:

| Target | Gradient interpolated | Fusion interpolated | Gradient sustained | Fusion sustained |
|---|---:|---:|---:|---:|
| N80 | 860.3 | 828.5 | 877 | 845 |
| N90 | 1004.0 | 966.5 | 1005 | not reached |
| N95 | not reached | not reached | not reached | not reached |

Fusion reaches N80 earlier on the mean curve. Its N90 crossing is transient:
the curve regresses above the N90 threshold by 1005 labels. Gradient-MaxDet
reaches and sustains N90 at 1005. Seed-wise N80/N90 directions are also mixed:
Fusion is later on seed 157 but earlier on seed 6101. These crossings therefore
do not overturn the worse paired AULC and endpoint results.

## Selection and representation mechanism

Fusion genuinely changes the selected batches. Mean batch overlap with the
matched Gradient-MaxDet trajectory is 17.1% for seed 157 and 13.1% for seed
6101 (15.1% overall), with some rounds sharing no selected rows.

The latent block is much lower-rank and more internally correlated than the
gradient block:

| Seed | Gradient effective rank | Latent effective rank | Fused effective rank | Gradient abs kernel corr. | Latent abs kernel corr. | Fused abs kernel corr. |
|---:|---:|---:|---:|---:|---:|---:|
| 157 | 26.54 | 2.95 | 11.77 | 0.207 | 0.743 | 0.532 |
| 6101 | 20.55 | 2.74 | 9.62 | 0.214 | 0.762 | 0.572 |

Because the low-rank latent block receives half the normalized kernel weight,
the fused spectrum is less diverse than the gradient-only spectrum even though
the feature vector is wider. This is consistent with large selection changes
that do not improve predictive efficiency.

## Compute accounting

Each seed used 21 new fits and one legally reused round-zero fit. Excluding the
reused historical round-zero work:

| Seed | New training | New gradient extraction | Selector | New fits | New gradient extractions |
|---:|---:|---:|---:|---:|---:|
| 157 | 10,178.3 s | 1,619.9 s | 3.70 s | 21 | 20 |
| 6101 | 7,625.5 s | 1,886.2 s | 4.14 s | 21 | 20 |

Latent extraction and fusion construction were not separately clocked in the
frozen run. The original zero placeholders must not be interpreted as measured
zero cost; `compute_cost_audit.csv` marks both values unavailable. This is a
reporting limitation, not a selection or outcome ambiguity.

## Figures and result tables

Key figures are under the study's `figures/` directory:

- `nrmse_by_seed_and_mean.png`;
- `nrmse_aulc_comparison.png`;
- `endpoint_rmse_by_seed_and_mean.png`;
- `selection_overlap.png`;
- `representation_diagnostics.png`.

The complete per-budget metrics and audits are under `results/`, including
`learning_curve_metrics.csv`, `paired_primary_nrmse.csv`,
`phase_nrmse_aulc.csv`, `labels_to_full_reference_posthoc.csv`,
`gradient_maxdet_selection_overlap.csv`,
`representation_diagnostics_summary.csv`, and `compute_cost_audit.csv`.

## Conclusion

### Evidence

Fusion changes the acquisition geometry substantially, but it has worse NRMSE
AULC and worse final NRMSE in both matched seeds. Both endpoint RMSEs and R2s
also regress on average. The only favorable signals are earlier mean N80 and a
late-phase improvement in seed 6101; neither is stable across seeds, and the
mean N90 crossing is not sustained.

### Interpretation

At fixed `alpha=0.5`, the additional latent geometry does not translate into
better label efficiency. Equal block energy gives a highly correlated,
low-effective-rank latent kernel substantial weight, apparently displacing
useful gradient diversity rather than complementing it.

### Speculation

A smaller latent weight could preserve more gradient diversity, but this study
does not test that hypothesis. The current evidence is insufficient to justify
an alpha sweep or U50-Fusion expansion.

## Next decision

Do not promote `gradient_latent_fusion_maxdet`, do not expand it to five seeds,
and do not run the planned alpha or U50-Fusion variants. Retain Gradient-MaxDet
as the stronger method for this matched developmental comparison.
