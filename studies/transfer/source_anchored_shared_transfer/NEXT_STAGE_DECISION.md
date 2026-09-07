# Source-anchored shared transfer: final decision

**CURRENT_QGEOGNN_REPRESENTATION_NOT_SUFFICIENT_FOR_LOW_LABEL_CROSS_COLUMN_TRANSFER**

Outcome D under the frozen decision rule; A/B/C are false. This is a negative
result for the tested source-initialized, low-label training recipe, not a
proof that the 128D representation lacks useful information. The unchanged
linear/ReLU head can itself express positive scale-only by scaling its
weights/biases. Expressivity, optimization and small-sample estimation are
not separately identified by these results.

## Completed scope and evidence

120 frozen contexts (8g/25g/40g, row/compound, five seeds, budgets
30/50/70/100), 480 new fits, five reused reference methods and 1080 metric
records. Zero failed fits, missing contexts, nonfinite predictions or donor
target labels. All predictions were frozen before this stage's test read.
Historical test exposure makes this developmental evidence.

N1 trains `backbone.convs.4`, condition completion and target head (36,387
parameters); N2 trains the complete backbone, condition completion and target
head (458,952). M1/M2 have exactly the matched trainable modules and add a
source-train batch loss through an independent frozen source head. All start
from the same qualified source function. Lambda=1, Adam lr=1e-4, raw-mL loss,
maximum 500 epochs and validation-only early stopping remain frozen.

## Label efficiency

Five-seed mean normalized AULC; lower is better. Conditional EA and policy
are historical frozen references, not newly selected or tuned learners.

| Column / protocol | Scale | Shrinkage | Conditional EA | Conditional policy | N0 head | N1 shallow | N2 full | M1 shallow | M2 full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8g row | 0.730 | 0.754 | 0.755 | 0.753 | 0.679 | 0.725 | 1.256 | 0.923 | 1.207 |
| 8g compound | 0.735 | 0.728 | 0.752 | 0.753 | 0.725 | 0.823 | 1.399 | 1.015 | 1.405 |
| 25g row | 2.060 | 2.021 | 1.966 | 1.951 | 3.376 | 2.117 | 2.947 | 2.939 | 4.517 |
| 25g compound | 1.607 | 1.524 | 1.479 | 1.484 | 3.050 | 2.067 | 3.172 | 3.297 | 4.402 |
| 40g row | 3.635 | 3.286 | 3.089 | 3.123 | 7.964 | 3.480 | 5.373 | 5.818 | 8.787 |
| 40g compound | 3.601 | 3.704 | 3.417 | 3.704 | 7.719 | 3.959 | 7.236 | 6.211 | 9.118 |

In 25g and 40g compound, all four new neural arms lose AULC to the strongest
fixed calibration reference in 5/5 seeds. Even N1, the best new neural arm
by mean AULC, is 39.81% and 15.86% worse than conditional EA. N1 improves over
N0 but this does not support replacing calibration. M1 worsens matched N1
AULC by 59.47%/56.89%; M2 worsens matched N2 by 38.79%/26.01%. M1 wins 0/5
paired seeds in both columns; M2 wins 0/5 and 1/5. No new arm passes the
all-calibration gate in even one context, at either primary endpoint.

8g is a narrower task with a competitive historical N0. N1 row AULC is close
to scale-only, but compound deteriorates; M2's 3.92% AULC improvement over N2
on 8g row is below the material threshold and does not replicate. These local
results do not justify a blanket statement that every neural method fails
on every target task.

## Budget-100 absolute error

Entries are V1/V2 RMSE and V1/V2 MAE in mL, then combined RMS NRMSE,
averaged over five seeds. These are purchased planned budgets; compound
acquisition uses the original whole-compound actual counts.

| Context | Method | RMSE V1 / V2 | MAE V1 / V2 | Combined NRMSE |
| --- | --- | ---: | ---: | ---: |
| 8g row | Scale | 6.20 / 9.24 | 3.82 / 5.52 | 0.690 |
| 8g row | N1 | 6.21 / 9.70 | 3.56 / 5.74 | 0.703 |
| 8g compound | Shrinkage | 5.99 / 8.59 | 3.26 / 4.75 | 0.660 |
| 8g compound | N1 | 6.86 / 10.27 | 3.52 / 5.15 | 0.765 |
| 25g row | Conditional EA | 17.78 / 25.30 | 9.13 / 15.04 | 1.948 |
| 25g row | N1 | 17.38 / 25.66 | 10.16 / 14.90 | 1.929 |
| 25g compound | Scale | 14.76 / 21.34 | 10.33 / 15.01 | 1.624 |
| 25g compound | Shrinkage | 13.95 / 19.76 | 8.91 / 12.49 | 1.525 |
| 25g compound | Conditional EA | 13.01 / 19.76 | 8.17 / 12.82 | 1.456 |
| 25g compound | N1 | 19.55 / 26.28 | 11.94 / 16.03 | 2.102 |
| 25g compound | N2 | 26.57 / 41.19 | 17.16 / 26.17 | 2.996 |
| 25g compound | M1 | 27.82 / 44.20 | 17.24 / 27.29 | 3.165 |
| 25g compound | M2 | 38.53 / 61.28 | 25.81 / 41.83 | 4.385 |
| 40g row | Conditional EA | 29.99 / 36.19 | 17.31 / 20.71 | 3.128 |
| 40g row | N1 | 29.84 / 36.95 | 17.67 / 22.28 | 3.133 |
| 40g compound | Scale | 34.74 / 42.71 | 24.33 / 28.51 | 3.640 |
| 40g compound | Shrinkage | 33.57 / 40.97 | 20.08 / 23.08 | 3.511 |
| 40g compound | Conditional EA | 31.84 / 43.46 | 18.64 / 28.43 | 3.441 |
| 40g compound | N1 | 37.46 / 48.27 | 21.99 / 28.53 | 3.976 |
| 40g compound | N2 | 64.98 / 89.61 | 42.65 / 58.18 | 7.044 |
| 40g compound | M1 | 54.52 / 75.14 | 30.87 / 42.96 | 5.906 |
| 40g compound | M2 | 86.41 / 120.96 | 60.42 / 85.59 | 9.404 |

N1's row performance is much closer to calibration than its compound
performance. Relative to conditional EA, its 25g row combined gain is only
0.97%, with V2 RMSE worse; 40g row combined error is slightly worse. There
is no replicated >=5% or >=10% all-reference high-budget gain. Large-column
absolute accuracy is not solved. Full per-method/per-context R2, RMSE and MAE
are in [budget100_mean_metrics.csv](budget100_mean_metrics.csv).

## Source preservation is not target improvement

The fixed 416-row source probe initially has RMSE 2.49/3.81 mL and R2
0.866/0.935. All-budget/five-seed means after compound adaptation:

| Column | N1 source RMSE | M1 source RMSE | N2 source RMSE | M2 source RMSE | N2 -> M2 combined NRMSE drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25g | 22.23 / 46.90 | 8.44 / 16.01 | 32.36 / 60.93 | 5.05 / 8.81 | 3.674 -> 0.317 |
| 40g | 45.80 / 86.08 | 17.72 / 33.52 | 55.47 / 98.38 | 5.03 / 9.69 | 6.329 -> 0.342 |

Both anchored variants reduce source drift in 5/5 seeds in every one of the
six column/protocol contexts (averaged over budgets per seed). Ordinary FT
substantially loses the original source function, but preserving it does not
improve target learning here. This is consistent with source/target conflict
or an overly restrictive anchoring recipe; it does not identify which is
causal or justify post-test lambda tuning. Source loss and target loss share
units, but lambda=1 does not guarantee equal gradient strength.

Target AULC SD on 25g compound is N1/M1 0.341/0.623 and N2/M2 0.815/0.865.
On 40g compound it is 0.384/0.674 and 2.076/0.463. M2 therefore reduces
40g full-FT dispersion but at substantially worse mean error. None passes
the joint source-preservation, target-variance and nonworse-mean gate for C.
The five partitions overlap; this SD is not pure optimization variance.

Source-head parameter drift is exactly zero in all 480 fits. On compound,
N2/M2 backbone L2 drift is 3.973/3.464 (25g) and 7.775/5.481 (40g);
target-head drift is 0.273/0.293 and 0.548/0.541. These parameter norms do not
substitute for functional source performance and exclude BN running buffers.

Training-cap hits (out of 120 fits/arm): N1 19, N2 15, M1 34, M2 1.
Selections in the first five epochs: N1 0, N2 8, M1 0, M2 12. M1 hits
500 epochs in 15/20 40g-row fits; this is a limitation of the tested recipe,
not permission to extend training after test. Source replay draws total
2,164,678 for M1 and 1,520,792 for M2, sampled from 3330 historical source
training rows. Per-fit unique counts and purchased/train/validation counts
are retained in the ledgers and training audit.

## Error distribution and EA failure

At budget100 compound, V1 source-q50 top-decile RMSE for Scale/N1/M1 is
30.08/48.01/70.90 mL (25g) and 45.84/56.88/119.89 mL (40g). Neural transfer
does not fix the high-q50 tail. There are real local gains: 40g V1 low/mid
q50 MAE is 16.37/22.58 for Scale and 11.53/12.11 for N1, while high-q50 MAE
worsens from 31.88 to 37.86. N1 improves high-EA V1 RMSE from 27.77 to
10.28 mL, but low-EA worsens from 47.44 to 56.35. It has not resolved the
EA-dependent failure across the distribution.

Compound-macro V1/V2 MAE for 25g Scale/N1/M1 is 10.50/15.33,
13.31/18.27, 19.40/31.09 mL. For 40g it is 24.67/29.21, 22.66/29.70,
33.10/46.61; shrinkage achieves 20.48/23.94. Thus scale-only is not best
in every subgroup or metric, but the neural arms do not establish a more
comprehensive advantage over strong calibration. Median relative errors
also have local improvements; for example 40g N1 V1/V2 is 0.234/0.204
versus Scale 0.367/0.282 and shrinkage 0.240/0.184. No single output can be
used to conceal the other. All strata, median absolute errors, compound
macro RMSE and the preregistered 1 mL relative-error floor are reported in
the [full interpretation](RESULT_INTERPRETATION.md) and accompanying CSVs.

## Baseline and next research decision

Retain **scale-only / local identity shrinkage** as the defensible primary
point-transfer baseline family. Preserve frozen conditional references and
the competitive 8g head-only comparator. Do not install a per-context
test-selected oracle or promote N1/N2/M1/M2 as the main learner for AL.
No new neural method stably beats the strong calibration references.

The next research priority, if separately authorized, is a preregistered
column-conditioned shared-representation study with explicit label accounting
and independent compound/batch confirmation. Controlled adaptive readout is
a distinct alternative intervention, not an automatic combined change.
Collecting independent repeats, tail coverage and crossed column/flow data
would help distinguish modeling limitations from data limitations. The
current results do not establish an irreducible noise floor.

Do not begin UQ + Active Transfer now; independent transfer validation and
the existing quantile/UQ qualification remain prerequisites. No AL, new
architecture, feature/LR/lambda sweep or donor-label adaptation was started.
`target-compound holdout != source-unseen molecular OOD`; mass, flow and
column specification remain confounded, with no causal effect claim.

Historical **NO_ADDITIONAL_COMPLEXITY_JUSTIFIED_FOR_TESTED_CALIBRATION_EXTENSIONS**
remains valid in its original scope. This completed stage fills the missing
current-V2 25g/40g shallow/full comparison and separately tests source anchoring.
It does not reverse or rewrite the earlier calibration experiments.
