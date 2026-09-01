# A1a — Hybrid One-Step Batch-Diversity Control

Status: `COMPLETED_STOPPED`. Mechanism gate: **FAIL**. A1b is `NOT_AUTHORIZED`.

## Question and frozen intervention

Across five new row-level outer seeds, one frozen Round-0 K=3 source-free ensemble created one exact Top-25% uncertainty shortlist. The treatment selected 25 rows by the frozen farthest-first Hybrid rule; ten deterministic controls independently selected 25 rows at random from that same shortlist. Every arm used the same L0, validation, test, scaler, member seeds, and initial parameter hashes within its outer seed. Only the selected label set changed. S1 supplied no acquisition input.

The independent replication unit is five outer datasets, not the fifty within-seed controls. The primary gain is `Round-0 NRMSE - after-batch NRMSE`; larger is better.

## Exact primary results

| Outer seed | Baseline | Hybrid after | Hybrid gain | Random gains 0–9 | Random mean | Random median | Hybrid − mean | Hybrid − median | Beat count | Percentile |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 17 | 0.662270 | 0.614944 | 0.047326 | 0.003962, 0.026631, 0.024728, 0.001662, 0.057490, -0.046641, 0.014715, 0.056263, 0.035206, 0.017141 | 0.019116 | 0.020935 | 0.028211 | 0.026392 | 8/10 | 0.80 |
| 137 | 0.519874 | 0.570306 | -0.050432 | -0.013585, -0.043340, -0.001458, 0.006726, 0.043184, -0.002741, 0.014045, -0.015947, 0.019380, 0.012720 | 0.001898 | 0.002634 | -0.052331 | -0.053066 | 0/10 | 0.00 |
| 941 | 0.816761 | 0.768518 | 0.048243 | 0.083755, -0.001635, 0.030394, 0.033751, 0.094332, 0.031594, 0.026266, 0.032539, 0.038622, 0.012816 | 0.038243 | 0.032066 | 0.010000 | 0.016177 | 8/10 | 0.80 |
| 2027 | 0.532779 | 0.496906 | 0.035873 | 0.018951, 0.007659, 0.027564, 0.066517, 0.069338, 0.038528, 0.000722, 0.058245, 0.014627, 0.041386 | 0.034354 | 0.033046 | 0.001519 | 0.002827 | 5/10 | 0.50 |
| 4099 | 0.664844 | 0.655677 | 0.009166 | 0.073907, 0.021093, 0.102397, -0.015963, 0.010145, 0.015910, 0.048735, 0.019965, 0.032250, 0.027615 | 0.033606 | 0.024354 | -0.024439 | -0.015188 | 1/10 | 0.10 |

`arm_metrics.csv` preserves the full-precision values. It was transcribed from the formal runner's machine-emitted completion records after the first compact-output implementation omitted that intermediate table; a deterministic check confirms that it exactly reproduces every aggregate in `seed_summary.csv`.

## Preregistered mechanism gate

All four conditions were required:

| Condition | Observed | Pass? |
|---|---:|:---:|
| Hybrid gain above control median in at least 4/5 seeds | 3/5 | No |
| Mean across seeds of Hybrid minus control mean > 0 | -0.007408 | No |
| Median across seeds of Hybrid minus control median > 0 | +0.002827 | Yes |
| Hybrid beats at least 8/10 controls in at least 3/5 seeds | 2/5 | No |

Therefore `diversity_mechanism_supported = false`. Under a shared Round-0 uncertainty shortlist, this experiment did not find stable one-step learning-advantage evidence for farthest-first selection. This does not erase the historical E2 row-level pilot, establish that diversity never helps, compare uncertainty filtering against the full pool, or generalize to novel compounds.

## Stop decision and audit

A1b is not authorized. No compound run, advanced batch method, active-transfer study, or additional diagnostic was started. All 180 fits are represented in `fit_audit.csv` (60 arms × 3 members). Initialization hashes, validation hashes, and scaler hashes are consistent within seed/member comparisons; test truth was not used for checkpoint selection. All selected batches contain 25 unique U0 rows from the exact shared shortlist. Runtime checkpoints, histories, fit results, predictions, and resume state were deleted after finalization: 587,719,959 bytes deleted; retained compact files were 658,555 bytes before final placement/manifest accounting.
