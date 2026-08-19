# E2 Compound Seed-42 Preflight

This directory contains only source-free Round-0 K=3 diagnostics and a Round-1 acquisition dry-run for Random, Coverage, Ensemble and Hybrid. No Round-1 model was trained and the full compound pilot was not started.

## Partition checks

- L0 train/validation, U0 and test counts are 318/57/3375/413; validation is included in the 375-label budget.
- Test canonical compounds have zero overlap with L0 train, L0 validation or U0.
- Sample IDs are unique and stable; the scaler uses L0 train only.

## Round-0 signals

| Signal | Spearman | AUROC | Enrichment | AUSE |
|---|---:|---:|---:|---:|
| Ensemble | 0.625 | 0.902 | 6.154 | 0.057 |
| Quantile Width | 0.549 | 0.907 | 5.917 | 0.066 |
| Latent Distance | 0.550 | 0.848 | 4.822 | 0.077 |
| Random | 0.008 | 0.488 | 0.740 | 0.223 |

Quantile Width remains a strong compound source-free risk-ranking signal, but this preflight does not promote it into the preregistered four-strategy primary comparison.

## Acquisition dry-run

All four methods selected 25 unique U0 samples with no test leakage. Coverage repeated deterministically. Hybrid selected only from the frozen Top-25% Ensemble subset (844 candidates) and used the same farthest-first implementation as Coverage. These checks authorize review before a full compound pilot; they do not constitute a compound learning-curve result.
