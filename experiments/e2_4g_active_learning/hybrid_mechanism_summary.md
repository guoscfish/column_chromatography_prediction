# E2 Row Hybrid Mechanism Audit

## What The Finalized Row Pilot Shows

Hybrid has lower mean normalized AULC than Ensemble, and its mean within-batch pairwise latent distance is 17.313 versus 14.877 for Ensemble. The paired Hybrid–Ensemble diversity effect is reported in `batch_diversity_paired_effects.csv` using the three outer seeds as the statistical unit.

This co-occurrence is consistent with the hypothesis that uncertainty filtering followed by diversity selection reduces batch redundancy. It does **not** prove that diversity selection caused the AULC improvement: Hybrid and Ensemble selected sets differ in more than one property.

## Selected Error And Learning Gain

`selected_error_vs_learning_gain.csv` defines next-round improvement as `NRMSE_t - NRMSE_t+1`. Its Spearman summaries are descriptive only: rounds within an outer seed are not independent and the sample is small. The results therefore neither establish causal mediation nor justify a strict claim that prediction risk equals, or fails to equal, training utility.

## Required Causal Control (Not Run Here)

The direct future control is **uncertainty-filter-random**: retain the same Ensemble Top-25% candidate subset as Hybrid, select 25 samples uniformly at random within that subset, and compare it with farthest-first Hybrid. This isolates the contribution of diversity selection more cleanly. It is registered here only; no such active-learning curve was run.
