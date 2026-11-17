# Six learning curves, including both Free versions

The requested dual-expert curve is omitted. All six curves use the same two seeds, 157 and 6101. Each point is the arithmetic mean of saved validation NRMSE; lower is better. Each curve ends at the last budget completed by both seeds. No smoothing, interpolation, extrapolation or best-so-far envelope is applied.

Free (new) is the completed CW-visible Sol 6.1/xhigh rerun. Free (historical) and LCMD16+Free16 retain historical mixed-model trajectories. Both LCMD controls use center/width-gradient geometry, as in the prior five-method figure. These are descriptive validation comparisons.

six_methods.png/pdf/svg show all available matched-seed points. six_methods_common_budget.png/pdf/svg show the common L333–L1005 interval. validation_curves.csv retains individual seed values and fit-audit hashes; mean_curves.csv contains means and sample standard deviations (ddof=1). snapshot.json records input hashes, method identities and endpoints. Only saved summaries and fit audits were read; no training, model calls or test-truth reads.

Reproduce with the original saved audits available locally:

```sh
/Users/fish/miniforge3/envs/fish/bin/python scripts/studies/plot_six_methods_both_free_20261112.py --output reports/<fresh-directory>
```
