# QGeoGNN-V2 4g row Gradient-LCMD-TP

Status: `COMPLETED / STRONG_POSITIVE`.

LCMD beat the Random median in 5/5 outer seeds; mean after-batch combined NRMSE improvement was 31.69%. See `FINAL_REPORT.md`, the frozen `PREREGISTRATION.md`, and `results/seed_summary.csv`.

## Historical boundary

- Under the legacy E2 row protocol, Coverage/Hybrid beat Random in 3/3 seeds; that was pilot evidence, not a current-V2 result.
- A1a applied farthest-first inside a shared Top-25% uncertainty shortlist and failed its mechanism gate. It rejected only that uncertainty-shortlist-to-diversity mechanism, not 4g active learning in general.
- The legacy predictor had a condition-feature reachability defect. Its E2/A1a numbers are not pooled with this study.
- This experiment uses the qualified standalone QGeoGNN-V2 and full-pool, sketched full-network Gradient-LCMD-TP.

Method basis: Holzmüller et al., [JMLR 24(164)](https://www.jmlr.org/papers/v24/22-0937.html), with the corrected TP semantics documented by the [author implementation](https://github.com/dholzmueller/bmdal_reg).
