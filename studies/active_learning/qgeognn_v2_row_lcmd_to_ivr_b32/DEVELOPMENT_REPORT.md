# Two-seed development result

The authorized staged run completed seeds 157 and 6101 from the exact frozen
LCMD-L653 state. Both trajectories reached 1005, remained test-blind until all
12 continuation points were frozen, and were then evaluated in a separate
development report. No other seed was executed.

The switch keeps the LCMD prefix exactly. Mean full reconstructed AULC is
0.614967 for LCMD->IVR versus 0.619050 for continued LCMD. Mean late AULC
(653--1005) is 0.527366 versus 0.535162, and mean endpoint combined NRMSE is
0.502290 versus 0.518751. Endpoint V1/V2 RMSE are 3.482/5.123 versus
3.537/5.505. These are useful directional signals, but the paired late-AULC
result is not stable: seed 157 improves by -0.02023 while seed 6101 worsens by
0.00464. The endpoint NRMSE improves in both seeds, by -0.03268 and -0.00024;
the second difference is too small to treat as a meaningful win by itself.

Against pure Kernel-IVR, the switch is also mixed. Seed 157 has slightly worse
late AULC (+0.00444) but slightly better endpoint NRMSE (-0.00249); seed 6101
has substantially better late AULC (-0.05247) and endpoint (-0.01671).

| Scope | Method | Full AULC | Early AULC | Late AULC | NRMSE @1005 | V1 RMSE | V2 RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| development mean | LCMD | 0.619050 | 0.711328 | 0.535162 | 0.518751 | 3.537 | 5.505 |
| development mean | pure IVR | 0.614458 | 0.683845 | 0.551379 | 0.511890 | 3.540 | 5.234 |
| development mean | LCMD->IVR | 0.614967 | 0.711328 | 0.527366 | 0.502290 | 3.482 | 5.123 |

Target crossings are in `development/targets.csv`. Cohort mean T80 is 705.57
for the switch, 693.34 for LCMD and 819.53 for pure IVR. Cohort mean T90 is
871.49 for the switch and 962.35 for pure IVR; LCMD is censored at 1005.
The switch's T90 first observed budget is 877 and its sustained-through-final
budget is 973. T95 is censored for all three cohort curves. Per-seed values
show the same mixed pattern: switch T90 is censored for seed 157 but reaches
first observed 813 and sustained 941 for seed 6101.

The development decision is **stop before the remaining three seeds**. The
hypothesis receives partial support for endpoint and T90 behavior, but the
defining late-stage AULC advantage is not directionally consistent across the
two development splits, and pure-IVR comparison is also split. This is not
enough evidence to spend the remaining 33 fits and 30 gradient extractions,
and it does not justify an adaptive controller.

The most plausible interpretation is that IVR's late benefit depends on the
trajectory-specific posterior geometry. LCMD-L653 can make that geometry
useful for one split and less useful for another; a fixed switch at 653 is not
a generally reliable stage rule. Endpoint improvement may also reflect
late predictor behavior rather than a stable acquisition advantage.

## Adaptive diagnostic

`development/observable_state_diagnostic.csv` records only validation fit
audits, gradient norm summaries, IVR predicted variance reduction and candidate
pool size. It reads no test truth or hidden U labels. It currently leaves
effective rank, LCMD coverage and cross-method candidate overlap explicitly
uncomputed rather than inventing proxies. The available two-seed diagnostic
does not establish a stable validation plateau / IVR-gain transition: the
signal is too split and the fixed switch itself was selected post hoc.

Do not train RL or a bandit. A future adaptive study would need a new
validation-only protocol with a predeclared one-way `StageAdaptiveSwitchPolicy`
and an untouched confirmation cohort. It should require a sustained validation
plateau plus a stable IVR marginal-gain condition, with no switch-back.
