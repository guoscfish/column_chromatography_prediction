# Phase 0: endpoint-aware evaluation review

2026-09-18. Completed without new fits or test-truth access. This is retrospective
analysis of published results on seeds 157, 887, 2357, 6101, 12203, not a new
confirmation cohort. Historical files and decisions are unchanged.

## Prediction quality and label efficiency

| Method | NRMSE@1005 | V1 RMSE | V2 RMSE | V1 R2 | V2 R2 | NRMSE AULC | V1 R2 AULC | V2 R2 AULC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | .543301 | 4.146006 | 6.873136 | .672247 | .773242 | .624260 | .577877 | .664102 |
| Hybrid | .409709 | 3.294481 | 4.804172 | .787133 | .887548 | .479354 | .720649 | .820343 |
| LCMD | .408614 | 3.241494 | 4.922502 | .797100 | .882749 | .479539 | .729368 | .819585 |
| Kernel-IVR | .395913 | 3.168245 | 4.645669 | .806557 | .895395 | .500070 | .705014 | .800568 |

LCMD and Hybrid improve the whole curve relative to Random, including both R2
endpoints. Their NRMSE-AULC wins remain 5/5. IVR has the best final mean metrics,
but its NRMSE and both R2 AULCs are worse than LCMD's. There is no single winner
across the three reporting layers and no weighted overall score.

All three active methods also win against Random on both endpoint R2 AULCs in
5/5 seeds. LCMD's mean paired R2-AULC gains are +0.151491 (V1) and +0.155483 (V2),
with descriptive bootstrap intervals [0.123913, 0.172335] and
[0.114191, 0.198227]. Hybrid's corresponding gains are +0.142772 and +0.156241;
IVR's are +0.127137 and +0.136467. These are paired within the same split, not
comparisons between unmatched seed cohorts.

`NO_CLEAR_SEQUENTIAL_AL_GAIN` is the frozen LCMD-versus-Hybrid classification
fall-through, not evidence of no AL benefit. The original classification is
preserved. The new report does not silently replace its preregistered decision.

## Corrected definitions

- AULC is the trapezoidal integral divided by the observed budget span. R2 AULC
  is higher-is-better; error AULC is lower-is-better. "Normalized" describes the
  budget axis, not division by initial error or full-data gap.
- Error gap closure is `(E0-E)/(E0-Efull)`; R2 closure is
  `(R2-R20)/(R2full-R20)`. Values are not clipped to [0,1]. Nonpositive full-data
  gaps are undefined, not successful target attainment.
- N80/N90/N95 are reported for every metric, separately for V1 and V2, per seed
  and on the cohort-mean curve. These two aggregation orders are not equivalent.
- Each target has first interpolated, first observed, and sustained-through-final
  observed crossing. A last-point sustained crossing has no subsequent evidence
  of persistence. Unreached targets are censored with no extrapolation.
- Incremental saving uses `(Nrandom-Nmethod)/(Nrandom-333)` at the same metric,
  target and crossing rule. It is undefined when either curve is censored or
  Random needs zero additional labels. Shared validation adds 416 labels to all
  active budgets; it is excluded from incremental experimental saving.
- RMSE and R2 share squared-error information on a fixed test split; they are not
  independent confirmations. Their gap targets differ because RMSE is nonlinear.

For combined NRMSE at Random@1005, Random first reaches the threshold at 941,
LCMD and IVR at 493, Hybrid at 525. Actual-grid incremental savings are 448/608 =
73.68%, 73.68%, and 416/608 = 68.42%, respectively. The historical 79.27% LCMD
figure instead uses an interpolated 472.33 against a fixed 1005-label reference.
Both are descriptive estimates from this offline benchmark, not measured savings
in a prospective laboratory deployment.

Endpoint target conclusions also differ: LCMD's V1 RMSE first crosses N90 at
941 but does not retain it at 1005; V2 RMSE never reaches N90. IVR sustains V1
RMSE N90 from 973 and V2 RMSE N90 from 941. Hybrid first crosses V2 RMSE N90 at
909 but loses it later. R2 thresholds are reached earlier in several cases;
they must not be substituted for error thresholds.

## Robustness and cost

`paired_effects.csv` includes directional wins, ties, mean/median paired delta,
sample standard deviation, and a descriptive 95% paired bootstrap interval
(10,000 resamples, fixed RNG). Five overlapping row splits are not five
independent chemical datasets; the intervals are not a confirmatory significance
claim. All per-seed values remain available. There is no post-hoc winner gate.

| Method | Fits, all 5 seeds | Epochs | Fit time, hours | Gradient extraction, hours |
|---|---:|---:|---:|---:|
| Random | 110 | 22559 | 19.18 | 0 |
| Hybrid | 320 | 82123 | 56.18 | 0 |
| LCMD | 110 | 28339 | 16.06 | 3.39 |
| IVR | 110 | 27582 | 7.79 | 1.59 |

Hybrid's 210 extra ensemble fits are included. Historical round elapsed time
overlaps fit/gradient time and is not added to those components. Fit time includes
validation and prediction; standalone ensemble-inference timing was not recorded
and is marked missing. Selector timing is available for IVR only. Runtime
comparisons are descriptive because machine load differed. No historical cost
was incurred again for this report; original IVR reused five initial fits and
ran 105 new fits. Full-data references are shared evaluation overhead, excluded
from acquisition-method operating costs.

## Scope corrections and next control

B16/B32/B333 results use different final label budgets and cannot isolate a batch
size effect. The Phase 1 control fixes 333 initial + 320 new = 653 active labels.
Static LCMD uses one initial ordered B320 sequence; Adaptive LCMD recomputes after
each B32. Both use identical scratch training. OneShot and Static share the same
final model because their final ordered training data and initialization match.

IVR already uses a regression V-optimal/BAIT-like objective. The mechanism audit
does not establish numerical instability or a greedy optimizer bottleneck.
CW-LCMD's negative development result and innovation's selection-only results do
not justify additional coordinate variants. Quantile width is a mixed proxy,
not established epistemic uncertainty or calibrated aleatoric truth.

Artifacts: `results/complete_results.csv`, the tidy per-seed tables, targets,
costs, source hashes in `provenance.json`, and `figures/learning_curves.png`.
