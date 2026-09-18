# IVR mechanism audit: final report

Decision: `STOP_BEFORE_GATE_C`.

All 25 frozen states were audited. No training or label reveal was performed.
The gates measure reproducibility and surrogate behavior, not predictive improvement.

| Representation | Comparison | Median rho | P10 rho | Mean overlap | P10 overlap | Pass |
|---|---|---:|---:|---:|---:|---|
| block | cross_seed_512 | 0.9921 | 0.9880 | 0.6438 | 0.5312 | False |
| scalar | 512_vs_2048_same_seed | 0.9946 | 0.9892 | 0.6871 | 0.5625 | False |
| scalar | cross_seed_512 | 0.9885 | 0.9794 | 0.5767 | 0.4688 | False |

Forward/backward positive gain: 7/25 snapshots; median relative batch gain 0.000000.
Multi-output IVR beats same-state LCMD on its own surrogate: 25/25 snapshots.
Candidate eligible for a separately sealed training experiment: None.

See PROTOCOL.md and config.json for formulas, fixed gates, limitations and representation scaling.
Raw low-rank kernels do not establish numerical instability. Scalar/block objectives are not directly comparable.
Marginal variance-reduction scores need not be monotone. No test-error gate was applied.
No automatic dimension/noise/regularization search or COMPOUND expansion is allowed.
