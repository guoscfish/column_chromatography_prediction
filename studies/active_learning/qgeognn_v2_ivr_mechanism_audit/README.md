# IVR mechanism audit

This is a label-free exploratory audit of representation stability and batch
optimization on 25 previously frozen QGeoGNN-IVR states. It is not a new
predictive-performance benchmark.

- [Frozen design](PROTOCOL.md) and [numeric rules](config.json).
- [Overall decision](decision.json) and [overall report](FINAL_REPORT.md).
- [Gate A report](gate_a/FINAL_REPORT.md): representation and numerical stability.
- [Gate B report](gate_b/FINAL_REPORT.md): forward/backward and two-output IVR.
- [Stability figure](figures/representation_stability.png).
- `results/`: aggregate spectra, selection statistics, paired comparisons,
  mechanism outcomes and explicitly descriptive controls.
- `runtime/seed_*/round_*/`: checksummed gradient banks, full eigenvalue
  spectra, initial scores, selected candidates and update traces. These
  large artifacts remain local, following the repository's runtime policy.

## Reproduction

Use the qualified `fish` environment and the same frozen source artifacts.

```sh
python scripts/studies/run_ivr_mechanism_audit.py --run
python scripts/studies/report_ivr_mechanism_diagnostics.py
```

The main command resumes only compatible completed snapshots and rejects
changes to sealed code, config or inputs. `--report` validates all 25 receipts
and rebuilds the primary summaries without extracting gradients again.
The descriptive reporter requires the complete primary decision and does
not change its gates. Its LCMD stability control was added after the first
snapshot; endpoint geometry and duplicate diagnostics were also added during
execution. None are independent confirmation or retrospective gate overrides.

## Interpretation

The covariance updates are checked against direct reinversion. Scalar IVR
and two-output IVR optimize different surrogate risks, so their raw risk
numbers cannot rank predictive accuracy. Transferred-batch objective gaps
are signed differences relative to a greedy batch, not regret against a
global optimum, and may be negative.

The kernels and L/U membership come from existing scalar-IVR trajectories.
This controls the comparison of selectors at a fixed state; it cannot show
the result of a multi-output method's own closed-loop training trajectory.
No full sequential or COMPOUND experiment is launched by this audit command.
