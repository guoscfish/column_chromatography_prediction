# Gate A: representation stability

Completed: five frozen seeds x rounds 1, 5, 10, 15, 20.
300 scalar configurations (4 dimensions x 3 mappings x 25 states),
plus 75 shared-parameter two-output configurations at 512D.

The scalar audit covers 256, 512, 1024 and 2048 dimensions.
Multi-output cross-dimension stability has NOT been measured.
No new fits, labels, validation outcomes or test outcomes were used.

| Representation | Comparison | Median rho | Mean batch overlap | Median batch regret | Gate |
|---|---|---:|---:|---:|---|
| block | cross_seed_512 | 0.992130 | 0.6438 | 0.7601% | False |
| scalar | 512_vs_2048_same_seed | 0.994588 | 0.6871 | 0.4002% | False |
| scalar | cross_seed_512 | 0.988484 | 0.5767 | 0.8951% | False |

## Numerical checks

Largest regularized precision condition number: 1204.948.
Largest direct versus incremental covariance-factor error: 6.800e-15.
Largest historical-feature reconstruction relative error: 5.263e-07.
Smallest historical-score Spearman: 0.9999999969.

With mean per-experiment squared feature norm fixed to one,
`P = I + sum_L J_i.T J_i` has minimum eigenvalue at least one and
maximum eigenvalue at most `1 + sum_R ||J_i||_F^2 = 3331`.
Thus a large condition number of the UNREGULARIZED Gram matrix does
not imply an unstable inverse in the implemented posterior.

## Descriptive LCMD control (added after first snapshot)

- 512_vs_2048_same_seed: mean LCMD overlap 0.4779; IVR 0.6871.
- cross_seed_512: mean LCMD overlap 0.3367; IVR 0.5767.

A failed 0.75 mean-overlap gate is an engineering-gate result,
not proof that an acquisition method is scientifically invalid.
LCMD and transferred-objective controls contextualize this threshold;
they do not retroactively change the sealed decision.

Full spectra/scores and all selected candidates are stored under runtime;
aggregate tables are in ../results. See ../PROTOCOL.md and ../config.json.
