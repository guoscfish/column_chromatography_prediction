# QGeoGNN-V2 Sequential B=32 Active Learning

This study is the sequential extension of the established confirmation cohort.
It compares one nested Random trajectory, V2-Hybrid, and Gradient-LCMD for each
of seeds 157, 887, 2357, 6101, and 12203.

Commit A freezes implementation, tests, splits, metrics, targets, and decision
logic. It contains no formal training or test result. The formal run has 21
complete acquisitions after the shared 333-label initial state and ends at
1,005 active labels. Shared validation always costs another 416 labels.

Primary active-learning evidence is normalized AULC, labels-to-Random@1005,
and incremental new-experiment saving. Endpoint RMSE, MAE, and R2 are secondary
predictor diagnostics.

See `PREREGISTRATION.md` for frozen scientific choices and `PROTOCOL.md` for
the execution, resume, freeze, and test-reveal barriers.
