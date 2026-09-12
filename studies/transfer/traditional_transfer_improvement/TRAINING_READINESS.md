# Training Readiness

Status: **TRAINING_READY_FOR_FORMAL_5SEED_ROW**

Audited baseline HEAD: b1be9685ae6465d310793959bf1445e9226b4ec8.
The superseded 19-row pilot remains INVALID_PILOT_SUPERSEDED.

Corrected pilot_v2: 408 graph-covered filtered rows; exact frozen target ROW
seed 769539383; 320 train / 5 validation / 83 test. Qualified source seed 42
is separate. All training rows are used, test truth is excluded, and source
checkpoint preprocessing is inherited with verified hashes.

Historical shallow inventory exactly matches the original wrapper and historical
36,387-parameter count. Stage-A best inheritance, BN buffers/affine separation,
L-BFGS closure updates, loss scale invariance and all other readiness contracts
pass. Full repository tests: **377 passed**, 195 warnings, 85.48s in
conda fish. See pilot_v2/pytest_full.log and readiness_gate.json.

Validation combined NRMSE: P0 0.840366, P1 0.659585, P2 0.543798, P3 0.301062.
P2/P3 BN buffer drift is zero. All predictions are finite. P0/P1/P2 have
quantile crossing; the retained loss is a soft penalty, not hard ordering.
Five validation rows and unconverged Adam stages limit performance conclusions.

Recommend keeping P0/P1 controls and prioritizing P2/P3 in a separately authorized
25g + 40g x five-seed ROW run. This readiness result does not launch or authorize
that run. This task stops at the pilot. Full evidence and execution recovery
notes are in pilot_v2/PILOT_REPORT_VALIDATION_ONLY.md.
