# Training Readiness

Status: **TRAINING_READY** for the frozen 1-seed validation pilot.

The protocol, explicit BN policy, validation-only adaptation contract, staged
checkpoint inheritance, dimensionally coherent endpoint scaling, and a small
P0-P3 manifest runner are present. Source-stat BN is reapplied after every
training-mode transition, and L-BFGS uses a full-batch closure.

The runner connects the audited 25g canonical table and graph cache, creates a
deterministic row split, and computes target scales from train rows. The
protected current-V2 source checkpoint is
`studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt`.
The one-seed pilot completed for P0-P3 with single-thread settings
(`KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). The
cache contains 19 usable rows; the resulting 15/1/3 train/valid/test split and
all validation trajectories are recorded in `pilot_artifact_manifest.json`.
No test scoring or test-label access occurred.

The next authorized step is the formal preregistered 5-seed run; it was not
started in this task.
