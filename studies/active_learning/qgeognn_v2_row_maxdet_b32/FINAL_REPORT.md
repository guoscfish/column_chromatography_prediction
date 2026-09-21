# Gradient-MaxDet and fixed-U50 Gradient-MaxDet — final row study

Status: **COMPLETE_DEVELOPMENT_EVIDENCE**. This is development evidence on the historically
exposed row cohort, not independent confirmation.

The lowest cohort-mean whole-curve combined-NRMSE AULC is `hybrid`. The
lowest mean endpoint combined NRMSE at 1005 labels is `gradient_maxdet`. Full
metrics, per-seed differences, early/middle/late partial AULCs, censored target
crossings, mechanism overlaps, and compute accounting are under `results/`.

All 88 new-method prediction points and 42 acquisition batches crossed one
global pre-test barrier before labels were opened. Historical Random, LCMD,
Hybrid, Kernel-IVR, matched full-data references, and exact reusable round-zero
artifacts were not retrained.
