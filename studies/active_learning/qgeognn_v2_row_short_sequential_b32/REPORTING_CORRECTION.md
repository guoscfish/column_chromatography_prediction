# Post-freeze reporting correction

The experimental protocol in `protocol.json` and all 16 checkpoint/prediction pairs in `global_pre_test_freeze.json` were frozen before test truth access. Their source-code and artifact hashes remain the original experiment record.

After evaluation, the first reporting pass incorrectly classified Center/Width-LCMD as `PROMISING_FOR_525`. It evaluated an improvement against Gradient-LCMD before the explicit STOP rule against the strongest baseline, Gradient-MaxDet. Both seeds have worse AULC and NRMSE@429 than Gradient-MaxDet, so STOP must take precedence. The reporting function was corrected, tested with a regression case, and the tables/report/figures were regenerated exclusively from the previously frozen metric CSV. No fit, acquisition, prediction, or test-label reveal was repeated.

The original reporting-source SHA-256 recorded in `protocol.json` is `abfaea41cf20a570d0a40bcb2c68346ee73950cd100f0810d4668c24a4b8b19a`. The current reporting-source hash intentionally differs; this is an auditable post-freeze reporting-only amendment, not a silent replacement of the executed protocol. The experiment code and all runtime artifacts remain bound to the frozen protocol hash.
