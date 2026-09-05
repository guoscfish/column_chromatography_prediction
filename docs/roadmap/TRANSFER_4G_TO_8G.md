# 4g→8g transfer roadmap

The final standalone source is qualified for ordinary point transfer. [The baseline study](../../studies/transfer/4g_to_8g/TRANSFER_BASELINE_REPORT.md) compares zero-shot, affine, target-head-only, last2 and full fine-tuning. Five frozen T1 partitions, random nested budgets 30/50/70/100 (eight validation labels included), 574 unthresholded target rows and source-train preprocessing are fixed. Test outcomes never select checkpoints or tune methods.

The source is final 4g row seed 42, chosen in preregistration. The source checkpoint hash and complete protocol are in the study directory. The study reuses split manifests, not historical performance conclusions. T1/T1b/G0/S1 are `HISTORICAL_LEGACY_PREDICTOR_EVIDENCE`.

Next work follows [the measured decision](../NEXT_STAGE_DECISION.md). Independent validation must precede confirmatory transfer claims. UQ/head qualification runs in parallel to ordinary transfer and is required before active acquisition. No acquisition or adapter sweep belongs to this first baseline.
