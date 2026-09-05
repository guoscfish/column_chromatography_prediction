# Final standalone 4g qualification

Six runs only: row/compound × seeds 42, 525, 1101. Reuse the existing qualification manifests byte-for-byte; never regenerate. The 4163-row / 217-compound domain retains V1 ≤ 60 and V2 ≤ 120.

Use direct seeded standalone initialization and the unchanged R2 effective network/head/loss. Adam lr 0.001, weight decay 0, batch 2048, max 1000 epochs, patience 100, deterministic epoch shuffle, target weights 1:1; fit all preprocessing on training rows. Only validation selects checkpoints; test is evaluated after reload of the frozen best checkpoint. The standalone initializer consumes RNG only for effective modules; it is not claimed to recreate the diagnostic canonical Legacy initialization.

Judge stability, train/validation/test gaps, row predictive signal, and explainable compound generalization from the measured results; impose no arbitrary R² performance threshold. Quantile diagnostics do not block ordinary point transfer. Fix source selection now to row seed 42, regardless of the other runs' test results. Do not tune the head, backbone, data thresholds, or optimizer.
