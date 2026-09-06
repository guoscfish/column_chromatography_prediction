# Next transfer residual diagnostics

Decision: **`NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA`**.

Start with [the measured interpretation and stop decision](RESULT_INTERPRETATION.md), then the [complete metrics report](NEXT_TRANSFER_DIAGNOSTICS_REPORT.md). The [pre-experiment audit](../../../NEXT_TRANSFER_MODEL_AUDIT.md) explains which ideas were already tested and why only these two diagnostic families were added.

All 120 frozen cross-column contexts completed. Train-only monotone calibration did not establish a stable gain. Shared column calibration improved compound portfolio AULC by 9.04% versus affine, but only 1.42% versus scale-only and 1.64% versus an independent regularization control. These gains do not meet the frozen material/stability gate. This does not establish an irreducible noise floor or a causal readout bottleneck.

The qualified predictor, original cross-column results, splits and source checkpoint remain unchanged. The three-column portfolio uses matched total purchased labels, including validation; target-compound holdouts are purged from donor training. Historical test exposure is acknowledged. No source-unseen OOD, physical mass/flow law, or new-column extrapolation is claimed.

No Active Learning, full fine-tuning, adapter width sweep, Clean restart or adaptive readout experiment was run. No method was added after test evaluation.

Reproduction commands and artifact links are in [RESULT_INTERPRETATION.md](RESULT_INTERPRETATION.md). Runtime source inference is cached without labels; frozen predictions, fit coefficients, validation scores, label IDs and result tables are tracked. `protocol.json` locks the source, source code, input data and pre-experiment audit; changes stop silent reuse.
