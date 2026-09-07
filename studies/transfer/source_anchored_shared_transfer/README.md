# Source-anchored shared representation transfer

Completed: 120 contexts, 480 new fits, 1080 metric records; no failed fits.
Decision: **CURRENT_QGEOGNN_REPRESENTATION_NOT_SUFFICIENT_FOR_LOW_LABEL_CROSS_COLUMN_TRANSFER**
(tested recipe, developmental evidence). Source anchoring retains the source
function but does not yield a better low-label target learner. Retain
scale-only / local identity shrinkage as the main point-transfer baseline.
Read the [scientific report](RESULT_INTERPRETATION.md),
[final decision](NEXT_STAGE_DECISION.md) and [engineering checks](ENGINEERING_CHECKS.md).

Frozen design: [MODEL_PREREGISTRATION.md](MODEL_PREREGISTRATION.md).
Implementation/evidence audit: [PRE_EXPERIMENT_AUDIT.md](PRE_EXPERIMENT_AUDIT.md).
Qualified final V2, 8g/25g/40g, row/compound, five seeds, nested budgets
30/50/70/100: 120 contexts and four new matched neural methods (480 fits).

The runtime checkpoint/log directory is ignored by git. Compact blind
predictions, per-context manifests, fit/label audits and scientific tables
are retained. Completed context hashes and the protocol must match before
reuse. Partial fits resume from the last 25-epoch checkpoint; source sampling
is a deterministic function of seed and epoch. No test-driven model additions.

Run from the repository root in the existing `conda fish` environment:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR=/tmp/source-anchor-mpl
conda run --no-capture-output -n fish python -m pytest -q tests
conda run --no-capture-output -n fish python scripts/studies/evaluate_source_anchored_transfer.py --lock
conda run --no-capture-output -n fish python scripts/studies/run_source_anchored_transfer.py --execute --workers 4
conda run --no-capture-output -n fish python scripts/studies/evaluate_source_anchored_transfer.py
conda run --no-capture-output -n fish python scripts/studies/summarize_source_anchored_transfer.py
```

The evaluator refuses incomplete or changed global/context prediction freezes.
It writes the first test-read event after verifying every context. Source
probe truth is diagnostic only and is also absent from fitting. Target data
features and purchased label cells are read separately. Other target columns'
labels are never used by a focal learner. Source-train replay is historical
knowledge outside the purchased target-label budget.

The first frozen development context was `8g/row/769539383/30`. It is part of
the final design, not an additional tuning dataset. The smoke was used only
for execution correctness and its full-length fits are reused without tuning.

`KMP_DUPLICATE_LIB_OK=TRUE` is the existing local duplicate-OpenMP workaround.
The implementation remains CPU-qualified with one torch thread per worker.
The low-level library deprecation, optional charset and graph node-inference
warnings are recorded in the runtime logs; finite gradients/predictions and
model/label contracts are explicitly checked.
