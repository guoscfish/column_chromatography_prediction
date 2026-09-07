# Pre-test evaluator implementation correction

During the blind formal fitting phase, before any new target-test truth read,
the evaluator was reviewed against the already frozen scientific design.
The original evaluator contract is retained as `evaluation_protocol_initial.json`.
The corrected implementation is re-locked in `evaluation_protocol.json`.

1. Outcome C requires that calibration actually has lower mean AULC in the
   replicated stabilization contexts. Merely failing the joint material gate
   is not evidence that calibration has lower error. The correction enforces
   the preregistered wording “calibration remains stronger”; it does not change
   the 5%/4-of-5 thresholds, replication rules, methods, or fit/selection recipe.
2. Evaluation source standard deviations are read directly from the already
   hash-locked qualified source checkpoint. They no longer depend on an
   additional unprotected historical JSON lookup. Values are unchanged.

No training code, source checkpoint, predictions, test metrics, purchased
labels, model selection rule or scientific preregistration was changed.
No target test was used to identify these implementation issues.
