# Implementation audit

- The numerical fusion is parameterized and the alpha=.8 protocol records
  `0.8*K_gradient_normalized + 0.2*K_latent_normalized`.
- The kernel-equivalence test verifies `Phi Phi^T = 0.8 Kg + 0.2 Kh`.
- Feature-cache contracts include alpha, checkpoint hashes, ordered sample IDs,
  labeled and unlabeled hashes, dimensions, block scales, and feature hashes.
- Round-0 reuse passed split, source, graph cache, checkpoint, state, sample
  order, preprocessing, normalization, target-scale, gradient-cache, and latent
  cache hash checks for both seeds.
- Every later round was trained, extracted, fused, and selected on its own
  alpha=.8 trajectory. No alpha=.5 or Gradient model was reused after round 0.
- Every selected batch contains 32 unique IDs from the current unlabeled pool;
  trajectory transitions are validated before labels are revealed.
- Both matched splits were copied from the frozen baseline and their hashes are
  recorded in `reuse_audit.json`.
- The global barrier contains 22 frozen prediction points. Test truth was first
  read only after both 11-point trajectories were frozen.
- Acquisition always used uncentered latent features. Centered latent was
  computed only as a post-extraction diagnostic.

## Verification

- `tests/active_learning_v2`: 121 passed.
- All four user-supplied Gradient/alpha=.5 NRMSE@653 reference values match the
  metrics recomputed from frozen predictions to better than `1e-9`.
- A post-run artifact audit rehashed every alpha=.8 checkpoint and prediction,
  checked every selection against its round-specific unlabeled pool, and proved
  that all post-round0 checkpoint hashes differ from both historical
  trajectories.
- `git diff --check` reports four Markdown hard-break lines in the generated
  report template. They are retained because the source file hash is part of
  the frozen pre-test protocol; changing whitespace after test reveal would
  invalidate exact code provenance.
