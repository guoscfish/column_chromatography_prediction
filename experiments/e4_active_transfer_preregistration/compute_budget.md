# Compute budget (estimate)

The pilot has 3 seeds × 5 strategies × 16 budgets × 3 members = 720 target fits (plus zero-shot/full-data controls). Using the E2 observed upper bound of 56,337 seconds / 297 fits gives an intentionally conservative upper bound of about 136,500 seconds (38 hours); last2+head fine-tuning is expected to be lower. Plan for 720 best checkpoints and roughly 1–5 GB depending on checkpoint compression and prediction artifacts. Measure actual wall time before any expansion.
