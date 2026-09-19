# Execution notes

The first invocation started 2026-09-18 after acquisition freeze at 10:45 UTC.
It exited before all trajectories completed. Six new fits (seeds 157 and 887,
rounds 2 through 4) were already complete and are resumed with artifact checks.
No new test evaluation had occurred. Partial round-5 attempts, where present,
are preserved in `runtime/seed_*/incomplete_attempts` and restarted from the
registered initialization.

Completed-fit timing excludes work lost in incomplete attempts, which did not
write a final fit audit. That overhead is unmeasured, not zero. Calendar elapsed
time in `execution_time.json` covers the successful invocation only and must not
be presented as the full elapsed duration including the first invocation.
