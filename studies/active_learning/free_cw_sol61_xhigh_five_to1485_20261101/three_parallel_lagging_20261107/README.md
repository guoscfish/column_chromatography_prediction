# Three concurrent seeds, lagging seeds first

The user authorized continuing the original CW-visible Free32 experiment to 36 rounds / L1485 for each of the five seeds, with at most **three seed/API workers and three expensive compute leases**, prioritizing fewer completed rounds. Ties use ascending seed number. A worker completes one round, then scheduling is reconsidered; healthy in-flight work is not preempted for ranking.

This independent overlay composes the complete retained upstream-error, immediate-retry, HTTP403, HTTP/2, persistent, and frozen scientific execution chain. The original sources and historical authorizations remain unchanged. `authorization.json` pins this overlay and its immediate predecessor; predecessor validation recursively checks the original bindings and preserved fatal receipts.

The three compute leases are `locks/new_free_resource.lock`, `_1.lock`, and `_2.lock`. The original lease counts adopted compute toward the new cap. All three retain the original NoCW priority wait. NoCW is never signaled or changed.

Migration adopts existing workers, starts no new workers while three or more are occupied, and retires an old selector only after freezing its exact verified PID and confirming a terminal failed stream, no pending requests, and no compute phase. A healthy request or compute task is retained. Each transition records a handover receipt. The retained 660-second no-stream watchdog also applies to new workers. Local failed-child restart cooldowns are persisted for 60 seconds; transport retry delay remains zero. Cumulative receipts are reused, not reset. All five independent existing credentials, full model inputs, gpt-6.1-sol/xhigh, scientific validation, and training/split protocol remain unchanged.

Initial completed rounds: 157=15, 887=29, 2357=27, 6101=27, 12203=24 (122/180). First priority is 157, 12203, 2357. Recent failures were HTTP429. Migration does not itself resolve provider failures and does not count retries as accepted batches.

Run `test_overlay.py` for six offline scheduling/resource tests. CLI `controller_overlay.py validate` checks the full immutable chain. Launch with the existing fish Python, original three authorization flags, KMP_DUPLICATE_LIB_OK=TRUE and OMP/OPENBLAS/MKL_NUM_THREADS=2. `active_runtime.json` and this directory's `launch.json` identify the current controller and authorization digest. Do not restart an older entry point.
