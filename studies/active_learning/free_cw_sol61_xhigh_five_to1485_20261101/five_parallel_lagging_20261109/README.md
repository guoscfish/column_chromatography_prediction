# Five concurrent seed workers and compute leases

On 2026-11-09 the user requested: “现在开始五个并行吧”. In the existing
selection-and-computation context, this raises both resource limits from four
to five. Only resource scheduling changes; the frozen scientific trajectory,
gpt-6.1-sol/xhigh, five original independent credentials and complete selection
input remain inherited.

This entry composes `four_parallel_lagging_20261109/controller_overlay.py` and
its entire predecessor chain. Existing four-way and manual-resume one-round
workers retain their processes, requests and locks until they finish normally.
The scheduler includes adopted workers and starting children in the five-worker
limit. Each completed round triggers ranking by completed rounds, then seed.

Five compute leases share the existing original and `_1`, `_2`, `_3` paths and
add `new_free_resource_4.lock`. Thus adopted calculations count toward the same
limit. The unchanged NoCW training-priority wait runs before and after acquiring
a lease. No NoCW process is modified or signaled. Existing transport recovery,
660-second stream watchdog, zero transport backoff and 60-second local restart
cooldown are retained. New 401/402 errors remain fatal; only the five historical
401 receipts pinned in the manual-resume authorization retain their exception.

`authorization.json` binds this source and its immediate predecessor source and
authorization. `handover_preflight.json` preserves prior worker/stream states,
unresolved-request paths, cumulative attempts and completed artifact hashes.
`launch.json` records the replacement controller and caffeinate binding without
overwriting earlier launches. Resolve the latest runtime through
`active_runtime.json`.

Two offline tests verify ranking, five lease exclusion including adopted compute,
blocking a sixth computation, releasing/reacquiring leases and retaining NoCW
priority waits. Five inherited historical-401 tests also pass. The composed CLI
validator returns `VALID`; these checks make no API requests and train no model.

The five-minute monitor follows the five-way runtime. Monitoring output and
metrics never enter selection input or memory. `caffeinate -i -w` prevents idle
sleep while the controller lives; it does not establish lid-close sleep protection.
