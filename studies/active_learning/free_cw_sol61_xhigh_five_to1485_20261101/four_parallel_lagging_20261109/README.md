# Four concurrent seed workers and compute leases

The user explicitly confirmed on 2026-11-09: “是，选点和计算都改为4路”.
This overlay supersedes only the previous resource limits. It composes the
unchanged manual-resume overlay and the complete earlier authorization chain.
The model, effort, five credentials, full selection input, scientific code,
accepted batches, trained models and cumulative attempts remain inherited.
New HTTP 401/402 responses remain fatal; only the five historical receipts
pinned by the manual-resume authorization retain their existing exception.

The controller dispatches at most four seed workers, counting adopted workers
and children that are still starting. It reorders unfinished seeds by completed
round count, then seed number, after every round. Four shared compute leases
include the original three paths and `new_free_resource_3.lock`; existing
computations therefore count toward the same limit. The original NoCW training
priority wait runs before and after acquisition. Healthy old workers finish
their current round without interruption. Adopted workers exiting without a new
completion receive the same 60-second local restart cooldown.

`authorization.json` binds this source and its immediate predecessor source and
authorization. `handover_preflight.json` records worker identities, pending
requests, cumulative attempts and completed artifact hashes before changing the
controller. `launch.json` records the new controller and caffeinate binding.
The old launch and all historical records are preserved. Always resolve the
current entry and launch through the study's `active_runtime.json`.

Offline validation: two resource/ranking tests and five inherited manual-resume
tests passed; the composed CLI validator returned `VALID`. These tests send no
API requests and perform no training. A held original compute lease plus three
new leases blocks a fifth computation, and released leases can be reacquired.

The five-minute monitor uses the four-way configuration and verifies frozen
completion hashes. Its summaries and metrics are never selection input.
`caffeinate -i -w` prevents idle sleep while the controller lives; it does not
establish protection against sleep on closing the lid.
