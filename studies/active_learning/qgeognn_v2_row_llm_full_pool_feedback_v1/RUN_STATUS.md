# Run status: full-pool feedback LLM study

The protocol is frozen and validates against code and source hashes. The
historical pure-CW trajectory for both seeds was audited and reused. Both
full-pool random trajectories completed all six batches to L525: 12 new
fits, 384 selected labels across the two seeds, and 12 freeze-before-reveal
receipts. Total new random-arm training time recorded by fit audits is
2,020.3 seconds (seed 157: 975.7; seed 6101: 1,044.6). These are summed
training times, not wall time. Every batch has 32 unique, legal U_t IDs;
the 16 random supplements came from the entire remaining pool.

Both independent LLM trajectories are **awaiting selector configuration at
round zero**. Their per-seed packets and full-pool query catalogs were
constructed without revealing pending CW or candidate responses. Each packet
has 333 L_t IDs with observed response access, 16 pending CW cards, 128
numeric reference IDs, and 2,981 searchable legal supplemental candidates.
There have been zero formal LLM model calls and zero new LLM-arm label reveals.
The current process has no `OPENAI_API_KEY` for the frozen `gpt-5.4`
Responses API selector. No development-chat selections or random fallback
were substituted.

The new study has **not** produced learning curves, label-AULC, L525
comparisons, or chemistry-feedback conclusions. `report` correctly refuses
to open test truth while either LLM trajectory is incomplete. Therefore the
questions “better than random?” and “better than CW?” remain unanswered for
this version. The prior 128-candidate screen remains a separate completed
result with its original decision intact.

To finish, make the API key available to the experiment process without
putting it in Git or the task chat, run the two `cw16_llm16_full_pool`
commands in [README.md](README.md), then run `report`. Each command resumes
from its own frozen seed/method state; the model and query budgets remain
fixed in [PROTOCOL.md](PROTOCOL.md). Do not inspect new test metrics or
revise selection settings between rounds.

Engineering verification: 18 relevant tests passed; real-data boundary
audit for both seeds passed; seed 157 random trajectory was rerun and reused
its completed artifacts. See `preflight_boundary_audit.json`,
`random_trajectory_audit.json`, and the per-round selection directories.
