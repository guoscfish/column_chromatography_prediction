# Post-freeze reporting correction

Both trajectories, all 20 new fits, all 22 checkpoint/test-X prediction points,
and `global_pre_test_freeze.json` were complete before test truth was opened.
The post-freeze evaluator then wrote every result CSV, both figures, and the
completed `decision.json`. It failed only while rendering Markdown tables for
`FINAL_REPORT.md`, because this environment does not install pandas' optional
`tabulate` dependency.

The correction replaces `DataFrame.to_markdown` with a local deterministic
Markdown formatter and adds a recovery path that reads the already-written
post-freeze result tables. Recovery does not construct `RestrictedLabelStore`,
does not read test truth again, and does not change any metric, decision,
selection, checkpoint, prediction, fit audit, or global freeze artifact.

- Original sealed reporting source SHA-256:
  `ea80cfac1383105b2f1a40d846cb3081ba8e38afd84a9e8fe6a199960879f655`
- Original seal SHA-256:
  `6a4b320810e61e1fa8224d7d1d49bc9746ab50b7f4b32036e824e0ac055db466`
- Failure boundary: `FINAL_REPORT.md` table formatting after results and decision
- Scientific effect: none; presentation-only recovery

The final artifact manifest binds this note, the corrected reporter, the final
report, all aggregate result tables, figures, protocol, decision, and freeze.
