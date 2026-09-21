# Authorized staged execution, 2026-09-21

The subsequent user instruction explicitly changes the evaluation barrier in
the sealed PROTOCOL.md: complete the development subset (seeds 157 and 6101),
freeze both full 653..1005 trajectories, then read only these two seeds' test
truth to decide whether to run seeds 887, 2357 and 12203. The original seal
and its protocol remain intact so this departure is visible. This amendment
does not change the acquisition, predictor, split, target scale, budget grid,
source artifact, or original code hashes. The development analysis script is
separate from the sealed acquisition runner; it verifies both complete freezes
before test access and records its own SHA256 in its output.

The two chosen seeds are the exact development subset of the prior MaxDet and
Fusion studies. Their already published test outcomes, and any new interim
test outcome, make the go/no-go decision exploratory. If this subset suggests
a continuation, the other three row splits are still overlapping and exposed
through previous studies; they are not independent confirmation.

Decision rule: inspect both paired seed directions for late AULC, full AULC,
endpoint combined NRMSE and both endpoint RMSEs. Require a directionally
consistent improvement in late AULC over LCMD and no material full-AULC or
endpoint deterioration, with evidence of endpoint progress toward pure IVR or
a stronger sustained N90 crossing. Differences around 0.001 alone are weak.
If either seed substantially deteriorates and the mean late AULC/endpoint does
not improve, stop the full cohort and report the negative result. Borderline
or mixed evidence remains uncertain, and is not a success claim.

`--development-report` reveals test truth only for the two verified, fully
frozen development trajectories. It writes under `development/` and cannot
modify any baseline output. The original five-seed `--report` remains behind
the all-five pre-test freeze. The staged report precedes any continuation on
the other three seeds. No adaptive policy is trained in this phase.
