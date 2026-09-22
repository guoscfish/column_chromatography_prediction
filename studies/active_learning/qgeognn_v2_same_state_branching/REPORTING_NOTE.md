# Post-freeze reporting note

The sealed experiment completed, froze all predictions, revealed test truth once, and wrote its canonical CSV/JSON outputs successfully. Its first Markdown rendering used pandas `to_markdown`, but the optional `tabulate` dependency was unavailable and a temporary compatibility shim rendered DataFrame column names character-by-character.

`scripts/studies/finalize_qgeognn_v2_same_state_branching_report.py` repairs only the presentation from frozen CSV/JSON outputs. It does not open source data, checkpoints, predictions, runtime label stores, or test labels, and it does not modify sealed experiment code or numerical results.
