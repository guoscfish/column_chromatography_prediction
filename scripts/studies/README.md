# Study entrypoints

Current development: `run_qgeognn_v3_2_row_llm_scientist.py` (Scientist V3.2, seed157, Free-LLM32, six rounds to L525).

`resume_qgeognn_v3_2_interrupted.py` is an explicit, one-use process-interruption recovery tool. Its default is an offline exact replay with zero model calls. `--execute` requires every saved audit event, artifact and request hash to match, charges the unknown in-flight transport attempt, and preserves all scientific/repair budgets. It cannot resume a terminal validation or transport failure. The recorded V3.2 run now has a terminal repair-budget failure at L365; this tool cannot extend that trajectory.

`run_qgeognn_v3_row_llm_scientist.py` and `run_qgeognn_v3_loop.py` reproduce the frozen V3.1 pilot. The V3.1 loop is not a V3.2 entrypoint; use `run_qgeognn_v3_2_loop.py` for the newly authorized six-round V3.2 run.

Legacy reproduction: `run_qgeognn_v2_row_llm_scientist.py`, `run_qgeognn_v2_row_llm_dialog.py`, `run_qgeognn_v2_row_llm_full_pool.py`, and `run_qgeognn_v2_row_llm16_screen.py`. Historical numerical studies retain their named runners. V2 source modules remain common dependencies; do not delete them based on runner age.

New studies retain README, protocol, manifests, compact reports and CSV/JSON summaries in Git. Large catalogs, requests, turns, audit streams and checkpoints stay in ignored local study storage with hashes in the manifest. Existing tracked pilot artifacts are retained for audit.
