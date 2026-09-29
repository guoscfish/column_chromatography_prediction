# Study entrypoints

Current development: `run_qgeognn_v3_2_row_llm_scientist.py` (Scientist V3.2, seed157, Free-LLM32, one round only).

`run_qgeognn_v3_row_llm_scientist.py` and `run_qgeognn_v3_loop.py` reproduce the frozen V3.1 pilot. The loop is not a V3.2 entrypoint.

Legacy reproduction: `run_qgeognn_v2_row_llm_scientist.py`, `run_qgeognn_v2_row_llm_dialog.py`, `run_qgeognn_v2_row_llm_full_pool.py`, and `run_qgeognn_v2_row_llm16_screen.py`. Historical numerical studies retain their named runners. V2 source modules remain common dependencies; do not delete them based on runner age.

New studies retain README, protocol, manifests, compact reports and CSV/JSON summaries in Git. Large catalogs, requests, turns, audit streams and checkpoints stay in ignored local study storage with hashes in the manifest. Existing tracked pilot artifacts are retained for audit.
