#!/bin/sh
set -eu
cd /Users/fish/Documents/GitHub/column_chromatography_prediction
exec caffeinate -i env KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python -u scripts/studies/run_qgeognn_v2_row_llm_scientist.py run --revision astra_high_full_pool_20261001 --seed 157 --method free_llm32_scientist_v2 --status-interval 30
