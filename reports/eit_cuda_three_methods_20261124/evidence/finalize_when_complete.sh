#!/usr/bin/env bash
set -euo pipefail
root=/1/column_chromatography_prediction
suite="$root/formal_cuda/row4g_b32_20261124"
python="$root/environment/venv/bin/python"
cd "$root/worktree"
exec 9>"$suite/analysis-finalizer.lock"
flock -n 9 || exit 0
if [[ -s "$suite/analysis_final/analysis_status.json" ]]; then exit 0; fi
printf '%s %s\n' "$(date -Is)" 'Waiting for all fifteen formal trajectories'
until python3 - "$suite/queue.json" <<'PY'
import json,sys
tasks=[t for t in json.load(open(sys.argv[1]))["tasks"] if not t["qualification"]]
raise SystemExit(not (len(tasks)==15 and all(t["state"]=="succeeded" for t in tasks)))
PY
do
    sleep 60
done
out="$suite/analysis_final"
if [[ -e "$out" ]]; then out="$suite/analysis_final_retry_$(date +%Y%m%dT%H%M%S)"; fi
export CUDA_VISIBLE_DEVICES=
export GRADIO_ANALYTICS_ENABLED=False MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec "$python" -u -m scripts.analyze_formal_qgeognn_cuda --runs "$suite" --output "$out" --test
