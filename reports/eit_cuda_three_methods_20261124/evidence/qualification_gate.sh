#!/usr/bin/env bash
set -euo pipefail
root=/1/column_chromatography_prediction
suite="$root/formal_cuda/row4g_b32_20261124"
python="$root/environment/venv/bin/python"
cd "$root/worktree"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLBACKEND=Agg GRADIO_ANALYTICS_ENABLED=False CUBLAS_WORKSPACE_CONFIG=:4096:8
manage() { "$python" -m scripts.studies.queue_qgeognn_cuda --root "$suite" "$@"; }
trap 'manage release random || true' EXIT
proof="$suite/qualification"
while true; do
    check=$(python3 - "$suite/queue.json" <<'PY'
import json,sys,time
q=json.load(open(sys.argv[1]))
qual=[t for t in q["tasks"] if t["qualification"]]
if q["cutoff"] and time.time() >= q["cutoff"] - 45:
    print("cutoff")
elif all(t["state"] in ("succeeded", "failed") for t in qual):
    states={t["name"]: t["state"] for t in qual}
    gpu=next(t["attempts"][-1]["gpu"] for t in qual if t["name"]=="qual-hybrid-157-b")
    print(states["qual-cw_lcmd-157-a"], states["qual-hybrid-157-a"], states["qual-hybrid-157-b"], gpu)
else:
    print("wait")
PY
)
    if [[ "$check" == cutoff ]]; then exit 0; fi
    if [[ "$check" != wait ]]; then break; fi
    sleep 15
done
read -r cw hybrid_a hybrid_b gpu <<< "$check"

if [[ "$cw" == succeeded ]]; then
    if CUDA_VISIBLE_DEVICES= "$python" - "$proof" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from src.qgeognn_al.active_learning_v2 import sequential_runner as seq
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner import select_current_cw_lcmd
from src.qgeognn_al.training.predictor import atomic_json
base=Path(sys.argv[1])
run=base/"qual-cw_lcmd-157-a"
directory=run/"rounds/cw_lcmd/round_00"
source=seq._json(directory/"contract.json")
seq._verify_round_artifacts(directory, source)
next_dir=run/"rounds/cw_lcmd/round_01"
seq._verify_round_artifacts(next_dir, seq._json(next_dir/"contract.json"))
items=directory/"acquisition_artifacts"
audit=seq._json(items/"gradient_audit.json")
gradient=audit["gradient_audit"]
selected=pd.read_csv(items/"selected_next_batch.csv")
labeled=np.load(directory/"labeled_indices.npy", allow_pickle=False)
pool=np.load(directory/"unlabeled_indices.npy", allow_pickle=False)
with np.load(items/"current_center_width_gradient_features.npz", allow_pickle=False) as arrays:
    np.testing.assert_array_equal(arrays["canonical_indices"], np.r_[labeled,pool])
    values=arrays["features"]
    if values.shape != (3330,512) or not np.isfinite(values).all():
        raise RuntimeError("CW features have unexpected shape or nonfinite values")
    replay=select_current_cw_lcmd(values,len(labeled))
positions=selected.previous_U_t_position.to_numpy()
np.testing.assert_array_equal(positions,replay["selected_pool_positions"])
np.testing.assert_array_equal(selected.canonical_index.to_numpy(),pool[positions])
initial_ids=np.load(directory/"labeled_ids.npy", allow_pickle=False).astype(str).tolist()
round1_ids=np.load(next_dir/"labeled_ids.npy", allow_pickle=False).astype(str).tolist()
selected_ids=selected.sample_id.astype(str).tolist()
if (len(set(selected_ids)) != 32 or round1_ids != initial_ids+selected_ids
    or not gradient["all_finite"] or source["input"]["test_truth_access_count"] != 0):
    raise RuntimeError("CW validation or selection contract failed")
timing=seq._json(directory/"acquisition_artifacts/contract.json")["metadata"]
atomic_json(base/"cw_lcmd_qualification.json", {
    "seed":157, "eligible_selected":len(selected_ids), "feature_rows":len(values),
    "feature_dimension":values.shape[1], "nonzero_rows":gradient["rows"]-gradient["zero_norm_rows"],
    "matches_original_ordered_selection":True, "L333_to_L365_sealed":True,
    "gradient_extraction_seconds":timing["gradient_extraction_seconds"],
    "selector_seconds":timing["selector_seconds"],
    "fit_seconds":seq._json(directory/"model/fit_audit.json")["training_seconds"]})
print("CW_QUALIFICATION_PASSED", flush=True)
PY
    then
        printf '%s\n' "CW proof complete"
    else
        cw=failed
    fi
fi
if [[ "$hybrid_a" == succeeded && "$hybrid_b" == succeeded ]]; then
    if CUDA_VISIBLE_DEVICES="$gpu" "$python" -m scripts.studies.verify_qgeognn_cuda_hybrid \
        --first "$proof/qual-hybrid-157-a" --second "$proof/qual-hybrid-157-b" \
        --output "$proof/hybrid_qualification.json" &&
        python3 - "$proof/hybrid_qualification.json" <<'PY'
import json,sys
q=json.load(open(sys.argv[1]))
checks=q["independent_training"]
if not (checks["selected_order_equal"] and all(
    member["matched_train_inputs"] and member["checkpoint_state_hash_equal"]
    for member in checks["members"]
)):
    raise RuntimeError("independent deterministic Hybrid reruns diverged")
PY
    then
        manage admit hybrid "$proof/hybrid_qualification.json"
    fi
fi
if [[ "$cw" == succeeded ]]; then
    manage admit cw_lcmd "$proof/cw_lcmd_qualification.json"
fi
