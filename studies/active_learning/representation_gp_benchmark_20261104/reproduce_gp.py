"""Same-input numeric replay; these checks are not independent benchmark fits."""
import hashlib
import json
import os
from pathlib import Path
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key]="1"

import numpy as np
from gp_models import FixedRepresentationGP,ProjectedMaternGP

HERE=Path(__file__).resolve().parent


def main():
    import torch
    torch.use_deterministic_algorithms(True)
    path=HERE/"cache/seed_157_L333.npz"
    with np.load(path,allow_pickle=False) as z:
        x=z["gnn"].copy();y=z["train_y"][:,0].copy()
    checks=[]
    for tag,model in [("gnn_matern52",FixedRepresentationGP(kernel="matern52",max_iter=40,seed=17)),
                      ("gnn_projected_init17",ProjectedMaternGP(projection_dim=64,max_iter=40,seed=17))]:
        original=json.loads((HERE/"results/seed_157_L333"/(tag+".json")).read_text())
        predictions=[r for r in original["predictions"] if r["endpoint"]=="V1"]
        model.fit(x[:333],y)
        mean,latent=model.predict(x[589:])
        observation=latent+model.noise
        prior=[np.array([r[k] for r in predictions]) for k in ("mean_ml","latent_variance_ml2","observation_variance_ml2")]
        actual=[mean,latent,observation]
        differences=[float(np.max(np.abs(a-b))) for a,b in zip(actual,prior)]
        valid=all(np.allclose(a,b,rtol=1e-8,atol=1e-10) for a,b in zip(actual,prior))
        checks.append({"model":tag,"seed":157,"budget":333,"endpoint":"V1","rows":len(mean),
            "same_input_tolerance_passed":valid,"mean_max_absolute_difference_ml":differences[0],
            "latent_variance_max_absolute_difference_ml2":differences[1],
            "observation_variance_max_absolute_difference_ml2":differences[2],
            "bitwise_equal":all(np.array_equal(a,b) for a,b in zip(actual,prior))})
        if not valid:raise ValueError("same-input GP numerical replay failed")
    result={"scope":"same-runtime engineer verification; exclude from the 20-state statistical matrix",
        "cache_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"dtype":"float64","checks":checks,
        "new_gnn_fits":0,"new_labels_requested":0,"extra_gp_replay_fits":len(checks)}
    (HERE/"reproduction_check.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
