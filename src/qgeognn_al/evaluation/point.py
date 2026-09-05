"""Six-output point metrics with explicit source-scale normalization."""
import math
import numpy as np


def point_metrics(truth, prediction, scales):
    result = {}
    for i, target in enumerate(("V1", "V2")):
        error = truth[:, i] - prediction[:, 3*i+1]
        total = np.square(truth[:, i] - truth[:, i].mean()).sum()
        result.update({f"{target}_r2": float(1 - np.square(error).sum()/total) if total else float("nan"),
                       f"{target}_rmse": float(np.sqrt(np.square(error).mean())), f"{target}_mae": float(np.abs(error).mean())})
    result["combined_normalized_rmse"] = float(math.sqrt(.5*sum((result[f"{t}_rmse"]/scales[t])**2 for t in ("V1", "V2"))))
    result["all_outputs_finite"] = bool(np.isfinite(prediction).all())
    return result
