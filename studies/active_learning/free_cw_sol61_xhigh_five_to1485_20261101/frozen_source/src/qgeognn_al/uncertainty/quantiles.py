"""Descriptive quantile diagnostics; these never train or alter the predictor."""
import numpy as np
from scipy.stats import spearmanr

def quantile_metrics(frame, target):
    truth = frame[f"{target}_true"].to_numpy(float)
    q10 = frame[f"{target}_q10"].to_numpy(float)
    q50 = frame[f"{target}_q50"].to_numpy(float)
    q90 = frame[f"{target}_q90"].to_numpy(float)
    error = np.abs(truth - q50)
    width = q90 - q10
    uncertainty = np.maximum(width, 0)
    top_count = max(1, int(np.ceil(.2 * len(error))))
    top = error[np.argsort(-uncertainty, kind="stable")[:top_count]]
    correlation = spearmanr(uncertainty, error).statistic
    return {
        "rows": len(frame),
        "top_uncertainty_rows": top_count,
        "negative_width_rate": float(np.mean(width < 0)),
        "crossing_rate": float(np.mean((q10 > q50) | (q50 > q90))),
        "q10_pinball_loss": float(np.mean(np.maximum(.1 * (truth-q10), -.9 * (truth-q10)))),
        "q50_rmse": float(np.sqrt(np.mean(np.square(truth-q50)))),
        "q50_mae": float(error.mean()),
        "q90_pinball_loss": float(np.mean(np.maximum(.9 * (truth-q90), -.1 * (truth-q90)))),
        "empirical_coverage": float(np.mean((truth >= q10) & (truth <= q90))),
        "interval_width": float(width.mean()),
        "nonnegative_interval_width": float(uncertainty.mean()),
        "uncertainty_error_spearman": float(correlation) if np.isfinite(correlation) else None,
        "top_20pct_uncertainty_error_enrichment": float(top.mean()/error.mean()) if len(top) and error.mean() else None,
    }
