"""Two bounded calibration diagnostics; the qualified predictor is unchanged."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class MonotoneCalibration:
    centers: np.ndarray
    scales: np.ndarray
    knots: np.ndarray
    coefficients: np.ndarray
    output_scales: np.ndarray
    strength: float

    @staticmethod
    def design(x, knots):
        low, high = knots
        return np.column_stack([
            np.ones(len(x)), np.minimum(x, low),
            np.clip(x - low, 0., high - low), np.maximum(x - high, 0.),
        ])

    def predict(self, source):
        values = (np.asarray(source, dtype=float) - self.centers) / self.scales
        return np.column_stack([
            self.design(values[:, j], self.knots[j]) @ self.coefficients[j]
            * self.output_scales[j] for j in range(2)
        ])

    def audit(self):
        return {
            "strength": self.strength, "centers": self.centers.tolist(),
            "input_scales": self.scales.tolist(), "knots_standardized": self.knots.tolist(),
            "knots_source_ml": (self.knots * self.scales[:, None] + self.centers[:, None]).tolist(),
            "coefficients": self.coefficients.tolist(),
            "slopes_ml_per_source_ml": (self.coefficients[:, 1:] * self.output_scales[:, None]
                                        / self.scales[:, None]).tolist(),
            "output_scales": self.output_scales.tolist(),
        }


def fit_monotone(source, truth, output_scales, strength):
    """Train-only knots/standardization and nonnegative slopes, with linear tails.

    Objective: mean squared normalized error + strength * sum(diff(slopes)**2).
    Validation and test arrays are deliberately absent from this interface.
    """
    source, truth = np.asarray(source, float), np.asarray(truth, float)
    output_scales = np.asarray(output_scales, float)
    if source.shape != truth.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("aligned (n,2) arrays required")
    if len(source) < 4 or not np.isfinite(source).all() or not np.isfinite(truth).all():
        raise ValueError("insufficient or non-finite training data")
    if strength < 0 or output_scales.shape != (2,) or np.any(output_scales <= 0):
        raise ValueError("invalid regularization or normalization")
    centers, scales = source.mean(0), source.std(0)
    if np.any(scales < 1e-8):
        raise ValueError("degenerate source range")
    values = (source - centers) / scales
    knots = np.quantile(values, [1/3, 2/3], axis=0).T
    if np.any(np.diff(knots, axis=1) < 1e-8):
        raise ValueError("degenerate training knots")
    difference = np.array([[0., 1., -1., 0.], [0., 0., 1., -1.]])
    coefficients = []
    for j in range(2):
        design = MonotoneCalibration.design(values[:, j], knots[j])
        augmented = np.vstack([design / np.sqrt(len(source)), np.sqrt(strength) * difference])
        labels = np.r_[truth[:, j] / output_scales[j] / np.sqrt(len(source)), [0., 0.]]
        fit = lsq_linear(augmented, labels, bounds=([-np.inf, 0., 0., 0.], np.inf),
                         tol=1e-12, max_iter=1000)
        if not fit.success:
            raise RuntimeError(f"monotone solver failed: {fit.message}")
        coefficients.append(fit.x)
    return MonotoneCalibration(centers, scales, knots, np.asarray(coefficients), output_scales, float(strength))


@dataclass
class ColumnCalibration:
    coefficients: np.ndarray
    mass_ratios: np.ndarray
    source_scales: np.ndarray

    def predict(self, source, column_index):
        source = np.asarray(source, float)
        coefficient = self.coefficients[column_index]
        return (source / self.source_scales * coefficient[:, 0] + coefficient[:, 1]) * (
            self.mass_ratios[column_index] * self.source_scales)


def fit_column_calibration(sources, truths, mass_ratios, source_scales, strength, *, shared=True):
    """Separate affine coefficients coupled only through a quadratic penalty.

    Shared penalty = sum_c ||theta_c - mean(theta)||^2. The local control instead
    shrinks each theta toward the fixed mass-ratio identity (1,0). At lambda=0
    the design is block diagonal and exactly equals three independent affines.
    Each column contributes mean squared error, after fixed mass/source scaling.
    """
    count = len(sources)
    source_scales, mass_ratios = np.asarray(source_scales, float), np.asarray(mass_ratios, float)
    if count != 3 or len(truths) != count or mass_ratios.shape != (count,):
        raise ValueError("exactly three observed columns required")
    if strength < 0 or np.any(mass_ratios <= 0) or source_scales.shape != (2,) or np.any(source_scales <= 0):
        raise ValueError("invalid penalty or fixed scales")
    result = np.empty((count, 2, 2))
    coupling = np.kron(np.eye(count) - np.ones((count, count)) / count, np.eye(2))
    for j in range(2):
        rows, labels = [], []
        for c, (source, truth) in enumerate(zip(sources, truths)):
            source, truth = np.asarray(source, float), np.asarray(truth, float)
            if source.shape != truth.shape or source.ndim != 2 or source.shape[1] != 2:
                raise ValueError("aligned (n,2) donor arrays required")
            if len(source) < 2 or not np.isfinite(source).all() or not np.isfinite(truth).all():
                raise ValueError("insufficient or non-finite donor data")
            design = np.column_stack([source[:, j] / source_scales[j], np.ones(len(source))])
            if np.linalg.matrix_rank(design) != 2:
                raise ValueError("rank deficient donor affine design")
            block = np.zeros((len(source), 2 * count))
            block[:, c*2:c*2+2] = design / np.sqrt(len(source))
            rows.append(block)
            labels.append(truth[:, j] / (mass_ratios[c] * source_scales[j] * np.sqrt(len(source))))
        penalty = coupling if shared else np.eye(2 * count)
        prior = np.zeros(2 * count) if shared else np.tile([1., 0.], count)
        design = np.vstack([*rows, np.sqrt(strength) * penalty])
        label = np.r_[np.concatenate(labels), np.sqrt(strength) * prior]
        result[:, j, :] = np.linalg.lstsq(design, label, rcond=None)[0].reshape(count, 2)
    return ColumnCalibration(result, mass_ratios, source_scales)


def donor_training_rows(context, focal_column, protocol):
    """Keep original focal roles; purge held-out focal compounds from donors."""
    focal = context.loc[context.column.eq(focal_column)]
    protected = set(focal.loc[focal.role.isin(["validation", "test"]), "canonical_smiles"])
    train = context.loc[context.role.eq("gradient_train")].copy()
    if protocol == "compound":
        train = train.loc[~train.canonical_smiles.isin(protected)]
        if set(train.canonical_smiles) & protected:
            raise RuntimeError("cross-column compound leakage")
    elif protocol != "row":
        raise ValueError(protocol)
    return train


def validation_score(truth, prediction, scales):
    """RMS of source-normalized per-output RMSE, matching checkpoint selection."""
    return float(np.sqrt(np.mean(np.square((np.asarray(truth) - prediction) / scales))))
