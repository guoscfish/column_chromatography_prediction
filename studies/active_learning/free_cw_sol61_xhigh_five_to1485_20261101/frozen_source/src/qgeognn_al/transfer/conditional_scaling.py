"""A fixed three-coefficient conditional slope and its additive control."""
from dataclasses import dataclass

import numpy as np


@dataclass
class ConditionalFit:
    coefficients: np.ndarray
    ea_mean: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    source_scales: np.ndarray
    mass_ratio: float
    interaction: bool
    penalty: float

    def predict(self, source, ea):
        u = np.asarray(source, float)/self.source_scales
        e = np.asarray(ea, float)-self.ea_mean
        raw = u*e[:, None] if self.interaction else np.tile(e[:, None], (1, 2))
        third = (raw-self.feature_mean)/self.feature_std
        result = np.column_stack([np.column_stack([u[:, j], np.ones(len(u)), third[:, j]]) @ self.coefficients[j]
                                  for j in range(2)])
        return result*self.mass_ratio*self.source_scales

    def audit(self):
        return {"coefficients": self.coefficients.tolist(), "ea_mean": self.ea_mean,
                "feature_mean": self.feature_mean.tolist(), "feature_std": self.feature_std.tolist(),
                "source_scales": self.source_scales.tolist(), "mass_ratio": self.mass_ratio,
                "interaction": self.interaction, "penalty": self.penalty}


def fit_conditional(source, truth, ea, source_scales, mass_ratio, penalty, *, interaction=True):
    source, truth = np.asarray(source, float), np.asarray(truth, float)
    ea, scales = np.asarray(ea, float), np.asarray(source_scales, float)
    if source.shape != truth.shape or source.ndim != 2 or source.shape[1] != 2 or ea.shape != (len(source),):
        raise ValueError("aligned source/truth/condition required")
    if len(source)<4 or not all(np.isfinite(z).all() for z in [source, truth, ea, scales]):
        raise ValueError("non-finite or insufficient training data")
    if scales.shape != (2,) or np.any(scales<=0) or mass_ratio<=0 or penalty<0:
        raise ValueError("invalid normalization or penalty")
    u = source/scales
    e = ea-ea.mean()
    raw = u*e[:, None] if interaction else np.tile(e[:, None], (1, 2))
    mean, std = raw.mean(0), raw.std(0)
    std[std<1e-8] = 1.
    third = (raw-mean)/std
    coefficients = []
    for j in range(2):
        design = np.column_stack([u[:, j], np.ones(len(u)), third[:, j]])
        augmented = np.vstack([design/np.sqrt(len(u)), np.sqrt(penalty)*np.eye(3)])
        label = np.r_[truth[:, j]/(mass_ratio*scales[j]*np.sqrt(len(u))), np.sqrt(penalty)*np.array([1., 0., 0.])]
        coefficients.append(np.linalg.lstsq(augmented, label, rcond=None)[0])
    return ConditionalFit(np.asarray(coefficients), float(ea.mean()), mean, std, scales, float(mass_ratio), interaction, float(penalty))
