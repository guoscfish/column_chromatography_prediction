"""Endpoint-aligned hierarchical and frozen-latent Center/Width transfer.

HIER and the latent block are solved in one weighted least-squares problem in
V1/V2 space; there is no fitted anchor followed by a residual model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .controlled_conditional_extension import center_width_inverse, center_width_transform, project_physical


def _safe_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.abs(values) < 1e-10, 1.0, values)


def endpoint_scales(truth: np.ndarray) -> np.ndarray:
    """Fit endpoint scales from the supplied authorized labels only."""
    truth = np.asarray(truth, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != 2 or len(truth) < 2:
        raise ValueError("a nonempty two-endpoint training target is required")
    return _safe_scale(truth.std(axis=0, ddof=0))


@dataclass(frozen=True)
class HierarchicalBasis:
    center_scale: float
    width_scale: float
    ea_mean: float
    center_interaction_mean: float
    center_interaction_scale: float
    width_interaction_mean: float
    width_interaction_scale: float

    @classmethod
    def fit(cls, source: np.ndarray, ea: np.ndarray) -> "HierarchicalBasis":
        center, width = center_width_transform(source)
        c, w = center[:, 0], width[:, 0]
        ea = np.asarray(ea, dtype=float).reshape(-1)
        cs = float(_safe_scale(np.asarray([c.std(ddof=0)]))[0])
        ws = float(_safe_scale(np.asarray([w.std(ddof=0)]))[0])
        em = float(ea.mean())
        ci, wi = (c / cs) * (ea - em), (w / ws) * (ea - em)
        return cls(cs, ws, em, float(ci.mean()), float(_safe_scale(np.asarray([ci.std(ddof=0)]))[0]),
                   float(wi.mean()), float(_safe_scale(np.asarray([wi.std(ddof=0)]))[0]))

    def transform(self, source: np.ndarray, ea: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center, width = center_width_transform(source)
        c, w = center[:, 0] / self.center_scale, width[:, 0] / self.width_scale
        e = np.asarray(ea, dtype=float).reshape(-1) - self.ea_mean
        cx = np.column_stack([c, np.ones(len(c)), (c * e - self.center_interaction_mean) / self.center_interaction_scale])
        wx = np.column_stack([w, np.ones(len(w)), (w * e - self.width_interaction_mean) / self.width_interaction_scale])
        return cx, wx


def _expanded(design: np.ndarray, labels: np.ndarray, columns: tuple[str, ...], latent: np.ndarray | None) -> np.ndarray:
    latent_dim = 0 if latent is None else latent.shape[1]
    result = np.zeros((len(design), 3 + 3 * len(columns) + latent_dim), dtype=float)
    result[:, :3] = design
    for index, name in enumerate(columns):
        selected = labels == name
        result[selected, 3 + 3 * index:6 + 3 * index] = design[selected]
    if latent_dim:
        result[:, 3 + 3 * len(columns):] = latent
    return result


def _joint_solve(center_x: np.ndarray, width_x: np.ndarray, truth: np.ndarray, ratios: np.ndarray,
                 scales: np.ndarray, center_penalty: np.ndarray, width_penalty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve MSE(V1)/s1^2 + MSE(V2)/s2^2 plus block penalties."""
    n, pc, pw = len(truth), center_x.shape[1], width_x.shape[1]
    c, w = center_x * ratios[:, None], width_x * ratios[:, None]
    v1 = np.column_stack([c, -0.5 * w]) / scales[0]
    v2 = np.column_stack([c, +0.5 * w]) / scales[1]
    design = np.vstack([v1, v2]) / np.sqrt(n)
    target = np.r_[truth[:, 0] / scales[0], truth[:, 1] / scales[1]] / np.sqrt(n)
    penalty = np.r_[center_penalty, width_penalty]
    prior = np.zeros(pc + pw, dtype=float)
    prior[0] = prior[pc] = 1.0
    augmented = np.vstack([design, np.diag(np.sqrt(penalty))])
    coefficients = np.linalg.lstsq(augmented, np.r_[target, np.sqrt(penalty) * prior], rcond=None)[0]
    return coefficients[:pc], coefficients[pc:]


@dataclass(frozen=True)
class HierarchicalCWV2Fit:
    columns: tuple[str, ...]
    mass_ratios: dict[str, float]
    basis: HierarchicalBasis
    center_coefficients: np.ndarray
    width_coefficients: np.ndarray
    center_lambda: float
    width_lambda: float
    scales: np.ndarray

    def predict_raw(self, source: np.ndarray, ea: np.ndarray, labels: Sequence[str]) -> np.ndarray:
        labels = np.asarray(labels, dtype=str).reshape(-1)
        cx, wx = self.basis.transform(source, ea)
        ce, we = _expanded(cx, labels, self.columns, None), _expanded(wx, labels, self.columns, None)
        ratios = np.asarray([self.mass_ratios[label] for label in labels], dtype=float)
        return center_width_inverse((ce @ self.center_coefficients) * ratios,
                                    (we @ self.width_coefficients) * ratios)

    def predict(self, source: np.ndarray, ea: np.ndarray, labels: Sequence[str]) -> np.ndarray:
        return project_physical(self.predict_raw(source, ea, labels))[0]


def fit_hierarchical_cw_v2(source: np.ndarray, truth: np.ndarray, ea: np.ndarray,
                           labels: Sequence[str], columns: Sequence[str], mass_ratios: dict[str, float],
                           center_lambda: float, width_lambda: float, *, scales: np.ndarray | None = None) -> HierarchicalCWV2Fit:
    source, truth = np.asarray(source, float), np.asarray(truth, float)
    labels, names = np.asarray(labels, dtype=str), tuple(str(x) for x in columns)
    if set(labels) - set(names) or set(mass_ratios) != set(names):
        raise ValueError("column labels and mass-ratio contract do not match")
    basis = HierarchicalBasis.fit(source, ea)
    cx, wx = basis.transform(source, ea)
    ce, we = _expanded(cx, labels, names, None), _expanded(wx, labels, names, None)
    ratios = np.asarray([mass_ratios[label] for label in labels], dtype=float)
    used_scales = endpoint_scales(truth) if scales is None else _safe_scale(np.asarray(scales, float))
    cp = np.r_[np.repeat(.1, 3), np.repeat(float(center_lambda), 3 * len(names))]
    wp = np.r_[np.repeat(.1, 3), np.repeat(float(width_lambda), 3 * len(names))]
    cc, wc = _joint_solve(ce, we, truth, ratios, used_scales, cp, wp)
    return HierarchicalCWV2Fit(names, {k: float(v) for k, v in mass_ratios.items()}, basis, cc, wc,
                               float(center_lambda), float(width_lambda), used_scales)


def _direct_features(source: np.ndarray, ea: np.ndarray, latent: np.ndarray) -> np.ndarray:
    center, width = center_width_transform(source)
    return np.column_stack([center[:, 0], width[:, 0], np.asarray(ea, float).reshape(-1), np.asarray(latent, float)])


@dataclass(frozen=True)
class DirectLatentFit:
    model_center: object
    model_width: object
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    latent_components: int | None
    mass_ratios: dict[str, float]

    def predict(self, source: np.ndarray, ea: np.ndarray, latent: np.ndarray, labels: Sequence[str]) -> np.ndarray:
        values = (_direct_features(source, ea, latent) - self.feature_mean) / self.feature_scale
        ratios = np.asarray([self.mass_ratios[str(label)] for label in labels], dtype=float)
        center = np.asarray(self.model_center.predict(values)).reshape(-1) * ratios
        width = np.asarray(self.model_width.predict(values)).reshape(-1) * ratios
        return project_physical(center_width_inverse(center, width))[0]


def fit_direct_latent(source: np.ndarray, truth: np.ndarray, ea: np.ndarray, latent: np.ndarray,
                      labels: Sequence[str], mass_ratios: dict[str, float], *, kind: str, strength: float) -> DirectLatentFit:
    labels = np.asarray(labels, dtype=str)
    ratios = np.asarray([mass_ratios[x] for x in labels], dtype=float)
    values = _direct_features(source, ea, latent)
    mean, scale = values.mean(0), _safe_scale(values.std(0, ddof=0))
    x = (values - mean) / scale
    yc, yw = center_width_transform(truth)
    if kind == "ridge":
        center_model, width_model = Ridge(alpha=float(strength)), Ridge(alpha=float(strength))
        components = None
    elif kind == "pls":
        center_model = PLSRegression(n_components=int(strength), scale=False, max_iter=1000)
        width_model = PLSRegression(n_components=int(strength), scale=False, max_iter=1000)
        components = int(strength)
    else:
        raise ValueError("kind must be ridge or pls")
    center_model.fit(x, yc[:, 0] / ratios)
    width_model.fit(x, yw[:, 0] / ratios)
    return DirectLatentFit(center_model, width_model, mean, scale, components,
                           {k: float(v) for k, v in mass_ratios.items()})


@dataclass(frozen=True)
class JointHierLatentFit:
    columns: tuple[str, ...]
    mass_ratios: dict[str, float]
    basis: HierarchicalBasis
    center_coefficients: np.ndarray
    width_coefficients: np.ndarray
    latent_mean: np.ndarray
    latent_scale: np.ndarray
    pca: PCA | None
    delta_lambda_center: float
    delta_lambda_width: float
    latent_lambda_center: float | None
    latent_lambda_width: float | None
    scales: np.ndarray

    def predict_raw(self, source: np.ndarray, ea: np.ndarray, latent: np.ndarray, labels: Sequence[str]) -> np.ndarray:
        labels = np.asarray(labels, dtype=str)
        cx, wx = self.basis.transform(source, ea)
        h = np.asarray(latent, float)
        if self.pca is not None:
            h = self.pca.transform(h)
        h = (h - self.latent_mean) / self.latent_scale
        if self.latent_lambda_center is None:
            h = None
        ce, we = _expanded(cx, labels, self.columns, h), _expanded(wx, labels, self.columns, h)
        ratios = np.asarray([self.mass_ratios[x] for x in labels], dtype=float)
        return center_width_inverse((ce @ self.center_coefficients) * ratios,
                                    (we @ self.width_coefficients) * ratios)

    def predict(self, source: np.ndarray, ea: np.ndarray, latent: np.ndarray, labels: Sequence[str]) -> np.ndarray:
        return project_physical(self.predict_raw(source, ea, latent, labels))[0]


def fit_joint_hier_latent(source: np.ndarray, truth: np.ndarray, ea: np.ndarray, latent: np.ndarray,
                          labels: Sequence[str], columns: Sequence[str], mass_ratios: dict[str, float], *,
                          delta_lambda_center: float, delta_lambda_width: float,
                          latent_lambda_center: float | None, latent_lambda_width: float | None,
                          pca_components: int | None = None, scales: np.ndarray | None = None) -> JointHierLatentFit:
    """Fit both bases simultaneously; ``None`` latent penalties force beta=0."""
    source, truth, latent = np.asarray(source, float), np.asarray(truth, float), np.asarray(latent, float)
    labels, names = np.asarray(labels, dtype=str), tuple(str(x) for x in columns)
    basis = HierarchicalBasis.fit(source, ea)
    cx, wx = basis.transform(source, ea)
    pca = None
    if (latent_lambda_center is None) != (latent_lambda_width is None):
        raise ValueError("both latent penalties must be None together")
    if latent_lambda_center is None:
        h = None
        latent_mean, latent_scale = np.zeros(latent.shape[1]), np.ones(latent.shape[1])
    else:
        if pca_components is not None:
            pca = PCA(n_components=min(int(pca_components), len(latent) - 1, latent.shape[1]), random_state=0).fit(latent)
            latent = pca.transform(latent)
        latent_mean, latent_scale = latent.mean(0), _safe_scale(latent.std(0, ddof=0))
        h = (latent - latent_mean) / latent_scale
    ce, we = _expanded(cx, labels, names, h), _expanded(wx, labels, names, h)
    ratios = np.asarray([mass_ratios[x] for x in labels], dtype=float)
    used_scales = endpoint_scales(truth) if scales is None else _safe_scale(np.asarray(scales, float))
    cp = np.r_[np.repeat(.1, 3), np.repeat(float(delta_lambda_center), 3 * len(names))]
    wp = np.r_[np.repeat(.1, 3), np.repeat(float(delta_lambda_width), 3 * len(names))]
    if h is not None:
        cp = np.r_[cp, np.repeat(float(latent_lambda_center), h.shape[1])]
        wp = np.r_[wp, np.repeat(float(latent_lambda_width), h.shape[1])]
    cc, wc = _joint_solve(ce, we, truth, ratios, used_scales, cp, wp)
    return JointHierLatentFit(names, {k: float(v) for k, v in mass_ratios.items()}, basis, cc, wc,
                              latent_mean, latent_scale, pca, float(delta_lambda_center),
                              float(delta_lambda_width), None if latent_lambda_center is None else float(latent_lambda_center),
                              None if latent_lambda_width is None else float(latent_lambda_width), used_scales)


__all__ = ["DirectLatentFit", "HierarchicalCWV2Fit", "JointHierLatentFit", "endpoint_scales",
           "fit_direct_latent", "fit_hierarchical_cw_v2", "fit_joint_hier_latent"]
