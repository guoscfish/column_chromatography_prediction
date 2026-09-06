"""Label-role-independent statistics for the scaling-failure audit."""
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import GroupKFold

from .calibration import fit_affine, fit_scale_only

DESCRIPTORS = ("MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotatableBonds", "RingCount")
CONDITIONS = ("EA_fraction", "solvent_DCM", "loading_ul", "amount_density_ul", "loading_volume_ul")
BIN_NAMES = ("low", "medium", "high", "extreme_tail")


def match_key(row, *, relaxed=False):
    a, b = (Fraction(str(value)) for value in str(row["PE/EA"]).split("/"))
    if a+b <= 0:
        raise ValueError("invalid eluent ratio")
    fields = [str(row["canonical_smiles"]), str(b/(a+b)), str(row["loading solvent"])]
    fields += [str(Fraction(str(row[name]))) for name in
               ("Density g/ml", "V/ul", "Volume of loading solvent/ul")]
    if not relaxed:
        fields.append(str(Fraction(str(row["Flow mL/min"]))))
    return tuple(fields)


def source_bins(source, train_source):
    cuts = np.quantile(train_source, [1/3, 2/3, .9])
    return np.array(BIN_NAMES)[np.searchsorted(cuts, source, side="right")], cuts


def safe_ratio(truth, source, floor=.5):
    return np.divide(truth, source, out=np.full_like(np.asarray(truth, float), np.nan),
                     where=np.asarray(source) >= floor)


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5 or np.std(x[valid]) < 1e-10 or np.std(y[valid]) < 1e-10:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def partial_rank(x, y, source, groups=None):
    x, y, source = (np.asarray(z, float) for z in (x, y, source))
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(source)
    if valid.sum() < 8:
        return float("nan")
    x, y, source = (rankdata(z[valid]) for z in (x, y, source))
    design = np.column_stack([np.ones(len(x)), source])
    if groups is not None:
        dummy = pd.get_dummies(np.asarray(groups)[valid], dtype=float).to_numpy()
        design = np.column_stack([design, dummy[:, 1:]])
    if len(x)-np.linalg.matrix_rank(design) < 4:
        return float("nan")
    rx = x-design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y-design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(rx) < 1e-8 or np.std(ry) < 1e-8:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def out_of_fold_calibration(source, truth, groups):
    """Every diagnostic residual comes from a fit excluding its entire compound."""
    source, truth, groups = np.asarray(source), np.asarray(truth), np.asarray(groups)
    count = min(5, len(np.unique(groups)))
    if count < 2:
        raise ValueError("compound cross-fitting needs at least two groups")
    scale, affine = np.empty_like(truth, dtype=float), np.empty_like(truth, dtype=float)
    fold_ids = np.empty(len(truth), dtype=int)
    for fold, (train, valid) in enumerate(GroupKFold(count).split(source, groups=groups)):
        if set(groups[train]) & set(groups[valid]):
            raise RuntimeError("cross-fitting group overlap")
        scale[valid] = fit_scale_only(truth[train], source[train], source[valid]).prediction
        affine[valid] = fit_affine(truth[train], source[train], source[valid]).prediction
        fold_ids[valid] = fold
    return scale, affine, fold_ids


def standardize_condition_contrast(frame, feature):
    """Source-bin standardized high-minus-low ratio, observed common support only."""
    work = frame.loc[frame.ratio.notna()].copy()
    if feature == "solvent_DCM":
        work["level"] = work[feature]
    else:
        work["level"] = (work[feature] > work[feature].median()).astype(int)
    differences = []
    for _, group in work.groupby("source_bin"):
        low, high = group.loc[group.level.eq(0)], group.loc[group.level.eq(1)]
        if min(len(low), len(high)) >= 3 and min(low.canonical_smiles.nunique(), high.canonical_smiles.nunique()) >= 2:
            differences.append(high.ratio.mean()-low.ratio.mean())
    denominator = float(work.ratio.median()) if len(work) else np.nan
    return {"common_source_bins": len(differences),
            "relative_standardized_contrast": float(np.mean(differences)/denominator)
            if len(differences) >= 2 and abs(denominator) > 1e-8 else np.nan}


def neighborhood_consistency(molecules, seed):
    data = molecules.dropna(subset=["ratio_mean", *DESCRIPTORS])
    if len(data) < 8:
        return {"neighbor_rho": np.nan, "permutation_p": np.nan, "molecules": len(data)}
    values = data[list(DESCRIPTORS)].to_numpy(float)
    std = values.std(0)
    std[std < 1e-8] = 1
    values = (values-values.mean(0))/std
    distances = np.square(values[:, None]-values[None, :]).sum(2)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :3]
    residual = data.ratio_mean.to_numpy()
    rho = correlation(residual, residual[neighbors].mean(1))
    if not np.isfinite(rho):
        return {"neighbor_rho": rho, "permutation_p": np.nan, "molecules": len(data)}
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(199):
        shuffled = rng.permutation(residual)
        null.append(correlation(shuffled, shuffled[neighbors].mean(1)))
    scale_rho = correlation(data.residual_mean, data.residual_mean.to_numpy()[neighbors].mean(1)) if "residual_mean" in data else np.nan
    return {"neighbor_rho": rho, "neighbor_scale_residual_rho": scale_rho,
            "permutation_p": float((1+np.sum(np.asarray(null) >= rho))/200), "molecules": len(data)}
