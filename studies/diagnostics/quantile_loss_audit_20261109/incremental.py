"""Fixed group-fold ridge diagnostics, never an AL policy or hyperparameter search."""
from __future__ import annotations
import hashlib
import numpy as np
from .quantiles import ranking_metrics


def fixed_group_folds(groups, count=5):
    if type(count) is not int or count < 2:
        raise ValueError('at least two folds required')
    result=np.asarray([int.from_bytes(hashlib.sha256(str(g).encode()).digest()[:8],'big') % count for g in groups],dtype=int)
    if len(set(result))<2:
        raise ValueError('at least two represented group folds required')
    return result


def oof_ridge(features, outcome, groups, *, fold_count=5, ridge=1.):
    x,y=np.asarray(features,float),np.asarray(outcome,float)
    if x.ndim !=2 or y.shape!=(len(x),) or len(groups)!=len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or ridge<=0:
        raise ValueError('finite aligned diagnostic data and positive fixed ridge required')
    fold=fixed_group_folds(groups,fold_count);out=np.empty_like(y)
    for value in sorted(set(fold)):
        test=fold==value;train=~test
        if train.sum()<2:
            raise ValueError('insufficient held-out training rows')
        mean=x[train].mean(0);scale=x[train].std(0);scale[scale<1e-12]=1.
        z=(x[train]-mean)/scale;center=y[train].mean()
        weights=np.linalg.solve(z.T@z+ridge*np.eye(x.shape[1]), z.T@(y[train]-center))
        out[test]=center+((x[test]-mean)/scale)@weights
    return out,fold


def incremental_width_diagnostic(base_features, width_features, error, groups):
    base,width=np.asarray(base_features,float),np.asarray(width_features,float)
    if width.ndim !=2 or len(width)!=len(base):raise ValueError('width rows must align')
    left,fold=oof_ridge(base,error,groups);right,other=oof_ridge(np.column_stack([base,width]),error,groups)
    assert np.array_equal(fold,other)
    y=np.asarray(error,float)
    lrmse=float(np.sqrt(np.square(y-left).mean()));rrmse=float(np.sqrt(np.square(y-right).mean()))
    result={'rows':len(y),'groups':len(set(groups)),'folds_represented':len(set(fold)),
            'baseline_oof_rmse':lrmse,'with_width_oof_rmse':rrmse,'rmse_delta_with_minus_base':rrmse-lrmse,
            'baseline_oof_mae':float(np.abs(y-left).mean()),'with_width_oof_mae':float(np.abs(y-right).mean()),
            'ridge':1.,'fold_definition':'SHA256(canonical_smiles) mod 5; all standardization fit on fold train only',
            'interpretation':'retrospective group-OOF diagnostic; no causal or AULC claim'}
    for prefix,score in [('baseline',left),('with_width',right)]:
        result.update({prefix+'_'+key:value for key,value in ranking_metrics(y,score).items()})
    return result


def seed_bootstrap_interval(values, *, repetitions=2000, seed=20261109):
    v=np.asarray(values,float)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():raise ValueError('finite seed-level values required')
    rng=np.random.default_rng(seed);means=v[rng.integers(0,len(v),size=(repetitions,len(v)))].mean(1)
    return {'independent_seed_count':len(v),'mean':float(v.mean()),'lower95':float(np.quantile(means,.025)),'upper95':float(np.quantile(means,.975)),
            'scope':'random-seed variability conditional on this reused dataset and fixed budget; not population or learning-efficiency confidence'}
