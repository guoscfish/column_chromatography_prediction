"""Descriptive raw-six-output diagnostics; never sorts or changes a predictor."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def _vectors(truth, quantiles):
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
        raise ValueError('expected finite nonempty raw [q10,q50,q90] rows')
    if truth is None:
        return None, q
    y = np.asarray(truth, dtype=np.float64)
    if y.shape != (len(q),) or not np.isfinite(y).all():
        raise ValueError('truth must match explicitly authorized rows')
    return y, q


def ranking_metrics(error, score):
    error = np.asarray(error, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if error.ndim != 1 or score.shape != error.shape or not len(error) or not np.isfinite(error).all() or not np.isfinite(score).all() or np.any(error < 0):
        raise ValueError('finite aligned nonnegative errors and ranking scores required')
    count = max(1, int(np.ceil(.2 * len(error))))
    ranking = np.argsort(-score, kind='stable')
    correlation = spearmanr(score, error).statistic if np.ptp(score) and np.ptp(error) else np.nan
    mean = float(error.mean())
    removal = np.linspace(0., .9, 10)
    oracle = np.argsort(-error, kind='stable')
    gaps = []
    for fraction in removal:
        removed = min(len(error)-1, int(np.floor(fraction*len(error))))
        gaps.append((error[ranking[removed:]].mean()-error[oracle[removed:]].mean())/mean if mean else 0.)
    return {'spearman': float(correlation) if np.isfinite(correlation) else None,
            'top20_count': count, 'top20_error_enrichment': float(error[ranking[:count]].mean()/mean) if mean else None,
            'ause': float(np.trapezoid(gaps, removal)),
            'ause_definition': 'integral normalized retained-MAE minus oracle over removal fractions 0..0.9, 10 fixed points'}


def quantile_summary(truth, quantiles):
    y, q = _vectors(truth, quantiles)
    a, b, c = q.T
    width = c-a
    result = {'rows': len(q), 'q10_gt_q50_rate': float(np.mean(a>b)),
              'q50_gt_q90_rate': float(np.mean(b>c)),
              'any_crossing_rate': float(np.mean((a>b)|(b>c))),
              'negative_width_rate': float(np.mean(a>c)),
              'q10_eq_q50_rate': float(np.mean(a==b)), 'q50_eq_q90_rate': float(np.mean(b==c)),
              'q10_eq_q90_rate': float(np.mean(a==c)),
              'signed_width_mean': float(width.mean()), 'signed_width_median': float(np.median(width)),
              'clipped_width_mean': float(np.maximum(width,0).mean()),
              'clipped_width_median': float(np.median(np.maximum(width,0))), 'raw_outputs_sorted': False}
    if y is None:
        result['target_metrics_status'] = 'NOT_MEASURED'
        return result
    error = np.abs(y-b)
    result.update(q10_pinball=float(np.maximum(.1*(y-a), -.9*(y-a)).mean()),
                  q50_mse=float(np.square(y-b).mean()), q50_rmse=float(np.sqrt(np.square(y-b).mean())),
                  q50_mae=float(error.mean()), q90_pinball=float(np.maximum(.9*(y-c), -.1*(y-c)).mean()),
                  central80_coverage=float(np.mean((y>=a)&(y<=c))))
    for name, score in [('signed_width',width),('clipped_width',np.maximum(width,0))]:
        result.update({name+'_'+key:value for key,value in ranking_metrics(error,score).items()})
    return result


def sorting_diagnostic(truth, quantiles):
    y,q = _vectors(truth,quantiles)
    if y is None:
        raise ValueError('sorting error diagnosis requires authorized truth')
    changed = np.sort(q, axis=1)[:,1]-q[:,1]
    return {'q50_changed_rate': float(np.mean(changed != 0)),
            'q50_change_mean': float(changed.mean()), 'q50_change_abs_mean': float(np.abs(changed).mean()),
            'q50_mae_raw': float(np.abs(y-q[:,1]).mean()),
            'q50_mae_sorted': float(np.abs(y-(q[:,1]+changed)).mean()),
            'q50_rmse_raw': float(np.sqrt(np.square(y-q[:,1]).mean())),
            'q50_rmse_sorted': float(np.sqrt(np.square(y-(q[:,1]+changed)).mean())),
            'postprocessing_only': True}


def score_overlap(left, right):
    left, right = np.asarray(left,float), np.asarray(right,float)
    if left.ndim != 1 or left.shape != right.shape or not len(left) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('aligned finite scores required')
    count=max(1,int(np.ceil(.2*len(left))))
    a=set(np.argsort(-left,kind='stable')[:count]);b=set(np.argsort(-right,kind='stable')[:count])
    corr=spearmanr(left,right).statistic if np.ptp(left) and np.ptp(right) else np.nan
    return {'spearman':float(corr) if np.isfinite(corr) else None,'top20_jaccard':len(a&b)/len(a|b),'top20_overlap_fraction':len(a&b)/count}
