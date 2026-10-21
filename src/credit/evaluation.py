"""Discrimination, probability quality and transparent hypothetical decisions."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, roc_curve, precision_recall_curve, auc


def _probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Predictions must be a nonempty finite 1-D probability array in [0,1].')
    return p


def _labeled(y, p):
    p = _probabilities(p)
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be aligned 1-D binary values.')
    return y, p


def _decision_settings(threshold, ratio):
    if np.ndim(threshold) or not np.isfinite(threshold):
        raise ValueError('Threshold must be a finite scalar.')
    if isinstance(ratio, (bool, np.bool_)) or np.ndim(ratio) or not np.isfinite(ratio) or ratio <= 0:
        raise ValueError('Cost ratio must be a positive finite scalar.')


def metrics(y, p):
    y, p = _labeled(y, p)
    if np.unique(y).size != 2:
        raise ValueError('Discrimination metrics require both label classes.')
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    return {'auc': float(roc_auc_score(y, p)), 'ks': float(np.max(tpr-fpr)),
            'average_precision': float(average_precision_score(y, p)),
            'pr_auc_trapezoid': float(auc(recall, precision)),
            'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p)),
            'mean_predicted': float(np.mean(p)), 'default_rate': float(np.mean(y))}


def per_row_cost(y, p, threshold, ratio):
    y, p = _labeled(y, p)
    _decision_settings(threshold, ratio)
    reject = p >= threshold
    return ((y == 0) & reject).astype(float) + ratio * ((y == 1) & ~reject)


def policy(y, p, threshold, ratio):
    y, p = _labeled(y, p)
    _decision_settings(threshold, ratio)
    approved = p < threshold
    tp = int((~approved & (y == 1)).sum()); fp = int((~approved & (y == 0)).sum())
    fn = int((approved & (y == 1)).sum()); tn = int((approved & (y == 0)).sum())
    return {'threshold': float(threshold), 'cost_per_customer': float(per_row_cost(y,p,threshold,ratio).mean()),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp/(tp+fp) if tp+fp else 0.,
            'recall': tp/(tp+fn) if tp+fn else 0.,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'approval_rate': float(approved.mean()),
            'approved_default_rate': float(y[approved].mean()) if approved.any() else None,
            'bad_capture': float((~approved & (y == 1)).sum() / (y == 1).sum()) if (y == 1).any() else None}


def choose_threshold(y, p, ratio):
    """Exact empirical minimization over tied score blocks; lower rejection on ties.

    Starts with reject-all, then approves entire equal-probability blocks. Thus
    no unattainable decision is created by splitting ties.
    """
    y, p = _labeled(y, p)
    _decision_settings(0., ratio)
    order = np.argsort(p, kind='stable'); ys, ps = y[order], p[order]
    total = float((y == 0).sum())
    costs = total + np.cumsum(np.where(ys == 1, ratio, -1))
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps)-1]
    vals = np.r_[total, costs[ends]]
    thresholds = np.r_[0., np.nextafter(ps[ends], np.inf)]
    best = np.flatnonzero(np.isclose(vals, vals.min(), rtol=0, atol=1e-10))[-1]
    return float(thresholds[best])


def approval_threshold(p_validation, target):
    """Largest attainable validation approval count within floor(target * N).

    Approval is p < threshold. Whole tied-score blocks remain together;
    the requested fraction is a capacity ceiling, not an exact quota.
    """
    p = _probabilities(p_validation)
    if np.ndim(target) or not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError('Target approval must lie in [0,1].')
    count = int(np.floor(target * len(p)))
    ordered = np.sort(p)
    return float(np.nextafter(ordered[-1], np.inf) if count == len(p) else ordered[count])


def reliability(y, p):
    y, p = _labeled(y, p)
    frame = pd.DataFrame({'y': y, 'p': p})
    frame['bin'] = pd.cut(frame.p, np.linspace(0,1,11), include_lowest=True, labels=False)
    return frame.groupby('bin').agg(count=('y','size'), predicted=('p','mean'), observed=('y','mean')).reset_index()


def bootstrap(y, p, reference, threshold, reference_threshold, ratio, groups, repeats, seed):
    """Paired cluster bootstrap, retaining all rows of duplicate-feature groups.

    Conditional on already fitted models and selected policies; excludes training
    and model-selection uncertainty. Differences use the same sampled clusters.
    """
    y, p = _labeled(y, p)
    reference = _probabilities(reference)
    _decision_settings(threshold, ratio)
    _decision_settings(reference_threshold, ratio)
    groups = np.asarray(groups)
    if len(reference) != len(y) or groups.ndim != 1 or len(groups) != len(y) or pd.isna(groups).any():
        raise ValueError('Reference predictions and nonmissing groups must align with labels.')
    if isinstance(repeats, (bool, np.bool_)) or not isinstance(repeats, (int, np.integer)) or repeats < 2:
        raise ValueError('Use at least two integer bootstrap repetitions.')
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    chunks = np.split(order, np.flatnonzero(np.diff(inv[order]))+1)
    rng = np.random.default_rng(seed); rows=[]
    a = per_row_cost(y,p,threshold,ratio); b = per_row_cost(y,reference,reference_threshold,ratio)
    for _ in range(repeats):
        idx = np.concatenate([chunks[k] for k in rng.integers(0,len(uniq),len(uniq))])
        if np.unique(y[idx]).size != 2:
            continue
        rows.append([roc_auc_score(y[idx],p[idx]), roc_auc_score(y[idx],p[idx])-roc_auc_score(y[idx],reference[idx]),
                     float((b[idx]-a[idx]).mean()), float(a[idx].mean())])
    if len(rows) < 2:
        raise ValueError('Too few valid bootstrap samples.')
    result={}
    for name, values in zip(['auc','auc_gain_vs_woe','cost_saving_vs_woe','cost_per_customer'],np.asarray(rows).T):
        result[name]={'lower':float(np.quantile(values,.025)), 'upper':float(np.quantile(values,.975))}
    return result
