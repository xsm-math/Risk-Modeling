"""Discrimination, probability quality and transparent hypothetical decisions."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, roc_curve


def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return {'auc': float(roc_auc_score(y, p)), 'ks': float(np.max(tpr-fpr)),
            'average_precision': float(average_precision_score(y, p)),
            'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p)),
            'mean_predicted': float(np.mean(p)), 'default_rate': float(np.mean(y))}


def per_row_cost(y, p, threshold, ratio):
    reject = np.asarray(p) >= threshold
    y = np.asarray(y)
    return ((y == 0) & reject).astype(float) + ratio * ((y == 1) & ~reject)


def policy(y, p, threshold, ratio):
    y, p = np.asarray(y), np.asarray(p)
    approved = p < threshold
    return {'threshold': float(threshold), 'cost_per_customer': float(per_row_cost(y,p,threshold,ratio).mean()),
            'approval_rate': float(approved.mean()),
            'approved_default_rate': float(y[approved].mean()) if approved.any() else None,
            'bad_capture': float((~approved & (y == 1)).sum() / (y == 1).sum()) if (y == 1).any() else None}


def choose_threshold(y, p, ratio):
    """Exact empirical minimization over tied score blocks; lower rejection on ties.

    Starts with reject-all, then approves entire equal-probability blocks. Thus
    no unattainable decision is created by splitting ties.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError('Cost ratio must be positive.')
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError('Nonempty, aligned finite predictions required.')
    order = np.argsort(p, kind='stable'); ys, ps = y[order], p[order]
    total = float((y == 0).sum())
    costs = total + np.cumsum(np.where(ys == 1, ratio, -1))
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps)-1]
    vals = np.r_[total, costs[ends]]
    thresholds = np.r_[0., np.nextafter(ps[ends], np.inf)]
    best = np.flatnonzero(np.isclose(vals, vals.min(), rtol=0, atol=1e-10))[-1]
    return float(thresholds[best])


def approval_threshold(p_validation, target):
    return float(np.quantile(p_validation, target, method='higher'))


def reliability(y, p):
    frame = pd.DataFrame({'y': y, 'p': p})
    frame['bin'] = pd.cut(frame.p, np.linspace(0,1,11), include_lowest=True, labels=False)
    return frame.groupby('bin').agg(count=('y','size'), predicted=('p','mean'), observed=('y','mean')).reset_index()


def bootstrap(y, p, reference, threshold, reference_threshold, ratio, groups, repeats, seed):
    """Paired cluster bootstrap, retaining all rows of duplicate-feature groups.

    Conditional on already fitted models and selected policies; excludes training
    and model-selection uncertainty. Differences use the same sampled clusters.
    """
    y, p, reference = map(np.asarray, (y,p,reference))
    groups = np.asarray(groups)
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
