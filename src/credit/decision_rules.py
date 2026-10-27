"""Label-free batch capacity rules and explicit retrospective cost accounting."""
import hashlib
import numpy as np

from .evaluation import _probabilities


def row_keys(ids,n):
    ids=np.asarray(ids)
    if ids.ndim!=1 or len(ids)!=n or len(np.unique(ids))!=n:
        raise ValueError('Unique aligned row IDs required for deterministic ties')
    return ids


def capacity_count(n,fraction):
    if isinstance(fraction,(bool,np.bool_)) or not np.isscalar(fraction) or not np.isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError('Capacity fraction must lie in [0,1]')
    return int(np.floor(n*fraction))


def hard_capacity(p,ids,fraction,ratio=None,risk_budget=None):
    """At most floor(q*N), lowest PD first; optional positive-gain/risk gates.

    IDs break equal-score ties without labels. Without a cost gate this fills
    capacity as a ranking-only control, not as a minimum-cost policy. With
    homogeneous costs and unit loss weight, a sorted feasible prefix is optimal.
    """
    p=_probabilities(p);ids=row_keys(ids,len(p));cap=capacity_count(len(p),fraction)
    if ratio is not None and (isinstance(ratio,(bool,np.bool_)) or not np.isfinite(ratio) or ratio<=0):
        raise ValueError('Positive cost ratio required')
    if risk_budget is not None and (isinstance(risk_budget,(bool,np.bool_)) or not np.isfinite(risk_budget) or risk_budget<0):
        raise ValueError('Nonnegative unit expected-loss budget required')
    order=np.lexsort((ids,p));chosen=np.zeros(len(p),bool)
    order=order[:cap]
    if ratio is not None:order=order[p[order]<1/(1+ratio)]
    if risk_budget is not None:
        # The budget is an upper bound on predicted mean loss, not realized loss.
        order=order[np.cumsum(p[order])<=risk_budget+1e-12]
    chosen[order]=True
    return chosen


def grouped_batches(groups,ids,target,seed):
    """Outcome-blind deterministic packing; duplicate financial groups intact."""
    groups=np.asarray(groups).astype(str);ids=row_keys(ids,len(groups))
    if type(target) is not int or target<1:raise ValueError('Positive integer batch size required')
    buckets={}
    for i,g in enumerate(groups):buckets.setdefault(g,[]).append(i)
    ordering=sorted(buckets,key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    result=[];current=[]
    for g in ordering:
        members=sorted(buckets[g],key=lambda i:ids[i])
        if current and len(current)+len(members)>target:result.append(np.array(current));current=[]
        current.extend(members)
        if len(current)>=target:result.append(np.array(current));current=[]
    if current:result.append(np.array(current))
    if sorted(np.concatenate(result).tolist())!=list(range(len(groups))):raise AssertionError('Batch coverage failed')
    return result


def cost_rows(y,p,approved,ratio):
    p=_probabilities(p);y=np.asarray(y);a=np.asarray(approved)
    if y.shape!=p.shape or a.shape!=p.shape or a.dtype!=bool or not np.isin(y,[0,1]).all():
        raise ValueError('Aligned binary labels and Boolean decisions required')
    if not np.isfinite(ratio) or ratio<=0:raise ValueError('Positive cost ratio required')
    realized=(~a)*(1-y)+a*y*ratio
    expected=(~a)*(1-p)+a*p*ratio
    return realized.astype(float),expected


def decision_summary(y,p,approved,ratio,ids):
    realized,expected=cost_rows(y,p,approved,ratio)
    a=np.asarray(approved);y=np.asarray(y);ids=row_keys(ids,len(y))
    # Publish aggregates/hashes only. Row decisions remain in ignored local/.
    return {'n':len(y),'approved':int(a.sum()),'approval_rate':float(a.mean()),
        'actual_cost':float(realized.mean()),'predicted_cost':float(expected.mean()),
        'approved_defaults':int(y[a].sum()),'predicted_approved_defaults':float(np.asarray(p)[a].sum()),
        'approved_default_rate':float(y[a].mean()) if a.any() else None,
        'approval_ids_sha256':hashlib.sha256(','.join(map(str,sorted(ids[a].tolist()))).encode()).hexdigest()}


def paired_group_interval(values,groups,repeats=500,seed=172):
    values=np.asarray(values,float);groups=np.asarray(groups).astype(str)
    if values.ndim!=1 or values.shape!=groups.shape or not np.isfinite(values).all() or repeats<2:
        raise ValueError('Aligned finite values/groups and repeated draws required')
    _,inverse=np.unique(groups,return_inverse=True)
    totals=np.bincount(inverse,weights=values);sizes=np.bincount(inverse)
    draws=np.random.default_rng(seed).integers(0,len(sizes),size=(repeats,len(sizes)))
    means=totals[draws].sum(1)/sizes[draws].sum(1)
    return float(values.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))


def order_diagnostics(raw,calibrated,ids):
    raw=_probabilities(raw);cal=_probabilities(calibrated);ids=row_keys(ids,len(raw))
    if raw.shape!=cal.shape:raise ValueError('Aligned predictions required')
    order=np.lexsort((ids,raw));r=raw[order];c=cal[order];distinct=np.diff(r)>0
    raw_blocks=int(len(np.unique(raw)));cal_blocks=int(len(np.unique(cal)))
    return {'rows':len(raw),'raw_blocks':raw_blocks,'calibrated_blocks':cal_blocks,
        'nondecreasing_on_observed_rows':bool((np.diff(c)>=0).all()),
        'strict_between_observed_raw_blocks':bool((np.diff(c)[distinct]>0).all()),
        'identical_raw_ties_remain_ties':bool((np.diff(c)[~distinct]==0).all()),
        'same_score_id_order':bool(np.array_equal(order,np.lexsort((ids,cal))))}
