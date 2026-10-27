"""Feasible credit-cost allocation with explicit hypothetical shared resources."""
import hashlib
from time import perf_counter
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from .evaluation import _probabilities
from .decision_rules import row_keys


def hypothetical_weights(groups,seed,sigma=.8):
    """Seeded, label-free scenario weights; not observed loan amounts or EAD."""
    result=[]
    for g in np.asarray(groups).astype(str):
        h=hashlib.sha256(f'{seed}:{g}'.encode()).digest()
        u=(int.from_bytes(h[:8],'big')+.5)/2**64
        v=(int.from_bytes(h[8:16],'big')+.5)/2**64
        # Clip away double-precision rounding to exactly 0/1.
        u=np.clip(u,1e-15,1-1e-15)
        z=np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)
        result.append(np.clip(np.exp(-.5*sigma*sigma+sigma*z),.25,4.))
    return np.array(result)


def inputs(p,weight,ids,K,funding,risk,ratio):
    p=_probabilities(p);w=np.asarray(weight,float);ids=row_keys(ids,len(p))
    if w.shape!=p.shape or not np.isfinite(w).all() or (w<=0).any():raise ValueError('Positive aligned weights required')
    if type(K) is not int or not 0<=K<=len(p):raise ValueError('Invalid count cap')
    if not np.isfinite([funding,risk,ratio]).all() or min(funding,risk)<0 or ratio<=0:raise ValueError('Invalid resource/cost settings')
    gain=w*(1-(1+ratio)*p)
    matrix=np.vstack([np.ones(len(p)),w,ratio*p*w]);limits=np.array([K,funding,risk],float)
    return p,w,ids,gain,matrix,limits


def feasible(mask,matrix,limits,tolerance=1e-7):
    a=np.asarray(mask)
    return bool(a.dtype==bool and a.shape==(matrix.shape[1],) and (matrix@a<=limits+tolerance).all())


def greedy(p,weight,ids,K,funding,risk,ratio=5,priority='pd'):
    p,w,ids,gain,A,b=inputs(p,weight,ids,K,funding,risk,ratio)
    if priority not in {'pd','gain'}:raise ValueError('Unknown greedy priority')
    order=np.lexsort((ids,p if priority=='pd' else -gain))
    chosen=np.zeros(len(p),bool);used=np.zeros(3)
    for j in order:
        if gain[j]>0 and (used+A[:,j]<=b+1e-10).all():chosen[j]=True;used+=A[:,j]
    if not feasible(chosen,A,b):raise AssertionError('Greedy feasibility failed')
    return chosen


def allocate(p,weight,ids,K,funding,risk,ratio=5,time_limit=2.,relative_gap=1e-4):
    p,w,ids,gain,A,b=inputs(p,weight,ids,K,funding,risk,ratio)
    if not np.isfinite(time_limit) or time_limit<=0 or not 0<=relative_gap<=1:raise ValueError('Invalid solver controls')
    start=perf_counter()
    starts=[greedy(p,w,ids,K,funding,risk,ratio,priority=s) for s in ['pd','gain']]
    best=max(starts,key=lambda a:gain@a);best_gain=float(gain@best)
    active=np.flatnonzero(gain>0)
    if not len(active) or K==0 or funding==0:
        return best,{'solver_status':0,'solver_message':'Trivial feasible optimum','planning_gain':best_gain,
            'gain_upper_bound':best_gain,'relative_gain_gap':0.,'absolute_gain_gap':0.,
            'planning_precision_met':True,'greedy_retained':False,'no_solver_incumbent':False,'solve_seconds':perf_counter()-start}
    result=milp(-gain[active],integrality=np.ones(len(active)),bounds=Bounds(0,1),
        constraints=LinearConstraint(A[:,active],-np.inf,b),options={'time_limit':time_limit,'mip_rel_gap':relative_gap})
    no_incumbent=True;retained=True
    if result.x is not None and np.isfinite(result.x).all() and np.max(abs(result.x-np.rint(result.x)))<=1e-5:
        trial=np.zeros(len(p),bool);trial[active]=np.rint(result.x).astype(bool)
        if feasible(trial,A,b):
            no_incumbent=False
            if gain@trial>=best_gain:best=trial;best_gain=float(gain@best);retained=False
    bound=getattr(result,'mip_dual_bound',None)
    upper=max(best_gain,-float(bound)) if bound is not None and np.isfinite(bound) and result.status in (0,1) else None
    absolute=max(0.,upper-best_gain) if upper is not None else None
    relative=absolute/max(abs(best_gain),1e-12) if absolute is not None else None
    precision=absolute is not None and (absolute<=1e-7 or relative<=relative_gap+1e-8)
    if not feasible(best,A,b):raise AssertionError('MILP incumbent failed resource checks')
    return best,{'solver_status':int(result.status),'solver_message':str(result.message),'planning_gain':best_gain,
        'gain_upper_bound':upper,'relative_gain_gap':relative,'absolute_gain_gap':absolute,
        'planning_precision_met':bool(precision),'greedy_retained':retained,
        'no_solver_incumbent':no_incumbent,'solve_seconds':perf_counter()-start}


def accounting(y,p,w,a,ratio=5):
    y=np.asarray(y);p=_probabilities(p);w=np.asarray(w,float);a=np.asarray(a)
    if (y.shape!=p.shape or w.shape!=p.shape or a.shape!=p.shape or a.dtype!=bool or not np.isin(y,[0,1]).all()
            or not np.isfinite(w).all() or (w<=0).any() or not np.isfinite(ratio) or ratio<=0):
        raise ValueError('Aligned valid allocation accounting required')
    actual=w*((~a)*(1-y)+a*ratio*y)
    predicted=w*((~a)*(1-p)+a*ratio*p)
    return actual,predicted
