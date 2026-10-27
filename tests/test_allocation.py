import itertools
import numpy as np
import pytest
from src.credit.allocation import allocate,greedy,inputs,feasible,hypothetical_weights,accounting


def enumerate_best(p,w,K,F,R,ratio=5):
    *_,gain,A,b=inputs(p,w,np.arange(len(p)),K,F,R,ratio)
    choices=[np.array(a,bool) for a in itertools.product([0,1],repeat=len(p))]
    return max(float(gain@a) for a in choices if feasible(a,A,b))


def test_independent_enumeration_exposes_low_pd_greedy_failure():
    p=np.array([.01,.02,.02]);w=np.array([6.,5.,5.]);ids=np.arange(3)
    a=greedy(p,w,ids,2,10,1.,5,'pd');b,d=allocate(p,w,ids,2,10,1.,5,relative_gap=0.)
    assert list(a)==[True,False,False] and list(b)==[False,True,True]
    assert d['planning_gain']==pytest.approx(enumerate_best(p,w,2,10,1.))
    assert d['absolute_gain_gap']==pytest.approx(0,abs=1e-7)


@pytest.mark.parametrize('funding,risk',[(0,0),(2,.05),(3,.4),(5,100)])
def test_unit_cost_greedy_matches_enumeration_and_milp(funding,risk):
    p=np.array([0,.02,.05,.1,.2]);w=np.ones(5);ids=np.arange(5)
    a=greedy(p,w,ids,4,funding,risk);b,d=allocate(p,w,ids,4,funding,risk,relative_gap=0.)
    g=w*(1-6*p)
    assert float(g@a)==pytest.approx(enumerate_best(p,w,4,funding,risk))
    assert float(g@b)==pytest.approx(float(g@a))


def test_milp_retains_at_least_both_checked_greedy_gains():
    p=np.array([.01,.02,.06,.11,.14]);w=np.array([3,2,2,1,4.]);ids=np.arange(5)
    a,d=allocate(p,w,ids,3,5,.7)
    g=w*(1-6*p)
    assert float(g@a)>=max(float(g@greedy(p,w,ids,3,5,.7,priority=k)) for k in ['pd','gain'])-1e-9


def test_hypothetical_weights_keep_groups_and_permutations():
    g=np.array(['a','b','a','c']);w=hypothetical_weights(g,2)
    assert w[0]==w[2] and ((w>=.25)&(w<=4)).all()
    np.testing.assert_array_equal(hypothetical_weights(g[::-1],2),w[::-1])


def test_weighted_cost_accounting_and_reject_all_identity():
    y=np.array([0,1]);p=np.array([.1,.2]);w=np.array([2.,3.]);a=np.array([True,False])
    actual,pred=accounting(y,p,w,a)
    np.testing.assert_allclose(actual,[0,0]);np.testing.assert_allclose(pred,[1,2.4])
    baseline=w*(1-p);gain=w*(1-6*p)
    assert baseline.sum()-pred.sum()==pytest.approx(gain@a)


def test_no_positive_gain_does_not_force_capacity_to_fill():
    a,d=allocate([.5,.9],[1,1],[1,2],2,2,10)
    assert not a.any() and d['planning_precision_met']


def test_invalid_resource_inputs_rejected():
    with pytest.raises(ValueError):allocate([.1,.2],[1,-1],[1,2],2,2,1)
    with pytest.raises(ValueError):allocate([.1,.2],[1,1],[1,2],2,-1,1)
