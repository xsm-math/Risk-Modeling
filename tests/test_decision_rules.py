import itertools
import numpy as np
import pytest
import pandas as pd
from src.credit.decision_study import baseline_metric_errors
from src.credit.decision_rules import (hard_capacity,grouped_batches,cost_rows,
    decision_summary,order_diagnostics,paired_group_interval)


def test_capacity_and_label_free_tie_breaking_under_permutation():
    p=np.array([.2,.1,.1,.4]);ids=np.array([7,3,2,1])
    a=hard_capacity(p,ids,.5)
    assert set(ids[a])=={2,3}
    for perm in [np.array([3,2,1,0]),np.array([1,3,0,2])]:
        assert set(ids[perm][hard_capacity(p[perm],ids[perm],.5)])=={2,3}
    assert hard_capacity(p,ids,0).sum()==0
    assert hard_capacity(p,ids,1).sum()==4


def test_homogeneous_cost_and_risk_prefix_matches_independent_enumeration():
    p=np.array([.01,.09,.15,.3,.7]);ids=np.arange(5);r=5;budget=.13
    a=hard_capacity(p,ids,.8,r,budget)
    def objective(mask):return ((~mask)*(1-p)+mask*r*p).sum()
    feasible=[np.array(b,dtype=bool) for b in itertools.product([0,1],repeat=5)
              if sum(b)<=4 and np.array(b)@p<=budget+1e-12]
    assert objective(a)==pytest.approx(min(objective(x) for x in feasible))
    assert a.sum()==2


def test_capacity_only_can_fill_negative_gain_while_cost_gate_stops():
    p=np.array([.2,.5,.9]);ids=np.arange(3)
    assert hard_capacity(p,ids,1).sum()==3
    assert hard_capacity(p,ids,1,5).sum()==0


def test_monotone_transform_preserves_capacity_membership():
    p=np.array([.1,.1,.2,.4,.8]);ids=np.arange(5)[::-1]
    q=p**2
    assert np.array_equal(hard_capacity(p,ids,.6),hard_capacity(q,ids,.6))
    assert order_diagnostics(p,q,ids)['same_score_id_order']


def test_diagnostics_detect_tie_merging_and_rank_reversal():
    assert not order_diagnostics([.1,.2,.3],[.2,.2,.4],[1,2,3])['strict_between_observed_raw_blocks']
    assert not order_diagnostics([.1,.2,.3],[.4,.3,.2],[1,2,3])['nondecreasing_on_observed_rows']


def test_batches_keep_groups_intact_and_ignore_input_order():
    groups=np.array(['a','b','b','c','d','d','d']);ids=np.array([7,6,5,4,3,2,1])
    batches=grouped_batches(groups,ids,2,3)
    assigned={g:j for j,b in enumerate(batches) for g in groups[b]}
    for j,b in enumerate(batches):assert all(assigned[g]==j for g in groups[b])
    assert sorted(np.concatenate(batches).tolist())==list(range(7))
    perm=np.arange(7)[::-1]
    other=grouped_batches(groups[perm],ids[perm],2,3)
    assert [set(ids[b]) for b in batches]==[set(ids[perm][b]) for b in other]


def test_accounting_and_empty_approval():
    y=np.array([0,1,0]);p=np.array([.1,.2,.4]);a=np.array([True,True,False])
    actual,expected=cost_rows(y,p,a,5)
    np.testing.assert_allclose(actual,[0,5,1]);np.testing.assert_allclose(expected,[.5,1,.6])
    s=decision_summary(y,p,np.zeros(3,bool),5,np.arange(3))
    assert s['approved_default_rate'] is None and s['actual_cost']==pytest.approx(2/3)


def test_group_bootstrap_exact_zero_and_constant_effect():
    assert paired_group_interval([0,0,0],['a','a','b'])==(0.,0.,0.)
    assert paired_group_interval([2,2,2],['a','a','b'])==(2.,2.,2.)


def test_baseline_reproduction_keeps_strict_tolerance_for_study_models():
    names=['woe_lr','woe_lr_sigmoid','random_forest','random_forest_sigmoid','xgboost']
    old=pd.DataFrame({'brier':[.2]*5},index=names);new=old.copy()
    new.loc['xgboost','brier']+=7e-9
    assert baseline_metric_errors(old,new)[1]==0
    new.loc['woe_lr','brier']+=1e-8
    with pytest.raises(ValueError):baseline_metric_errors(old,new)


@pytest.mark.parametrize('kwargs',[{'fraction':-1},{'fraction':True},{'fraction':.5,'ratio':0},{'fraction':.5,'risk_budget':-1}])
def test_invalid_controls_rejected(kwargs):
    with pytest.raises(ValueError):hard_capacity([.1,.2],[1,2],**kwargs)
