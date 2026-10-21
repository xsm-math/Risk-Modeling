"""Regression cases found by the 2026-10-21 technical audit."""
import numpy as np
import pytest
import pandas as pd
from src.credit.evaluation import (policy, metrics, per_row_cost, choose_threshold,
                                   approval_threshold, reliability, bootstrap)


@pytest.mark.parametrize('bad', [[np.nan,.2], [np.inf,.2], [-.1,.2], [1.1,.2], [[.1],[.2]], []])
def test_invalid_probabilities_cannot_produce_decisions_or_metrics(bad):
    for fn in [lambda:policy([1,0],bad,.5,5), lambda:per_row_cost([1,0],bad,.5,5),
               lambda:choose_threshold([1,0],bad,5),lambda:metrics([1,0],bad),
               lambda:reliability([1,0],bad),lambda:approval_threshold(bad,.5)]:
        with pytest.raises(ValueError):fn()


@pytest.mark.parametrize('y', [[0,2],[0,np.nan],[0],[[0],[1]]])
def test_invalid_labels_cannot_silently_change_costs(y):
    with pytest.raises(ValueError):policy(y,[.2,.8],.5,5)
    with pytest.raises(ValueError):choose_threshold(y,[.2,.8],5)


def test_cost_equals_confusion_accounting_at_probability_boundaries():
    y=np.array([1,0,1,0]);p=np.array([0.,1.,1.,0.])
    for t in [0.,.5,1.,np.nextafter(1.,np.inf)]:
        d=policy(y,p,t,5)
        assert d['cost_per_customer']==(d['fp']+5*d['fn'])/len(y)


@pytest.mark.parametrize('target',[0.,.1,.5,.6,.9,1.])
@pytest.mark.parametrize('p',[[0.,.2,1.],[.2,.2,.2,.2],[.1,.2,.2,.8]])
def test_approval_ceiling_matches_independent_tied_block_enumeration(target,p):
    p=np.array(p);t=approval_threshold(p,target)
    capacity=int(np.floor(target*len(p)))
    attainable=[int((p<c).sum()) for c in np.r_[0.,np.nextafter(np.unique(p),np.inf)]]
    assert int((p<t).sum())==max(n for n in attainable if n<=capacity)
    if target==1:assert (p<t).all()


def test_bootstrap_rejects_misaligned_or_missing_clusters():
    for groups in [[0],[0,np.nan]]:
        with pytest.raises(ValueError):bootstrap([0,1],[.2,.8],[.3,.7],.5,.5,5,groups,10,1)
    with pytest.raises(ValueError):bootstrap([0,1],[.2,.8],[.3],.5,.5,5,[0,1],10,1)


def test_tree_probability_csv_preserves_metrics_when_read_as_float64(tmp_path):
    from src.credit.experiment import write_predictions
    p=np.array([.123456789,.876543211,.99999994,.000000119],dtype=np.float32)
    y=np.array([0,1,1,0]);path=tmp_path/'predictions.csv'
    write_predictions(path,np.arange(4),y,{'tree':p})
    restored=pd.read_csv(path,float_precision='round_trip')
    np.testing.assert_array_equal(restored.tree.to_numpy(),p.astype(np.float64))
    expected=metrics(y,p);actual=metrics(restored.y,restored.tree)
    for key in expected:assert actual[key]==pytest.approx(expected[key],abs=1e-15)
