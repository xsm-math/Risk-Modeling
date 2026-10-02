"""Offline guarantees for the public-data pipeline (no network in unit tests)."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits
from src.credit.data import FINANCIAL, TARGET, features, split_data, load_data
from src.credit.models import fit_candidates
from src.credit.evaluation import choose_threshold, per_row_cost, policy, approval_threshold, bootstrap


def fixture_frame(n=400):
    rng=np.random.default_rng(91)
    df=pd.DataFrame(rng.normal(size=(n,len(FINANCIAL))),columns=FINANCIAL)
    df['LIMIT_BAL']=rng.integers(1,100,n)*1000
    df['ID']=np.arange(n);df[TARGET]=np.tile([0,0,0,1],n//4)
    return df


def test_duplicate_groups_never_cross_four_part_split():
    df=fixture_frame()
    df.loc[350:359,FINANCIAL]=df.loc[:9,FINANCIAL].to_numpy()
    splits,m=split_data(df,31)
    assert m.groupby('group')['split'].nunique().max()==1
    assert sorted(np.concatenate(list(splits.values())))==list(range(len(df)))
    assert all(set(df.iloc[i][TARGET])=={0,1} for i in splits.values())
    _,same=split_data(df,31)
    pd.testing.assert_frame_equal(m,same)


def test_features_exclude_labels_ids_demographics_and_are_row_local():
    df=fixture_frame();x=features(df)
    modified=df.copy();modified[TARGET]=1-df[TARGET];modified.ID+=900000
    modified['SEX']=2;modified['AGE']=123
    pd.testing.assert_frame_equal(x,features(modified))
    pd.testing.assert_frame_equal(x.iloc[:3],features(df.iloc[:3]))
    assert len(x.columns)==24 and not set(['ID',TARGET,'SEX','AGE'])&set(x.columns)


def test_invalid_input_and_modified_source_fail_closed(tmp_path):
    df=fixture_frame();df.loc[0,'LIMIT_BAL']=0
    with pytest.raises(ValueError):features(df)
    p=tmp_path/'bad.zip';p.write_bytes(b'not the pinned UCI archive')
    with pytest.raises(ValueError,match='checksum'):load_data(p)
    with pytest.raises(FileNotFoundError):load_data(tmp_path/'missing.zip')


@pytest.mark.parametrize('ratio',[1,2,5,20.5])
def test_threshold_matches_exhaustive_tied_score_search(ratio):
    rng=np.random.default_rng(77);y=rng.integers(0,2,100);p=rng.choice([.1,.2,.5,.9],100)
    threshold=choose_threshold(y,p,ratio)
    candidates=np.r_[0,np.nextafter(np.unique(p),np.inf)]
    loss=np.array([per_row_cost(y,p,t,ratio).sum() for t in candidates])
    assert per_row_cost(y,p,threshold,ratio).sum()==pytest.approx(loss.min())
    assert threshold==candidates[np.flatnonzero(np.isclose(loss,loss.min()))[-1]]


def test_policy_edges_and_score_ties():
    y=np.array([0,1,0,1]);p=np.full(4,.2)
    assert policy(y,p,0,5)['approval_rate']==0
    assert policy(y,p,1,5)['approval_rate']==1
    assert policy(y,p,0,5)['approved_default_rate'] is None
    assert approval_threshold(p,.5)==.2
    assert policy(y,p,.2,5)['approval_rate']==0


def test_calibration_does_not_refit_base_and_serialization_is_exact(tmp_path):
    df=fixture_frame();X=features(df);y=df[TARGET].to_numpy()
    config={'lr_C':1.,'seed':22,'tree':{'max_iter':3,'max_leaf_nodes':3}}
    with threadpool_limits(limits=1):
        a=fit_candidates(X.iloc[:250],y[:250],X.iloc[250:325],y[250:325],config)
        b=fit_candidates(X.iloc[:250],y[:250],X.iloc[250:325],1-y[250:325],config)
        for name in ['linear_lr','woe_lr','hist_gbdt']:
            np.testing.assert_allclose(a[name].predict(X.iloc[325:]),b[name].predict(X.iloc[325:]),rtol=0,atol=0)
            assert a[name].estimator is a[name+'_sigmoid'].estimator
        path=tmp_path/'model.joblib';joblib.dump(a,path)
        restored=joblib.load(path)
        for name in a:
            np.testing.assert_allclose(a[name].predict(X.iloc[325:]),restored[name].predict(X.iloc[325:]),rtol=0,atol=0)


def test_cluster_bootstrap_paired_identity_has_zero_difference():
    y=np.tile([0,1],30);p=np.tile([.2,.6],30);groups=np.arange(60)//2
    a=bootstrap(y,p,p,.3,.3,5,groups,20,90)
    assert a['auc_gain_vs_woe']=={'lower':0.,'upper':0.}
    assert a['cost_saving_vs_woe']=={'lower':0.,'upper':0.}
