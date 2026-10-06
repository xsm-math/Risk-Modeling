"""Decision, SQL time ordering, score scale and drift invariants."""
import json
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from threadpoolctl import threadpool_limits
from src.credit.data import STATUS, BILLS, PAYMENTS, features
from src.credit.models import fit_candidates
from src.credit.diagnostics import fixed_psi, export_scorecard
from src.credit.sql_features import execute, validate
from src.credit.evaluation import policy, metrics
from tests.test_credit import fixture_frame


def test_sql_uses_latest_three_periods_and_preserves_negative_bills(tmp_path):
    df=fixture_frame(4)
    df.loc[:,STATUS]=np.array([[-2,0,1,2,0,0]]*4)
    df.loc[:,BILLS]=np.array([[-100,0,100,200,300,400]]*4)
    df.loc[:,PAYMENTS]=np.array([[10,20,30,40,50,60]]*4)
    sql=execute(df)
    assert list(sql.delayed_months_3)==[1]*4
    assert list(sql.payment_total_3)==[60]*4
    assert list(sql.delayed_months)==[2]*4
    assert list(sql.max_delay)==[2]*4
    np.testing.assert_allclose(sql.payment_bill_ratio,210/1001)
    assert (sql.utilization_latest<0).all()
    result=validate(df,tmp_path)
    assert all(r['max_absolute_error']<1e-12 for r in result)


def test_psi_reference_identity_extremes_and_explicit_unknown():
    ref=np.arange(100,dtype=float)
    zero,bins=fixed_psi(ref,ref)
    assert zero==0 and bins.reference_count.sum()==100
    shift,details=fixed_psi(ref,np.r_[np.full(99,9999.),np.nan])
    assert shift>.25 and details.current_count.sum()==100
    value,details=fixed_psi([0,1,0,1],[0,1,999,np.nan],categorical=True)
    assert value>0
    assert details.loc[details.bin=='unknown','current_count'].iloc[0]==1
    assert details.loc[details.bin=='missing','current_count'].iloc[0]==1
    same,details=fixed_psi([np.nan,np.nan],[np.nan,np.nan])
    assert same==0 and details.reference_count.sum()==2


def test_additive_scorecard_probability_and_odds_scale(tmp_path):
    df=fixture_frame();X=features(df);y=df['default payment next month'].to_numpy()
    config={'lr_C':1.,'seed':22,'tree':{'max_iter':3,'max_leaf_nodes':3}}
    with threadpool_limits(limits=1):
        models=fit_candidates(X.iloc[:250],y[:250],X.iloc[250:325],y[250:325],config)
        spec=export_scorecard(models['woe_lr'],X.iloc[:250],y[:250],X.iloc[325:],tmp_path)
    assert spec['max_probability_reconstruction_error']<1e-12
    assert expit((spec['offset']-600)/spec['factor'])==pytest.approx(1/21)
    assert expit((spec['offset']-620)/spec['factor'])==pytest.approx(1/41)
    full=json.loads((tmp_path/'scorecard.json').read_text())
    bins=pd.read_csv(tmp_path/'scorecard_bins.csv')
    assert len(full['features'])==24
    assert bins.groupby('feature').train_count.sum().eq(250).all()
    iv=pd.read_csv(tmp_path/'iv_coefficients.csv').set_index('feature').iv
    np.testing.assert_allclose(bins.groupby('feature').iv_component.sum().reindex(iv.index),iv,rtol=0,atol=1e-12)


def test_weighting_and_new_tree_models_share_frozen_train_and_calibration():
    df=fixture_frame();X=features(df);y=df['default payment next month'].to_numpy()
    cfg={'lr_C':1.,'seed':22,'tree':{'max_iter':3,'max_leaf_nodes':3},'imbalance_ablation':True,
         'random_forest':{'n_estimators':5,'max_depth':3,'min_samples_leaf':10,'class_weight':'balanced_subsample'},
         'xgboost':{'n_estimators':5,'max_depth':2,'scale_pos_weight':1.0}}
    with threadpool_limits(limits=1):
        models=fit_candidates(X.iloc[:250],y[:250],X.iloc[250:325],y[250:325],cfg)
        assert len(models)==12
        for name in ['random_forest','xgboost','linear_lr_balanced']:
            assert models[name].estimator is models[name+'_sigmoid'].estimator
            p=models[name].predict(X.iloc[325:])
            assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
        from xgboost import DMatrix
        booster=models['xgboost'].estimator.get_booster();sample=DMatrix(X.iloc[325:])
        np.testing.assert_allclose(booster.predict(sample,pred_contribs=True).sum(axis=1),
                                  booster.predict(sample,output_margin=True),rtol=1e-5,atol=1e-5)


def test_confusion_metrics_and_ap_are_not_silent_pr_auc_aliases():
    y=np.array([0,0,1,1]);p=np.array([.1,.6,.2,.9])
    d=policy(y,p,.5,5)
    assert (d['tp'],d['fp'],d['fn'],d['tn'])==(1,1,1,1)
    assert d['precision']==d['recall']==d['f1']==.5
    assert d['cost_per_customer']==1.5
    result=metrics(y,p)
    assert result['average_precision']!=result['pr_auc_trapezoid']
