import numpy as np
import pandas as pd
import pytest

from src import train
from src.risk.features import WOEEncoder, scorecard
from src.risk.model import LogisticRegressionGD, sigmoid
from src.risk.monitor import monitor_report, judge
from src.risk.metrics import psi


def test_clean_deduplicates_without_learning_imputation():
    df = pd.DataFrame({'customer_id': ['a', 'a', 'b'], 'age': [999, 30, 20],
                       'debt_ratio': [2., .5, np.nan]})
    result = train.clean(df)
    assert list(result.index) == [0, 1]
    assert np.isnan(result.loc[0, 'age'])
    assert np.isnan(result.loc[1, 'debt_ratio'])
    assert result.loc[0, 'debt_ratio'] == 1.2


def test_raw_coefficients_reconstruct_probabilities_and_scores():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(100, 3)) * [2, 20, .2] + [4, 100, 1]
    y = (X[:, 0] > 4).astype(int)
    model = LogisticRegressionGD().fit(X, y)
    w, b = model.raw_parameters()
    np.testing.assert_allclose(model.predict_proba(X), sigmoid(X @ w + b))
    p = model.predict_proba(X)
    factor = 20 / np.log(2)
    scores = 600 - factor * (np.log(20) + X @ w + b)
    np.testing.assert_allclose(scores, 600 - factor * (np.log(20) + np.log(p/(1-p))))
    card = scorecard(w, b, ['a', 'b', 'c'])
    reconstructed = card.iloc[0]['得分贡献'] + card.iloc[-1]['得分贡献'] + X @ card.iloc[1:-1]['得分贡献'].to_numpy()
    np.testing.assert_allclose(reconstructed, scores, atol=1.0)


def test_refit_resets_scaling():
    X = np.arange(40.).reshape(20, 2)
    y = np.tile([0, 1], 10)
    model = LogisticRegressionGD().fit(X, y).fit(X + 100, y)
    fresh = LogisticRegressionGD().fit(X + 100, y)
    np.testing.assert_allclose(model.predict_proba(X + 100), fresh.predict_proba(X + 100))


def test_woe_merged_intervals_match_training_mapping():
    X = pd.DataFrame({'x': np.arange(120.)})
    y = pd.Series(([0] * 9 + [1]) * 12)
    enc = WOEEncoder(n_bins=8, min_bin_bad=4).fit(X, y, ['x'])
    w = enc.transform(X, ['x'])
    assert len(enc.rules['x']['woe_by_interval']) == len(enc.edges('x')) - 1
    np.testing.assert_allclose(w[::3], enc.transform(X.iloc[::3], ['x']))
    assert np.isfinite(enc.transform(pd.DataFrame({'x': [-100, 1000, np.nan]}), ['x'])).all()


def test_sparse_and_constant_psi_detect_shift():
    assert psi(np.ones(100), np.ones(100)) == 0
    assert psi(np.ones(100), np.full(100, 2.)) > .25
    assert psi(np.r_[np.zeros(99), 1], np.ones(100)) > .25
    with pytest.raises(ValueError):
        psi(np.array([]), np.ones(10))


def test_monitor_scores_and_boundaries():
    base = pd.DataFrame({'x': np.arange(100.)})
    table, alarm = monitor_report(base, base, ['x'], np.ones(100), np.full(100, 2.))
    assert alarm and '__score__' in table.feature.tolist()
    assert judge(.1) == '需关注' and judge(.25) == '显著漂移'
    with pytest.raises(ValueError):
        monitor_report(base, base, ['x'], train_scores=np.ones(100))


def test_pipeline_fits_woe_only_on_fitting_partition(tmp_path, monkeypatch):
    original = WOEEncoder.fit
    observed = []
    def checked_fit(self, X, y, features):
        observed.append(len(X))
        assert X.notna().all().all()
        return original(self, X, y, features)
    monkeypatch.setattr(WOEEncoder, 'fit', checked_fit)
    monkeypatch.setattr(train, 'OUT_DIR', tmp_path)
    result = train.run()
    assert observed == [result['n_fit']]
    assert result['n_fit'] + result['n_validation'] + result['n_test'] == result['n_total']
    pred = pd.read_csv(tmp_path / 'predictions.csv')
    assert np.isfinite(pred['score']).all()
    assert result['auc_test'] == result['auc_rank_check']
    assert result['cost_threshold_best'] == 400 * result['confusion_threshold_best']['fp'] + 8200 * result['confusion_threshold_best']['fn']
