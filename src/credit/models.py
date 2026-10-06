"""Fixed-capacity baselines and holdout logit-sigmoid calibration."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from src.risk.features import WOEEncoder


class WOETransform(TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        self.names_ = list(X.columns)
        self.encoder_ = WOEEncoder(n_bins=5, min_bin_pct=.05, min_bin_bad=30)
        self.encoder_.fit(X, pd.Series(y), self.names_)
        return self

    def transform(self, X):
        return self.encoder_.transform(X, self.names_)


def build_models(config):
    lr = lambda: LogisticRegression(C=config['lr_C'], max_iter=3000, solver='lbfgs')
    models = {
        'linear_lr': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), lr()),
        'woe_lr': make_pipeline(WOETransform(), StandardScaler(), lr()),
        'hist_gbdt': HistGradientBoostingClassifier(**config['tree'], early_stopping=False,
                                                   random_state=config['seed']),
    }
    if config.get('imbalance_ablation', False):
        models['linear_lr_balanced'] = make_pipeline(SimpleImputer(strategy='median'),
            StandardScaler(), LogisticRegression(C=config['lr_C'], max_iter=3000,
                                                 class_weight='balanced'))
    if 'random_forest' in config:
        models['random_forest'] = RandomForestClassifier(**config['random_forest'],
                                                       random_state=config['seed'], n_jobs=2)
    if 'xgboost' in config:
        from xgboost import XGBClassifier
        models['xgboost'] = XGBClassifier(**config['xgboost'], objective='binary:logistic',
            eval_metric='logloss', tree_method='hist', random_state=config['seed'], n_jobs=2)
    return models


class ProbabilityModel:
    def __init__(self, estimator, calibration=None):
        self.estimator = estimator
        self.calibration = calibration

    def predict(self, X):
        p = self.estimator.predict_proba(X)[:, 1]
        if self.calibration is not None:
            p = self.calibration.predict_proba(logit(np.clip(p, 1e-6, 1-1e-6)).reshape(-1, 1))[:, 1]
        return np.clip(p, 1e-9, 1-1e-9)


def fit_candidates(X_train, y_train, X_cal, y_cal, config):
    candidates = {}
    for name, estimator in build_models(config).items():
        estimator.fit(X_train, y_train)
        base = ProbabilityModel(estimator)
        candidates[name] = base
        calibrator = LogisticRegression(C=1e6, max_iter=1000)
        calibrator.fit(logit(base.predict(X_cal)).reshape(-1, 1), y_cal)
        candidates[name + '_sigmoid'] = ProbabilityModel(estimator, calibrator)
    return candidates
