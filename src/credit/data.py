"""Verified UCI download, strict schema and duplicate-group-aware holdouts."""
from pathlib import Path
from io import BytesIO
import hashlib
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

URL = 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
SHA256 = '56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602'
TARGET = 'default payment next month'
STATUS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILLS = [f'BILL_AMT{i}' for i in range(1, 7)]
PAYMENTS = [f'PAY_AMT{i}' for i in range(1, 7)]
FINANCIAL = ['LIMIT_BAL'] + STATUS + BILLS + PAYMENTS
RAW_FEATURES = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE'] + STATUS + BILLS + PAYMENTS


def load_data(path, download=False):
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError('Run python -m src.credit download first; no synthetic fallback.')
        payload = urllib.request.urlopen(URL, timeout=60).read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError('UCI archive checksum mismatch; source requires review.')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError('Local UCI archive checksum mismatch.')
    with zipfile.ZipFile(BytesIO(payload)) as z:
        with z.open('default of credit card clients.xls') as f:
            df = pd.read_excel(f, header=1, engine='xlrd')
    if list(df.columns) != ['ID'] + RAW_FEATURES + [TARGET] or len(df) != 30000:
        raise ValueError('Unexpected UCI schema or row count.')
    if not df.ID.is_unique or df.isna().any().any() or set(df[TARGET]) != {0, 1}:
        raise ValueError('Unexpected IDs, missing values or labels.')
    return df


def features(df):
    """Only credit-account behaviour; no ID, target or demographic predictors.

    Deterministic row-local transformations, with no fitted statistics. Negative
    bills and nonpositive status codes are preserved; they are not missing values.
    """
    x = df[FINANCIAL].astype(float).copy()
    if not np.isfinite(x.to_numpy()).all() or (x.LIMIT_BAL <= 0).any():
        raise ValueError('Financial fields must be finite and credit limits positive.')
    x['utilization_latest'] = x.BILL_AMT1 / x.LIMIT_BAL
    x['utilization_mean'] = x[BILLS].mean(axis=1) / x.LIMIT_BAL
    x['delayed_months'] = (x[STATUS] >= 1).sum(axis=1)
    x['max_delay'] = x[STATUS].clip(lower=0).max(axis=1)
    x['payment_bill_ratio'] = x[PAYMENTS].sum(axis=1) / (x[BILLS].clip(lower=0).sum(axis=1) + 1)
    return x


def split_data(df, seed):
    # Group on model inputs, so identical usable records cannot cross holdouts.
    groups = pd.util.hash_pandas_object(df[FINANCIAL], index=False).to_numpy()
    folds = np.full(len(df), -1, dtype=int)
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(cv.split(df, df[TARGET], groups)):
        folds[idx] = k
    role = np.where(folds < 6, 'train', np.where(folds == 6, 'calibration',
                    np.where(folds == 7, 'validation', 'test')))
    manifest = pd.DataFrame({'ID': df.ID.to_numpy(), 'group': groups, 'fold': folds, 'split': role})
    if manifest.groupby('group')['split'].nunique().max() != 1:
        raise AssertionError('Duplicate group crosses split.')
    return {name: np.flatnonzero(role == name) for name in ['train', 'calibration', 'validation', 'test']}, manifest
