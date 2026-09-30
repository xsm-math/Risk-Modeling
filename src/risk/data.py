"""模拟信贷数据生成与 CSV 读取。系数大小不可跨不同量纲直接比较。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# 各原始量纲下的数据生成系数；绝对值不能直接比较重要性
TRUE_WEIGHTS: dict[str, float] = {
    "age": -0.020,
    "monthly_income": -0.000035,
    "debt_ratio": 1.900,
    "credit_score": -0.0050,
    "delinq_30d_2y": 0.420,
    "delinq_90d_2y": 0.700,
    "inquiry_6m": 0.130,
    "credit_history_months": -0.0035,
    "utilization": 1.100,
    "loan_amount": 0.0000060,
    "employment_years": -0.045,
    "num_open_accounts": 0.020,
}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def generate_credit_data(n_samples: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """生成一份有真实信号、有缺失、有异常值的信贷数据。

    设计要点（每一项都对应风控的真实坑）：
    - debt_ratio / utilization 对生成 logit 的系数为正
    - delinq_90d_2y 是强信号，但只有小部分人有值 → 天然稀疏
    - 故意注入 3% 缺失值 + 1% 年龄异常值，用来展示清洗能力
    """
    n = n_samples or CFG.n_samples
    rng = np.random.default_rng(seed if seed is not None else CFG.seed)

    age = np.clip(rng.normal(38, 10, n), 20, 70)
    monthly_income = np.clip(rng.lognormal(9.0, 0.5, n), 2000, 200000)
    debt_ratio = np.clip(rng.beta(2.0, 3.0, n), 0, 1.5)
    credit_score = np.clip(rng.normal(650, 70, n), 350, 850)
    # 逾期次数：越高的分数逾期越少（制造多重共线性，用于演示特征筛选）
    delinq_30d_2y = np.clip(rng.poisson(np.clip(1.2 - (credit_score - 650) / 600, 0.05, 3), n), 0, 8)
    delinq_90d_2y = np.where(rng.random(n) < 0.12, rng.integers(1, 4, n), 0)
    inquiry_6m = np.clip(rng.poisson(2.2, n), 0, 15)
    credit_history_months = np.clip(rng.normal(140, 60, n), 6, 400)
    utilization = np.clip(rng.beta(3.0, 3.0, n), 0, 1.2)
    loan_amount = np.clip(rng.lognormal(10.2, 0.6, n), 3000, 500000)
    employment_years = np.clip(rng.gamma(2.0, 2.5, n), 0, 40)
    num_open_accounts = np.clip(rng.poisson(6, n), 0, 30)

    df = pd.DataFrame({
        "customer_id": [f"C{i:07d}" for i in range(n)],
        "age": age,
        "monthly_income": monthly_income,
        "debt_ratio": debt_ratio,
        "credit_score": credit_score,
        "delinq_30d_2y": delinq_30d_2y,
        "delinq_90d_2y": delinq_90d_2y,
        "inquiry_6m": inquiry_6m,
        "credit_history_months": credit_history_months,
        "utilization": utilization,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "num_open_accounts": num_open_accounts,
    })

    # 真实 logit + 一点噪声（确保 AUC 不会虚高到 1.0）
    logit = np.full(n, CFG.base_logit)
    for col, w in TRUE_WEIGHTS.items():
        logit = logit + w * df[col].to_numpy()
    logit += rng.normal(0, CFG.noise_scale, n)
    p = _sigmoid(logit)
    df["is_default"] = (rng.random(n) < p).astype(int)

    # 注入缺失与异常：真实数据永远不干净
    for col in ("monthly_income", "employment_years", "debt_ratio"):
        idx = rng.choice(n, size=int(0.03 * n), replace=False)
        df.loc[idx, col] = np.nan
    df.loc[rng.choice(n, size=int(0.01 * n), replace=False), "age"] = 999

    return df


def load_data(path: str | Path | None = None, force_regenerate: bool = False) -> pd.DataFrame:
    """优先读 `data/raw/credit.csv`；不存在则生成并落盘（保证可复现）。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv = Path(path) if path else RAW_DIR / "credit.csv"
    if csv.exists() and not force_regenerate:
        return pd.read_csv(csv)
    df = generate_credit_data()
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    return df
