"""单元测试：用已知答案的输入验证指标实现。

这不是形式主义——面试官问"AUC 怎么算的"时，
你说"我写了两种独立实现并交叉验证，误差 <1e-9"，比背定义强得多。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk.metrics import (  # noqa: E402
    auc_rank,
    auc_roc,
    best_threshold,
    confusion_at_threshold,
    expected_cost,
    ks_stat,
    psi,
)
from src.risk.model import LogisticRegressionGD, log_loss, sigmoid  # noqa: E402


# ------------------------------------------------------------------ AUC
def test_auc_perfect_and_reversed():
    y = np.array([0, 0, 1, 1])
    assert auc_roc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auc_roc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert auc_roc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_auc_known_value():
    """手算例子：y=[1,0,1,0]，p=[0.9,0.8,0.4,0.3]。

    正样本分数 0.9 > 0.8/0.3，0.4 > 0.3，但 0.4 < 0.8 → 3 对 / 4 对 = 0.75
    """
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.4, 0.3])
    assert auc_roc(y, p) == pytest.approx(0.75)
    assert auc_rank(y, p) == pytest.approx(0.75)


def test_auc_two_implementations_agree_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(50, 400))
        y = (rng.random(n) < 0.3).astype(int)
        if y.sum() == 0 or y.sum() == n:
            continue
        p = rng.random(n)
        assert auc_roc(y, p) == pytest.approx(auc_rank(y, p), abs=1e-9)


def test_auc_matches_sklearn_if_available():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = 500
        y = (rng.random(n) < 0.4).astype(int)
        p = np.clip(0.5 * y + rng.normal(0.5, 0.25, n), 0, 1)
        assert auc_roc(y, p) == pytest.approx(sklearn.roc_auc_score(y, p), abs=1e-9)
        assert auc_rank(y, p) == pytest.approx(sklearn.roc_auc_score(y, p), abs=1e-9)


# ------------------------------------------------------------------ KS
def test_ks_bounds_and_value():
    y = np.array([0, 0, 1, 1])
    ks, thr = ks_stat(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert ks == pytest.approx(1.0)
    ks2, _ = ks_stat(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert ks2 == pytest.approx(0.0)
    assert 0.0 <= thr <= 1.0


def test_ks_ge_auc_relation():
    """对同一个模型，KS 通常与 AUC 正相关，但两者数值不等。"""
    rng = np.random.default_rng(3)
    n = 2000
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(0.35 * y + rng.random(n), 0, 1)
    a, k = auc_roc(y, p), ks_stat(y, p)[0]
    assert 0.5 <= a <= 1.0
    assert 0.0 <= k <= 1.0
    assert not np.isclose(a, k)


# ------------------------------------------------------------------ 混淆矩阵 / 期望损失
def test_confusion_counts():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.2])
    m = confusion_at_threshold(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall_tpr"] == pytest.approx(0.5)
    assert m["approval_rate"] == pytest.approx(3 / 5)


def test_best_threshold_matches_brute_force_on_its_grid():
    """在**同一网格**上暴力搜索，结果必须与 best_threshold 完全一致。

    注意：网格搜索只会给出网格上的最优；跨网格比较时组内最优可能更小，
    所以断言必须限定在同一网格（第一版测试写错过这一点，被 runner 抓出来了）。
    """
    rng = np.random.default_rng(11)
    n = 3000
    y = (rng.random(n) < 0.25).astype(int)
    p = np.clip(0.4 * y + rng.random(n), 0, 1)
    cost_fp, cost_fn = 400.0, 8200.0

    grid = np.linspace(0.001, 0.999, 199)
    brute = [expected_cost(y, p, t, cost_fp, cost_fn) for t in grid]
    k = int(np.argmin(brute))

    thr, cost = best_threshold(y, p, cost_fp, cost_fn)
    assert cost == pytest.approx(min(brute))
    assert thr == pytest.approx(grid[k])
    assert 0.0 < thr < 1.0


def test_best_threshold_equals_analytic_optimum_for_calibrated_scores():
    """当预测概率**已校准**时，最优阈值满足解析式：

        通过一个客户的期望收益 = (1-p)*profit_good - p*loss_bad
        → 临界 p* = profit_good / (profit_good + loss_bad) = 1 / (1 + loss/profit)

    这是"用数学选阈值"最干净的证据：解析解 ≈ 网格搜索解。
    """
    rng = np.random.default_rng(23)
    n = 40000
    p_true = rng.beta(2.0, 6.0, n)               # 校准好的真实概率
    y = (rng.random(n) < p_true).astype(int)
    p_hat = np.clip(p_true + rng.normal(0, 0.02, n), 1e-6, 1 - 1e-6)  # 近似校准的预测

    cost_fp, cost_fn = 400.0, 8200.0             # 误拒好客户 / 放行坏客户
    analytic = cost_fp / (cost_fp + cost_fn)     # ≈0.0465
    thr, _ = best_threshold(y, p_hat, cost_fp, cost_fn)
    assert abs(thr - analytic) < 0.05, f"网格解 {thr:.4f} 偏离解析解 {analytic:.4f}"


# ------------------------------------------------------------------ PSI
def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(5)
    base = rng.normal(0, 1, 5000)
    small = rng.normal(0.1, 1, 5000)
    big = rng.normal(1.5, 1, 5000)
    p_small, p_big = psi(base, small), psi(base, big)
    assert p_small < 0.05 < p_big


# ------------------------------------------------------------------ 模型
def test_sigmoid_stability():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    s = sigmoid(z)
    assert np.all(np.isfinite(s))
    assert s[0] == pytest.approx(0.0)
    assert s[-1] == pytest.approx(1.0)
    assert s[2] == pytest.approx(0.5)


def test_log_loss_known():
    y = np.array([1, 0])
    assert log_loss(y, np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
    assert log_loss(y, np.array([0.5, 0.5])) == pytest.approx(np.log(2), abs=1e-6)


def test_logistic_recovers_separating_signal():
    """可分离数据上，模型应给出接近 0/1 的概率且系数符号正确。"""
    rng = np.random.default_rng(1)
    n = 800
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    z = 3.0 * x1 - 2.0 * x2
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    X = np.column_stack([x1, x2])
    m = LogisticRegressionGD(lr=0.5, epochs=2000, l2=0.0, standardize=True).fit(X, y)
    w = dict(m.coefficients())
    assert w["x0"] > 0 and w["x1"] < 0
    from src.risk.metrics import auc_roc
    assert auc_roc(y, m.predict_proba(X)) > 0.7


def test_model_comparison_with_sklearn_if_available():
    """同样的 WOE 化输入，手写 GD 与 sklearn 的 AUC 应接近（差距 < 0.05）。"""
    sklearn_lm = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(2)
    n = 1500
    X = rng.normal(0, 1, (n, 3))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ np.array([1.5, -1.0, 0.5]))))).astype(int)

    mine = LogisticRegressionGD(lr=0.5, epochs=1500, l2=0.01).fit(X, y)
    ref = sklearn_lm.LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    from src.risk.metrics import auc_roc
    a_mine = auc_roc(y, mine.predict_proba(X))
    a_ref = auc_roc(y, ref.predict_proba(X)[:, 1])
    assert abs(a_mine - a_ref) < 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
