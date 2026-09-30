"""NumPy 逻辑回归：梯度下降、L2 正则与数值稳定预测。"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定版 sigmoid：分段处理避免 exp 溢出。"""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticRegressionGD:
    """梯度下降 + L2 的 logistic regression。

    参数
    ----
    lr : 学习率
    epochs : 最大迭代轮数
    l2 : L2 正则强度（不含截距项）
    tol : 损失改善小于 tol 时提前停止
    standardize : 是否对输入特征按拟合集均值与标准差进行标准化
    """

    def __init__(self, lr: float = 0.5, epochs: int = 400, l2: float = 0.0,
                 tol: float = 1e-7, standardize: bool = True, verbose: bool = False):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.tol = tol
        self.standardize = standardize
        self.verbose = verbose
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self.feature_names: list[str] = []

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return X
        if self._mu is None:
            self._mu = X.mean(axis=0)
            self._sigma = X.std(axis=0)
            self._sigma[self._sigma < 1e-12] = 1.0
        return (X - self._mu) / self._sigma

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self._mu = self._sigma = None
        self.loss_history = []
        Xs = self._scale(X)
        self.w = np.zeros(d)
        self.b = 0.0
        prev = np.inf

        for ep in range(self.epochs):
            p = sigmoid(Xs @ self.w + self.b)
            grad_w = Xs.T @ (p - y) / n + self.l2 * self.w
            grad_b = float(np.mean(p - y))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            p = sigmoid(Xs @ self.w + self.b)
            loss = log_loss(y, p) + 0.5 * self.l2 * float(np.dot(self.w, self.w))
            self.loss_history.append(loss)

            if self.verbose and (ep % 50 == 0 or ep == self.epochs - 1):
                msg = f"  epoch {ep:4d}  loss={loss:.6f}"
                if X_val is not None and y_val is not None:
                    msg += f"  val_auc={_quick_auc(y_val, self.predict_proba(X_val)):.4f}"
                print(msg)

            if abs(prev - loss) < self.tol:
                if self.verbose:
                    print(f"  提前停止于 epoch {ep}（损失改善 < {self.tol}）")
                break
            prev = loss
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("请先调用 fit()")
        Xs = self._scale(np.asarray(X, dtype=float))
        return sigmoid(Xs @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def raw_parameters(self) -> tuple[np.ndarray, float]:
        """Return coefficients and intercept in the original input (WOE) scale."""
        if self.w is None:
            raise RuntimeError("Call fit first")
        if not self.standardize:
            return self.w.copy(), float(self.b)
        w = self.w / self._sigma
        return w, float(self.b - self._mu @ w)

    def coefficients(self) -> list[tuple[str, float]]:
        w, _ = self.raw_parameters()
        names = self.feature_names or [f"x{i}" for i in range(len(w))]
        return sorted(zip(names, map(float, w)), key=lambda kv: abs(kv[1]), reverse=True)

    def odds_ratios(self) -> list[tuple[str, float]]:
        """exp(系数)：在 WOE 输入下，WOE 每增加 1，odds 变为原来的 exp(beta) 倍。"""
        return [(n, float(np.exp(v))) for n, v in self.coefficients()]


def _quick_auc(y: np.ndarray, p: np.ndarray) -> float:
    """内部用的轻量 AUC（避免循环导入）。"""
    from .metrics import auc_rank
    return auc_rank(y, p)
