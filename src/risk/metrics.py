"""NumPy 实现的排序、分类、成本与分布稳定性指标。"""
from __future__ import annotations

import numpy as np


def confusion_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    """给定阈值返回混淆矩阵。约定：prob >= thr 判为坏客户(拒绝)。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = (np.asarray(y_prob, dtype=float) >= thr).astype(float)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))  # 抓到坏客户
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))  # 误拒好客户
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))  # 放过坏客户
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))  # 正确通过

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = TPR = 坏客户召回
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": float(thr),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall_tpr": recall,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "f1": f1,
        "approval_rate": (tn + fn) / len(y_true) if len(y_true) else 0.0,  # 通过率
    }


def roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """手写 ROC：按分数从高到低排序，逐点累计 TPR / FPR。

    返回 (fpr, tpr, thresholds)，thresholds 对应每个切分点。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    P = float(np.sum(y_sorted == 1))
    N = float(np.sum(y_sorted == 0))
    if P == 0 or N == 0:
        raise ValueError("ROC 需要同时存在正负样本")

    # 只在分数变化处取切分点（避免同分导致阶梯失真）
    distinct = np.where(np.diff(p_sorted))[0]
    idx = np.r_[distinct, len(p_sorted) - 1]

    tps = np.cumsum(y_sorted == 1)[idx]
    fps = np.cumsum(y_sorted == 0)[idx]
    tpr = np.r_[0.0, tps / P]
    fpr = np.r_[0.0, fps / N]
    thr = np.r_[np.inf, p_sorted[idx]]
    return fpr, tpr, thr


def auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC = P(随机取一个正样本，其分数高于随机取的一个负样本)。

    用梯形法对 ROC 积分；等价的 Mann-Whitney U 形式见 auc_rank()。
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.trapezoid(tpr, fpr))


def auc_rank(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC 的秩和(Mann-Whitney U)形式，用于交叉验证 auc_roc 的实现是否正确。

    AUC = (sum(rank of positives) - P(P+1)/2) / (P*N)，rank 用平均秩处理同分。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    P = float(np.sum(y_true == 1))
    N = float(np.sum(y_true == 0))
    if P == 0 or N == 0:
        raise ValueError("AUC 需要同时存在正负样本")

    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.empty(len(y_prob), dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=float)
    # 同分取平均秩
    sorted_p = y_prob[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1

    return float((np.sum(ranks[y_true == 1]) - P * (P + 1) / 2.0) / (P * N))


def ks_stat(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """KS = max(TPR - FPR)，返回 (ks, 取到最大值的阈值)。

    风控最常用指标：它直接对应"切一刀时，好坏样本分布的最大分离度"。
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    gap = tpr - fpr
    k = int(np.argmax(gap))
    return float(gap[k]), float(thr[k])


def expected_cost(y_true: np.ndarray, y_prob: np.ndarray, thr: float,
                  cost_fp: float, cost_fn: float) -> float:
    """给定阈值的期望总代价 = FP*误拒代价 + FN*误放代价。

    这就是"用数学选阈值"：在阈值网格上最小化期望代价，而不是拍脑袋用 0.5。
    """
    m = confusion_at_threshold(y_true, y_prob, thr)
    return m["fp"] * cost_fp + m["fn"] * cost_fn


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                   cost_fp: float, cost_fn: float,
                   grid: int = 199) -> tuple[float, float]:
    """在 (0,1) 网格上搜索最小化期望代价的阈值，返回 (阈值, 最小期望代价)。"""
    thresholds = np.linspace(0.001, 0.999, grid)
    costs = [expected_cost(y_true, y_prob, t, cost_fp, cost_fn) for t in thresholds]
    k = int(np.argmin(costs))
    return float(thresholds[k]), float(costs[k])


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10,
        eps: float = 1e-6) -> float:
    """PSI = sum((actual% - expected%) * ln(actual% / expected%))。

    监控用：模型上线后如果特征分布漂移，PSI 会变大。
    经验阈值：<0.1 稳定；0.1~0.25 需关注；>0.25 显著漂移，考虑重训。
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    # 分箱边界取自期望(训练)分布的分位数
    if expected.size == 0 or actual.size == 0 or not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
        raise ValueError("PSI requires nonempty finite arrays")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 2:
        values = np.unique(expected)
        if len(values) == 1:
            v = values[0]
            edges = np.array([-np.inf, v, np.nextafter(v, np.inf), np.inf])
        else:
            edges = np.r_[-np.inf, values[:-1] + np.diff(values) / 2, np.inf]
    else:
        edges[0], edges[-1] = -np.inf, np.inf

    e_cnt, _ = np.histogram(expected, bins=edges)
    a_cnt, _ = np.histogram(actual, bins=edges)
    e_pct = np.clip(e_cnt / max(e_cnt.sum(), 1), eps, None)
    a_pct = np.clip(a_cnt / max(a_cnt.sum(), 1), eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
