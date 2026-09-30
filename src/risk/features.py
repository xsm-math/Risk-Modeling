"""数值分箱、WOE/IV 与评分卡。缺失值使用拟合集的中位数填充。"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def _quantile_edges(x: pd.Series, n_bins: int) -> np.ndarray:
    """在**原始值空间**上取分箱边界（去重后可能少于 n_bins 个箱）。

    关键坑：零膨胀变量（如"近两年逾期90天+次数"有 ~88% 是 0）直接用分位数，
    会算出大量**相同**的边界，pd.cut 去重后可能只剩 1~2 个箱，
    于是这个真实信号很强的变量 IV 会变成 0，被误判成"无用特征"丢掉。

    处理：把 0 与 非0 拆成两段分别取分位边界，0 内部不再细分。
    这样"有逾期记录 vs 无记录"的结构被保留，且边界是**原始值**，
    对任何新数据（训练/验证/测试/线上）都一致 —— 不会出现同值不同箱。
    """
    v = x.astype(float).to_numpy()
    zero_frac = float(np.mean(v == 0))
    edges: list[float] = [-np.inf]

    if 0.05 < zero_frac < 0.95:
        # 零值/非零值之间放一个边界，把 0 单独切成一段
        edges.append(0.0)
        nz = v[v != 0]
        k = max(n_bins - 1, 1)
        if nz.size >= k:
            qs = np.quantile(nz, np.linspace(0, 1, k + 1))
            edges.extend(float(t) for t in qs[1:-1])
    else:
        non_inf = v[np.isfinite(v)]
        if non_inf.size == 0:
            return np.array([])
        qs = np.quantile(non_inf, np.linspace(0, 1, n_bins + 1))
        edges.extend(float(t) for t in qs[1:-1])

    edges.append(np.inf)
    return np.unique(np.asarray(edges, dtype=float))


def _bins_from_edges(x: pd.Series, edges: np.ndarray) -> pd.Series:
    """按原始值边界切箱，返回 0..k-1 的整数箱号。"""
    return pd.cut(x.astype(float), bins=edges, include_lowest=True,
                  labels=False, duplicates="drop")


def _woe_iv_table(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """给定 (bin, y) 计算每箱 WOE/IV 及总 IV。"""
    g = df.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    g = g.rename(columns={"count": "total", "sum": "bad"})
    g["good"] = g["total"] - g["bad"]
    tot_bad = max(g["bad"].sum(), 1)
    tot_good = max(g["good"].sum(), 1)
    g["bad_pct"] = g["bad"] / tot_bad
    g["good_pct"] = g["good"] / tot_good
    g["woe"] = np.log((g["bad_pct"] + EPS) / (g["good_pct"] + EPS))
    g["iv"] = (g["bad_pct"] - g["good_pct"]) * g["woe"]
    return g, float(g["iv"].sum())


def _merge_small_bins(bins: pd.Series, y: pd.Series, min_pct: float,
                      min_bad: int = 0) -> pd.Series:
    """合并过小的箱，保证每个箱的 WOE 估计是稳定的。

    两个约束（第二个常被忽略，但对低坏账率数据是致命的）：
    1. 每箱样本占比 >= min_pct
    2. 每箱坏样本数 >= min_bad —— 否则 WOE = ln((bad%/good%)) 的分子接近 0，
       估计会剧烈抖动（0 个坏样本时 WOE 直接被 ε 决定），在小数据集上直接导致
       训练集表现虚高、测试集崩掉。

    合并策略：反复找到"最差的箱"，并入相邻箱，直到满足约束或只剩 2 箱。
    """
    n = len(bins)
    min_count = max(int(min_pct * n), 1)
    codes = bins.astype("category").cat.codes.to_numpy() if hasattr(bins, "cat") \
        else pd.Series(bins).astype("category").cat.codes.to_numpy()
    y_arr = np.asarray(y)

    while True:
        uniq = np.sort(pd.unique(codes))
        if len(uniq) <= 2:
            break
        stats = {int(c): (int(np.sum(codes == c)), int(np.sum((codes == c) & (y_arr == 1))))
                 for c in uniq}
        bad = [c for c, (cnt, nbad) in stats.items()
               if cnt < min_count or nbad < min_bad]
        if not bad:
            break
        # 选"最差"的箱：优先坏样本数最少，其次样本数最少
        c = min(bad, key=lambda k: (stats[k][1], stats[k][0]))
        order = list(uniq)
        pos = order.index(c)
        target = order[pos - 1] if pos > 0 else order[pos + 1]
        codes = np.where(codes == c, target, codes)

    # 重新编码为 0..k-1，保持原有顺序
    uniq = np.sort(pd.unique(codes))
    remap = {int(old): new for new, old in enumerate(uniq)}
    return pd.Series([remap[int(c)] for c in codes], index=bins.index, name=bins.name)


def woe_binning(x: pd.Series, y: pd.Series, n_bins: int = 5,
                min_bin_pct: float = 0.05,
                min_bin_bad: int = 25) -> tuple[pd.Series, pd.DataFrame, float]:
    """对单个数值特征做分箱 + WOE 编码。

    返回 (woe_series, woe_iv_table, iv)

    流程：分位切箱（零值单独成段）→ 合并过小/坏样本过少的箱 → 计算 WOE/IV → 映射回样本。
    """
    x = pd.Series(x).reset_index(drop=True).astype(float)
    y = pd.Series(y).reset_index(drop=True)

    filled = x.fillna(x.median())
    edges = _quantile_edges(filled, n_bins)
    if len(edges) < 2:
        # 常量特征：只有一箱，WOE=0，IV=0
        woe_vals = pd.Series(np.zeros(len(x)))
        table = pd.DataFrame({"total": [len(x)], "bad": [int(y.sum())], "woe": [0.0], "iv": [0.0]})
        return woe_vals, table, 0.0

    b = _bins_from_edges(filled, edges)
    b = _merge_small_bins(b, y, min_bin_pct, min_bin_bad)

    tmp = pd.DataFrame({"bin": b, "y": y.values})
    table, iv = _woe_iv_table(tmp)
    mapping = table["woe"].to_dict()
    woe_vals = b.map(mapping).astype(float)
    # 缺失值单独成一箱时 WOE 可能为 nan —— 用 0 兜底（中性）
    woe_vals = woe_vals.fillna(0.0)
    return woe_vals, table, iv


def iv_of(x: pd.Series, y: pd.Series, n_bins: int = 5, min_bin_pct: float = 0.05) -> float:
    """只要 IV 值（用于特征筛选）。"""
    return woe_binning(x, y, n_bins, min_bin_pct)[2]


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "无用(<0.02)"
    if iv < 0.1:
        return "弱(0.02-0.1)"
    if iv < 0.3:
        return "中(0.1-0.3)"
    if iv < 0.5:
        return "强(0.3-0.5)"
    return "过强(>0.5，怀疑泄漏)"


def scorecard(w: np.ndarray, b: float, feature_names: list[str],
              base_score: float = 600.0, base_odds: float = 20.0,
              pdo: float = 20.0) -> pd.DataFrame:
    """把 logistic 系数翻译成评分卡。

    factor = pdo / ln(2)；offset = base_score - factor * ln(base_odds)
    score = offset - factor * (b + sum(w_i * woe_i))
    """
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    rows = [{"特征": "(基准)", "WOE": np.nan, "系数": np.nan, "得分贡献": round(offset, 2)}]
    for name, wi in zip(feature_names, w):
        rows.append({
            "特征": name,
            "WOE": np.nan,
            "系数": round(float(wi), 4),
            "得分贡献": round(-factor * float(wi), 2),  # WOE 每 +1 的分数变化
        })
    rows.append({"特征": "(截距)", "WOE": np.nan, "系数": round(float(b), 4),
                 "得分贡献": round(-factor * float(b), 2)})
    return pd.DataFrame(rows)


class WOEEncoder:
    """把分箱规则**拟合一次、到处复用**的 WOE 编码器。

    为什么必须做成对象？
    训练/验证/测试三份数据必须用**完全相同**的分箱边界和 WOE 映射，
    否则同一特征在不同集合里含义不同，验证集选出的阈值就会失效。

    内部结构（关键设计）：
    - `edges`            : 原始值空间的箱边界
    - `woe_by_interval`  : 长度 = len(edges)-1 的数组，直接给出**每个原始区间**的 WOE

    为什么用 `woe_by_interval` 而不是 bin→WOE 字典？
    合并小箱会让最终箱数少于原始区间数（例如 5 个区间合成 4 个箱）。
    如果用字典查表，多出来的那个区间码会查不到，一旦用 `fillna(0)` 兜底，
    就可能静默地把一整段样本的 WOE 置成 0，导致信息丢失。
    用数组按区间直接索引后，每个区间都有明确取值，不可能出现查不到的情况。
    """

    def __init__(self, n_bins: int = 5, min_bin_pct: float = 0.05, min_bin_bad: int = 25):
        self.n_bins_target = n_bins
        self.min_bin_pct = min_bin_pct
        self.min_bin_bad = min_bin_bad
        self.rules: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, features: list[str]) -> "WOEEncoder":
        y = pd.Series(y).reset_index(drop=True)
        for c in features:
            x = pd.Series(X[c].to_numpy(dtype=float)).reset_index(drop=True)
            fill_value = float(x.median())
            filled = x.fillna(fill_value)
            edges = _quantile_edges(filled, self.n_bins_target)

            if len(edges) < 2:
                self.rules[c] = {"fill": fill_value, "edges": None,
                                 "woe_by_interval": np.zeros(1), "iv": 0.0,
                                 "n_bins_final": 1}
                continue

            n_intervals = len(edges) - 1
            raw_codes = _bins_from_edges(filled, edges)
            merged_codes = _merge_small_bins(raw_codes, y, self.min_bin_pct, self.min_bin_bad)
            table, iv = _woe_iv_table(pd.DataFrame({"bin": merged_codes, "y": y.values}))
            woe_of_merged = table["woe"].to_dict()

            # 把「合并后的箱」摊平回「原始区间」：每个原始区间都有确定的 WOE
            interval_to_merged = {}
            for raw_code, merged_code in zip(raw_codes.to_numpy(), merged_codes.to_numpy()):
                interval_to_merged.setdefault(int(raw_code), int(merged_code))
            woe_by_interval = np.array(
                [woe_of_merged[interval_to_merged[i]] if i in interval_to_merged else 0.0
                 for i in range(n_intervals)], dtype=float)

            self.rules[c] = {
                "fill": fill_value,
                "edges": edges,
                "woe_by_interval": woe_by_interval,
                "iv": float(iv),
                "n_bins_final": int(table.shape[0]),
            }
        return self

    def transform(self, X: pd.DataFrame, features: list[str]) -> np.ndarray:
        """把 X 编码成 WOE 矩阵（列顺序 = features 顺序）。

        使用训练时保存的**原始值边界**与区间 WOE 数组，
        因此训练/验证/测试/线上对同一个取值给出完全相同的 WOE。
        """
        cols = []
        for c in features:
            rule = self.rules[c]
            x = pd.Series(X[c].to_numpy(dtype=float)).reset_index(drop=True).fillna(rule["fill"])
            if rule["edges"] is None:
                cols.append(np.zeros(len(x)))
                continue
            codes = _bins_from_edges(x, rule["edges"])
            # 超出训练集范围的值（含 NaN 转成的箱外码）落到最近区间，绝不静默置 0
            codes = codes.fillna(0 if len(rule["woe_by_interval"]) else 0).astype(int)
            codes = codes.clip(0, len(rule["woe_by_interval"]) - 1)
            cols.append(rule["woe_by_interval"][codes.to_numpy()])
        return np.column_stack(cols)

    def iv(self, feature: str) -> float:
        return float(self.rules[feature]["iv"])

    def bin_mapping(self, feature: str) -> dict[int, float]:
        """返回 原始区间码 -> WOE（便于把分箱结果写进报告）。"""
        arr = self.rules[feature]["woe_by_interval"]
        return {i: float(v) for i, v in enumerate(arr)}

    def edges(self, feature: str):
        return self.rules[feature]["edges"]

    def n_bins(self, feature: str) -> int:
        return int(self.rules[feature].get("n_bins_final", 0))
