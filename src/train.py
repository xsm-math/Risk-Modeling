"""端到端训练流水线：清洗 → 分箱/WOE → 建模 → 评估 → 阈值优化 → 报告落盘。

跑法：
    python -m src.train                # 默认配置
    python -m src.train --verbose      # 打印每 50 轮损失
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # 支持 python src/train.py 直接跑
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk.config import CFG  # noqa: E402
from src.risk.data import load_data  # noqa: E402
from src.risk.features import WOEEncoder, iv_strength, scorecard  # noqa: E402
from src.risk.metrics import (  # noqa: E402
    auc_rank,
    auc_roc,
    best_threshold,
    confusion_at_threshold,
    expected_cost,
    ks_stat,
    psi,
)
from src.risk.monitor import monitor_report
from src.risk.model import LogisticRegressionGD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
LABEL = "is_default"
ID_COLS = ["customer_id"]


# --------------------------------------------------------------------- 清洗
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply fixed domain rules; imputation is fitted later on the fitting set only."""
    df = df.drop_duplicates(subset=["customer_id"]).copy().reset_index(drop=True)
    df.loc[df["age"] > 100, "age"] = np.nan
    df["debt_ratio"] = df["debt_ratio"].clip(upper=1.2)
    return df


def stratified_split(y: pd.Series, test_size: float, seed: int):
    """分层切分：保证训练/测试的坏样本率一致（否则指标不可比）。"""
    rng = np.random.default_rng(seed)
    idx_bad = np.where(y.to_numpy() == 1)[0]
    idx_good = np.where(y.to_numpy() == 0)[0]
    rng.shuffle(idx_bad)
    rng.shuffle(idx_good)
    n_test_bad = int(len(idx_bad) * test_size)
    n_test_good = int(len(idx_good) * test_size)
    test_idx = np.r_[idx_bad[:n_test_bad], idx_good[:n_test_good]]
    train_idx = np.r_[idx_bad[n_test_bad:], idx_good[n_test_good:]]
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


# --------------------------------------------------------------------- 主流程
def run(verbose: bool = False) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = clean(load_data())
    features = [c for c in df.columns if c not in ID_COLS + [LABEL]]
    y = df[LABEL]

    train_idx, test_idx = stratified_split(y, CFG.test_size, CFG.seed)
    X_train_raw = df.loc[train_idx, features]
    X_test_raw = df.loc[test_idx, features]
    y_train = y.loc[train_idx].to_numpy()
    y_test = y.loc[test_idx].to_numpy()

    # ---- 训练集内部再切验证集（用于选阈值，避免用测试集调参）
    fit_pos, val_pos = stratified_split(pd.Series(y_train), CFG.val_size, CFG.seed + 1)
    y_fit = y_train[fit_pos]
    y_val = y_train[val_pos]
    medians = X_train_raw.iloc[fit_pos].median().fillna(0.0)
    X_train_raw = X_train_raw.fillna(medians)
    X_test_raw = X_test_raw.fillna(medians)
    enc_all = WOEEncoder(n_bins=CFG.max_bins, min_bin_pct=CFG.min_bin_pct,
                         min_bin_bad=CFG.min_bin_bad)
    enc_all.fit(X_train_raw.iloc[fit_pos], pd.Series(y_fit), features)

    iv_rows = [{
        "feature": c,
        "iv": round(enc_all.iv(c), 4),
        "strength": iv_strength(enc_all.iv(c)),
    } for c in features]
    iv_df = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)

    kept = [c for c in features if enc_all.iv(c) >= CFG.iv_threshold]
    if not kept:
        raise RuntimeError("所有特征 IV 都低于阈值，请调低 CFG.iv_threshold")

    # 入模特征使用同一个 encoder（rules 已按特征名索引）
    enc = enc_all

    # ---- 三份数据复用同一个已拟合编码器
    Xw_fit = enc.transform(X_train_raw.iloc[fit_pos], kept)
    Xw_val = enc.transform(X_train_raw.iloc[val_pos], kept)
    Xw_test = enc.transform(X_test_raw, kept)

    # ---- 建模
    model = LogisticRegressionGD(lr=CFG.lr, epochs=CFG.epochs, l2=CFG.l2,
                                 tol=CFG.tol, standardize=True, verbose=verbose)
    model.feature_names = kept
    model.fit(Xw_fit, y_fit, Xw_val, y_val)

    p_val = model.predict_proba(Xw_val)
    p_test = model.predict_proba(Xw_test)

    # ---- 评估
    auc_val, auc_test = auc_roc(y_val, p_val), auc_roc(y_test, p_test)
    auc_rank_test = auc_rank(y_test, p_test)  # 交叉验证两种 AUC 实现一致
    ks_val, ks_thr = ks_stat(y_val, p_val)
    ks_test, _ = ks_stat(y_test, p_test)

    # ---- 阈值优化（在验证集上选，在测试集上报告）
    thr_best, cost_val_best = best_threshold(y_val, p_val,
                                             CFG.total_cost_fp, CFG.total_cost_fn)
    m_default = confusion_at_threshold(y_test, p_test, 0.5)
    m_best = confusion_at_threshold(y_test, p_test, thr_best)
    cost_default = expected_cost(y_test, p_test, 0.5, CFG.total_cost_fp, CFG.total_cost_fn)
    cost_best = expected_cost(y_test, p_test, thr_best, CFG.total_cost_fp, CFG.total_cost_fn)
    cost_approve_all = expected_cost(y_test, p_test, 1.01, CFG.total_cost_fp, CFG.total_cost_fn)
    cost_reject_all = expected_cost(y_test, p_test, -0.01, CFG.total_cost_fp, CFG.total_cost_fn)

    # Reference distributions come exclusively from the fitting set.
    fit_raw = X_train_raw.iloc[fit_pos]
    psi_df, alarm = monitor_report(fit_raw, X_test_raw, features,
                                  model.predict_proba(Xw_fit), p_test)
    shifted = X_test_raw.copy()
    shifted["monthly_income"] *= 1.3
    drift_df, drift_alarm = monitor_report(fit_raw, shifted, features,
        model.predict_proba(Xw_fit), model.predict_proba(enc.transform(shifted, kept)))
    drift_df.to_csv(OUT_DIR / "drift_simulation.csv", index=False)

    # ---- 落盘
    (OUT_DIR / "iv_table.csv").write_text(iv_df.to_csv(index=False, lineterminator=chr(10)), encoding="utf-8-sig")
    (OUT_DIR / "psi_table.csv").write_text(psi_df.to_csv(index=False, lineterminator=chr(10)), encoding="utf-8-sig")
    raw_w, raw_b = model.raw_parameters()
    sc = scorecard(raw_w, raw_b, kept)
    (OUT_DIR / "scorecard.csv").write_text(sc.to_csv(index=False, lineterminator=chr(10)), encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "customer_id": df.loc[test_idx, "customer_id"].to_numpy(),
        "y_true": y_test,
        "prob": p_test,
        "score": 600 - 20 / np.log(2) * (np.log(20) + Xw_test @ raw_w + raw_b),
        "decision": np.where(p_test >= thr_best, "拒绝", "通过"),
    })
    (OUT_DIR / "predictions.csv").write_text(
        pred_df.to_csv(index=False, lineterminator=chr(10), float_format="%.6f"), encoding="utf-8-sig")

    summary = {
        "data_source": "synthetic logistic DGP; not real customer data",
        "seed": CFG.seed, "n_fit": len(fit_pos), "n_validation": len(val_pos),
        "monitor_alarm": alarm, "drift_simulation_alarm": drift_alarm,
        "n_total": int(len(df)), "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "bad_rate_all": round(float(y.mean()), 4),
        "bad_rate_train": round(float(y_train.mean()), 4),
        "bad_rate_test": round(float(y_test.mean()), 4),
        "n_features_kept": len(kept),
        "features_kept": kept,
        "features_dropped": [c for c in features if c not in kept],
        "auc_val": round(auc_val, 4), "auc_test": round(auc_test, 4),
        "auc_rank_check": round(auc_rank_test, 4),
        "ks_val": round(ks_val, 4), "ks_test": round(ks_test, 4), "ks_threshold": round(ks_thr, 4),
        "threshold_05": 0.5,
        "threshold_best": float(thr_best),
        "cost_assumption": {
            "cost_false_reject_good_customer": CFG.total_cost_fp,
            "cost_approve_bad_customer": CFG.total_cost_fn,
        },
        "cost_approve_all": round(cost_approve_all, 2),
        "cost_reject_all": round(cost_reject_all, 2),
        "cost_threshold_05": round(cost_default, 2),
        "cost_threshold_best": round(cost_best, 2),
        "saving_vs_05": round(cost_default - cost_best, 2),
        "saving_vs_05_pct": round((cost_default - cost_best) / cost_default * 100, 2) if cost_default else 0.0,
        "confusion_threshold_05": {k: round(v, 4) if isinstance(v, float) else v for k, v in m_default.items()},
        "confusion_threshold_best": {k: round(v, 4) if isinstance(v, float) else v for k, v in m_best.items()},
        "psi_max": round(float(psi_df["psi"].max()), 4),
        "psi_alert_features": psi_df.loc[psi_df["psi"] >= CFG.psi_alert, "feature"].tolist(),
        "psi_warn_features": psi_df.loc[psi_df["psi"] >= CFG.psi_warn, "feature"].tolist(),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(summary, iv_df, psi_df, sc, model, enc,
                 X_train_raw.iloc[fit_pos], y_fit)
    return summary


def write_report(summary, iv_df, psi_df, sc, model, enc, X_fit, y_fit):
    s = summary
    def table(df):
        return "| " + " | ".join(df.columns) + " |\n| " + " | ".join(["---"] * len(df.columns)) + " |\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in df.round(4).itertuples(index=False, name=None))
    report = f"""# 信用风险建模实验报告

由 `python -m src.train` 生成。数据为固定随机种子的模拟样本，成本为实验假设，不代表真实业务收益。

## 数据与实验设计

共 {s['n_total']} 条样本，坏样本率 {s['bad_rate_all']:.2%}。
拟合集 {s['n_fit']} / 验证集 {s['n_validation']} / 测试集 {s['n_test']}。
按标签分层随机切分。固定规则修正年龄、截尾负债率、按客户去重；中位数、WOE/IV 和模型仅在拟合集学习。
验证集仅用于阈值网格选择，测试集用于本次最终评估。历史版本曾按同一数据的测试结果挑选分箱配置，因此本结果不能视作从未使用过的独立验证。

## 排序能力与成本策略

| 指标 | 结果 |
| --- | --- |
| 测试 AUC | {s['auc_test']:.4f} |
| 测试 KS（最大 TPR-FPR） | {s['ks_test']:.4f} |
| 验证 AUC | {s['auc_val']:.4f} |
| 保留特征数 | {s['n_features_kept']} |
| 验证集选择阈值 | {s['threshold_best']:.6f} |
| 0.5 阈值测试成本 | {s['cost_threshold_05']:,.0f} |
| 所选阈值测试成本 | {s['cost_threshold_best']:,.0f} |
| 相对 0.5 成本变化（降低为正） | {s['saving_vs_05_pct']:.2f}% |
| 全通过 / 全拒绝成本 | {s['cost_approve_all']:,.0f} / {s['cost_reject_all']:,.0f} |

成本 = 400 × FP + 8200 × FN；这是已知标签上的离线成本核算。阈值搜索为 199 个网格点，不能保证连续域最优。
所选策略通过率 {s['confusion_threshold_best']['approval_rate']:.2%}，坏样本召回率 {s['confusion_threshold_best']['recall_tpr']:.2%}。

## 特征筛选

{table(iv_df)}

## 评分卡

先将标准化系数还原至原始 WOE 尺度，再计算 score = 600 - 20/ln(2) × [ln(20) + logit(p)]。
好坏比 20:1 对应 600 分，好坏比翻倍加 20 分。CSV 中的贡献为 WOE 每增加 1 的分数变化，不是各箱的最终分值。

{table(sc.drop(columns=['WOE']))}

## 分布监控

{table(psi_df)}

PSI ≥ 0.10 关注，≥ 0.25 告警。阈值是演示规则，告警后需要排查原因。
收入乘 1.3 的漂移实验告警：{s['drift_simulation_alarm']}，详见 `drift_simulation.csv`。
随机切分的 PSI 只能说明两份样本的分布差异，不能证明跨期稳定或线上表现。

## 局限

- 模拟数据来自 logistic 机制，对逻辑回归有利，不能据此判断真实业务表现。
- 测试坏样本较少，单次切分的指标和成本改善存在较大不确定性。
- 历史分箱配置接触过测试结果；后续需要新数据、嵌套验证或独立时间外验证。
- 尚无真实数据验证、拒绝推断、概率校准、模型持久化或线上部署。
- WOE 不强制单调；分箱合并到两箱即停止，可能仍不满足最小坏样本数。
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproducible synthetic credit risk experiment")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(verbose=args.verbose), ensure_ascii=False, indent=2))
