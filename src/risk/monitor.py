"""Feature and score distribution monitoring against the fitting sample."""
import numpy as np
import pandas as pd
from .config import CFG
from .metrics import psi

def judge(value: float) -> str:
    return "显著漂移" if value >= CFG.psi_alert else ("需关注" if value >= CFG.psi_warn else "稳定")

def monitor_report(train_df, current_df, features, train_scores=None, current_scores=None):
    if (train_scores is None) != (current_scores is None):
        raise ValueError("Both score arrays must be supplied together")
    rows = []
    for name in features:
        value = psi(train_df[name].to_numpy(), current_df[name].to_numpy(), bins=CFG.psi_bins)
        rows.append({"feature": name, "psi": value, "judge": judge(value)})
    if train_scores is not None:
        value = psi(np.asarray(train_scores), np.asarray(current_scores), bins=CFG.psi_bins)
        rows.append({"feature": "__score__", "psi": value, "judge": judge(value)})
    table = pd.DataFrame(rows, columns=["feature", "psi", "judge"])
    table = table.sort_values("psi", ascending=False).reset_index(drop=True)
    return table, bool((table["judge"] == "显著漂移").any())
