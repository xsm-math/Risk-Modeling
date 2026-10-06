"""Execute SQL on observed UCI histories and verify parity with Python features."""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from .data import STATUS, BILLS, PAYMENTS, features


def execute(df, sql_path=None):
    histories=[]
    for k,(status,bill,payment) in enumerate(zip(STATUS,BILLS,PAYMENTS)):
        histories.append(pd.DataFrame({'customer_id':df.ID.to_numpy(), 'period_index':6-k,
            'status':df[status].to_numpy(), 'bill_amount':df[bill].to_numpy(),
            'payment_amount':df[payment].to_numpy(), 'credit_limit':df.LIMIT_BAL.to_numpy()}))
    path=Path(sql_path or Path(__file__).resolve().parents[2]/'sql/behavior_features.sql')
    with sqlite3.connect(':memory:') as con:
        pd.concat(histories,ignore_index=True).to_sql('credit_history',con,index=False)
        result=pd.read_sql_query(path.read_text(encoding='utf-8'),con)
    return result


def validate(df, out):
    result=execute(df).set_index('customer_id');python=features(df).set_index(df.ID).sort_index()
    names=['utilization_latest','utilization_mean','delayed_months','max_delay','payment_bill_ratio']
    rows=[]
    for name in names:
        np.testing.assert_allclose(result[name],python[name],rtol=1e-12,atol=1e-12)
        rows.append({'feature':name,'rows_checked':len(df),
                     'max_absolute_error':float(np.max(np.abs(result[name]-python[name])))})
    pd.DataFrame(rows).to_csv(out/'sql_parity.csv',index=False)
    # Publish only aggregate SQL outputs; no customer histories.
    result[['delayed_months_3','payment_total_3']].describe().to_csv(out/'sql_aggregate_summary.csv')
    return rows
