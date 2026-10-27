"""Verify published aggregate evidence; optionally replay private row decisions."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.credit.decision_study import sha,load_frozen,load_rows,batch_schemes
from src.credit.decision_rules import decision_summary


def number_key(value):
    return str(int(value)) if float(value).is_integer() else str(value)


def verify(out,baseline=None):
    out=Path(out);f=json.loads((out/'selection.json').read_text());p=f['protocol']
    load_frozen(p,out,baseline)
    m=json.loads((out/'manifest.json').read_text())
    if m['signature']!=f['signature']:raise ValueError('Source manifest inconsistent')
    if any(sha(out/n)!=v for n,v in m['files'].items()):raise ValueError('Published artifact changed')
    if (out/'figure_manifest.json').exists():
        figure=json.loads((out/'figure_manifest.json').read_text())
        if figure['script_sha256']!=sha(ROOT/'scripts/plot_decision_research.py',True):raise ValueError('Figure source changed')
    v=json.loads((out/'test_verification.json').read_text())
    if v['selection_sha256']!=sha(out/'selection.json'):raise ValueError('Selection changed')
    b=pd.read_csv(out/'batch_decisions.csv',float_precision='round_trip')
    hard=b[b.rule!='validation_cutoff']
    if (hard.approved>hard.cap).any() or hard.excess_over_cap.sum()!=0:raise ValueError('Hard capacity violated')
    result={'public_checks':'passed','batch_decision_cells':len(b),'hard_capacity_cells':len(hard),
        'hard_capacity_violations':0,'private_row_replay':False,'published_test_rows':v['test_rows']}
    if baseline is not None:
        if sha(out/'local/test_actions.npz')!=v['private_actions_sha256']:raise ValueError('Private masks changed')
        rows=load_rows(baseline,'test');actions=np.load(out/'local/test_actions.npz');ids=rows.ID.to_numpy();y=rows.y.to_numpy()
        costs=pd.read_csv(out/'cost_rules.csv',float_precision='round_trip')
        for r in costs.itertuples():
            a=actions[f'{r.model}|whole|{r.rule}|{number_key(r.ratio)}']
            np.testing.assert_array_equal(a,rows[r.model].to_numpy()<r.threshold)
            s=decision_summary(y,rows[r.model],a,r.ratio,ids)
            for key in ['actual_cost','predicted_cost','approved','approved_defaults']:
                if abs(s[key]-getattr(r,key))>1e-10:raise ValueError('Per-row cost accounting differs')
        schemes=batch_schemes(rows,p)
        for r in b.itertuples():
            idx=schemes[r.scheme][r.batch];prob=rows[r.model].to_numpy()[idx]
            a=actions[f'{r.model}|{r.scheme}|{r.rule}|{r.fraction}'][idx]
            if r.rule=='validation_cutoff':
                expected=prob<f['capacity_thresholds'][r.model][str(r.fraction)]
            else:
                order=sorted(range(len(idx)),key=lambda j:(prob[j],ids[idx[j]]))[:r.cap]
                if r.rule=='cost_then_capacity':order=[j for j in order if prob[j]<1/(1+p['primary_ratio'])]
                expected=np.zeros(len(idx),bool);expected[order]=True
            np.testing.assert_array_equal(a,expected)
            s=decision_summary(y[idx],prob,a,p['primary_ratio'],ids[idx])
            if s['approval_ids_sha256']!=r.approval_ids_sha256 or abs(s['actual_cost']-r.actual_cost)>1e-10:
                raise ValueError('Batch row decisions differ from published aggregates')
        agreement=pd.read_csv(out/'action_agreement.csv')
        for r in agreement.itertuples():
            if (actions[f'{r.model}|{r.decision}']!=actions[f'{r.model}_sigmoid|{r.decision}']).sum()!=r.different_rows:
                raise ValueError('Published action mismatch count differs')
        result.update(private_row_replay=True,cost_rules_replayed=len(costs),capacity_rows_replayed=len(b),
            action_agreements_replayed=len(agreement))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--baseline',type=Path)
    args=p.parse_args();print(json.dumps(verify(args.run_dir,args.baseline),indent=2))


if __name__=='__main__':main()
