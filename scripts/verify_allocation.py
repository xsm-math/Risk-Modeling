"""Audit allocation resource ledgers and optionally replay private memberships."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.credit.allocation_study import study_signature
from src.credit.decision_study import sha,load_rows
from src.credit.decision_rules import grouped_batches,paired_group_interval
from src.credit.allocation import hypothetical_weights
from scripts.verify_decisions import verify as verify_prior


def verify(out,prior,baseline=None):
    out=Path(out);prior=Path(prior);verify_prior(prior,baseline)
    f=json.loads((out/'selection.json').read_text());p=f['protocol'];m=json.loads((out/'manifest.json').read_text())
    if f['signature']!=study_signature(p,prior) or m['signature']!=f['signature']:raise ValueError('Allocation source/prior changed')
    if any(sha(out/n)!=v for n,v in m['files'].items()):raise ValueError('Published allocation artifact changed')
    if (out/'figure_manifest.json').exists():
        figure=json.loads((out/'figure_manifest.json').read_text())
        if figure['script_sha256']!=sha(ROOT/'scripts/plot_decision_research.py',True):raise ValueError('Figure source changed')
    v=json.loads((out/'test_verification.json').read_text())
    if sha(out/'selection.json')!=v['selection_sha256']:raise ValueError('Frozen design changed')
    d=pd.read_csv(out/'batch_allocations.csv',float_precision='round_trip')
    for used,cap in [('approved','count_cap'),('funding_used','funding_cap'),('predicted_loss_used','predicted_loss_cap')]:
        if (d[used]>d[cap]+1e-7).any():raise ValueError('Shared resource violation')
    keys=['batch_size','weight','model','funding_fraction','risk_rate','batch']
    g=d.pivot(index=keys,columns='algorithm',values='planning_gain')
    if (g.milp+1e-7<g[['pd_greedy','gain_greedy']].max(1)).any():raise ValueError('MILP worsened checked greedy')
    unit=g.loc[g.index.get_level_values('weight')=='unit']
    if (unit.milp-unit.pd_greedy).abs().max()>1e-6:raise ValueError('Unit prefix optimum not reproduced')
    milp=d[d.algorithm=='milp']
    if not milp.planning_precision_met.eq(True).all():raise ValueError('MILP precision gate unmet')
    result={'public_checks':'passed','allocation_batch_cells':len(d),'milp_problems':len(milp),
        'resource_violations':0,'milp_precision_unmet':0,'unit_prefix_matches_planning_gain':True,'private_row_replay':False}
    if baseline is not None:
        if sha(out/'local/test_actions.npz')!=v['private_actions_sha256']:raise ValueError('Private allocation masks changed')
        rows=load_rows(baseline,'test');actions=np.load(out/'local/test_actions.npz');y=rows.y.to_numpy();groups=rows.group.to_numpy()
        weights={'unit':np.ones(len(rows)),'hypothetical_heterogeneous':hypothetical_weights(groups,p['weight_seed'],p['weight_sigma'])}
        schemes={n:grouped_batches(groups,rows.ID,n,p['batch_seed']) for n in p['batch_sizes']}
        def key(r,algorithm):return f'{r.batch_size}|{r.weight}|{r.model}|{r.funding_fraction}|{r.risk_rate}|{algorithm}'
        for r in d.itertuples():
            idx=schemes[r.batch_size][r.batch];a=actions[key(r,r.algorithm)][idx]
            w=weights[r.weight][idx];prob=rows[r.model].to_numpy()[idx];label=y[idx]
            actual=float(w[(~a)&(label==0)].sum()+p['ratio']*w[a&(label==1)].sum())
            predicted=float((w*((~a)*(1-prob)+a*p['ratio']*prob)).sum())
            for observed,recorded in [(a.sum(),r.approved),(a@w,r.funding_used),
                (a@(p['ratio']*prob*w),r.predicted_loss_used),(actual,r.actual_cost_sum),(predicted,r.predicted_cost_sum)]:
                if abs(observed-recorded)>1e-7:raise ValueError('Private allocation ledger differs')
        effects=pd.read_csv(out/'paired_effects.csv',float_precision='round_trip')
        for r in effects.itertuples():
            w=weights[r.weight];label=y
            aa=actions[key(r,'milp')];bb=actions[key(r,r.reference)]
            ca=w*((~aa)*(1-label)+aa*p['ratio']*label);cb=w*((~bb)*(1-label)+bb*p['ratio']*label)
            mean,lo,hi=paired_group_interval(ca-cb,groups,p['bootstrap_repeats'],p['bootstrap_seed'])
            np.testing.assert_allclose([mean,lo,hi],[r.milp_minus_reference_actual_cost,r.ci95_low,r.ci95_high],rtol=0,atol=1e-10)
        result.update(private_row_replay=True,paired_effects_recomputed=len(effects))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--prior',type=Path,default=ROOT/'reports/decision_study/2026-10-27');parser.add_argument('--baseline',type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.run_dir,args.prior,args.baseline),indent=2))


if __name__=='__main__':main()
