"""Follow-on hypothetical budget experiment; no new model selection or refit."""
import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .decision_study import ROOT,sha,write_json,write_csv,load_frozen,load_rows,signature,packages
from .decision_rules import grouped_batches,capacity_count,paired_group_interval
from .allocation import hypothetical_weights,inputs,feasible,greedy,allocate,accounting


def protocol_check(p):
    if p['models']!=['random_forest','random_forest_sigmoid'] or p['primary_model']!='random_forest_sigmoid':raise ValueError('Inherited RF study required')
    if p['weight_scenarios']!=['unit','hypothetical_heterogeneous']:raise ValueError('Declare unit and hypothetical weights')
    for key in ['funding_fractions','risk_rates']:
        if not p[key] or len(p[key])!=len(set(p[key])) or any(isinstance(x,bool) or not np.isfinite(x) or not 0<=x<=1 for x in p[key]):raise ValueError('Invalid resource grid')
    if not 0<=p['capacity_fraction']<=1 or p['ratio']<=0 or p['time_limit']<=0 or not 0<=p['relative_gain_gap']<=1:raise ValueError('Invalid cost/solve controls')
    if p['primary_funding_fraction'] not in p['funding_fractions'] or p['primary_risk_rate'] not in p['risk_rates']:raise ValueError('Primary case outside grid')
    if not p['batch_sizes'] or any(type(x) is not int or x<1 for x in p['batch_sizes']):raise ValueError('Invalid workloads')
    return p


def study_signature(p,prior):
    previous=json.loads((prior/'selection.json').read_text())
    return {'design':p,'prior_signature':signature(previous['protocol']),
        'prior_selection_sha256':sha(prior/'selection.json'),'prior_test_verification_sha256':sha(prior/'test_verification.json'),
        'allocation_source':sha(ROOT/'src/credit/allocation.py',True),
        'study_source':sha(ROOT/'src/credit/allocation_study.py',True)}


def verify_prior(prior,baseline):
    f=json.loads((prior/'selection.json').read_text());load_frozen(f['protocol'],prior,baseline)
    v=json.loads((prior/'test_verification.json').read_text())
    if sha(prior/'selection.json')!=v['selection_sha256'] or any(sha(prior/n)!=h for n,h in v['files'].items()):raise ValueError('Prior completed evidence changed')
    return f


def run(p,prior,baseline,out):
    if any(out.iterdir()):raise FileExistsError('Use an empty allocation output')
    previous=verify_prior(prior,baseline);stamp=study_signature(p,prior)
    # Freeze before this stage obtains evaluation rows. The cohort is nevertheless
    # already published/inspected; this is not claimed as an untouched holdout.
    write_json(out/'selection.json',{'signature':stamp,'protocol':p,'runtime':packages(),
        'frozen_at':datetime.now(timezone.utc).isoformat(),'input_hashes':previous['input_hashes'],
        'primary_comparison':'MILP minus feasible low-PD greedy in the declared inherited-calibrated primary case',
        'no_test_reselection':True})
    rows=load_rows(baseline,'test');ids=rows.ID.to_numpy();y=rows.y.to_numpy();groups=rows.group.to_numpy()
    weights={'unit':np.ones(len(rows)), 'hypothetical_heterogeneous':hypothetical_weights(groups,p['weight_seed'],p['weight_sigma'])}
    schemes={n:grouped_batches(groups,ids,n,p['batch_seed']) for n in p['batch_sizes']}
    records=[];actions={};actual_vectors={};planning_gains={}
    for size,weight,model,F_rate,R_rate in itertools.product(p['batch_sizes'],p['weight_scenarios'],p['models'],p['funding_fractions'],p['risk_rates']):
        w=weights[weight];prob=rows[model].to_numpy()
        masks={k:np.zeros(len(rows),bool) for k in ['pd_greedy','gain_greedy','milp']}
        gains={k:0. for k in masks}
        for batch,idx in enumerate(schemes[size]):
            K=capacity_count(len(idx),p['capacity_fraction']);F=F_rate*w[idx].sum();R=p['ratio']*R_rate*w[idx].sum()
            local={}
            for algorithm,priority in [('pd_greedy','pd'),('gain_greedy','gain')]:
                start=perf_counter();a=greedy(prob[idx],w[idx],ids[idx],K,F,R,p['ratio'],priority)
                local[algorithm]=(a,{'solver_status':-1,'planning_precision_met':None,'solve_seconds':perf_counter()-start,
                    'gain_upper_bound':None,'relative_gain_gap':None,'absolute_gain_gap':None,'greedy_retained':False,'no_solver_incumbent':False})
            local['milp']=allocate(prob[idx],w[idx],ids[idx],K,F,R,p['ratio'],p['time_limit'],p['relative_gain_gap'])
            *_,gain,A,b=inputs(prob[idx],w[idx],ids[idx],K,F,R,p['ratio'])
            for algorithm,(a,diagnostics) in local.items():
                if not feasible(a,A,b):raise AssertionError('Shared resource constraint failed')
                masks[algorithm][idx]=a
                ac,pc=accounting(y[idx],prob[idx],w[idx],a,p['ratio'])
                value=float(gain@a);gains[algorithm]+=value
                records.append({'batch_size':size,'weight':weight,'model':model,'funding_fraction':F_rate,'risk_rate':R_rate,
                    'batch':batch,'algorithm':algorithm,'n':len(idx),'count_cap':K,'funding_cap':float(F),'predicted_loss_cap':float(R),
                    'approved':int(a.sum()),'funding_used':float(a@w[idx]),'predicted_loss_used':float(a@(p['ratio']*prob[idx]*w[idx])),
                    'realized_loss':float(a@(p['ratio']*y[idx]*w[idx])),'actual_cost_sum':float(ac.sum()),'predicted_cost_sum':float(pc.sum()),
                    **diagnostics,'planning_gain':value})
            if weight=='unit' and abs(local['milp'][1]['planning_gain']-float(gain@local['pd_greedy'][0]))>1e-6:
                raise AssertionError('Unit-weight sorting should match MILP planning optimum')
        for algorithm,a in masks.items():
            key=f'{size}|{weight}|{model}|{F_rate}|{R_rate}|{algorithm}'
            actions[key]=a;actual_vectors[key]=accounting(y,prob,w,a,p['ratio'])[0];planning_gains[key]=gains[algorithm]
        write_csv(pd.DataFrame(records),out/'progress.csv')
        print(f'allocation: batch={size} {weight} {model} funding={F_rate} risk={R_rate}; {len(records)} batch cells',flush=True)
    df=pd.DataFrame(records)
    write_csv(df,out/'batch_allocations.csv');(out/'progress.csv').unlink()
    effects=[]
    for size,weight,model,F_rate,R_rate in itertools.product(p['batch_sizes'],p['weight_scenarios'],p['models'],p['funding_fractions'],p['risk_rates']):
        prefix=f'{size}|{weight}|{model}|{F_rate}|{R_rate}|'
        for reference in ['pd_greedy','gain_greedy']:
            delta=actual_vectors[prefix+'milp']-actual_vectors[prefix+reference]
            mean,lo,hi=paired_group_interval(delta,groups,p['bootstrap_repeats'],p['bootstrap_seed'])
            effects.append({'batch_size':size,'weight':weight,'model':model,'funding_fraction':F_rate,'risk_rate':R_rate,
                'reference':reference,'milp_minus_reference_actual_cost':mean,'ci95_low':lo,'ci95_high':hi,
                'planning_gain_improvement':planning_gains[prefix+'milp']-planning_gains[prefix+reference],
                'different_rows':int((actions[prefix+'milp']!=actions[prefix+reference]).sum()),
                'primary_case':bool(size==p['primary_batch_size'] and weight==p['primary_weight'] and model==p['primary_model']
                    and F_rate==p['primary_funding_fraction'] and R_rate==p['primary_risk_rate'] and reference=='pd_greedy')})
    write_csv(pd.DataFrame(effects),out/'paired_effects.csv')
    (out/'local').mkdir(exist_ok=True);np.savez_compressed(out/'local/test_actions.npz',**actions)
    if study_signature(p,prior)!=stamp:raise ValueError('Source/prior study changed during allocation')
    verify_prior(prior,baseline)
    write_json(out/'test_verification.json',{'selection_sha256':sha(out/'selection.json'),
        'files':{n:sha(out/n) for n in ['batch_allocations.csv','paired_effects.csv']},
        'private_actions_sha256':sha(out/'local/test_actions.npz'),'test_rows':len(rows),'no_test_reselection':True})
    report(p,out)


def report(p,out):
    f=json.loads((out/'selection.json').read_text());v=json.loads((out/'test_verification.json').read_text())
    if sha(out/'selection.json')!=v['selection_sha256'] or any(sha(out/n)!=h for n,h in v['files'].items()):raise ValueError('Allocation evidence changed')
    df=pd.read_csv(out/'batch_allocations.csv',float_precision='round_trip');effects=pd.read_csv(out/'paired_effects.csv',float_precision='round_trip')
    keys=['batch_size','weight','model','funding_fraction','risk_rate','algorithm']
    summary=df.groupby(keys,as_index=False).agg(batches=('batch','size'),rows=('n','sum'),approved=('approved','sum'),
        actual_cost=('actual_cost_sum','sum'),predicted_cost=('predicted_cost_sum','sum'),planning_gain=('planning_gain','sum'),
        solve_seconds=('solve_seconds','sum'))
    summary['actual_cost_per_row']=summary.actual_cost/summary.rows
    summary['predicted_cost_per_row']=summary.predicted_cost/summary.rows
    write_csv(summary,out/'summary.csv')
    text=['# Hypothetical shared funding and expected-loss allocation','',p['scope'],'',
        'All cost weights are either one or seeded, clipped lognormal scenario weights, shared by duplicate financial groups. They are not observed EAD, loan amounts or losses.',
        'An approved default costs 5*weight, a rejected nondefault costs weight. Funding uses weight; predicted loss uses 5*PD*weight. Count, funding and predicted-risk caps are checked for every algorithm.',
        'Only upper bounds are imposed, so reject-all is feasible. No infeasibility or guaranteed realized-loss claim follows from a tight budget.', '',
        '## Predeclared primary comparison','','```',effects[effects.primary_case].to_string(index=False),'```','',
        '## All conditional paired effects','','```',effects.to_string(index=False),'```','',
        'Positive planning gain improvement uses the same probabilities and feasible set. It is not a realized-cost gain. Actual-cost differences use common historical labels; negative favors MILP.',
        'Bootstrap intervals resample duplicate groups with fixed decision memberships, not refits, new optimized portfolios or new future cohorts. Batches are artificial, overlapping workload schemes of the same people.',
        'Unit weights admit an exact sorted positive-benefit prefix; MILP is unnecessary. Heterogeneous weights can create discrete packing effects; gain per funding unit equals 1-6*PD here, so it is the same ordering as PD rather than an extra distinct baseline.',
        'The MILP preserves the best checked PD/gain-greedy solution if its returned candidate is absent or worse. Reported bounds certify only the forecast-conditioned planning problem at its configured tolerance.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    write_json(out/'manifest.json',{'signature':f['signature'],'runtime':f['runtime'],
        'files':{x.name:sha(x) for x in sorted(out.iterdir()) if x.is_file() and x.name!='manifest.json'},
        'private_rows_published':False,'weights_are_hypothetical':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/allocation_study.json')
    parser.add_argument('--baseline',type=Path,default=ROOT/'reports/generated/decision_baseline')
    parser.add_argument('--prior',type=Path,default=ROOT/'reports/decision_study/2026-10-27')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=protocol_check(json.loads(args.protocol.read_text()));out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    run(p,args.prior.resolve(),args.baseline.resolve(),out)


if __name__=='__main__':main()
