"""Retrospective calibration/action study with frozen rules and hard batch caps."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import choose_threshold,approval_threshold,metrics
from .decision_rules import (hard_capacity,grouped_batches,decision_summary,cost_rows,
    paired_group_interval,order_diagnostics,capacity_count)

ROOT=Path(__file__).resolve().parents[2]
SOURCE_FILES=['src/credit/data.py','src/credit/models.py','src/credit/evaluation.py',
    'src/credit/experiment.py','src/risk/features.py','src/credit/decision_rules.py','src/credit/decision_study.py']


def sha(path,normalize=False):
    value=Path(path).read_bytes()
    return hashlib.sha256(value.replace(b'\r\n',b'\n') if normalize else value).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def write_csv(frame,path):
    frame.to_csv(path,index=False,float_format='%.17g',lineterminator='\n')


def signature(p):
    return {'protocol':hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),
        'sources':{n:sha(ROOT/n,True) for n in SOURCE_FILES},'baseline_config':sha(ROOT/'configs/credit.json',True)}


def packages():
    return {n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','xgboost']}


def validate_protocol(p):
    if p['models']!=['woe_lr','woe_lr_sigmoid','random_forest','random_forest_sigmoid']:raise ValueError('Four declared model variants required')
    for key in ['cost_ratios','capacity_fractions','unit_risk_rates','batch_sizes']:
        if not p[key] or len(set(p[key]))!=len(p[key]) or any(isinstance(v,bool) or not np.isfinite(v) for v in p[key]):raise ValueError('Invalid design grid')
    if any(x<=0 for x in p['cost_ratios']) or p['primary_ratio'] not in p['cost_ratios']:raise ValueError('Invalid costs')
    if any(not 0<=x<=1 for x in p['capacity_fractions']+p['unit_risk_rates']+[p['risk_capacity_fraction']]):raise ValueError('Invalid capacity/risk rate')
    if any(type(x) is not int or x<1 for x in p['batch_sizes']):raise ValueError('Invalid batch size')
    if type(p['bootstrap_repeats']) is not int or p['bootstrap_repeats']<2:raise ValueError('Invalid bootstrap')
    return p


def input_hashes(baseline):
    names=['manifest.json','selection.json','test_metrics.csv','model_policies.csv',
        'local/processed/split_manifest.csv','local/processed/validation_predictions.csv','local/processed/test_predictions.csv']
    return {n:sha(baseline/n,True) for n in names}


def load_rows(baseline,stage):
    rows=pd.read_csv(baseline/f'local/processed/{stage}_predictions.csv',float_precision='round_trip')
    manifest=pd.read_csv(baseline/'local/processed/split_manifest.csv',dtype={'group':str})
    rows=rows.merge(manifest[['ID','group','split']],on='ID',validate='one_to_one')
    if rows.empty or not rows.split.eq(stage).all() or not rows.ID.is_unique:raise ValueError('Prediction/split IDs inconsistent')
    return rows


def baseline_check(baseline):
    old=pd.read_csv(ROOT/'reports/credit/test_metrics.csv').set_index('model')
    new=pd.read_csv(baseline/'test_metrics.csv').set_index('model').reindex(old.index)
    error=float(np.max(np.abs(old.to_numpy(float)-new.to_numpy(float))))
    before=json.loads((ROOT/'reports/credit/manifest.json').read_text())
    after=json.loads((baseline/'manifest.json').read_text())
    if error>1e-12 or before['split_manifest_sha256']!=after['split_manifest_sha256']:
        raise ValueError('Historical baseline did not reproduce; review before extension')
    for name,value in before['code_sha256'].items():
        if sha(ROOT/name,True)!=value:raise ValueError('Original benchmark source changed')
    return {'maximum_metric_difference':error,'split_manifest_sha256':after['split_manifest_sha256'],
        'source_archive_sha256':after['source_sha256'],'runtime':after['runtime_packages'],
        'original_selected_model':after['selection']['model'],'historical_split_reused':True}


def validate_stage(p,baseline,out):
    if any(out.iterdir()):raise FileExistsError('Use an empty study output')
    stamp=signature(p);inputs=input_hashes(baseline);reproduced=baseline_check(baseline)
    rows=load_rows(baseline,'validation');cuts={};caps={};diagnostics=[]
    for name in p['models']:
        cuts[name]={str(r):choose_threshold(rows.y,rows[name],r) for r in p['cost_ratios']}
        caps[name]={str(q):approval_threshold(rows[name],q) for q in p['capacity_fractions']}
    for base in ['woe_lr','random_forest']:
        diagnostics.append({'model':base,**order_diagnostics(rows[base],rows[base+'_sigmoid'],rows.ID)})
    write_csv(pd.DataFrame(diagnostics),out/'validation_order.csv')
    write_json(out/'baseline_reproduction.json',reproduced)
    if stamp!=signature(p) or inputs!=input_hashes(baseline):raise ValueError('Inputs changed during validation')
    write_json(out/'selection.json',{'protocol':p,'signature':stamp,'input_hashes':inputs,'runtime':packages(),
        'frozen_at':datetime.now(timezone.utc).isoformat(),'cost_thresholds':cuts,'capacity_thresholds':caps,
        'tie_rule':'Hard capacity: ascending PD then immutable ID; may split equal-score blocks. Historical thresholds keep complete blocks.',
        'validation_files':{n:sha(out/n) for n in ['validation_order.csv','baseline_reproduction.json']}})


def load_frozen(p,out,baseline=None):
    f=json.loads((out/'selection.json').read_text())
    if f['signature']!=signature(p) or f['protocol']!=p:raise ValueError('Frozen study source/protocol changed')
    if any(sha(out/n)!=v for n,v in f['validation_files'].items()):raise ValueError('Validation evidence changed')
    if baseline is not None and f['input_hashes']!=input_hashes(baseline):raise ValueError('Private input predictions changed')
    return f


def batch_schemes(rows,p):
    return {'whole':[np.arange(len(rows))],**{f'groups_{n}':grouped_batches(rows.group,rows.ID,n,p['batch_seed']) for n in p['batch_sizes']}}


def test_stage(p,baseline,out):
    if (out/'test_metrics.csv').exists():raise FileExistsError('Evaluation evidence already exists')
    frozen=load_frozen(p,out,baseline)
    if frozen['runtime']!=packages():raise ValueError('Runtime changed between validation and evaluation')
    rows=load_rows(baseline,'test');all_metrics=[];costs=[];batch_rows=[];order=[];effects=[];agreements=[];risk_rows=[];actions={}
    ids=rows.ID.to_numpy();y=rows.y.to_numpy();groups=rows.group.to_numpy()
    for name in p['models']:
        prob=rows[name].to_numpy();all_metrics.append({'model':name,**metrics(y,prob)})
        for r in p['cost_ratios']:
            thresholds={'validation_min_cost':frozen['cost_thresholds'][name][str(r)],'probability_cost_rule':1/(1+r),'threshold_0.5':.5}
            for rule,t in thresholds.items():
                a=prob<t;actions[f'{name}|whole|{rule}|{r}']=a
                costs.append({'model':name,'ratio':r,'rule':rule,'threshold':t,**decision_summary(y,prob,a,r,ids)})
    schemes=batch_schemes(rows,p)
    for scheme,batches in schemes.items():
        for name in p['models']:
            prob=rows[name].to_numpy()
            for q in p['capacity_fractions']:
                masks={r:np.zeros(len(rows),bool) for r in ['validation_cutoff','fill_capacity','cost_then_capacity']}
                for j,idx in enumerate(batches):
                    limits={'validation_cutoff':prob[idx]<frozen['capacity_thresholds'][name][str(q)],
                        'fill_capacity':hard_capacity(prob[idx],ids[idx],q),
                        'cost_then_capacity':hard_capacity(prob[idx],ids[idx],q,p['primary_ratio'])}
                    cap=capacity_count(len(idx),q)
                    for rule,a in limits.items():
                        masks[rule][idx]=a
                        batch_rows.append({'scheme':scheme,'batch':j,'model':name,'fraction':q,'rule':rule,'cap':cap,
                            'excess_over_cap':max(0,int(a.sum())-cap),'unused_cap':max(0,cap-int(a.sum())),
                            **decision_summary(y[idx],prob[idx],a,p['primary_ratio'],ids[idx])})
                        if rule!='validation_cutoff' and a.sum()>cap:raise AssertionError('Hard batch capacity failed')
                for rule,a in masks.items():actions[f'{name}|{scheme}|{rule}|{q}']=a
            for rate in p['unit_risk_rates']:
                a=np.zeros(len(rows),bool)
                for j,idx in enumerate(batches):
                    budget=rate*len(idx)
                    b=hard_capacity(prob[idx],ids[idx],p['risk_capacity_fraction'],p['primary_ratio'],budget)
                    a[idx]=b;s=decision_summary(y[idx],prob[idx],b,p['primary_ratio'],ids[idx])
                    if s['predicted_approved_defaults']>budget+1e-9:raise AssertionError('Unit predicted-loss budget failed')
                    risk_rows.append({'scheme':scheme,'batch':j,'model':name,'unit_risk_rate':rate,
                        'risk_budget':budget,'realized_risk_excess':max(0,s['approved_defaults']-budget),**s})
                actions[f'{name}|{scheme}|risk_prefix|{rate}']=a
    for base in ['woe_lr','random_forest']:
        order.append({'model':base,**order_diagnostics(rows[base],rows[base+'_sigmoid'],ids),
            'raw_logit_clip_count':int(((rows[base]<1e-6)|(rows[base]>1-1e-6)).sum())})
        for key in [k for k in actions if k.startswith(base+'|')]:
            other=base+'_sigmoid'+key[len(base):]
            a,b=actions[key],actions[other]
            agreements.append({'model':base,'decision':key[len(base)+1:],
                'different_rows':int((a!=b).sum()),'total_rows':len(rows)})
        for rule in ['validation_min_cost','probability_cost_rule','threshold_0.5']:
            r=p['primary_ratio'];a=actions[f'{base}|whole|{rule}|{r}'];b=actions[f'{base}_sigmoid|whole|{rule}|{r}']
            ca,_=cost_rows(y,rows[base],a,r);cb,_=cost_rows(y,rows[base+'_sigmoid'],b,r)
            mean,lo,hi=paired_group_interval(cb-ca,groups,p['bootstrap_repeats'],p['bootstrap_seed'])
            effects.append({'model':base,'rule':rule,'ratio':r,'calibrated_minus_raw_cost':mean,
                'ci95_low':lo,'ci95_high':hi,'different_rows':int((a!=b).sum())})
    files={'test_metrics':all_metrics,'cost_rules':costs,'batch_decisions':batch_rows,'test_order':order,
        'paired_calibration_effects':effects,'action_agreement':agreements,'unit_risk_prefix':risk_rows}
    for n,data in files.items():write_csv(pd.DataFrame(data),out/f'{n}.csv')
    (out/'local').mkdir(exist_ok=True)
    np.savez_compressed(out/'local/test_actions.npz',**actions)
    if signature(p)!=frozen['signature'] or input_hashes(baseline)!=frozen['input_hashes']:raise ValueError('Inputs changed during evaluation')
    write_json(out/'test_verification.json',{'selection_sha256':sha(out/'selection.json'),
        'files':{f'{n}.csv':sha(out/f'{n}.csv') for n in files},'private_actions_sha256':sha(out/'local/test_actions.npz'),
        'test_rows':len(rows),'same_historical_split':True,'no_test_reselection':True})


def report_stage(p,out):
    f=load_frozen(p,out);v=json.loads((out/'test_verification.json').read_text())
    if sha(out/'selection.json')!=v['selection_sha256'] or any(sha(out/n)!=h for n,h in v['files'].items()):raise ValueError('Frozen evidence changed')
    d=pd.read_csv(out/'batch_decisions.csv',float_precision='round_trip')
    summary=d.groupby(['scheme','model','fraction','rule'],as_index=False).agg(
        batches=('batch','size'),rows=('n','sum'),approved=('approved','sum'),cap=('cap','sum'),
        excess=('excess_over_cap','sum'),violating_batches=('excess_over_cap',lambda x:int((x>0).sum())))
    summary['approval_rate']=summary.approved/summary.rows
    write_csv(summary,out/'capacity_summary.csv')
    cost=pd.read_csv(out/'cost_rules.csv');effect=pd.read_csv(out/'paired_calibration_effects.csv')
    agreement=pd.read_csv(out/'action_agreement.csv');ordering=pd.read_csv(out/'test_order.csv')
    text=['# Calibration, actions and hard batch capacity','',p['scope'],'',
        'All base model settings and the cohort split are inherited. Validation freezes cost and scalar capacity cutoffs. Hard batch policies use only current scores and immutable IDs; labels enter evaluation only.',
        'Capacity-only fills K as a ranking control. Cost-then-capacity stops when expected incremental benefit is nonpositive. Equal scores may be split by ID; this differs from historical whole-block scalar thresholds.', '',
        '## Primary hypothetical 1:5 cost results','','```',cost[cost.ratio==p['primary_ratio']].to_string(index=False),'```','',
        '## Paired group bootstrap: calibrated minus raw cost','','```',effect.to_string(index=False),'```','',
        'Intervals condition on already fitted models and frozen rules; no retraining or future-cohort uncertainty. Negative cost difference favors calibrated probabilities.', '',
        '## Observed ordering and per-row action agreement','','```',ordering.to_string(index=False),'```','',
        '```',agreement.to_string(index=False),'```','',
        '## Capacity summary','','```',summary.to_string(index=False),'```','',
        'Group-packed batches are artificial workload partitions of the same 5,998 people, not new independent cohorts. Duplicate financial groups stay within batches; repeated schemes reuse people.',
        'Unit-risk prefix rows enforce predicted expected defaults, not realized loss. Violations of realized counts do not imply the numerical predicted-risk constraint failed.',
        'All costs are normalized retrospective classification scenarios, not causal approval profits. Credit limits are not treated as EAD. Row predictions, IDs and decision masks stay in ignored local files.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    write_json(out/'manifest.json',{'signature':f['signature'],'runtime':f['runtime'],
        'files':{x.name:sha(x) for x in sorted(out.iterdir()) if x.is_file() and x.name!='manifest.json'},
        'private_rows_published':False,'historical_split_reused':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/decision_study.json')
    parser.add_argument('--baseline',type=Path,default=ROOT/'reports/generated/decision_baseline')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['run','validate','test','report'],default='run')
    args=parser.parse_args();p=validate_protocol(json.loads(args.protocol.read_text()))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);baseline=args.baseline.resolve()
    if args.stage in ['run','validate']:validate_stage(p,baseline,out)
    if args.stage in ['run','test']:test_stage(p,baseline,out)
    if args.stage in ['run','report']:report_stage(p,out)
    print(f'{args.stage}: {out}')


if __name__=='__main__':main()
