"""Derived figures only; preserve every frozen rule and row decision."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.credit.decision_study import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--decision',type=Path,required=True);p.add_argument('--allocation',type=Path,required=True)
    args=p.parse_args();first=args.decision;second=args.allocation
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'credit-decisions-v1'})
    def save(fig,out,name):
        fig.savefig(out/f'{name}.png',dpi=160);fig.savefig(out/f'{name}.svg',metadata={'Date':None});plt.close(fig)
    d=pd.read_csv(first/'cost_rules.csv',float_precision='round_trip')
    d=d[(d.ratio==5)&d.model.isin(['random_forest','random_forest_sigmoid'])]
    rules=['validation_min_cost','probability_cost_rule','threshold_0.5']
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for j,(name,label,color) in enumerate([('random_forest','RF raw','#64748b'),('random_forest_sigmoid','RF sigmoid','#0369a1')]):
        y=d[d.model==name].set_index('rule').loc[rules,'actual_cost']
        bars=ax.bar(np.arange(3)+(j-.5)*.34,y,width=.34,label=label,color=color);ax.bar_label(bars,fmt='%.4f',padding=3)
    ax.set_xticks(range(3),['Validation-selected cutoff','Theoretical cutoff 1/6','Fixed cutoff 0.5'])
    ax.set_ylim(0,.88);ax.set_ylabel('Hypothetical cost / customer');ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_title('Calibration and decision rules | historical cohort, no new holdout')
    save(fig,first,'calibration_cost')
    b=pd.read_csv(first/'batch_decisions.csv',float_precision='round_trip')
    b=b[(b.model=='random_forest_sigmoid')&(b.scheme=='groups_250')&(b.fraction==.6)]
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    for rule,label,color in [('validation_cutoff','Validation cutoff','#b45309'),('cost_then_capacity','Cost then hard capacity','#0369a1')]:
        g=b[b.rule==rule].sort_values('batch');ax.plot(g.batch,g.approved-g.cap,'o-',label=label,color=color)
    ax.axhline(0,color='#64748b',label='Hard capacity-only: exactly K',linewidth=2)
    ax.set(xlabel='Artificial group-packed batch',ylabel='Approvals minus floor(0.6 * batch size)',
        title='Current-batch capacity | same 5,998 historical people, target batches 250')
    ax.legend();ax.grid(alpha=.2);save(fig,first,'hard_capacity')
    e=pd.read_csv(second/'paired_effects.csv',float_precision='round_trip')
    s=pd.read_csv(second/'summary.csv',float_precision='round_trip')
    primary=e[e.primary_case].iloc[0]
    g=s[(s.model==primary.model)&(s.weight==primary.weight)&(s.batch_size==primary.batch_size)&
        (s.funding_fraction==primary.funding_fraction)&(s.risk_rate==primary.risk_rate)].set_index('algorithm')
    fig,axes=plt.subplots(1,2,figsize=(14,5.2),layout='constrained')
    names=['pd_greedy','gain_greedy','milp']
    for j,(metric,label,color) in enumerate([('predicted_cost_per_row','Predicted cost','#64748b'),('actual_cost_per_row','Retrospective cost','#0369a1')]):
        bars=axes[0].bar(np.arange(3)+(j-.5)*.34,g.loc[names,metric],width=.34,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.4f',padding=3,fontsize=8)
    axes[0].set_xticks(range(3),['PD greedy','Gain greedy','MILP']);axes[0].set_ylim(0,.73)
    axes[0].set_title('Predeclared primary hypothetical budget case');axes[0].set_ylabel('Normalized weighted cost / customer');axes[0].legend(loc='lower left')
    h=e[(e.model=='random_forest_sigmoid')&(e.weight=='hypothetical_heterogeneous')&(e.reference=='pd_greedy')].reset_index(drop=True)
    x=h.milp_minus_reference_actual_cost.to_numpy();lo=h.ci95_low.to_numpy();hi=h.ci95_high.to_numpy()
    labels=[f'n~{r.batch_size}, F={r.funding_fraction:.0%}, R={r.risk_rate:.0%}' for r in h.itertuples()]
    axes[1].errorbar(x,np.arange(len(h)),xerr=np.vstack([x-lo,hi-x]),fmt='o',color='#0369a1',capsize=3)
    axes[1].axvline(0,color='#64748b',linestyle='--');axes[1].set_yticks(range(len(h)),labels);axes[1].invert_yaxis()
    axes[1].set_xlabel('MILP minus PD-greedy cost / customer (negative favors MILP)')
    axes[1].set_title('Conditional paired group-bootstrap 95% intervals')
    for ax in axes:ax.grid(alpha=.15);ax.set_axisbelow(True)
    save(fig,second,'budget_effects')
    script_sha=sha(Path(__file__),True)
    for out,names,inputs in [(first,['calibration_cost','hard_capacity'],['cost_rules.csv','batch_decisions.csv']),
                            (second,['budget_effects'],['paired_effects.csv','summary.csv'])]:
        write_json(out/'figure_manifest.json',{'script_sha256':script_sha,'descriptive_only':True,
            'inputs':{n:sha(out/n) for n in inputs},'figures':{f'{n}.{ext}':sha(out/f'{n}.{ext}') for n in names for ext in ['png','svg']}})
        manifest=json.loads((out/'manifest.json').read_text())
        manifest['files']={x.name:sha(x) for x in sorted(out.iterdir()) if x.is_file() and x.name!='manifest.json'}
        write_json(out/'manifest.json',manifest)
    print('Derived figures saved; frozen inputs and decisions unchanged.')


if __name__=='__main__':main()
