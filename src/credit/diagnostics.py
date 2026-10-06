"""Aggregate quality audit, additive scorecard and fixed-reference drift tests."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_curve, precision_recall_curve
from .data import FINANCIAL, STATUS, BILLS, PAYMENTS, TARGET, features


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def quality_audit(df, train, out):
    rows = []
    for name in df.columns:
        s = df[name]; q = s.quantile([.01,.25,.5,.75,.99]); iqr = q.loc[.75]-q.loc[.25]
        rows.append({'feature': name, 'missing': int(s.isna().sum()),
            'nonfinite': int((~np.isfinite(s)).sum()), 'unique': s.nunique(),
            'min': s.min(), 'p01': q.loc[.01], 'median': q.loc[.5], 'p99': q.loc[.99], 'max': s.max(),
            'iqr_flags': int(((s < q.loc[.25]-1.5*iqr) | (s > q.loc[.75]+1.5*iqr)).sum())})
    pd.DataFrame(rows).to_csv(out/'data_quality.csv', index=False)
    frequencies = []
    for name in STATUS+['SEX','EDUCATION','MARRIAGE']:
        for value, count in df[name].value_counts().sort_index().items():
            frequencies.append({'feature': name, 'value': value, 'count': count,
                'documentation_flag': (name in STATUS and value in [-2,0]) or
                   (name == 'EDUCATION' and value not in [1,2,3,4]) or
                   (name == 'MARRIAGE' and value not in [1,2,3])})
    pd.DataFrame(frequencies).to_csv(out/'category_frequencies.csv', index=False)
    groups = pd.util.hash_pandas_object(df[FINANCIAL], index=False)
    conflicting = pd.DataFrame({'group': groups, 'y': df[TARGET]}).groupby('group').y.nunique()
    audit = {'rows': len(df), 'defaults': int(df[TARGET].sum()), 'default_rate': float(df[TARGET].mean()),
        'missing_cells': int(df.isna().sum().sum()), 'duplicate_ids': int(df.ID.duplicated().sum()),
        'duplicate_financial_rows': int(df[FINANCIAL].duplicated().sum()),
        'conflicting_label_groups': int((conflicting > 1).sum()),
        'negative_bill_cells': int((df[BILLS] < 0).sum().sum()),
        'negative_payment_cells': int((df[PAYMENTS] < 0).sum().sum()),
        'utilization_over_one': int((df.BILL_AMT1 > df.LIMIT_BAL).sum()),
        'nonpositive_limits': int((df.LIMIT_BAL <= 0).sum()),
        'policy': 'Flags are descriptive; negative bills and high utilization are retained. No test-based clipping.'}
    write_json(out/'data_quality.json', audit)
    # Label relationships and correlations use training records only.
    x = features(df.iloc[train]); y = df.iloc[train][TARGET].to_numpy()
    x.corr(method='spearman').to_csv(out/'train_spearman.csv')
    (pd.DataFrame({'status':df.iloc[train].PAY_0.to_numpy(), 'y':y})
        .groupby('status').agg(count=('y','size'), default_rate=('y','mean'))
        .to_csv(out/'train_default_by_status.csv'))
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axs[0,0].bar(['Non-default','Default'],[int((y==0).sum()),int(y.sum())],color=['#31688e','#d65f5f'])
    axs[0,0].set(title='Training class balance',ylabel='Records')
    for label in [0,1]:
        axs[0,1].hist(x.utilization_latest.to_numpy()[y==label],bins=np.linspace(-.2,2,45),
            density=True,histtype='step',label=str(label))
    axs[0,1].legend(title='Default');axs[0,1].set(title='Utilization (display range -0.2 to 2)',xlabel='Latest bill / limit')
    d=pd.read_csv(out/'train_default_by_status.csv');axs[1,0].bar(d.status,d.default_rate,color='#31688e')
    axs[1,0].set(title='Train-only default by latest status',xlabel='Raw PAY_0 code',ylabel='Default fraction')
    image=axs[1,1].imshow(x.corr(method='spearman'),vmin=-1,vmax=1,cmap='RdBu_r')
    axs[1,1].set(title='Train Spearman correlation',xticks=range(len(x.columns)),yticks=range(len(x.columns)))
    axs[1,1].set_xticklabels(x.columns,rotation=90,fontsize=5);axs[1,1].set_yticklabels(x.columns,fontsize=5)
    fig.colorbar(image,ax=axs[1,1]);fig.savefig(out/'eda.png',dpi=160);plt.close(fig)
    return audit


def export_scorecard(model, X_train, y_train, X_check, out):
    pipe = model.estimator
    transform, scaler, lr = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
    enc = transform.encoder_; names = transform.names_
    beta = lr.coef_[0]/scaler.scale_
    intercept = float(lr.intercept_[0]-np.dot(beta,scaler.mean_))
    factor = 20/np.log(2); offset = 600-factor*np.log(20)
    base_points = float(offset-factor*intercept)
    coefficients=[]; bins=[]; rules={}
    y=np.asarray(y_train)
    for j,name in enumerate(names):
        rule=enc.rules[name]; edges=rule['edges']; arr=rule['woe_by_interval']
        iv=float(rule['iv'])
        coefficients.append({'feature':name,'iv':iv,'woe_coefficient':float(beta[j]),
            'odds_multiplier_per_woe_unit':float(np.exp(beta[j])),
            'score_change_per_woe_unit':float(-factor*beta[j]),'final_bins':enc.n_bins(name)})
        if edges is None: edges=np.array([-np.inf,np.inf])
        values=X_train[name].fillna(rule['fill'])
        codes=pd.cut(values,edges,labels=False,include_lowest=True).to_numpy()
        rules[name]={'fill':float(rule['fill']),
            'edges':[str(v) if not np.isfinite(v) else float(v) for v in edges],
            'woe_by_interval':arr.tolist(),'beta':float(beta[j])}
        for k,woe in enumerate(arr):
            mask=codes==k; count=int(mask.sum()); bad=int(y[mask].sum())
            bins.append({'feature':name,'interval':k,'left_exclusive':edges[k],
                'right_inclusive':edges[k+1], 'train_count':count, 'train_defaults':bad,
                'train_good':count-bad,'bad_share':bad/y.sum(),
                'good_share':(count-bad)/(len(y)-y.sum()),
                'iv_component':float((bad/y.sum()-(count-bad)/(len(y)-y.sum()))*woe),
                'train_default_rate':bad/count if count else np.nan,
                'woe':float(woe),'points':float(-factor*beta[j]*woe)})
    pd.DataFrame(coefficients).sort_values('iv',ascending=False).to_csv(out/'iv_coefficients.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'scorecard_bins.csv',index=False)
    encoded=enc.transform(X_check,names)
    scores=base_points+encoded@(-factor*beta)
    reconstructed=expit((offset-scores)/factor)
    error=float(np.max(np.abs(reconstructed-model.predict(X_check))))
    np.testing.assert_allclose(reconstructed,model.predict(X_check),rtol=0,atol=1e-12)
    spec={'base_score':600,'base_odds_good_to_bad':20,'pdo':20,'factor':float(factor),
        'offset':float(offset),'intercept':intercept,'base_points':base_points,
        'woe_convention':'log(bad_share/good_share); legacy EPS smoothing',
        'missing_policy':'training median; dataset has no missing cells',
        'interval_convention':'(left,right]; outer boundaries infinite',
        'max_probability_reconstruction_error':error,'features':rules}
    write_json(out/'scorecard.json',spec)
    return {k:v for k,v in spec.items() if k!='features'}


def fixed_psi(reference, current, categorical=False):
    """Reference-only bins, explicit missing/unknown states and half-count smoothing."""
    r=pd.Series(reference); c=pd.Series(current)
    if categorical:
        levels=sorted(r.dropna().unique())
        rc=np.array([(r==v).sum() for v in levels]+[r.isna().sum(),0],dtype=float)
        cc=np.array([(c==v).sum() for v in levels]+[c.isna().sum(),(~c.isna() & ~c.isin(levels)).sum()],dtype=float)
        bounds=[str(v) for v in levels]+['missing','unknown']
    else:
        quantiles=r.dropna().quantile(np.arange(.1,1,.1)).dropna().to_numpy()
        edges=np.unique(np.r_[-np.inf,quantiles,np.inf])
        count=lambda s: pd.cut(s,edges,labels=False,include_lowest=True).value_counts().reindex(range(len(edges)-1),fill_value=0).to_numpy()
        rc=np.r_[count(r.dropna()),r.isna().sum()].astype(float)
        cc=np.r_[count(c.dropna()),c.isna().sum()].astype(float)
        bounds=[f'({a},{b}]' for a,b in zip(edges[:-1],edges[1:])]+['missing']
    rp=(rc+.5)/(rc.sum()+.5*len(rc)); cp=(cc+.5)/(cc.sum()+.5*len(cc))
    parts=(cp-rp)*np.log(cp/rp)
    return float(parts.sum()),pd.DataFrame({'bin':bounds,'reference_count':rc,'current_count':cc,
                                         'reference_share':rp,'current_share':cp,'psi_component':parts})


def stability_audit(model, df_train, df_test, threshold, out):
    ref=features(df_train); current=features(df_test)
    # Coherent transformation of raw inputs, then recompute all derived features.
    stressed=df_test.copy(); stressed.LIMIT_BAL=stressed.LIMIT_BAL*.7
    stressed.PAY_0=np.where(stressed.PAY_0>=0,np.minimum(stressed.PAY_0+1,8),stressed.PAY_0)
    shift=features(stressed)
    pr=model.predict(ref);pc=model.predict(current);ps=model.predict(shift)
    rows=[]; details=[]
    for scenario,x,p in [('same_cohort_test',current,pc),('simulated_limit_status_shift',shift,ps)]:
        for name in list(ref.columns)+['model_probability']:
            r=pr if name=='model_probability' else ref[name]
            c=p if name=='model_probability' else x[name]
            value,d=fixed_psi(r,c,categorical=name in STATUS)
            rows.append({'scenario':scenario,'feature':name,'psi':value,
                         'heuristic_level':'review' if value>=.25 else 'watch' if value>=.1 else 'low'})
            details.append(d.assign(scenario=scenario,feature=name))
    result=pd.DataFrame(rows);result.to_csv(out/'feature_psi.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(out/'psi_bins.csv',index=False)
    summary={'reference':'training population; all boundaries fixed before test',
        'stress_definition':'LIMIT_BAL x0.7; nonnegative latest status +1 capped at 8; derived fields recomputed',
        'same_cohort_approval_rate':float((pc<threshold).mean()),
        'stress_approval_rate':float((ps<threshold).mean()),
        'same_cohort_max_psi':float(result[result.scenario=='same_cohort_test'].psi.max()),
        'stress_max_psi':float(result[result.scenario!='same_cohort_test'].psi.max()),
        'limitation':'Stress has no new labels; no predictive accuracy or causal default effects are inferred. PSI cutoffs are heuristics.'}
    write_json(out/'stability.json',summary)
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(13,8),layout='constrained')
    pivot=result.pivot(index='feature',columns='scenario',values='psi').sort_values('simulated_limit_status_shift')
    axs[0].barh(pivot.index,pivot.same_cohort_test,color='#31688e')
    axs[0].set(title='Same-cohort test vs fixed training bins',xlabel='PSI (linear)',ylabel='')
    axs[1].barh(pivot.index,pivot.simulated_limit_status_shift,color='#ed8c2f')
    axs[1].set_xscale('symlog',linthresh=.01)
    axs[1].axvline(.1,color='gray',linestyle='--');axs[1].axvline(.25,color='red',linestyle='--')
    axs[1].set(title='Simulated shift vs same fixed bins',xlabel='PSI (symlog; linear below 0.01)',ylabel='')
    fig.savefig(out/'stability.png',dpi=160);plt.close(fig)
    return summary


def diagnostic_plots(out, y, predictions, selected, importance, coefficients):
    import matplotlib.pyplot as plt
    names=list(dict.fromkeys(['linear_lr','woe_lr','random_forest','xgboost',selected]))
    names=[n for n in names if n in predictions]
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for name in names:
        p=predictions[name];fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
        axs[0].plot(fpr,tpr,label=name);axs[1].plot(recall,precision,label=name)
    axs[0].plot([0,1],[0,1],'k--');axs[1].axhline(np.mean(y),color='black',linestyle='--')
    axs[0].set(title='Test ROC',xlabel='False positive rate',ylabel='True positive rate')
    axs[1].set(title='Test precision-recall',xlabel='Recall',ylabel='Precision')
    for ax in axs:ax.legend(fontsize=8)
    fig.savefig(out/'roc_pr.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    d=importance.head(12).sort_values('validation_auc_drop');axs[0].barh(d.feature,d.validation_auc_drop,xerr=d.repeat_sd)
    axs[0].set(title='Selected model: validation permutation importance',xlabel='AUC decrease')
    d=coefficients.head(12).sort_values('woe_coefficient');axs[1].barh(d.feature,d.woe_coefficient)
    axs[1].set(title='WOE LR coefficients (unstandardized WOE)',xlabel='Log odds change / WOE unit')
    fig.savefig(out/'explanations.png',dpi=160);plt.close(fig)
