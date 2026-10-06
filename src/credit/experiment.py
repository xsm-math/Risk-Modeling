"""Four-way, frozen-configuration experiment and reproducible research outputs."""
from pathlib import Path
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from .data import load_data, features, split_data, TARGET, FINANCIAL, SHA256, URL
from .models import fit_candidates
from .evaluation import (metrics, policy, choose_threshold, approval_threshold,
                         reliability, bootstrap)
from .diagnostics import quality_audit, export_scorecard, stability_audit, diagnostic_plots
from .sql_features import validate as validate_sql

ROOT = Path(__file__).resolve().parents[2]


def table(df, digits=4):
    df = df.copy()
    for c in df.select_dtypes('number'):
        df[c] = df[c].map(lambda v: f'{v:.{digits}f}' if pd.notna(v) else 'NA')
    return '| ' + ' | '.join(df.columns) + ' |\n| ' + ' | '.join(['---']*len(df.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str,row)) + ' |' for row in df.to_numpy())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')


def run(config_path=None, output=None):
    config_path = Path(config_path or ROOT/'configs/credit.json')
    out = Path(output or ROOT/'reports/credit'); out.mkdir(parents=True,exist_ok=True)
    config = json.loads(config_path.read_text()); ratio=config['primary_cost_ratio']
    df=load_data(ROOT/'data/raw/uci_credit.zip')
    X=features(df); y=df[TARGET].to_numpy()
    splits, manifest=split_data(df, config['seed'])
    # Row-level data and persisted models remain local, not published to Git.
    local=ROOT/'data/processed/credit'; local.mkdir(parents=True,exist_ok=True)
    manifest.to_csv(local/'split_manifest.csv',index=False,lineterminator='\n')
    t,c,v,h=[splits[n] for n in ['train','calibration','validation','test']]
    import matplotlib
    matplotlib.use('Agg')
    quality=quality_audit(df,t,out)
    sql_parity=validate_sql(df,out)
    with threadpool_limits(limits=2):
        candidates=fit_candidates(X.iloc[t],y[t],X.iloc[c],y[c],config)
        pv={name:m.predict(X.iloc[v]) for name,m in candidates.items()}
        validation=pd.DataFrame([{'model':n,**metrics(y[v],p)} for n,p in pv.items()])
        # Final model/calibration chosen before test prediction or label evaluation.
        selected=validation.sort_values(['log_loss','model']).iloc[0]['model']
        thresholds={n:choose_threshold(y[v],p,ratio) for n,p in pv.items()}
        policy_specs=[]
        for r in config['cost_ratios']:
            policy_specs.extend([
                {'policy':'validation_min_cost','ratio':r,'threshold':choose_threshold(y[v],pv[selected],r)},
                {'policy':'probability_cost_rule','ratio':r,'threshold':1/(1+r)},
                {'policy':'threshold_0.5','ratio':r,'threshold':.5},
                {'policy':'approve_all','ratio':r,'threshold':1.},
                {'policy':'reject_all','ratio':r,'threshold':0.}])
        approval_specs=[{'target_approval':a,'threshold':approval_threshold(pv[selected],a)} for a in config['approval_targets']]
        selection={'model':selected,'criterion':'minimum validation log_loss; model name breaks ties',
                   'primary_cost_ratio':ratio,'threshold':thresholds[selected], 'policy_specs':policy_specs,
                   'approval_specs':approval_specs, 'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest()}
        write_json(out/'selection.json',selection)
        ph={name:m.predict(X.iloc[h]) for name,m in candidates.items()}
        # Fixed prior-probability baseline: fitted on train only.
        ph['constant_prior']=np.full(len(h),y[t].mean())
        comparison=pd.DataFrame([{'model':n,**metrics(y[h],p)} for n,p in ph.items()])
        model_policies=pd.DataFrame([{'model':n,**policy(y[h],ph[n],thresholds[n],ratio)} for n in candidates])
        decisions=pd.DataFrame([{'model':n,'strategy':strategy,
            **policy(y[h],ph[n],cutoff,ratio)} for n in candidates
            for strategy,cutoff in [('validation_min_cost',thresholds[n]),('threshold_0.5',.5),('probability_cost_rule',1/(1+ratio))]])
        decisions.to_csv(out/'decision_metrics.csv',index=False)
        costs=pd.DataFrame([{**spec,**policy(y[h],ph[selected],spec['threshold'],spec['ratio'])} for spec in policy_specs])
        approvals=pd.DataFrame([{**spec,**policy(y[h],ph[selected],spec['threshold'],ratio)} for spec in approval_specs])
        intervals=bootstrap(y[h],ph[selected],ph['woe_lr'],thresholds[selected],thresholds['woe_lr'],ratio,
                            manifest.iloc[h]['group'].to_numpy(),config['bootstrap_repeats'],config['seed']+1)
        # Permutation interpretation uses validation only; not used for refitting.
        rng=np.random.default_rng(config['seed']+2); importance=[]
        base=roc_auc_score(y[v],pv[selected])
        for name in X.columns:
            drops=[]
            for _ in range(3):
                altered=X.iloc[v].copy(); altered[name]=rng.permutation(altered[name].to_numpy())
                drops.append(base-roc_auc_score(y[v],candidates[selected].predict(altered)))
            importance.append({'feature':name,'validation_auc_drop':np.mean(drops),'repeat_sd':np.std(drops)})
        importance=pd.DataFrame(importance).sort_values('validation_auc_drop',ascending=False)
        scorecard=export_scorecard(candidates['woe_lr'],X.iloc[t],y[t],X.iloc[h],out)
        stability=stability_audit(candidates[selected],df.iloc[t],df.iloc[h],thresholds[selected],out)
        tree_rows=[]
        for tree_name in ['random_forest','xgboost']:
            if tree_name in candidates:
                tree_rows.extend({'model':tree_name,'feature':f,'importance':float(value)}
                    for f,value in zip(X.columns,candidates[tree_name].estimator.feature_importances_))
        pd.DataFrame(tree_rows).to_csv(out/'tree_importance.csv',index=False)
        shap_summary=None
        if 'xgboost' in candidates:
            from xgboost import DMatrix
            booster=candidates['xgboost'].estimator.get_booster()
            sample=X.iloc[v[:512]]
            contributions=booster.predict(DMatrix(sample),pred_contribs=True,approx_contribs=False)
            margins=booster.predict(DMatrix(sample),output_margin=True)
            error=float(np.max(np.abs(contributions.sum(axis=1)-margins)))
            np.testing.assert_allclose(contributions.sum(axis=1),margins,rtol=1e-5,atol=1e-5)
            shap_summary={'model':'uncalibrated xgboost challenger','method':'native exact TreeSHAP',
                          'rows':len(sample),'space':'raw log odds','max_additivity_error':error}
            pd.DataFrame({'feature':X.columns,'mean_abs_shap':np.abs(contributions[:,:-1]).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False).to_csv(out/'xgboost_shap.csv',index=False)
        diagnostic_plots(out,y[h],ph,selected,importance,pd.read_csv(out/'iv_coefficients.csv'))
        # Monitor score distribution against calibration reference, with fixed bins.
        reference=candidates[selected].predict(X.iloc[c])
        edges=np.unique(np.r_[-np.inf,np.quantile(reference,np.arange(.1,1,.1)),np.inf])
        ref=np.histogram(reference,bins=edges)[0]+.5; cur=np.histogram(ph[selected],bins=edges)[0]+.5
        rp=ref/ref.sum();cp=cur/cur.sum()
        psi=float(np.sum((cp-rp)*np.log(cp/rp)))
        # Complete pipeline artifact, plus schema/version metadata. Load only trusted joblib files.
        artifact=ROOT/'artifacts'; artifact.mkdir(exist_ok=True)
        joblib.dump({'model':candidates[selected],'threshold':thresholds[selected],
                     'features':list(X.columns),'model_name':selected,'config':config,
                     'sklearn_version':sklearn.__version__},artifact/'credit_model.joblib')
        joblib.dump({'model':candidates['woe_lr'],'threshold':thresholds['woe_lr'],
                     'features':list(X.columns),'model_name':'woe_lr','config':config,
                     'scorecard':scorecard,'sklearn_version':sklearn.__version__},artifact/'credit_scorecard.joblib')
        restored=joblib.load(artifact/'credit_model.joblib')
        restored_probability=restored['model'].predict(X.iloc[h[:32]])
        roundtrip_error=float(np.max(np.abs(restored_probability-ph[selected][:32])))
        np.testing.assert_allclose(restored_probability,ph[selected][:32],rtol=0,atol=1e-12)
        pd.DataFrame({'ID':df.iloc[h].ID.to_numpy(),'y':y[h],**ph}).to_csv(local/'test_predictions.csv',index=False)
    split_summary=pd.DataFrame([{'split':n,'rows':len(idx),'defaults':int(y[idx].sum()),'default_rate':float(y[idx].mean()),
                                'groups':int(manifest.iloc[idx]['group'].nunique())} for n,idx in splits.items()])
    for name,frame in [('validation_metrics',validation),('test_metrics',comparison),('model_policies',model_policies),
                       ('cost_sensitivity',costs),('approval_frontier',approvals),('feature_importance',importance),('split_summary',split_summary)]:
        frame.to_csv(out/(name+'.csv'),index=False)
    cal=pd.concat([reliability(y[h],p).assign(model=n) for n,p in ph.items()],ignore_index=True)
    cal.to_csv(out/'reliability.csv',index=False)
    metadata={'dataset':'UCI Default of Credit Card Clients','doi':'10.24432/C55S3H','license':'CC BY 4.0',
              'source_url':URL,'source_sha256':SHA256,'n_rows':len(df),'raw_financial_features':len(FINANCIAL),
              'model_features':list(X.columns),'duplicate_financial_rows':int(df[FINANCIAL].duplicated().sum()),
              'split_manifest_sha256':hashlib.sha256((local/'split_manifest.csv').read_bytes()).hexdigest(),
              'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
              'bootstrap_repeats':config['bootstrap_repeats'],'test_score_psi_vs_calibration':psi,
              'artifact_roundtrip_max_error':roundtrip_error,'selection':selection,'intervals':intervals}
    import importlib.metadata
    metadata.update({'revision':config.get('revision','baseline'),
        'runtime_packages':{name:importlib.metadata.version(name) for name in
            ['numpy','pandas','scikit-learn','scipy','matplotlib','xlrd','joblib','threadpoolctl','xgboost']},
        'quality_audit':quality,'scorecard':scorecard,'stability':stability,'sql_parity':sql_parity})
    metadata['shap']=shap_summary
    write_json(out/'manifest.json',metadata)
    plot_results(out,comparison,cal,costs,approvals,selected,ratio)
    from .reporting import write_report
    write_report(out,comparison,validation,model_policies,costs,approvals,split_summary,metadata)
    print(json.dumps({'selected':selected,'test':comparison.set_index('model').loc[selected].to_dict(),
                      'policy':model_policies.set_index('model').loc[selected].to_dict(),'intervals':intervals},indent=2))
    return metadata


def plot_results(out,comparison,cal,costs,approvals,selected,ratio):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'credit-benchmark','font.size':10})
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    order=comparison.sort_values('auc');axs[0,0].scatter(order.auc,order.model,color='#31688e',s=55)
    axs[0,0].set(xlim=(.45,.9),xlabel='Test ROC AUC',title='Ranking: higher is better')
    for name in dict.fromkeys(['woe_lr',selected.removesuffix('_sigmoid'),selected,'xgboost']):
        d=cal[cal.model==name];axs[0,1].plot(d.predicted,d.observed,'o-',label=name)
    axs[0,1].plot([0,1],[0,1],'k--',alpha=.5);axs[0,1].legend(fontsize=8)
    axs[0,1].set(xlabel='Mean predicted probability',ylabel='Observed default fraction',title='Reliability: closer to diagonal is better')
    for name,g in costs.groupby('policy'):
        axs[1,0].plot(g.ratio,g.cost_per_customer,'o-',label=name)
    axs[1,0].legend(fontsize=7);axs[1,0].set(xlabel='Missed-default / false-rejection cost ratio',ylabel='Hypothetical cost / customer',title='Validation-selected rules evaluated on test')
    axs[1,1].plot(approvals.approval_rate,approvals.approved_default_rate,'o-',color='#21918c')
    axs[1,1].set(xlabel='Realized test approval rate',ylabel='Default fraction among approved',title='Approval cutoffs fixed on validation')
    fig.suptitle('Credit risk benchmark | '+selected+' | one historical cohort',fontsize=14)
    fig.savefig(out/'benchmark.svg', metadata={'Date': None})
    fig.savefig(out/'benchmark.png',dpi=160);plt.close(fig)
    svg = out/'benchmark.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
