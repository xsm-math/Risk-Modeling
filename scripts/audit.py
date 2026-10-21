"""Reproduce or verify the public-credit experiment and a shuffled-label control."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.credit.data import load_data,features,split_data,TARGET
from src.credit.evaluation import metrics,choose_threshold,approval_threshold
from src.credit.models import fit_candidates
from src.credit.experiment import run


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,help='Verify an existing isolated run; otherwise reproduce first.')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/generated/audit')
    parser.add_argument('--negative-control',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    run_dir=args.run_dir.resolve() if args.run_dir else out/'reproduction'
    config_path=ROOT/'configs/credit.json';config=json.loads(config_path.read_text(encoding='utf-8'))
    df=load_data(ROOT/'data/raw/uci_credit.zip',download=True)
    if args.run_dir is None:run(config_path,run_dir)
    meta=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    require(meta['selection']['config_sha256']==hashlib.sha256(config_path.read_bytes()).hexdigest(),'Run configuration differs.')
    for name,digest in meta['code_sha256'].items():
        require(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,f'Source differs: {name}')
    X=features(df);y=df[TARGET].to_numpy();splits,groups=split_data(df,config['seed'])
    all_indices=np.concatenate(list(splits.values()))
    require(np.array_equal(np.sort(all_indices),np.arange(len(df))),'Split coverage/overlap failed.')
    require(groups.groupby('group')['split'].nunique().max()==1,'Duplicate inputs crossed split boundaries.')
    local=run_dir/'local/processed'
    val=pd.read_csv(local/'validation_predictions.csv',float_precision='round_trip')
    test=pd.read_csv(local/'test_predictions.csv',float_precision='round_trip')
    require(np.array_equal(val.ID,df.iloc[splits['validation']].ID),'Validation rows differ.')
    require(np.array_equal(test.ID,df.iloc[splits['test']].ID),'Test rows differ.')
    require(np.array_equal(val.y,y[splits['validation']]) and np.array_equal(test.y,y[splits['test']]),'Saved labels differ.')
    names=[c for c in val if c not in ['ID','y']]
    validation=pd.DataFrame([{'model':n,**metrics(val.y,val[n])} for n in names])
    expected=validation.sort_values(['log_loss','model']).iloc[0]['model']
    require(expected==meta['selection']['model'],'Model selection does not match validation log loss.')
    threshold=choose_threshold(val.y,val[expected],config['primary_cost_ratio'])
    require(np.isclose(threshold,meta['selection']['threshold'],rtol=0,atol=1e-14),'Threshold differs from validation-only search.')
    published=pd.read_csv(run_dir/'test_metrics.csv').set_index('model')
    maximum_error=0.
    for name in published.index:
        calculated=metrics(test.y,test[name])
        for key,value in calculated.items():
            error=abs(value-published.loc[name,key]);maximum_error=max(maximum_error,error)
            require(np.isclose(value,published.loc[name,key],rtol=1e-10,atol=1e-12),f'Test metric mismatch: {name}/{key}')
    decisions=pd.read_csv(run_dir/'decision_metrics.csv')
    require(np.allclose(decisions.cost_per_customer,(decisions.fp+config['primary_cost_ratio']*decisions.fn)/len(test),rtol=0,atol=1e-12),'Cost and confusion ledger disagree.')
    approvals=[]
    for spec in meta['selection']['approval_specs']:
        target=spec['target_approval'];cutoff=approval_threshold(val[expected],target)
        require(np.isclose(cutoff,spec['threshold'],rtol=0,atol=1e-14),'Approval cutoff differs.')
        count=int((val[expected]<cutoff).sum())
        require(count<=int(np.floor(target*len(val))),'Validation capacity exceeded.')
        approvals.append({'target':target,'validation_approval':count/len(val),
                          'test_approval':float((test[expected]<cutoff).mean())})
    negative=None
    if args.negative_control:
        t,c,v,h=[splits[n] for n in ['train','calibration','validation','test']]
        rng=np.random.default_rng(config['seed']+999)
        with threadpool_limits(limits=2):
            shuffled=fit_candidates(X.iloc[t],rng.permutation(y[t]),X.iloc[c],rng.permutation(y[c]),config)
            shuffled_validation=rng.permutation(y[v]);shuffled_test=rng.permutation(y[h])
            pv={name:model.predict(X.iloc[v]) for name,model in shuffled.items()}
            scores=pd.DataFrame([{'model':n,**metrics(shuffled_validation,p)} for n,p in pv.items()])
            chosen=scores.sort_values(['log_loss','model']).iloc[0]['model']
            rows=[{'model':name,'selected_on_validation':name==chosen,
                   **{'validation_'+k:value for k,value in metrics(shuffled_validation,pv[name]).items()},
                   **{'test_'+k:value for k,value in metrics(shuffled_test,model.predict(X.iloc[h])).items()},
                   'original_label_test_auc_diagnostic':metrics(y[h],model.predict(X.iloc[h]))['auc']}
                  for name,model in shuffled.items()]
        pd.DataFrame(rows).to_csv(out/'negative_control.csv',index=False)
        negative={'seed':config['seed']+999,'selected_on_shuffled_validation_labels':chosen,
                  'test_auc':next(r['test_auc'] for r in rows if r['model']==chosen),
                  'design':'Labels independently permuted inside all four fixed partitions; split/features/model capacity unchanged. Selection uses permuted validation labels before evaluating permuted test labels. Original-label AUC is retained as a separate diagnostic, not a control pass criterion.',
                  'scope':'One diagnostic, not proof of no leakage, not an external test.'}
    result={'checks':'passed','rows':len(df),'selected':expected,'threshold':threshold,
            'maximum_metric_reconstruction_error':maximum_error,'approval_capacity':approvals,
            'code_sha256':meta['code_sha256'],'data_sha256':meta['source_sha256'],
            'negative_control':negative,
            'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
            'limitations':'Previously published cohort/split; no new temporal or external holdout. Bootstrap is conditional on fixed fitted models and rules.'}
    (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
