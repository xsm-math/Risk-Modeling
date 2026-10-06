"""Download, train and batch-score the real-data benchmark."""
import argparse
import json
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from .data import load_data, features
from .experiment import ROOT, run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('download')
    train=sub.add_parser('run');train.add_argument('--config');train.add_argument('--output')
    score=sub.add_parser('score');score.add_argument('--input',required=True);score.add_argument('--output',required=True)
    score.add_argument('--model',default=str(ROOT/'artifacts/credit_model.joblib'))
    args=parser.parse_args()
    if args.command=='download':
        df=load_data(ROOT/'data/raw/uci_credit.zip',download=True);print(json.dumps({'rows':len(df),'columns':len(df.columns)}))
    elif args.command=='run':
        run(args.config,args.output)
    else:
        # Only load artifacts you created/trust: joblib uses Python pickle.
        artifact=joblib.load(args.model);raw=pd.read_csv(args.input)
        X=features(raw)
        if list(X.columns)!=artifact['features']:raise ValueError('Feature schema differs from saved model.')
        with threadpool_limits(limits=2):p=artifact['model'].predict(X)
        result=pd.DataFrame({'row':range(len(raw)),'default_probability':p,'reject_flag':p>=artifact['threshold']})
        if 'scorecard' in artifact:
            from numpy import log
            spec=artifact['scorecard']
            result['credit_score']=spec['offset']-spec['factor']*log(p/(1-p))
        if 'ID' in raw:result.insert(0,'ID',raw.ID)
        result.to_csv(args.output,index=False)
        print(f'Scored {len(result)} rows using {artifact["model_name"]}; threshold={artifact["threshold"]:.6f}')

if __name__=='__main__':main()
