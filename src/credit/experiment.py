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
    manifest.to_csv(local/'split_manifest.csv',index=False)
    t,c,v,h=[splits[n] for n in ['train','calibration','validation','test']]
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
        restored=joblib.load(artifact/'credit_model.joblib')
        np.testing.assert_allclose(restored['model'].predict(X.iloc[h[:32]]),ph[selected][:32],rtol=0,atol=1e-12)
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
              'artifact_roundtrip_max_error':0.0,'selection':selection,'intervals':intervals}
    write_json(out/'manifest.json',metadata)
    plot_results(out,comparison,cal,costs,approvals,selected,ratio)
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
    for name in ['woe_lr','hist_gbdt','hist_gbdt_sigmoid']:
        d=cal[cal.model==name];axs[0,1].plot(d.predicted,d.observed,'o-',label=name)
    axs[0,1].plot([0,1],[0,1],'k--',alpha=.5);axs[0,1].legend(fontsize=8)
    axs[0,1].set(xlabel='Mean predicted probability',ylabel='Observed default fraction',title='Reliability: closer to diagonal is better')
    for name,g in costs.groupby('policy'):
        axs[1,0].plot(g.ratio,g.cost_per_customer,'o-',label=name)
    axs[1,0].legend(fontsize=7);axs[1,0].set(xlabel='Missed-default / false-rejection cost ratio',ylabel='Hypothetical cost / customer',title='Validation-selected rules evaluated on test')
    axs[1,1].plot(approvals.approval_rate,approvals.approved_default_rate,'o-',color='#21918c')
    axs[1,1].set(xlabel='Realized test approval rate',ylabel='Default fraction among approved',title='Approval cutoffs fixed on validation')
    fig.suptitle('Credit risk benchmark | '+selected+' | one historical cohort',fontsize=14)
    fig.savefig(out/'benchmark.svg', metadata={'Date': None});plt.close(fig)
    svg = out/'benchmark.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def write_report(out,comparison,validation,model_policies,costs,approvals,splits,meta):
    name=meta['selection']['model']; m=comparison.set_index('model').loc[name]
    p=model_policies.set_index('model').loc[name];ci=meta['intervals'];ratio=meta['selection']['primary_cost_ratio']
    base=comparison.set_index('model').loc['woe_lr'];basep=model_policies.set_index('model').loc['woe_lr']
    relevant=costs[costs.ratio==ratio][['policy','threshold','cost_per_customer','approval_rate','approved_default_rate','bad_capture']]
    report=f'''# 信用卡违约概率估计与成本敏感决策

## 研究问题与结论

研究采用 UCI 的30,000条历史信用卡记录，比较线性逻辑回归、WOE逻辑回归和梯度提升树，并检验独立概率校准及决策阈值的影响。它是已有信用卡客户的下一期风险研究，不是新客贷款审批系统，也不是生产收益评估。

按验证集 Log loss 最小值选出 **{name}**；测试集 AUC **{m.auc:.4f}**、KS **{m.ks:.4f}**、Average precision **{m.average_precision:.4f}**、Brier **{m.brier:.4f}**、Log loss **{m.log_loss:.4f}**。AUC的配对分组bootstrap 95%区间为 **[{ci['auc']['lower']:.4f}, {ci['auc']['upper']:.4f}]**。

相对未校准WOE逻辑回归，测试AUC差为 **{m.auc-base.auc:.4f}**，差值95%区间 **[{ci['auc_gain_vs_woe']['lower']:.4f}, {ci['auc_gain_vs_woe']['upper']:.4f}]**。不得根据测试表中其他模型的好坏重新改选主模型。

在“误拒非违约者成本=1、放行违约者成本={ratio}”的假设下，验证集选择的阈值为 **{p.threshold:.6f}**，测试通过率 **{p.approval_rate:.2%}**、通过群体违约率 **{p.approved_default_rate:.2%}**、违约捕获率 **{p.bad_capture:.2%}**。相对各自在验证集选阈值的WOE基准，每客户成本减少 **{basep.cost_per_customer-p.cost_per_customer:.4f}** 单位，95%区间 **[{ci['cost_saving_vs_woe']['lower']:.4f}, {ci['cost_saving_vs_woe']['upper']:.4f}]**。这不是实际节省金额，也不是拒绝行为的因果效果。

![Benchmark](benchmark.svg)

## 1. 数据与信息时点

来源：[UCI / DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H)，作者 I-Cheng Yeh，CC BY 4.0。账单、支付与还款状态来自2005年4—9月；预测目标为下一月违约标签。金额为新台币，不转换成人民币。模型限于已提供全部观察期信息后做预测；数据未提供逐字段实际可用时间，不能直接证明生产时点可用性。

ID只用于审计；SEX、EDUCATION、MARRIAGE、AGE不作预测变量。19个账户行为字段生成5个固定衍生特征，共24个输入。保留负账单以及非正还款状态码；未记录的码含义不擅自补全。状态在简单线性基准中按数值输入，这是其限制。数据中同模型原始输入的重复记录数为 {meta['duplicate_financial_rows']}，保留且整组分配；记录ID并不代表时间顺序。

## 2. 四部分验证

{table(splits)}

先按19个账户输入的哈希分组，再用固定种子的10折分层分组划分组成约60%训练、10%校准、10%验证、20%测试。分组会使人数与违约率略有不同。模型和WOE只拟合训练集；sigmoid仅拟合校准集；模型/校准方法与策略阈值仅在验证集选择；测试集只做冻结后的评估。没有把ID排序伪装为时间外验证。验证集同时承担模型与策略选择，存在选择乐观偏差；最终测试隔离，但本结果仍是单一划分、单一历史群体的条件结果。

## 3. 模型和概率质量

无测试集调参、无过采样、无类别重加权。模型容量固定在 configs/credit.json。树为sklearn直方图梯度提升，不称为XGBoost或LightGBM。概率校准在基模型logit上拟合sigmoid，未重训基础模型。

**验证集选择依据：**

{table(validation[['model','auc','brier','log_loss']])}

**冻结后的测试比较：**

{table(comparison[['model','auc','ks','average_precision','brier','log_loss']])}

AUC衡量排序，不能说明概率是准确的；Brier与Log loss同时受校准和区分能力影响，不能单独证明校准改善。图中的可靠性曲线和 reliability.csv 一并给出各概率区间的预测均值、实际违约率及样本数。校准不保证每次都改善，保留全部结果。

## 4. 策略及成本假设

拒绝定义为 p≥阈值；成本为误拒好客户数+成本比×放行坏客户数。经验阈值在验证集所有可达的同分数分组边界上搜索，包含全拒绝/全通过；成本相同取更宽松策略。概率完全校准且上述固定成本成立时，理论阈值为1/(1+成本比)。二者是不同规则。

主成本比为{ratio}时：

{table(relevant)}

成本敏感性完整记录在 cost_sensitivity.csv；所有成本比均预先指定，各自只用验证标签选阈值，不能依据测试成本反向选择现实业务成本比。

**固定通过率目标：**

{table(approvals)}

阈值由验证集分位数确定；同分不强行拆开，所以实际通过率不一定等于目标，测试分布变化也会影响它。批准群体的违约率是已有客户标签的子集统计，不是实际审批后果。

## 5. 不确定性、解释与监控

使用{meta['bootstrap_repeats']}次测试集分组bootstrap，同一重复特征组整组重采样，模型差值共享抽样。区间只覆盖固定训练结果与固定策略下的测试抽样变动，不覆盖重新训练、历史年代或未来经济环境的不确定性。

feature_importance.csv 在验证集置换每一输入3次，记录AUC下降；这是模型依赖的描述，不是因果效应。原始字段与衍生字段相关，独立置换可能制造不自然组合，也会低估相互替代变量的重要性。

分数PSI参考分布取校准集，其固定分位区间用于测试集，半计数平滑后PSI={meta['test_score_psi_vs_calibration']:.5f}。这里只是同群体不同子样本对照，不是跨期稳定性或线上漂移验证。旧模拟实验的漂移告警仍独立保留。

## 6. 可复现与边界

源文件SHA256：`{meta['source_sha256']}`。拆分哈希、配置哈希、软件版本见 manifest.json。完整流水线已导出本地模型并验证加载后预测一致；这属于批量推理能力，不等于上线部署。逐客数据、预测与模型文件默认不提交仓库。

该数据较早且只有一组历史客户；没有逐笔敞口、回收率、违约发生时间、拒绝客户标签或真实业务成本。未实现OOT、拒绝推断、预期损失金额估计、实时服务与前瞻检验。排除人口属性不代表消除了代理歧视或满足任何具体监管要求。真实业务还需要近期多期数据、特征时点审核、业务成本估计与监控流程。

旧模拟基线及其曾接触测试集的历史限制保留在 docs/methodology.md；新实验不使用旧样本和旧分箱配置，不能把两者指标拼成同一组结论。
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
