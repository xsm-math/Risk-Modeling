"""Research manuscript and README populated from executed artifacts."""
import json
from pathlib import Path
import pandas as pd


def table(df, digits=4):
    df=df.copy()
    for c in df.select_dtypes('number'):
        df[c]=df[c].map(lambda v:f'{v:.{digits}f}' if pd.notna(v) else 'NA')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join(
        '| '+' | '.join(map(str,row))+' |' for row in df.to_numpy())


def write_report(out, comparison, validation, policies, costs, approvals, splits, meta):
    selected=meta['selection']['model'];q=meta['quality_audit'];ratio=meta['selection']['primary_cost_ratio']
    ci=meta['intervals'];s=meta['scorecard'];d=meta['stability']
    decisions=pd.read_csv(out/'decision_metrics.csv')
    important=pd.read_csv(out/'feature_importance.csv').head(10)
    iv=pd.read_csv(out/'iv_coefficients.csv').head(12)
    psi=pd.read_csv(out/'feature_psi.csv')
    calibration=comparison[['model','brier','log_loss','mean_predicted','default_rate']]
    report=f'''# Credit default: scorecards, tree benchmarks and decisions

## Abstract

This extension studies {q['rows']:,} UCI existing-cardholder records. Six fixed-capacity
base estimators and their holdout sigmoid variants compare ranking, probability
quality and asymmetric decision costs. Validation log loss selects **{selected}**.
All tables below come from the executed pipeline. This is an applied research
exercise on a historical cohort, with normalized hypothetical costs.

## Dataset and quality

Yeh (2009), [UCI / DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H), CC BY 4.0.
Six observed months (April–September 2005) predict the next-month default label.
Amounts are NT dollars. There are {q['defaults']:,} defaults ({q['default_rate']:.2%}),
{q['missing_cells']} missing cells, {q['duplicate_ids']} duplicate IDs,
{q['duplicate_financial_rows']} duplicate financial rows and {q['conflicting_label_groups']}
duplicate-input groups with conflicting labels. All conflicting labels are preserved.
There are {q['negative_bill_cells']} negative bill entries and
{q['utilization_over_one']} latest bill/limit values above one. These are flags,
not automatic grounds for deletion. Nonpositive limits: {q['nonpositive_limits']}.

`data_quality.csv` includes missingness, finite-value checks, quantiles and IQR flags
for every raw field. `category_frequencies.csv` flags undocumented codes. Negative
bills are retained; status -2/0 and education/marriage codes outside the documented
sets are not silently relabeled. Label relationships and correlations use train only.
ID and four demographic fields are excluded from models; proxy bias remains possible.

![Training EDA](eda.png)

## Methodology and information boundaries

{table(splits)}

Group identical financial histories before assigning ten stratified grouped folds:
0–5 train, 6 calibration, 7 validation, 8–9 test. WOE, scaling and base estimators
fit train only. Sigmoid fits calibration only. Validation selects model and policies.
The selected configuration is written before test predictions. Hyperparameters are
fixed, not searched against this test set. The 2026-10-02 published split is reused
for continuity; this revision is **not a new untouched external holdout**.
The validation set has dual model/threshold selection duties, so its minima are optimistic.

## Scorecard formulation

The inherited encoder uses numeric quantile bins, zero-aware cuts, adjacent sparse-bin
merges, and log(bad share / good share) with EPS smoothing. It stops merging at two
bins; minimum counts are therefore diagnostics, not unconditional guarantees.
Missing values use train medians (the verified UCI file has none). No monotonicity
or regulatory suitability is claimed. IV is exported for interpretation, not used
to cherry-pick predictors. All 24 inputs enter the WOE baseline; correlated inputs
can share univariate signal.

```text
WOE_jk = log[(bad_jk / total_bad + eps)/(good_jk / total_good + eps)]
IV_j = sum_k (bad_share_jk - good_share_jk) * WOE_jk
logit(PD) = b + sum_j beta_j * WOE_j(bin(x_j))
beta_j = standardized_beta_j / training_scale_j
b = standardized_intercept - sum_j beta_j * training_mean_j
factor = PDO / log(2); offset = base_score - factor * log(base_odds)
score = offset - factor * logit(PD)
      = base_points + sum_j [-factor * beta_j * WOE_j]
```

Good:bad odds 20:1 correspond to 600 points; doubling good:bad odds adds 20 points.
Score probability reconstruction maximum absolute error:
**{s['max_probability_reconstruction_error']:.3e}**. The raw WOE model, not its
separate sigmoid variant, defines this scorecard. `scorecard_bins.csv` lists original
numeric intervals and their final merged WOE values; repeated values after merging
are intentional. Points are unrounded in the specification.

{table(iv)}

## ML benchmarks and imbalance

Fixed models: raw LR, WOE LR, histogram GBDT, balanced raw LR, balanced-subsample
Random Forest and XGBoost. Tree capacities are explicit in `configs/credit.json`.
XGBoost retains scale_pos_weight=1 to prioritize probability estimation; balanced
LR is an ablation and RF uses balanced bootstrap weights. No oversampling is used.
The sigmoid variants fit on a population-preserving calibration holdout; class
weights can alter raw PD estimates. Threshold choice is the primary cost-sensitive
strategy. Each learner receives the same rows, features and validation objective.

{table(validation[['model','auc','average_precision','brier','log_loss']])}

## Evaluation

{table(comparison[['model','auc','ks','average_precision','pr_auc_trapezoid','brier','log_loss']])}

Average precision uses non-interpolated recall increments; trapezoidal PR-AUC is
reported separately. Linear PR interpolation can inflate the constant/tied-score
baseline's area; AP is the primary PR summary. Precision/recall/F1 and confusion counts belong to a stated
threshold, not an intrinsic ranking metric. Selected model AUC 95% grouped-bootstrap
interval: [{ci['auc']['lower']:.4f}, {ci['auc']['upper']:.4f}]. Paired AUC difference
versus WOE interval: [{ci['auc_gain_vs_woe']['lower']:.4f}, {ci['auc_gain_vs_woe']['upper']:.4f}].
The {meta['bootstrap_repeats']} paired cluster replicates retain duplicate-input groups and only measure
test sampling uncertainty conditional on frozen training and selection.

![ROC and PR curves](roc_pr.png)

## Calibration and probability quality

{table(calibration)}

Mean PD, Brier and log loss accompany ten-bin reliability counts in `reliability.csv`.
Brier and log loss combine discrimination and calibration; neither alone establishes
calibration improvement. Sigmoid is fitted on clipped raw logits without refitting
the base learner. Calibration is selected only if validation log loss favors it;
test results are retained even when they worsen.

![Model, reliability and decisions](benchmark.svg)

## Decision threshold and costs

For false rejection cost 1 and missed-default cost r, observed normalized loss is
FP + r*FN. Expected rejection cost is (1-PD); expected approval cost is r*PD, so
the probability rule rejects when PD >= 1/(1+r). The empirical rule minimizes
validation cost over entire equal-score blocks including approve/reject-all;
ties prefer fewer rejections. Primary r={ratio} is fixed before evaluation.
This is not a currency estimate of PD*LGD*EAD: the dataset lacks actual losses
and exposures. LIMIT_BAL must not be substituted for contractual EAD.

{table(costs[costs.ratio==ratio][['policy','threshold','cost_per_customer','approval_rate','approved_default_rate','bad_capture']])}

All model-specific validation thresholds, including calibrated and weighted
ablations, are evaluated transparently:

{table(policies[['model','threshold','precision','recall','f1','cost_per_customer','approval_rate']])}

The paired cost saving versus WOE interval is
[{ci['cost_saving_vs_woe']['lower']:.4f}, {ci['cost_saving_vs_woe']['upper']:.4f}]
normalized units per record. Full ratio scenarios and 0.5 comparisons are in
`cost_sensitivity.csv` and `decision_metrics.csv`. Test labels never choose these rules.

## Stability and population shift

{table(psi.sort_values('psi',ascending=False).head(12))}

Training-fixed quantile bins (raw status codes use category bins) include explicit
missing/unknown states and half-count smoothing. Same-cohort maximum PSI is
{d['same_cohort_max_psi']:.4f}. A simulated limit x0.7 and latest nonnegative status
+1 scenario, with derived features recomputed, gives maximum PSI
{d['stress_max_psi']:.4f}; approval changes from {d['same_cohort_approval_rate']:.2%}
to {d['stress_approval_rate']:.2%}. This is a pipeline stress experiment with no new
labels, not observed temporal drift or default-performance evidence.
0.1/0.25 PSI triggers are review heuristics, not significance tests or universal standards.

![Stability](stability.png)

## Explanation and scorecard/tree trade-off

{table(important)}

Train WOE coefficients and bin points give additive model-specific explanations.
Validation permutation importance reports AUC decrease and repeat variability;
correlated histories and derived features complicate single-feature interpretations.
RF impurity and XGBoost gain importances are in `tree_importance.csv` and are not
directly comparable scales. Native exact TreeSHAP explains 512 validation records
of the XGBoost challenger in raw log-odds space; additivity is checked and aggregate
mean absolute contributions are in `xgboost_shap.csv`. SHAP is not a causal effect.

![Explanations](explanations.png)

Scorecards offer small additive rule tables and simple reason contributions;
boosting handles nonlinearities/interactions but requires a model runtime and more
validation of explanations. Either needs governed features, versioning, calibration,
monitoring and local review. Interpretability does not establish legal compliance.

## SQL and reproducibility

`sql/behavior_features.sql` uses CTEs, CASE, ROW_NUMBER and six-period aggregations
on actual UCI account histories. Five shared Python/SQL features are checked on
all {q['rows']:,} records (`sql_parity.csv`); three-period delinquency and payments
are additional SQL demonstrations, not silently added model inputs. Account age
cannot be constructed because account opening dates are unavailable.

Source SHA256: `{meta['source_sha256']}`. Config, split hash and versions are in
`manifest.json`; source modification fails closed. Models and customer predictions
remain local. The exported model is reloaded and prediction equality checked.
Run `python scripts/run_all.py` after installing the reproduction requirements.

## Limitations and future work

One old existing-cardholder cohort, a reused published split and no actual losses
limit external validity. There are no different customer snapshot dates for genuine
OOT, no rejected applicants for reject inference, and no production deployments.
Six within-row months cannot be treated as six independently labeled cohorts.
Future work should obtain recent dated cohorts, mature labels and point-in-time
features; compare rolling OOT windows, missingness stress, monotonic scorecards,
fairness diagnostics and losses by segment before considering prospective decisions.
The original synthetic experiment and its historical test-exposure caveat remain separate.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')


def write_readme(root):
    root=Path(root);out=root/'reports/credit'
    meta=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    selected=meta['selection']['model'];metrics=pd.read_csv(out/'test_metrics.csv')
    policies=pd.read_csv(out/'model_policies.csv');m=metrics.set_index('model').loc[selected]
    p=policies.set_index('model').loc[selected];base=policies.set_index('model').loc['woe_lr']
    names=list(dict.fromkeys(['linear_lr','woe_lr','random_forest','xgboost','hist_gbdt',selected]))
    display=metrics[metrics.model.isin(names)][['model','auc','ks','average_precision','brier','log_loss']]
    changes={'{{RESULT_TABLE}}':table(display),'{{SELECTED}}':selected,
        '{{AUC}}':f'{m.auc:.4f}','{{KS}}':f'{m.ks:.4f}','{{AP}}':f'{m.average_precision:.4f}',
        '{{BRIER}}':f'{m.brier:.4f}','{{THRESHOLD}}':f'{p.threshold:.6f}',
        '{{COST}}':f'{p.cost_per_customer:.4f}','{{WOE_COST}}':f'{base.cost_per_customer:.4f}',
        '{{APPROVAL}}':f'{p.approval_rate:.2%}','{{RECALL}}':f'{p.recall:.2%}',
        '{{PRECISION}}':f'{p.precision:.2%}','{{F1}}':f'{p.f1:.4f}',
        '{{BAD_RATE}}':f'{meta["quality_audit"]["default_rate"]:.2%}'}
    source=root/'docs/README.template.md'
    text=source.read_text(encoding='utf-8')
    for key,value in changes.items():text=text.replace(key,value)
    if '{{' in text:raise ValueError(f'Unfilled documentation placeholder: {source}')
    (root/'README.md').write_text(text,encoding='utf-8',newline='\n')
