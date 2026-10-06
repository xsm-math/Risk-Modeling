# Monitoring specification and implementation boundary

The repository exports local batch models and offline diagnostics. It has no
production endpoint, live alerts, real approvals or automatic retraining service.
The following specification describes how the implemented artifacts would fit into
a governed monitoring process.

## Before labels mature

| Check | Candidate monitoring signal | Existing implementation |
|---|---|---|
| Input validity | Required fields, type/range, nonfinite values, positive limit | Schema/checksum checks for source; financial-value checks for scoring |
| Availability | Event time and ingestion time no later than observation cutoff | Assumed observed UCI window; no actual ingestion timestamps |
| Population | Feature PSI, missing/unknown states, volumes and segment mix | Train-fixed PSI bins; explicit missing/unknown buckets in diagnostics |
| Output | Mean PD, score distribution, approval and threshold version | Model/threshold artifact and same-cohort/stress probability PSI |
| Serving | Latency, error rate, artifact version, rollback readiness | Artifact roundtrip check; no real serving infrastructure |

Track missingness and new codes separately, since a small aggregate PSI can hide
an operational problem. All transformations, bins and reference periods must be
versioned. Candidate PSI review heuristics (0.1, 0.25) require local sample-size and
seasonality assessment. Review a change in source quality or policy before choosing
to recalibrate/retrain; a trigger alone does not diagnose its cause.

## After labels mature

Compare monthly vintage and segment AUC, KS, AP, confusion metrics under the actual
threshold, predicted/observed PD, reliability curves, Brier and log loss. Record the
prediction cutoff, outcome horizon and label-maturity date. Partial labels can make
current-period bad rates look deceptively low. Measure actual losses with observed
EAD/LGD and business opportunity costs; do not replace them with credit limits.

Approval changes observed sample composition. Report coverage and selection bias,
and avoid treating rejected unlabeled applicants as known good/bad. A stable score
distribution does not rule out concept drift. Persistent miscalibration may support
calibration review; ranking deterioration may require feature/model review. New
champions should be evaluated on recent untouched cohorts, shadow runs and capacity
constraints, with retained rollback artifacts.

## Reject inference and OOT

This dataset contains no rejected applicant population and no separate observation
cutoffs for genuine OOT. Parceling, extrapolation or inverse propensity weighting
would require explicit assumptions and data on selection/coverage; predicted labels
are not truth. No fabricated rejected records are included. The six months within
each row are explanatory histories, not six independent labeled cohorts.
Future production data should include application/snapshot time, observed feature
availability, decision policy, matured outcomes and legally obtained representative
coverage before examining these methods.

## Exact interpretation artifacts

`reports/credit/iv_coefficients.csv`: conditional WOE-LR coefficients on original
WOE units. `scorecard_bins.csv`: per-input interval points, with intentionally
repeated points for merged intervals. `xgboost_shap.csv`: mean absolute exact
TreeSHAP contributions for the fixed XGBoost challenger, in raw log-odds space.
Do not assign these SHAP values to a different selected model or calibrated PD.
Correlation complicates importance and reason contributions; none is causal.
