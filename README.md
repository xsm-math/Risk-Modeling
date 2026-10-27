# Credit Risk Modeling

**Statistical Learning & Risk Decision** · Calibration, capacity and allocation

![Python: Research](https://img.shields.io/badge/Python-Research-236d91?style=flat-square) ![Focus: Statistics](https://img.shields.io/badge/Focus-Statistics-236d91?style=flat-square) ![Evidence: Reproducible](https://img.shields.io/badge/Evidence-Reproducible-258a75?style=flat-square)

[Portfolio](https://github.com/xsm-math) · [01 Inventory](https://github.com/xsm-math/inventory-allocation) · [02 Credit Risk](https://github.com/xsm-math/Risk-Modeling) · [03 Gold Allocation](https://github.com/xsm-math/invest)

[![Tests](https://github.com/xsm-math/Risk-Modeling/actions/workflows/tests.yml/badge.svg)](https://github.com/xsm-math/Risk-Modeling/actions/workflows/tests.yml)

| Project Summary | Historical credit-card default modeling from interpretable scores to calibrated probabilities and cost policies. |
|---|---|
| Research Question | When do more accurate probabilities change actions, and when do current-batch constraints require allocation optimization? |
| Methods | WOE/RF calibration, frozen cost rules, label-free hard capacities, exact unit-weight prefixes and shared-budget MILP, paired group uncertainty and row-level audits. |
| Key Results | RF calibration leaves validation-threshold row decisions unchanged but lowers theoretical 1:5-rule cost from **.7011 to .5420**. Two hard rules have **zero violations in 1,240 batch-decision cells**. Unit-weight sorting matches MILP; the declared heterogeneous-budget primary comparison does not establish retrospective cost improvement. |
| Evidence | [Calibration/capacity conclusions](docs/decision_results.md), [budget conclusions](docs/allocation_results.md), [frozen action comparisons](reports/decision_study/2026-10-27/action_agreement.csv), [budget effects](reports/allocation_study/2026-10-27/paired_effects.csv). Same historical cohort/split; costs and resource weights are hypothetical. |

![Calibration and decision costs](reports/decision_study/2026-10-27/calibration_cost.png)

## Decision conclusions

- **Ranking-only actions:** raw/calibrated WOE and RF give identical validation-
  threshold memberships at all five declared cost ratios, and identical capacity-
  filling memberships. Per-row comparisons establish this, not just equal AUC.
- **Probability-dependent actions:** the theoretical 1:5 threshold changes 2,682
  RF decisions after calibration and reduces hypothetical cost by .1591 per row,
  conditional paired 95% interval [.1266,.1912]. Fixed .5 instead worsens cost.
- **Hard operational capacities:** a validation cutoff is not a new-batch quota.
  At 60%, it exceeds the 5,998-row batch cap by 16; the label-free hard rules
  respect every declared batch cap. Artificial batch schemes reuse the same people.
- **Budget allocation:** unit weights have an exact sorted-prefix solution.
  Under hypothetical heterogeneous weights, all 480 MILPs meet resource and
  planning-precision checks, but the primary actual-cost difference versus strong
  PD greedy is -.001640 per row, interval [-.004152,.001030]. This supports retaining
  simple feasible PD greedy as the practical default for this declared design.

The extensions reproduce the original split and keep fixed learners. They are
retrospective mechanism studies, not new untouched/OOT evaluations, causal
approval profit or realized-loss guarantees. [Protocols](docs/decision_protocol.md)
and [budget assumptions](docs/allocation_protocol.md) distinguish prediction,
planning quality and retrospective outcomes. All 91 tests pass locally on the
pinned Python 3.12 runtime; CI also checks published evidence hashes and ledgers.

[Research report](reports/credit/REPORT.md) · [Protocol](docs/credit_protocol.md) · [Data dictionary](docs/credit_data_dictionary.md)

## Abstract

**Technical audit (2026-10-21):** [Findings, fixes and 68-test verification](docs/TECHNICAL_AUDIT.md) · [Shuffled-label control](reports/audit/2026-10-21/negative_control.csv). Isolated runs include their row records and models under the chosen output.

**Decision extension (2026-10-27):** [Calibration/capacity results](docs/decision_results.md)
and [budget results](docs/allocation_results.md). Original benchmark outputs below
remain preserved. Public-only verifiers check sources, hashes and aggregate
constraints; private replay additionally checks row memberships and cost effects.

```bash
python scripts/verify_decisions.py --run-dir reports/decision_study/2026-10-27
python scripts/verify_allocation.py --run-dir reports/allocation_study/2026-10-27
```

The manual `Credit calibration capacity and budget reproduction` workflow
downloads the pinned source, refits the original benchmark, runs the two decision
stages in order and performs full private-row verification. It publishes only
aggregate research artifacts, leaving customer rows/model binaries local.

This applied research project estimates next-month default risk for existing
credit-card customers using 30,000 public historical records. It compares an
additive WOE logistic scorecard with raw logistic regression, Random Forest,
XGBoost and histogram gradient boosting. Independent probability calibration,
cost-based thresholds, grouped uncertainty estimates and drift diagnostics
connect predictive performance to a transparent decision exercise.
All reported numbers are generated by executed experiments.

## Problem Definition

A useful risk model must rank customers, estimate credible default probabilities
and support decisions with explicit costs. A high AUC alone does not settle an
approval policy. We study normalized costs for rejecting a non-defaulting customer
and approving a defaulting customer; these scenarios are not actual bank profits.
The task concerns existing cardholders, not an identified causal effect of
accepting or rejecting new applicants.

## Statistical Formulation

For each numeric variable, the inherited encoder fits zero-aware quantile bins
and merges sparse adjacent bins on train. The convention is
`WOE = log(bad share / good share)`; IV sums `(bad share − good share) × WOE`.
EPS smoothing and the encoder's two-bin stopping rule are documented.
Missing values use train medians; the verified source has no missing cells.
No monotonicity constraint or IV-based test selection is claimed.

```text
logit(PD) = b + sum(beta_j * WOE_j)
factor = PDO / ln(2)
offset = base_score - factor * ln(base_odds_good_to_bad)
score = offset - factor * logit(PD)
```

600 points correspond to good:bad odds 20:1, with PDO=20: doubling good:bad odds
adds 20 points. Standardized LR coefficients are restored to WOE units before
conversion. [Bin points](reports/credit/scorecard_bins.csv),
[IV and coefficients](reports/credit/iv_coefficients.csv) and the
[portable score specification](reports/credit/scorecard.json) are exported.
The summed unrounded points must reconstruct the raw WOE model probability
within 1e-12. A separately calibrated WOE model is not silently substituted for this card.

## Data

[Yeh (2009), UCI Default of Credit Card Clients](https://doi.org/10.24432/C55S3H),
CC BY 4.0: six months of payment status, bills and payments from April–September
2005 and a next-month default label. Observed default rate: **22.12%**.
The official archive is downloaded with a pinned SHA256 and strict schema checks.
No synthetic fallback is used for this benchmark.

The quality audit reports missing cells, invalid/nonfinite values, duplicate
financial histories, conflicting labels, extreme quantiles and undocumented codes.
Negative bills and high utilization are preserved as potentially valid behavior.
ID and all four demographic fields are excluded from prediction. Identical
financial records stay in the same partition, including conflicting-label groups.
Full-population quality counts are descriptive; labeled feature analysis uses train only.

![Training quality and EDA](reports/credit/eda.png)

## Methodology

### ML Benchmarks

Six base estimators each receive a holdout sigmoid variant:

| Base estimator | Purpose and imbalance strategy |
|---|---|
| Linear LR | Standardized raw-feature probability baseline |
| WOE LR | Additive scorecard baseline; all 24 features retained |
| Balanced linear LR | `class_weight=balanced` ablation |
| Random Forest | 250 trees, depth 10, leaf minimum 30, balanced bootstrap weights |
| XGBoost | 250 shallow trees, `scale_pos_weight=1`, regularized probability learner |
| Histogram GBDT | Existing fixed-capacity benchmark retained for continuity |

No oversampling is used. Class weighting can distort raw probability levels;
independent calibration and cost thresholds are evaluated alongside it.
The same rows, 24 account inputs and probability objective apply across learners.
Actual parameter dictionaries, package versions and checksums accompany the results.

### Calibration

Sigmoid fits on clipped base logits using the calibration holdout without
refitting the base learner. Raw and calibrated variants are retained, including
negative results. Validation, not test performance, chooses the final variant.
Reliability plots include per-bin sample counts; Brier alone cannot distinguish
calibration from discrimination. See [probability metrics](reports/credit/test_metrics.csv)
and [reliability bins](reports/credit/reliability.csv).

### SQL Feature Engineering

[Executable SQLite SQL](sql/behavior_features.sql) unpivots actual six-period
histories, then uses CTEs, CASE and ROW_NUMBER for utilization, delinquency counts,
max delay, payment/bill ratios and recent-three-period aggregates.
Five shared features are compared with Python on all 30,000 records with 1e-12
tolerance. The extra three-period outputs illustrate SQL and do not alter the model
silently. Account age is unavailable because opening dates are absent.

## Experimental Design

Ten stratified group folds form approximately 60% train, 10% calibration, 10%
validation and 20% test. Train fits preprocessing and estimators; calibration fits
sigmoid mappings; validation chooses model/calibration by minimum log loss and
sets policies; test evaluates the frozen choices. Capacities and cost scenarios
are fixed in [configs/credit.json](configs/credit.json), without a large search.

**Revision provenance:** this extension reuses the previously published 2026-10-02
split for continuity. It does not claim a new untouched or external holdout.
Validation serves both model and threshold selection. Bootstrap intervals describe
conditional test sampling uncertainty, not the uncertainty of retraining or future cohorts.

### Metrics and uncertainty

ROC-AUC and KS measure ranking; average precision and trapezoidal PR-AUC are
reported separately. Precision, recall, F1 and confusion counts are tied to explicit
policies. Brier and log loss evaluate probability quality, alongside a constant
training-prior baseline. Five hundred paired group bootstrap samples quantify the
selected model's AUC, its difference versus WOE and its hypothetical cost difference.

![ROC and PR](reports/credit/roc_pr.png)

## Results

Validation selects **random_forest_sigmoid**; the following rows summarize frozen test results.
All twelve candidate variants and the constant-prior baseline remain in the CSV.

| model | auc | ks | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- | --- |
| linear_lr | 0.7662 | 0.4366 | 0.5185 | 0.1383 | 0.4402 |
| woe_lr | 0.7774 | 0.4308 | 0.5478 | 0.1343 | 0.4298 |
| hist_gbdt | 0.7924 | 0.4519 | 0.5631 | 0.1322 | 0.4219 |
| random_forest | 0.7917 | 0.4580 | 0.5700 | 0.1724 | 0.5251 |
| random_forest_sigmoid | 0.7917 | 0.4580 | 0.5700 | 0.1322 | 0.4216 |
| xgboost | 0.7936 | 0.4553 | 0.5728 | 0.1315 | 0.4201 |

The selected policy has precision **38.45%**, default recall **75.41%**,
F1 **0.5093** and approval rate **56.64%**. Its normalized cost is
**0.5387 per customer**, versus **0.5600** for the raw WOE baseline,
each using its own validation-selected cutoff. This trade-off is a retrospective
scenario, not realized savings or an estimated effect of rejecting customers.

![Benchmarks, reliability and policy scenarios](reports/credit/benchmark.svg)

Explanations include WOE coefficients and additive bin points, validation
permutation importance, RF impurity/XGBoost gain importance, and native exact
TreeSHAP for 512 XGBoost validation records. TreeSHAP additivity is verified in raw
log-odds space; explanations are descriptive, not causal.

| Consideration | WOE scorecard | Boosting |
|---|---|---|
| Representation | Additive bins and points | Nonlinear splits and interactions |
| Explanation | Direct bin contributions; coefficients conditional on other inputs | Permutation/SHAP subject to feature dependence |
| Deployment | Small rules and logistic arithmetic | Versioned model runtime |
| Governance | Audit binning and point mappings | Audit capacity, explanations and serving compatibility |
| Limit | Coarse bins, correlated inputs, unconstrained monotonicity | More operational complexity; gain importance can mislead |

## Robustness and Sensitivity Analysis

Feature and score PSI use training-fixed bins, missing/unknown states and
half-count smoothing. Compare the same-cohort test population with a clearly
labeled simulation: reduce credit limits by 30% and increase nonnegative latest
payment status by one; then recompute dependent features. This tests sensitivity
and monitoring logic, not observed future default accuracy.
PSI 0.1/0.25 thresholds are review heuristics. Small PSI does not establish stable
conditional risk or calibration. Genuine OOT cannot be constructed from this single cohort.

![PSI](reports/credit/stability.png)

[Cost sensitivity](reports/credit/cost_sensitivity.csv) varies false-rejection/missed-default ratios 1, 2, 5, 10 and 20.5; grouped bootstrap intervals retain duplicate-input groups. Calibration ablations preserve adverse results. These quantify conditional retrospective variation, not performance on a new cohort.

## Business Interpretation

The primary scenario costs a false rejection 1 unit and a missed default 5 units.
Empirical cost is `FP + 5 × FN`; validation selects a threshold over whole tied-score
blocks, including all-approve/all-reject. If probabilities and costs are accurate,
the expected-cost rule is `PD >= 1/(1+5)`. The selected empirical threshold is
**0.171438**. The report compares both rules, 0.5, all-approve and all-reject,
plus cost ratios 1, 2, 5, 10 and 20.5 and validation-defined approval targets.
Actual EAD and LGD are unavailable; normalized loss is not a currency estimate.

The model supports a transparent probability-to-cost exercise; changing the loss ratio changes the policy. Better ranking does not itself determine rejection capacity or actual lending profitability. The WOE scorecard remains an auditable additive benchmark, while calibrated tree models add probability and runtime complexity.

## Limitations

Historical Taiwan existing-cardholder data and a reused published split provide
limited external evidence. No real rejected applicant records, actual recovery
costs, mature multi-cohort labels, deployment or prospective financial outcomes
are available. Status -2/0 meanings are incompletely documented; LR treats status
numerically. Binning is not monotonic, IV is univariate, correlated inputs complicate
explanation, and excluding demographic fields does not establish fairness or legal compliance.
Reject inference is discussed as a selection-bias problem, without fabricated labels.

## Future Work

Obtain recent snapshot-dated cohorts and point-in-time feature availability; assess
rolling OOT ranking, calibration and cost policies. Compare monotonic scorecards,
feature-group ablations, segment fairness, missingness stress, capacity constraints
and loss estimates from observed EAD/LGD. Add label-maturity-aware monitoring and
shadow evaluation before any prospective decision use.

## Repository Structure

| Directory | Role |
|---|---|
| `src/credit/` | Verified loading, features, model/calibration, decisions, diagnostics, SQL checks, reporting |
| `src/risk/`, `src/train.py` | Original NumPy synthetic scoring study |
| `configs/` | Frozen experiment configuration |
| `data/raw/`, `data/processed/` | Local raw data and row-level audit records |
| `scripts/`, `sql/` | One-command workflow and executable behavior aggregation |
| `tests/` | Numerical, split, calibration, scorecard, PSI, SQL and serialization checks |
| `reports/credit/` | Published aggregate figures, tables and provenance |
| `docs/` | Experiment protocol, monitoring and scorecard methodology |

Original study: [synthetic baseline](SYNTHETIC_BASELINE.md) and
[historical limitations](docs/methodology.md), run via `python -m src.train`.
Its results must not be combined with the real-data benchmark.

References: [UCI metadata](https://www.archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients),
[scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html),
[XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).

## Reproducibility

Python 3.11/3.12; the published runtime is pinned in
[requirements-credit-repro.txt](requirements-credit-repro.txt).

```bash
git clone https://github.com/xsm-math/Risk-Modeling.git
cd Risk-Modeling
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-credit-repro.txt
python -m pytest -q
python scripts/audit.py --negative-control
```

The audit downloads/verifies data, trains, evaluates, reconstructs saved metrics and runs a four-partition shuffled-label control. Outputs live under `reports/generated/audit/`; committed reference tables and README are preserved. The explicit maintenance command `python scripts/run_all.py` still refreshes the reference report and README.
For a custom isolated experiment:

```bash
python -m src.credit download
python -m src.credit run --output reports/generated
```

All explicit experiment outputs also contain `local/processed` rows and `local/artifacts` models. Batch inference after the audit (replace the model path for a custom run):

```bash
python -m src.credit score --input customers.csv --output predictions.csv --model reports/generated/audit/reproduction/local/artifacts/credit_model.joblib
python -m src.credit score --input customers.csv --output scorecard_predictions.csv --model reports/generated/audit/reproduction/local/artifacts/credit_scorecard.joblib
```

The input needs the 19 raw account fields; labels are unnecessary. The scorecard
artifact also emits points. Load only trusted model artifacts. Source archives,
split IDs, individual predictions and fitted models stay local and are ignored by Git.



Presentation-only summary: `python scripts/portfolio_summary.py --project credit`. Input hashes are recorded in [the chart manifest](assets/portfolio/manifest.json). Regenerating the README uses [the same research template](docs/README.template.md).
