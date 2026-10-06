# Real-data experiment protocol: 2026-10-06 extension

This protocol is implemented in `src/credit/`, with fixed settings in `configs/credit.json`.
The original NumPy simulation remains a separate study (`python -m src.train`).

This extension reuses the published 2026-10-02 cohort and split for continuity.
It is not a new untouched external holdout. Runtime pins are preserved because
grouped partition assignments can differ between library versions. New model
capacities are fixed before evaluating the extension.

## Scope

Estimate next-month default risk for existing cardholders after all six observed
months have become available. A retrospective decision exercise illustrates the
consequences of fixed asymmetric error costs. It does not identify the effect of
rejecting customers and does not estimate realized bank profit.

## Information boundaries

1. Download the exact UCI archive and verify SHA256 before parsing.
2. Exclude ID and all four demographic variables from model predictors.
3. Group identical 19-variable financial histories before splitting. Preserve all
   labels, including conflicting labels within a group; never deduplicate by outcome.
4. Use folds 0–5 for training, 6 for calibration, 7 for selection, 8–9 for testing.
   `StratifiedGroupKFold` uses labels to balance folds, not to learn predictors.
5. Fit WOE, scaling and estimators only on training. No supervised feature selection.
6. Fit logit-sigmoid calibration only on calibration. No base-estimator refit.
7. Choose among twelve fixed candidates by validation log loss; break ties by name.
8. Choose cost thresholds and target-approval cutoffs using validation only. Save
   `selection.json` before obtaining test predictions and computing test metrics.
9. Evaluate all frozen candidates on test for transparent comparison, without
   changing the selected model after reading the test table.

Validation does double duty for model and policy selection; it is not an unbiased
performance estimate. Fixed hyperparameters avoid a large search budget. The
final holdout evaluates this selection procedure only on one historical cohort.
No IID customer split can substitute for time-out-of-sample validation.

## Models

- Linear logistic regression: median imputer and StandardScaler fitted on train;
  fixed C=1, no class weighting. Status codes treated numerically as a simple baseline.
- WOE logistic regression: the existing train-only binning implementation, five
  target bins, 5% minimum bin proportion and 30 minimum defaults (merging stops
  at two bins, so minimum counts are not unconditional guarantees), then scaled LR.
  Numeric binning is not monotonic and unknown status codes are preserved as values.
- Histogram gradient boosting: 150 iterations, 15 leaves, learning rate .06,
  minimum leaf size 40, L2=3, no early stopping or test-based tuning.
- Each model also has a separate sigmoid-calibrated variant. Calibration is a
  one-variable logistic regression on clipped base logits, C=1e6. The slight
  regularization and clipping mean it is not an unpenalized analytic transform.
- Balanced linear LR: same preprocessing with class_weight=balanced as an ablation.
- Random Forest: 250 trees, max depth 10, min leaf 30, sqrt feature subsampling,
  balanced_subsample weights. Raw weighted PD requires a calibration quality check.
- XGBoost: 250 trees, depth 3, rate .04, min child weight 30, row/column sampling .85,
  L2=5 and scale_pos_weight=1. Probability loss is prioritized over default reweighting.
- No oversampling; cost threshold strategy and weighting ablations are compared on
  the same four parts. Raw and sigmoid variants are retained even when worse.
- Constant prior probabilities estimated from training provide a no-feature baseline.

## Decisions

A rejected non-default costs 1 unit; an approved default costs r units, where
r is in {1, 2, 5, 10, 20.5}. These are scenarios, not measured financial costs.
The primary scenario r=5 is specified before testing. The empirical minimizer
searches all tied-score block boundaries; equal costs choose fewer rejections.
The probability-based rule uses 1/(1+r) under a calibrated-probability assumption.

The approval-target curve uses validation quantiles without labels. Because the
rule never splits ties, realized approval can differ from the desired target.
Both distribution changes and finite sample variation also affect test approval.

## Uncertainty and explanation

500 paired cluster bootstrap replicates resample duplicate-feature groups on test;
95% percentile intervals compare the selected model with uncalibrated WOE LR,
including each model's own validation-selected threshold. The bootstrap is conditional
on training and selection; it does not measure algorithm retraining uncertainty.
Validation permutation importance is descriptive and not used to choose features.
Correlated derived features and out-of-distribution permutations limit interpretation.
Score PSI compares calibration and test with calibration-fixed quantile edges and
half-count smoothing. It is not temporal drift evidence.

The extension additionally computes all feature and selected-score PSI with
training-fixed bins; status features use categorical buckets, including unknown
and missing states. A separate simulated shift uses LIMIT_BAL x0.7 and increments
nonnegative PAY_0 by one, capped at 8, then recomputes all derived features. It has
no new labels and is not OOT accuracy evidence. PSI .1/.25 triggers are heuristics.

WOE coefficients are restored to original WOE units and translated to additive
points (base score 600, good:bad odds 20:1, PDO 20). Unrounded bin-point sums must
reconstruct raw WOE PD at absolute tolerance 1e-12. IV is exported, not used to
select inputs. Missing values use train medians; monotonicity is not constrained.
Each original interval is exported even when adjacent intervals share merged WOE.

Validation-only permutation explanation retains repeat variability. RF impurity
and XGBoost gain importance are model-specific. Native exact TreeSHAP explains
512 XGBoost validation records in raw log-odds space with checked additivity;
it does not explain a different selected model or its calibrated probability.

AP and trapezoidal PR-AUC are reported separately. Tied constant scores illustrate
why trapezoidal PR interpolation can be misleading; AP is the headline PR measure.
Confusion counts, precision, recall and F1 are tied to stated thresholds.

The SQLite behavior query runs on real six-period histories. Five shared features
are compared with Python on all 30,000 rows at 1e-12. Three-period outputs are SQL
demonstrations, not additional silent model inputs. Account age is unavailable.

## Reproduction and provenance

Raw files, row-level split manifests/predictions and joblib artifacts stay local.
The published manifest records dataset, config and split hashes plus runtime versions.
The current run exports and reloads the model to verify prediction equivalence.
The manifest also records quality, scorecard/SQL parity and native SHAP checks.
`python scripts/run_all.py` regenerates research results and README together.
See `docs/monitoring.md` for implemented diagnostics and proposed production monitoring.
If a source checksum changes, stop and review provenance rather than silently accept it.
Subsequent experiments must be labelled as revisions; repeated tuning against the
published test results would invalidate its status as an untouched holdout.

## References

- Yeh, I. (2009), Default of Credit Card Clients, UCI, DOI https://doi.org/10.24432/C55S3H.
- UCI metadata and licence: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- Histogram boosting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
