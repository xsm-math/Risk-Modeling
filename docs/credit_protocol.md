# Real-data experiment protocol

This protocol is implemented in `src/credit/`, with fixed settings in `configs/credit.json`.
The original NumPy simulation remains a separate study (`python -m src.train`).

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
7. Choose among six fixed candidates by validation log loss; break ties by name.
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

## Reproduction and provenance

Raw files, row-level split manifests/predictions and joblib artifacts stay local.
The published manifest records dataset, config and split hashes plus runtime versions.
The current run exports and reloads the model to verify prediction equivalence.
If a source checksum changes, stop and review provenance rather than silently accept it.
Subsequent experiments must be labelled as revisions; repeated tuning against the
published test results would invalidate its status as an untouched holdout.

## References

- Yeh, I. (2009), Default of Credit Card Clients, UCI, DOI https://doi.org/10.24432/C55S3H.
- UCI metadata and licence: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- Histogram boosting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
