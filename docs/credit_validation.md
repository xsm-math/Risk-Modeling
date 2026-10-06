# Real-data validation record — 2026-10-06 extension

## Local verification

- 37 offline tests passed: the 32 prior tests plus scorecard scale/parity,
  SQL month-order/negative-bill parity, PSI identity/unknown/missing cases,
  weighted/tree candidates and TreeSHAP additivity, and explicit confusion/PR definitions.
- Full pinned-UCI experiment completed on 30,000 rows, seed 20261002.
- Four disjoint group-aware splits: train 17,990; calibration 3,013;
  validation 2,999; test 5,998. All 817 duplicate financial-feature rows stay
  with their matching groups; none cross split boundaries.
- Twelve model/calibration candidates evaluated; selection based solely on validation
  log loss before test prediction. Selected: sigmoid-calibrated Random Forest.
- Original linear LR, WOE LR and histogram GBDT results reproduced. Source and
  LF-normalized split-manifest hashes match the prior published experiment. The
  published test cohort remains reused, not a fresh independent external holdout.
- SQL features executed on all 30,000 real histories; five shared features have
  maximum absolute error zero against Python.
- Unrounded additive scorecard points reconstruct raw WOE probability to maximum
  absolute error 6.67e-16 (tolerance 1e-12). The 600/620 point odds scale is tested.
- Exact native XGBoost TreeSHAP checked on 512 validation records; maximum raw-logit
  additivity residual 1.91e-6 within the float32-aware tolerance.
- Feature/score PSI computed against fixed training bins for real test and separately
  labeled limit/status stress; missing and unknown categories retained explicitly.
- 500 paired test-cluster bootstrap replicates completed.
- Serialized model reloaded; first 32 test predictions matched to 1e-12 absolute tolerance.
- EDA, ROC/PR, explanation, PSI and benchmark figures rendered and inspected.
- Both label-free batch scoring paths checked on 100 actual test rows; scorecard
  artifact produces probability, points and decisions matching saved-model output.
- Generated README/report links and placeholders checked.

## What the offline tests establish

Duplicate-group isolation and deterministic split coverage; row-local features
independent of ID/label/demographics; checksum failure and absent-data failure;
exact threshold costs against independent exhaustive tied-score enumeration;
all-reject/all-approve and tie behaviour; calibration labels cannot change the
base estimator; persisted prediction equivalence; identical-model paired bootstrap
returns zero effect.

These checks do not establish temporal generalization, economic validity or
production readiness. Tests use small offline fixtures; the UCI experiment is a
separate full-data run, not a download dependency of CI unit tests. The repository's
GitHub Actions workflow installs both dependency sets and runs the tests plus the
original simulation; remote status is reported by Actions, not inferred from local success.

Pinned source/config/split hashes, runtime versions and full results are in
`reports/credit/manifest.json`. Reproduction dependencies are recorded separately
from the original synthetic experiment's dependency snapshot.
