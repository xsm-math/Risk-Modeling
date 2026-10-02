# Real-data validation record

## Local verification

- 32 offline tests passed (22 existing scorecard tests plus 10 new test cases,
  counting four cost-ratio parameterizations).
- Full pinned-UCI experiment completed on 30,000 rows, seed 20261002.
- Four disjoint group-aware splits: train 17,990; calibration 3,013;
  validation 2,999; test 5,998. All 817 duplicate financial-feature rows stay
  with their matching groups; none cross split boundaries.
- Six model/calibration candidates evaluated; selection based solely on validation
  log loss before test prediction. Selected: uncalibrated histogram gradient boosting.
- 500 paired test-cluster bootstrap replicates completed.
- Serialized model reloaded; first 32 test predictions matched to 1e-12 absolute tolerance.
- Four-panel SVG figure rendered and visually inspected for clipping and legibility.

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
