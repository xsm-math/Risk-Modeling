# Technical audit — 2026-10-21

Status: reproducible public-cohort research; external/temporal validation remains absent. Baseline `d5bf4d0`; 37 existing tests passed, updated suite passes 68 on Python 3.12.14.

## Scope

Reviewed official-archive checks, row-local features, duplicate-input grouped partitions, WOE preprocessing, fixed-capacity candidates, independent calibration, validation-only selection/thresholds, test evaluation, grouped bootstrap, SQL parity, scorecard/TreeSHAP, drift diagnostics, persistence and reproduction. Existing synthetic modules remain separate; their WOE dependency and tests were included.

## Findings and repairs

| Finding | Evidence before repair | Repair |
|---|---|---|
| Missing probabilities produced inconsistent ledgers | Labels [1,0], probabilities [NaN,.2] gave cost 2.5 but confusion counts implied 0 | Shared finite, binary, aligned-array validation for decisions/metrics/thresholds |
| Approval targets failed at endpoints or small samples | Target 100% on [.1,.2,.3] approved only two records; quantile rounding could exceed a ceiling | Greatest attainable whole-score-block approval count <= floor(target*N), including 0/100% |
| Explicit report output did not isolate all state | Row records and artifacts still used shared root paths | Explicit output now contains `local/processed` and `local/artifacts`; default legacy paths remain compatible |
| Prediction text precision obscured exact reconstruction | Float32 tree probabilities promoted by CSV readers could change recomputed metrics | Export 17 significant digits and read with round-trip precision |
| Reproduction lacked code fingerprints / executable aggregate checks | Dataset hashes existed but verification was fragmented | Separate `code_sha256`, dataset `source_sha256`, validation predictions and an audit CLI |

Rows/models under `local/` are ignored and are excluded from uploaded evidence. The audit checks split coverage and group separation, reconstructs all test metrics from saved probabilities, verifies selection from validation log loss, reproduces thresholds and reconciles confusion-based costs. Maximum metric reconstruction error: 8.33e-17.

## Real-data reproduction and negative control

The 30,000-row official archive was downloaded and checksum verified. Baseline and repaired full experiments were rerun. Selected model remains `random_forest_sigmoid`; its rounded test AUC 0.7917, KS 0.4580, Brier 0.1322 and normalized cost 0.5387/customer remain. Floating-point accumulation/serialization corrections may affect low-order digits of native float32 challenger metrics; all-model maximum reference differences are in [verification](../reports/audit/2026-10-21/verification.json).

The negative control independently permutes labels inside all four fixed partitions, retains features/model capacity, selects with permuted validation labels and evaluates once against permuted test labels. Seed 20262001; selected-control test AUC 0.508985. All 12 candidates are retained in [negative_control.csv](../reports/audit/2026-10-21/negative_control.csv). A separate original-label diagnostic is explicitly labeled. A single permutation near chance is not proof that leakage is impossible and is not a new external test.

## Reproduce

```bash
python -m pip install -r requirements-credit-repro.txt
python -m pytest -q
python scripts/audit.py --negative-control
```

Default audit output is `reports/generated/audit`. Use `--run-dir` to verify an already generated isolated run and `--output` to retain a separate check directory. Custom output models live under `<output>/local/artifacts`; pass that model path to `python -m src.credit score --model ...`. The manual `Full credit technical audit` workflow uploads aggregates only.

## Evidence and limits

[Machine audit](../reports/audit/2026-10-21/audit.json) · [Reproduced test metrics](../reports/audit/2026-10-21/reproduced_test_metrics.csv).

Training/calibration/validation/test responsibilities were inspected and tested. Validation performs both model and threshold selection. The historical split was previously published; this audit does not create an untouched holdout, OOT, new-applicant evaluation or actual bank loss evidence. Bootstrap remains conditional on fixed fitted models/rules. WOE lacks enforced monotonicity; TreeSHAP explains raw XGBoost rather than calibrated RF; PSI stress lacks new labels. External dated cohorts, mature outcomes, monetary EAD/LGD, fairness and production monitoring remain future work.
