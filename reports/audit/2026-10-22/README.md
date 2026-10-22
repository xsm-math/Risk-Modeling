# Reproduction and reference reconciliation — 2026-10-22

Reviewed source baseline: `a11dbea51059320031e88a88a4d816a280cccc63`.
This run changes no model capacity, feature set, split, selection rule or published reference result.
Python 3.12.14 on Windows; direct dependencies match `requirements-credit-repro.txt`.
The final test suite passed **70 tests** (68 inherited, 2 reference-comparison tests).

Executed from the repository root:

```bash
python -m pytest -q
python scripts/audit.py --negative-control --output reports/generated/review-2026-10-22
python scripts/compare_reference.py --run-dir reports/generated/review-2026-10-22/reproduction --output reports/generated/review-2026-10-22/reference_comparison.json
```

All exited successfully. [audit.json](audit.json) records data/code fingerprints,
group coverage, validation-only selection and threshold checks, metric reconstruction,
approval limits and shuffled-label control. [reference_comparison.json](reference_comparison.json)
records published-table hashes, data/split hash agreement, runtime and every nonzero aggregate difference.
Row-level records and persisted models remain in the ignored generated directory.

| Check | Observed result |
|---|---|
| Selected model | random_forest_sigmoid |
| Test AUC / KS / Brier | 0.7916953185 / 0.4579604563 / 0.1322027427 |
| Primary threshold | 0.1714384374 |
| Normalized 1:5 cost/customer | 0.5386795599 |
| Metric reconstruction max absolute error | 8.3266726847e-17 |
| Shuffled-label selected-model test AUC | 0.5089852870 |
| Published test-table max absolute difference | 6.6912530239e-9 |
| Published validation-table max absolute difference | 1.4648999153e-8 |
| Published policy-table max absolute difference | 1.4901161166e-8 |
| Split-table difference | 0 |

The historical native float32 XGBoost aggregates differ at low-order digits after
the earlier numerical/serialization repairs. Reference comparison uses absolute
tolerance 1e-7 (below float32 epsilon near one), preserves all nonzero differences,
and still rejects any changed integer count. The selected model's test metrics and
confusion counts agree. Reference tables are retained rather than regenerated.

This is reproduction of the same published historical cohort and split, not a new
untouched, temporal or external test. One permutation is a diagnostic, not proof
of absence of leakage. Conditional bootstrap intervals and hypothetical costs
retain the limitations in the [research evidence guide](../../../docs/research_evidence.md).
