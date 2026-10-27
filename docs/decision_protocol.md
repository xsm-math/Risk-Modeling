# Calibration and hard-capacity study, declared before its evaluation

This is a retrospective mechanism study on the already published UCI cohort and
split, not a fresh untouched holdout, OOT study or causal lending experiment.
First reproduce the original benchmark in an isolated generated directory, with
the pinned Python 3.12 requirements. Compare every historical test metric and
the original split hash before using the private row predictions.
The four study variants must match within 1e-12. An unused raw XGBoost float32
path is allowed 1e-7 arithmetic variation, recorded separately rather than
described as bit-for-bit reproduction. This does not relax the study-model gate.

Two fixed base learners, WOE LR and RF, each retain raw and sigmoid versions.
The original base models, training/calibration rows and hyperparameters stay
unchanged. The new study does not select a new model from its test results.
The selected RF sigmoid remains the inherited primary model.

`configs/decision_study.json` declares cost ratios 1,2,5,10,20.5, primary 5,
capacity fractions .5/.6/.7/.8/.9, and group-packed target batch sizes 250/1000.
The whole test population is a third workload. Group packing is based on a
fixed group-hash seed, never outcomes; identical financial records stay together.
These schemes reuse people and are not independent future cohorts.

Validation freezes empirical minimum-cost thresholds and scalar capacity cutoffs.
Evaluation compares those cutoffs with the theoretical 1/(1+r) and .5 rules.
Per-row action masks remain private; aggregate identity hashes and mismatch
counts establish action agreement, rather than inferring it from confusion counts.
Ranking diagnostics check actual raw ties, merged calibrated ties, strict order
and clipping. No universal monotonicity is assumed from the method name alone.

Two hard batch rules use scores and immutable ID only: capacity-only fills
floor(q*N) lowest-score positions as a ranking control; cost-then-capacity also
requires PD<1/(1+r). Both may split equal-score blocks by declared ID order.
This differs from the historical whole-score-block scalar policy. No fairness
guarantee is implied. Hard capacity is an upper bound, not a forced sales target.

A homogeneous expected-default budget diagnostic uses sorted positive-gain
prefixes at capacity .6 and budget .02*N or .05*N. It has an exact sorting
solution and needs no integer solver. Predicted expected defaults are bounded;
realized default counts need not respect that budget in a single realization.

Paired 500-draw group bootstrap intervals compare raw/calibrated cost at the
primary ratio; they condition on frozen models/rules, exclude fitting uncertainty
and do not correct for multiple comparisons. Report all directions, including
the negative effect of a poorly matched fixed .5 threshold.

Archive full hashes and aggregate results, never customer rows or model binaries.
All new career/interview materials belong outside the repository.

```bash
python -m src.credit download
python -m src.credit run --output reports/generated/decision_baseline
python -m src.credit.decision_study --stage run --output reports/decision_study/new
```

Use a fresh output; existing frozen evaluation evidence is not overwritten.
Baseline local files are required for full row-level verification. Numeric runs
on other hardware may differ slightly; recorded hashes describe the executed run.
