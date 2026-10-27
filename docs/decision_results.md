# Accurate probabilities and hard batch capacity answer different questions

[Protocol](decision_protocol.md) · [Frozen rules](../reports/decision_study/2026-10-27/selection.json) ·
[Paired effects](../reports/decision_study/2026-10-27/paired_calibration_effects.csv) ·
[Row-action agreement counts](../reports/decision_study/2026-10-27/action_agreement.csv).

This extension reproduces the original split and evaluates four fixed variants
on the same published historical cohort. It does not create an untouched test
cohort or identify a causal approval benefit. Row predictions and action masks
are retained privately; public tables contain aggregates and identity hashes.

## Calibration improves some rules, leaves others unchanged

The RF AUC remains 0.791695 while Brier improves from .172387 to .132203.
In all 5,998 observed test rows, raw and calibrated score-ID order agrees for
both RF and WOE. Raw ties stay tied, distinct raw blocks stay distinct, and no
raw probability reaches the calibration-input clipping limits in these rows.
This is an observed diagnostic, not a universal positive-slope guarantee.

| Primary 1:5 rule | RF raw cost | RF calibrated cost | Calibrated minus raw, paired group bootstrap 95% interval | Changed row decisions |
|---|---:|---:|---|---:|
| Each validation-selected threshold | .538680 | .538680 | 0 [0,0] | 0 |
| Theoretical PD threshold 1/6 | .701067 | .542014 | -.159053 [-.191236,-.126610] | 2,682 |
| Fixed PD threshold .5 | .546182 | .744748 | +.198566 [.162113,.238014] | 1,035 |

Free validation thresholds produce identical row decisions for both base models
at **all five** declared cost ratios. Capacity-only filling also yields identical
raw/calibrated membership at every capacity and workload scheme. Strictly
order-preserving rescaling supplies no new ranking action in these cases.

Fixed probability thresholds and predicted-loss budgets depend on probability
levels. The homogeneous whole-cohort prefix at capacity .6 and predicted-default
budget .02*N admits 635 with raw RF versus 1,829 with calibrated RF. Both obey
their own predicted budget, but the raw model's positive-benefit gate stops much
earlier. This illustrates a decision mechanism, not proof of future risk control.

Calibration is not uniformly beneficial: WOE's theoretical-rule cost difference
is +.001667, interval [-.004018,.007085]. All six primary comparisons are retained;
intervals are conditional, unadjusted and exclude retraining uncertainty.

## Validation cutoffs do not enforce a new batch cap

The historical scalar cutoff keeps score ties together and targets a validation
fraction. The new label-free batch rules enforce floor(q*N) in each current batch,
breaking equal-score ties by immutable ID. Capacity-only fills the available
positions as a ranking control; cost-then-capacity also stops at nonpositive
expected gain. These are distinct declared policies, not a silent replacement
of the historical protocol.

At the RF sigmoid 60% limit:

| Workload | Validation cutoff approvals | Sum of batch caps | Overflow summed over violating batches | Violating batches | Hard capacity-only approvals |
|---|---:|---:|---:|---:|---:|
| All 5,998 rows | 3,614 | 3,598 | 16 | 1/1 | 3,598 |
| Group-packed target 1,000 | 3,614 | 3,598 | 38 | 4/6 | 3,598 |
| Group-packed target 250 | 3,614 | 3,598 | 80 | 14/24 | 3,598 |

Total overflow across violating batches can exceed aggregate overflow because
spare slots elsewhere cannot repair a hard local capacity breach. In the 250-row
scheme, cost-then-capacity admits 3,315; filling all slots is not required by an
upper bound and may increase expected cost.

Across all four variants, five fractions and three workload schemes, both hard
rules have **zero violations in 1,240 batch-decision cells**. The scalar rules
violate 340 of 620 scenario cells. These are reused, constructed workloads, not
620 independent future cohorts or an estimate of future breach probability.

## Reproduction and verification

The original dataset/split hash matches. The four study variants reproduce
historical metrics within 1.11e-16. An unused raw XGBoost float32 metric varies by
6.69e-9, separately recorded under a 1e-7 allowance; it does not relax the 1e-12
study-variant tolerance. Original benchmark sources are checked against commit
`a11dbea`; the study's executed source was committed at `0a626ef` before evaluation.

```bash
python scripts/verify_decisions.py --run-dir reports/decision_study/2026-10-27
# Full row replay also requires the isolated baseline's ignored local files:
python scripts/verify_decisions.py --run-dir reports/decision_study/2026-10-27 --baseline reports/generated/decision_baseline
```

The private replay checked all 60 whole-cohort cost rules, 1,860 batch decisions
and 132 raw/calibrated action comparisons against saved row masks. Source and
artifact hashes are checked. Public-only verification cannot reconstruct private
row membership and reports that limitation explicitly.

The unit-risk budget is an expected-count constraint, not a guarantee on realized
defaults. The calibrated whole-cohort .05*N case predicts 299.883 defaults within
a 299.90 budget and realizes 300. This is not a numerical feasibility failure.
Future-cohort validity and retraining uncertainty remain untested here.
