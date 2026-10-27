# Follow-on budget study, after calibration and capacity checks

The first two stages completed and row-level verification passed before this
follow-on was designed. It reuses the same already inspected historical cohort;
it is not a fresh holdout. No model is refitted or selected from these results.
RF sigmoid remains the inherited primary predictor, raw RF a calibration control.

The protocol and source are committed before budget evaluation. Four combinations
of funding fractions .2/.4 and predicted-default-weight rates .02/.05 are crossed
with unit weights or seeded hypothetical heterogeneous weights. Two group-packed
workloads target 250 and 1000 rows. All methods see identical probabilities,
weights, count caps floor(.6*N), funding and expected-loss caps within each cell.
The weight is assigned by a seeded group hash using a clipped lognormal transform,
not taken from credit limits or presented as true EAD. No label enters the weight.

An approved default costs 5*w; a rejected nondefault costs w. Reject-all is a
feasible baseline, since only resource upper bounds are imposed. The objective
minimizes expected retrospective classification cost. Funding usage is sum(a*w),
predicted loss is sum(a*5*PD*w). Actual labels evaluate the frozen decisions only.
Predicted loss compliance is not an actual loss, quantile-risk or tail guarantee.

PD-greedy and absolute-gain-greedy accept only positive-benefit feasible additions.
MILP optimizes exactly the same objective and constraints, with 2 seconds and
1e-4 relative planning-gain gap. It retains the best checked greedy if the solver
cannot improve it. Independent small-case enumeration validates feasibility,
optimality and a discrete counterexample. Unit weights have a sorted-prefix
solution, so integer optimization should offer no planning gain there.

The primary comparison is MILP minus PD-greedy, inherited RF sigmoid, heterogeneous
weights, target batch 250, funding .2 and risk rate .02. It is declared to avoid
choosing the most favorable case after evaluation. All 64 paired comparisons
remain visible, including adverse and zero realized effects. Conditional paired
group bootstrap uses 500 draws and fixed memberships, not full refits or reoptimized
bootstrap portfolios. Workload schemes reuse people and are not independent cohorts.

```bash
python -m src.credit.allocation_study --output reports/allocation_study/new
```

It requires the completed prior study and private baseline predictions. The full
manual workflow regenerates both in order. Public tables/hashes are committed;
row predictions, action masks and model binaries stay under ignored local files.
