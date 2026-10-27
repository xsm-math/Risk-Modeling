# Hypothetical shared funding and expected-loss allocation

Frozen follow-on after the calibration/capacity study; reused historical rows, hypothetical cost/weight/resources, no causal lending profit or future risk guarantee.

All cost weights are either one or seeded, clipped lognormal scenario weights, shared by duplicate financial groups. They are not observed EAD, loan amounts or losses.
An approved default costs 5*weight, a rejected nondefault costs weight. Funding uses weight; predicted loss uses 5*PD*weight. Count, funding and predicted-risk caps are checked for every algorithm.
Only upper bounds are imposed, so reject-all is feasible. No infeasibility or guaranteed realized-loss claim follows from a tight budget.

## Predeclared primary comparison

```
 batch_size                     weight                 model  funding_fraction  risk_rate reference  milp_minus_reference_actual_cost  ci95_low  ci95_high  planning_gain_improvement  different_rows  primary_case
        250 hypothetical_heterogeneous random_forest_sigmoid               0.2       0.02 pd_greedy                          -0.00164 -0.004152    0.00103                   1.110433              73          True
```

## All conditional paired effects

```
 batch_size                     weight                 model  funding_fraction  risk_rate   reference  milp_minus_reference_actual_cost  ci95_low  ci95_high  planning_gain_improvement  different_rows  primary_case
        250                       unit         random_forest               0.2       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit         random_forest               0.2       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit         random_forest               0.2       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit         random_forest               0.2       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit         random_forest               0.4       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit         random_forest               0.4       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit         random_forest               0.4       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit         random_forest               0.4       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit random_forest_sigmoid               0.2       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit random_forest_sigmoid               0.2       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit random_forest_sigmoid               0.2       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit random_forest_sigmoid               0.2       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit random_forest_sigmoid               0.4       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit random_forest_sigmoid               0.4       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit random_forest_sigmoid               0.4       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250                       unit random_forest_sigmoid               0.4       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250 hypothetical_heterogeneous         random_forest               0.2       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250 hypothetical_heterogeneous         random_forest               0.2       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250 hypothetical_heterogeneous         random_forest               0.2       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250 hypothetical_heterogeneous         random_forest               0.2       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250 hypothetical_heterogeneous         random_forest               0.4       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250 hypothetical_heterogeneous         random_forest               0.4       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250 hypothetical_heterogeneous         random_forest               0.4       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250 hypothetical_heterogeneous         random_forest               0.4       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
        250 hypothetical_heterogeneous random_forest_sigmoid               0.2       0.02   pd_greedy                         -0.001640 -0.004152   0.001030                   1.110433              73          True
        250 hypothetical_heterogeneous random_forest_sigmoid               0.2       0.02 gain_greedy                         -0.009174 -0.028542   0.009592                 106.353657             967         False
        250 hypothetical_heterogeneous random_forest_sigmoid               0.2       0.05   pd_greedy                         -0.001640 -0.004152   0.001030                   1.110433              73         False
        250 hypothetical_heterogeneous random_forest_sigmoid               0.2       0.05 gain_greedy                         -0.009174 -0.028542   0.009592                 106.353657             967         False
        250 hypothetical_heterogeneous random_forest_sigmoid               0.4       0.02   pd_greedy                          0.001121 -0.003530   0.007367                   0.660937              65         False
        250 hypothetical_heterogeneous random_forest_sigmoid               0.4       0.02 gain_greedy                         -0.038648 -0.059366  -0.018677                 161.158826            1072         False
        250 hypothetical_heterogeneous random_forest_sigmoid               0.4       0.05   pd_greedy                          0.001986 -0.002493   0.007553                   0.666532              74         False
        250 hypothetical_heterogeneous random_forest_sigmoid               0.4       0.05 gain_greedy                         -0.015847 -0.034068   0.000548                  59.527595             781         False
       1000                       unit         random_forest               0.2       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit         random_forest               0.2       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit         random_forest               0.2       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit         random_forest               0.2       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit         random_forest               0.4       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit         random_forest               0.4       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit         random_forest               0.4       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit         random_forest               0.4       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit random_forest_sigmoid               0.2       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit random_forest_sigmoid               0.2       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit random_forest_sigmoid               0.2       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit random_forest_sigmoid               0.2       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit random_forest_sigmoid               0.4       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit random_forest_sigmoid               0.4       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit random_forest_sigmoid               0.4       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000                       unit random_forest_sigmoid               0.4       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000 hypothetical_heterogeneous         random_forest               0.2       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000 hypothetical_heterogeneous         random_forest               0.2       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000 hypothetical_heterogeneous         random_forest               0.2       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000 hypothetical_heterogeneous         random_forest               0.2       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000 hypothetical_heterogeneous         random_forest               0.4       0.02   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000 hypothetical_heterogeneous         random_forest               0.4       0.02 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000 hypothetical_heterogeneous         random_forest               0.4       0.05   pd_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000 hypothetical_heterogeneous         random_forest               0.4       0.05 gain_greedy                          0.000000  0.000000   0.000000                   0.000000               0         False
       1000 hypothetical_heterogeneous random_forest_sigmoid               0.2       0.02   pd_greedy                          0.000730 -0.001147   0.003145                   0.469028              23         False
       1000 hypothetical_heterogeneous random_forest_sigmoid               0.2       0.02 gain_greedy                         -0.005414 -0.025801   0.013371                 117.249061             983         False
       1000 hypothetical_heterogeneous random_forest_sigmoid               0.2       0.05   pd_greedy                          0.000730 -0.001147   0.003145                   0.469028              23         False
       1000 hypothetical_heterogeneous random_forest_sigmoid               0.2       0.05 gain_greedy                         -0.005414 -0.025801   0.013371                 117.249061             983         False
       1000 hypothetical_heterogeneous random_forest_sigmoid               0.4       0.02   pd_greedy                         -0.000350 -0.001751   0.000659                   0.178610              17         False
       1000 hypothetical_heterogeneous random_forest_sigmoid               0.4       0.02 gain_greedy                         -0.038094 -0.057262  -0.016714                 176.843615            1135         False
       1000 hypothetical_heterogeneous random_forest_sigmoid               0.4       0.05   pd_greedy                          0.002525 -0.001997   0.009006                   0.152405              25         False
       1000 hypothetical_heterogeneous random_forest_sigmoid               0.4       0.05 gain_greedy                         -0.022137 -0.042818  -0.003708                  66.904573             844         False
```

Positive planning gain improvement uses the same probabilities and feasible set. It is not a realized-cost gain. Actual-cost differences use common historical labels; negative favors MILP.
Bootstrap intervals resample duplicate groups with fixed decision memberships, not refits, new optimized portfolios or new future cohorts. Batches are artificial, overlapping workload schemes of the same people.
Unit weights admit an exact sorted positive-benefit prefix; MILP is unnecessary. Heterogeneous weights can create discrete packing effects; gain per funding unit equals 1-6*PD here, so it is the same ordering as PD rather than an extra distinct baseline.
The MILP preserves the best checked PD/gain-greedy solution if its returned candidate is absent or worse. Reported bounds certify only the forecast-conditioned planning problem at its configured tolerance.
