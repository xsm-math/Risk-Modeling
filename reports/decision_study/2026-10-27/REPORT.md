# Calibration, actions and hard batch capacity

Retrospective extension of the already published historical cohort and split; no new untouched or out-of-time holdout.

All base model settings and the cohort split are inherited. Validation freezes cost and scalar capacity cutoffs. Hard batch policies use only current scores and immutable IDs; labels enter evaluation only.
Capacity-only fills K as a ranking control. Cost-then-capacity stops when expected incremental benefit is nonpositive. Equal scores may be split by ID; this differs from historical whole-block scalar thresholds.

## Primary hypothetical 1:5 cost results

```
                model  ratio                  rule  threshold    n  approved  approval_rate  actual_cost  predicted_cost  approved_defaults  predicted_approved_defaults  approved_default_rate                                              approval_ids_sha256
               woe_lr    5.0   validation_min_cost   0.168039 5998      3641       0.607036     0.560020        0.580325                388                   408.149482               0.106564 e2ccc98a4f5594698c1cd5a5940f9134a9e34497a70f1b2a7cdfe378db353d2a
               woe_lr    5.0 probability_cost_rule   0.166667 5998      3601       0.600367     0.562688        0.580299                384                   401.457054               0.106637 db1602e48158b3906bff81c031203b0383f278d11dd5ba330a9216d39c1ca1ae
               woe_lr    5.0         threshold_0.5   0.500000 5998      5327       0.888129     0.756085        0.767189                865                   875.951057               0.162380 8603a7ffaa81acf62f7053754cc76b16bf55ccc14906364977bb216b7f701347
       woe_lr_sigmoid    5.0   validation_min_cost   0.170043 5998      3641       0.607036     0.560020        0.590772                388                   418.432031               0.106564 e2ccc98a4f5594698c1cd5a5940f9134a9e34497a70f1b2a7cdfe378db353d2a
       woe_lr_sigmoid    5.0 probability_cost_rule   0.166667 5998      3531       0.588696     0.564355        0.590587                374                   399.913892               0.105919 9e90c5247ad099c09ed330a958d86afd181aefde83823f6f7e29d1266145c083
       woe_lr_sigmoid    5.0         threshold_0.5   0.500000 5998      5343       0.890797     0.760420        0.780237                872                   891.501164               0.163204 af7200b327a1be296a44b0dcac1a14fdff9fba1ab665d86207f432973fec7207
        random_forest    5.0   validation_min_cost   0.401734 5998      3397       0.566355     0.538680        0.900286                326                   879.954115               0.095967 821445af84301036bb016e1f0e814c6e46e74123718bd26ff72133f48275d68a
        random_forest    5.0 probability_cost_rule   0.166667 5998       635       0.105869     0.701067        0.562291                 28                    81.738435               0.044094 6d5c66f4a09d2265dab321da6246b1b15dffc5ced5e182277faf86e897c9105e
        random_forest    5.0         threshold_0.5   0.500000 5998      4270       0.711904     0.546182        1.143209                479                  1268.296559               0.112178 7106780e9af838d35af504b44d3d3a5d5808ba05921ba8e265d88aa5418ea503
random_forest_sigmoid    5.0   validation_min_cost   0.171438 5998      3397       0.566355     0.538680        0.547838                326                   332.069303               0.095967 821445af84301036bb016e1f0e814c6e46e74123718bd26ff72133f48275d68a
random_forest_sigmoid    5.0 probability_cost_rule   0.166667 5998      3317       0.553018     0.542014        0.547679                316                   318.577427               0.095267 d9e00e9bf69f7914b431196ff505acc3836c50203e448bd08b30961646df8be2
random_forest_sigmoid    5.0         threshold_0.5   0.500000 5998      5305       0.884461     0.744748        0.757301                850                   859.462823               0.160226 7cfc0381e2356ffdefc26686e6d3df2fd3a4d3de88f003ab1480df77540fe93c
```

## Paired group bootstrap: calibrated minus raw cost

```
        model                  rule  ratio  calibrated_minus_raw_cost  ci95_low  ci95_high  different_rows
       woe_lr   validation_min_cost      5                   0.000000  0.000000   0.000000               0
       woe_lr probability_cost_rule      5                   0.001667 -0.004018   0.007085              70
       woe_lr         threshold_0.5      5                   0.004335  0.000245   0.009095              16
random_forest   validation_min_cost      5                   0.000000  0.000000   0.000000               0
random_forest probability_cost_rule      5                  -0.159053 -0.191236  -0.126610            2682
random_forest         threshold_0.5      5                   0.198566  0.162113   0.238014            1035
```

Intervals condition on already fitted models and frozen rules; no retraining or future-cohort uncertainty. Negative cost difference favors calibrated probabilities.

## Observed ordering and per-row action agreement

```
        model  rows  raw_blocks  calibrated_blocks  nondecreasing_on_observed_rows  strict_between_observed_raw_blocks  identical_raw_ties_remain_ties  same_score_id_order  raw_logit_clip_count
       woe_lr  5998        5185               5185                            True                                True                            True                 True                     0
random_forest  5998        5857               5857                            True                                True                            True                 True                     0
```

```
        model                           decision  different_rows  total_rows
       woe_lr        whole|validation_min_cost|1               0        5998
       woe_lr      whole|probability_cost_rule|1              16        5998
       woe_lr              whole|threshold_0.5|1              16        5998
       woe_lr        whole|validation_min_cost|2               0        5998
       woe_lr      whole|probability_cost_rule|2              11        5998
       woe_lr              whole|threshold_0.5|2              16        5998
       woe_lr        whole|validation_min_cost|5               0        5998
       woe_lr      whole|probability_cost_rule|5              70        5998
       woe_lr              whole|threshold_0.5|5              16        5998
       woe_lr       whole|validation_min_cost|10               0        5998
       woe_lr     whole|probability_cost_rule|10             120        5998
       woe_lr             whole|threshold_0.5|10              16        5998
       woe_lr     whole|validation_min_cost|20.5               0        5998
       woe_lr   whole|probability_cost_rule|20.5              22        5998
       woe_lr           whole|threshold_0.5|20.5              16        5998
       woe_lr        whole|validation_cutoff|0.5               0        5998
       woe_lr            whole|fill_capacity|0.5               0        5998
       woe_lr       whole|cost_then_capacity|0.5               0        5998
       woe_lr        whole|validation_cutoff|0.6               0        5998
       woe_lr            whole|fill_capacity|0.6               0        5998
       woe_lr       whole|cost_then_capacity|0.6              67        5998
       woe_lr        whole|validation_cutoff|0.7               0        5998
       woe_lr            whole|fill_capacity|0.7               0        5998
       woe_lr       whole|cost_then_capacity|0.7              70        5998
       woe_lr        whole|validation_cutoff|0.8               0        5998
       woe_lr            whole|fill_capacity|0.8               0        5998
       woe_lr       whole|cost_then_capacity|0.8              70        5998
       woe_lr        whole|validation_cutoff|0.9               0        5998
       woe_lr            whole|fill_capacity|0.9               0        5998
       woe_lr       whole|cost_then_capacity|0.9              70        5998
       woe_lr             whole|risk_prefix|0.02              44        5998
       woe_lr             whole|risk_prefix|0.05              59        5998
       woe_lr   groups_250|validation_cutoff|0.5               0        5998
       woe_lr       groups_250|fill_capacity|0.5               0        5998
       woe_lr  groups_250|cost_then_capacity|0.5               0        5998
       woe_lr   groups_250|validation_cutoff|0.6               0        5998
       woe_lr       groups_250|fill_capacity|0.6               0        5998
       woe_lr  groups_250|cost_then_capacity|0.6              42        5998
       woe_lr   groups_250|validation_cutoff|0.7               0        5998
       woe_lr       groups_250|fill_capacity|0.7               0        5998
       woe_lr  groups_250|cost_then_capacity|0.7              70        5998
       woe_lr   groups_250|validation_cutoff|0.8               0        5998
       woe_lr       groups_250|fill_capacity|0.8               0        5998
       woe_lr  groups_250|cost_then_capacity|0.8              70        5998
       woe_lr   groups_250|validation_cutoff|0.9               0        5998
       woe_lr       groups_250|fill_capacity|0.9               0        5998
       woe_lr  groups_250|cost_then_capacity|0.9              70        5998
       woe_lr        groups_250|risk_prefix|0.02              44        5998
       woe_lr        groups_250|risk_prefix|0.05              62        5998
       woe_lr  groups_1000|validation_cutoff|0.5               0        5998
       woe_lr      groups_1000|fill_capacity|0.5               0        5998
       woe_lr groups_1000|cost_then_capacity|0.5               0        5998
       woe_lr  groups_1000|validation_cutoff|0.6               0        5998
       woe_lr      groups_1000|fill_capacity|0.6               0        5998
       woe_lr groups_1000|cost_then_capacity|0.6              53        5998
       woe_lr  groups_1000|validation_cutoff|0.7               0        5998
       woe_lr      groups_1000|fill_capacity|0.7               0        5998
       woe_lr groups_1000|cost_then_capacity|0.7              70        5998
       woe_lr  groups_1000|validation_cutoff|0.8               0        5998
       woe_lr      groups_1000|fill_capacity|0.8               0        5998
       woe_lr groups_1000|cost_then_capacity|0.8              70        5998
       woe_lr  groups_1000|validation_cutoff|0.9               0        5998
       woe_lr      groups_1000|fill_capacity|0.9               0        5998
       woe_lr groups_1000|cost_then_capacity|0.9              70        5998
       woe_lr       groups_1000|risk_prefix|0.02              45        5998
       woe_lr       groups_1000|risk_prefix|0.05              58        5998
random_forest        whole|validation_min_cost|1               0        5998
random_forest      whole|probability_cost_rule|1            1035        5998
random_forest              whole|threshold_0.5|1            1035        5998
random_forest        whole|validation_min_cost|2               0        5998
random_forest      whole|probability_cost_rule|2            2232        5998
random_forest              whole|threshold_0.5|2            1035        5998
random_forest        whole|validation_min_cost|5               0        5998
random_forest      whole|probability_cost_rule|5            2682        5998
random_forest              whole|threshold_0.5|5            1035        5998
random_forest       whole|validation_min_cost|10               0        5998
random_forest     whole|probability_cost_rule|10            1469        5998
random_forest             whole|threshold_0.5|10            1035        5998
random_forest     whole|validation_min_cost|20.5               0        5998
random_forest   whole|probability_cost_rule|20.5             429        5998
random_forest           whole|threshold_0.5|20.5            1035        5998
random_forest        whole|validation_cutoff|0.5               0        5998
random_forest            whole|fill_capacity|0.5               0        5998
random_forest       whole|cost_then_capacity|0.5            2364        5998
random_forest        whole|validation_cutoff|0.6               0        5998
random_forest            whole|fill_capacity|0.6               0        5998
random_forest       whole|cost_then_capacity|0.6            2682        5998
random_forest        whole|validation_cutoff|0.7               0        5998
random_forest            whole|fill_capacity|0.7               0        5998
random_forest       whole|cost_then_capacity|0.7            2682        5998
random_forest        whole|validation_cutoff|0.8               0        5998
random_forest            whole|fill_capacity|0.8               0        5998
random_forest       whole|cost_then_capacity|0.8            2682        5998
random_forest        whole|validation_cutoff|0.9               0        5998
random_forest            whole|fill_capacity|0.9               0        5998
random_forest       whole|cost_then_capacity|0.9            2682        5998
random_forest             whole|risk_prefix|0.02            1194        5998
random_forest             whole|risk_prefix|0.05            2568        5998
random_forest   groups_250|validation_cutoff|0.5               0        5998
random_forest       groups_250|fill_capacity|0.5               0        5998
random_forest  groups_250|cost_then_capacity|0.5            2352        5998
random_forest   groups_250|validation_cutoff|0.6               0        5998
random_forest       groups_250|fill_capacity|0.6               0        5998
random_forest  groups_250|cost_then_capacity|0.6            2680        5998
random_forest   groups_250|validation_cutoff|0.7               0        5998
random_forest       groups_250|fill_capacity|0.7               0        5998
random_forest  groups_250|cost_then_capacity|0.7            2682        5998
random_forest   groups_250|validation_cutoff|0.8               0        5998
random_forest       groups_250|fill_capacity|0.8               0        5998
random_forest  groups_250|cost_then_capacity|0.8            2682        5998
random_forest   groups_250|validation_cutoff|0.9               0        5998
random_forest       groups_250|fill_capacity|0.9               0        5998
random_forest  groups_250|cost_then_capacity|0.9            2682        5998
random_forest        groups_250|risk_prefix|0.02            1179        5998
random_forest        groups_250|risk_prefix|0.05            2531        5998
random_forest  groups_1000|validation_cutoff|0.5               0        5998
random_forest      groups_1000|fill_capacity|0.5               0        5998
random_forest groups_1000|cost_then_capacity|0.5            2364        5998
random_forest  groups_1000|validation_cutoff|0.6               0        5998
random_forest      groups_1000|fill_capacity|0.6               0        5998
random_forest groups_1000|cost_then_capacity|0.6            2682        5998
random_forest  groups_1000|validation_cutoff|0.7               0        5998
random_forest      groups_1000|fill_capacity|0.7               0        5998
random_forest groups_1000|cost_then_capacity|0.7            2682        5998
random_forest  groups_1000|validation_cutoff|0.8               0        5998
random_forest      groups_1000|fill_capacity|0.8               0        5998
random_forest groups_1000|cost_then_capacity|0.8            2682        5998
random_forest  groups_1000|validation_cutoff|0.9               0        5998
random_forest      groups_1000|fill_capacity|0.9               0        5998
random_forest groups_1000|cost_then_capacity|0.9            2682        5998
random_forest       groups_1000|risk_prefix|0.02            1190        5998
random_forest       groups_1000|risk_prefix|0.05            2564        5998
```

## Capacity summary

```
     scheme                 model  fraction               rule  batches  rows  approved  cap  excess  violating_batches  approval_rate
groups_1000         random_forest       0.5 cost_then_capacity        6  5998       635 2999       0                  0       0.105869
groups_1000         random_forest       0.5      fill_capacity        6  5998      2999 2999       0                  0       0.500000
groups_1000         random_forest       0.5  validation_cutoff        6  5998      2983 2999      15                  4       0.497332
groups_1000         random_forest       0.6 cost_then_capacity        6  5998       635 3598       0                  0       0.105869
groups_1000         random_forest       0.6      fill_capacity        6  5998      3598 3598       0                  0       0.599867
groups_1000         random_forest       0.6  validation_cutoff        6  5998      3614 3598      38                  4       0.602534
groups_1000         random_forest       0.7 cost_then_capacity        6  5998       635 4198       0                  0       0.105869
groups_1000         random_forest       0.7      fill_capacity        6  5998      4198 4198       0                  0       0.699900
groups_1000         random_forest       0.7  validation_cutoff        6  5998      4175 4198      19                  1       0.696065
groups_1000         random_forest       0.8 cost_then_capacity        6  5998       635 4798       0                  0       0.105869
groups_1000         random_forest       0.8      fill_capacity        6  5998      4798 4798       0                  0       0.799933
groups_1000         random_forest       0.8  validation_cutoff        6  5998      4839 4798      45                  4       0.806769
groups_1000         random_forest       0.9 cost_then_capacity        6  5998       635 5398       0                  0       0.105869
groups_1000         random_forest       0.9      fill_capacity        6  5998      5398 5398       0                  0       0.899967
groups_1000         random_forest       0.9  validation_cutoff        6  5998      5416 5398      36                  4       0.902968
groups_1000 random_forest_sigmoid       0.5 cost_then_capacity        6  5998      2999 2999       0                  0       0.500000
groups_1000 random_forest_sigmoid       0.5      fill_capacity        6  5998      2999 2999       0                  0       0.500000
groups_1000 random_forest_sigmoid       0.5  validation_cutoff        6  5998      2983 2999      15                  4       0.497332
groups_1000 random_forest_sigmoid       0.6 cost_then_capacity        6  5998      3317 3598       0                  0       0.553018
groups_1000 random_forest_sigmoid       0.6      fill_capacity        6  5998      3598 3598       0                  0       0.599867
groups_1000 random_forest_sigmoid       0.6  validation_cutoff        6  5998      3614 3598      38                  4       0.602534
groups_1000 random_forest_sigmoid       0.7 cost_then_capacity        6  5998      3317 4198       0                  0       0.553018
groups_1000 random_forest_sigmoid       0.7      fill_capacity        6  5998      4198 4198       0                  0       0.699900
groups_1000 random_forest_sigmoid       0.7  validation_cutoff        6  5998      4175 4198      19                  1       0.696065
groups_1000 random_forest_sigmoid       0.8 cost_then_capacity        6  5998      3317 4798       0                  0       0.553018
groups_1000 random_forest_sigmoid       0.8      fill_capacity        6  5998      4798 4798       0                  0       0.799933
groups_1000 random_forest_sigmoid       0.8  validation_cutoff        6  5998      4839 4798      45                  4       0.806769
groups_1000 random_forest_sigmoid       0.9 cost_then_capacity        6  5998      3317 5398       0                  0       0.553018
groups_1000 random_forest_sigmoid       0.9      fill_capacity        6  5998      5398 5398       0                  0       0.899967
groups_1000 random_forest_sigmoid       0.9  validation_cutoff        6  5998      5416 5398      36                  4       0.902968
groups_1000                woe_lr       0.5 cost_then_capacity        6  5998      2999 2999       0                  0       0.500000
groups_1000                woe_lr       0.5      fill_capacity        6  5998      2999 2999       0                  0       0.500000
groups_1000                woe_lr       0.5  validation_cutoff        6  5998      2986 2999      12                  3       0.497833
groups_1000                woe_lr       0.6 cost_then_capacity        6  5998      3584 3598       0                  0       0.597533
groups_1000                woe_lr       0.6      fill_capacity        6  5998      3598 3598       0                  0       0.599867
groups_1000                woe_lr       0.6  validation_cutoff        6  5998      3575 3598       4                  2       0.596032
groups_1000                woe_lr       0.7 cost_then_capacity        6  5998      3601 4198       0                  0       0.600367
groups_1000                woe_lr       0.7      fill_capacity        6  5998      4198 4198       0                  0       0.699900
groups_1000                woe_lr       0.7  validation_cutoff        6  5998      4167 4198       6                  1       0.694732
groups_1000                woe_lr       0.8 cost_then_capacity        6  5998      3601 4798       0                  0       0.600367
groups_1000                woe_lr       0.8      fill_capacity        6  5998      4798 4798       0                  0       0.799933
groups_1000                woe_lr       0.8  validation_cutoff        6  5998      4857 4798      61                  5       0.809770
groups_1000                woe_lr       0.9 cost_then_capacity        6  5998      3601 5398       0                  0       0.600367
groups_1000                woe_lr       0.9      fill_capacity        6  5998      5398 5398       0                  0       0.899967
groups_1000                woe_lr       0.9  validation_cutoff        6  5998      5440 5398      42                  6       0.906969
groups_1000        woe_lr_sigmoid       0.5 cost_then_capacity        6  5998      2999 2999       0                  0       0.500000
groups_1000        woe_lr_sigmoid       0.5      fill_capacity        6  5998      2999 2999       0                  0       0.500000
groups_1000        woe_lr_sigmoid       0.5  validation_cutoff        6  5998      2986 2999      12                  3       0.497833
groups_1000        woe_lr_sigmoid       0.6 cost_then_capacity        6  5998      3531 3598       0                  0       0.588696
groups_1000        woe_lr_sigmoid       0.6      fill_capacity        6  5998      3598 3598       0                  0       0.599867
groups_1000        woe_lr_sigmoid       0.6  validation_cutoff        6  5998      3575 3598       4                  2       0.596032
groups_1000        woe_lr_sigmoid       0.7 cost_then_capacity        6  5998      3531 4198       0                  0       0.588696
groups_1000        woe_lr_sigmoid       0.7      fill_capacity        6  5998      4198 4198       0                  0       0.699900
groups_1000        woe_lr_sigmoid       0.7  validation_cutoff        6  5998      4167 4198       6                  1       0.694732
groups_1000        woe_lr_sigmoid       0.8 cost_then_capacity        6  5998      3531 4798       0                  0       0.588696
groups_1000        woe_lr_sigmoid       0.8      fill_capacity        6  5998      4798 4798       0                  0       0.799933
groups_1000        woe_lr_sigmoid       0.8  validation_cutoff        6  5998      4857 4798      61                  5       0.809770
groups_1000        woe_lr_sigmoid       0.9 cost_then_capacity        6  5998      3531 5398       0                  0       0.588696
groups_1000        woe_lr_sigmoid       0.9      fill_capacity        6  5998      5398 5398       0                  0       0.899967
groups_1000        woe_lr_sigmoid       0.9  validation_cutoff        6  5998      5440 5398      42                  6       0.906969
 groups_250         random_forest       0.5 cost_then_capacity       24  5998       635 2999       0                  0       0.105869
 groups_250         random_forest       0.5      fill_capacity       24  5998      2999 2999       0                  0       0.500000
 groups_250         random_forest       0.5  validation_cutoff       24  5998      2983 2999      74                 11       0.497332
 groups_250         random_forest       0.6 cost_then_capacity       24  5998       635 3598       0                  0       0.105869
 groups_250         random_forest       0.6      fill_capacity       24  5998      3598 3598       0                  0       0.599867
 groups_250         random_forest       0.6  validation_cutoff       24  5998      3614 3598      80                 14       0.602534
 groups_250         random_forest       0.7 cost_then_capacity       24  5998       635 4198       0                  0       0.105869
 groups_250         random_forest       0.7      fill_capacity       24  5998      4198 4198       0                  0       0.699900
 groups_250         random_forest       0.7  validation_cutoff       24  5998      4175 4198      49                 12       0.696065
 groups_250         random_forest       0.8 cost_then_capacity       24  5998       635 4798       0                  0       0.105869
 groups_250         random_forest       0.8      fill_capacity       24  5998      4798 4798       0                  0       0.799933
 groups_250         random_forest       0.8  validation_cutoff       24  5998      4839 4798      86                 15       0.806769
 groups_250         random_forest       0.9 cost_then_capacity       24  5998       635 5398       0                  0       0.105869
 groups_250         random_forest       0.9      fill_capacity       24  5998      5398 5398       0                  0       0.899967
 groups_250         random_forest       0.9  validation_cutoff       24  5998      5416 5398      50                 13       0.902968
 groups_250 random_forest_sigmoid       0.5 cost_then_capacity       24  5998      2987 2999       0                  0       0.497999
 groups_250 random_forest_sigmoid       0.5      fill_capacity       24  5998      2999 2999       0                  0       0.500000
 groups_250 random_forest_sigmoid       0.5  validation_cutoff       24  5998      2983 2999      74                 11       0.497332
 groups_250 random_forest_sigmoid       0.6 cost_then_capacity       24  5998      3315 3598       0                  0       0.552684
 groups_250 random_forest_sigmoid       0.6      fill_capacity       24  5998      3598 3598       0                  0       0.599867
 groups_250 random_forest_sigmoid       0.6  validation_cutoff       24  5998      3614 3598      80                 14       0.602534
 groups_250 random_forest_sigmoid       0.7 cost_then_capacity       24  5998      3317 4198       0                  0       0.553018
 groups_250 random_forest_sigmoid       0.7      fill_capacity       24  5998      4198 4198       0                  0       0.699900
 groups_250 random_forest_sigmoid       0.7  validation_cutoff       24  5998      4175 4198      49                 12       0.696065
 groups_250 random_forest_sigmoid       0.8 cost_then_capacity       24  5998      3317 4798       0                  0       0.553018
 groups_250 random_forest_sigmoid       0.8      fill_capacity       24  5998      4798 4798       0                  0       0.799933
 groups_250 random_forest_sigmoid       0.8  validation_cutoff       24  5998      4839 4798      86                 15       0.806769
 groups_250 random_forest_sigmoid       0.9 cost_then_capacity       24  5998      3317 5398       0                  0       0.553018
 groups_250 random_forest_sigmoid       0.9      fill_capacity       24  5998      5398 5398       0                  0       0.899967
 groups_250 random_forest_sigmoid       0.9  validation_cutoff       24  5998      5416 5398      50                 13       0.902968
 groups_250                woe_lr       0.5 cost_then_capacity       24  5998      2999 2999       0                  0       0.500000
 groups_250                woe_lr       0.5      fill_capacity       24  5998      2999 2999       0                  0       0.500000
 groups_250                woe_lr       0.5  validation_cutoff       24  5998      2986 2999      61                 12       0.497833
 groups_250                woe_lr       0.6 cost_then_capacity       24  5998      3535 3598       0                  0       0.589363
 groups_250                woe_lr       0.6      fill_capacity       24  5998      3598 3598       0                  0       0.599867
 groups_250                woe_lr       0.6  validation_cutoff       24  5998      3575 3598      55                  9       0.596032
 groups_250                woe_lr       0.7 cost_then_capacity       24  5998      3601 4198       0                  0       0.600367
 groups_250                woe_lr       0.7      fill_capacity       24  5998      4198 4198       0                  0       0.699900
 groups_250                woe_lr       0.7  validation_cutoff       24  5998      4167 4198      46                 11       0.694732
 groups_250                woe_lr       0.8 cost_then_capacity       24  5998      3601 4798       0                  0       0.600367
 groups_250                woe_lr       0.8      fill_capacity       24  5998      4798 4798       0                  0       0.799933
 groups_250                woe_lr       0.8  validation_cutoff       24  5998      4857 4798     103                 18       0.809770
 groups_250                woe_lr       0.9 cost_then_capacity       24  5998      3601 5398       0                  0       0.600367
 groups_250                woe_lr       0.9      fill_capacity       24  5998      5398 5398       0                  0       0.899967
 groups_250                woe_lr       0.9  validation_cutoff       24  5998      5440 5398      60                 16       0.906969
 groups_250        woe_lr_sigmoid       0.5 cost_then_capacity       24  5998      2999 2999       0                  0       0.500000
 groups_250        woe_lr_sigmoid       0.5      fill_capacity       24  5998      2999 2999       0                  0       0.500000
 groups_250        woe_lr_sigmoid       0.5  validation_cutoff       24  5998      2986 2999      61                 12       0.497833
 groups_250        woe_lr_sigmoid       0.6 cost_then_capacity       24  5998      3493 3598       0                  0       0.582361
 groups_250        woe_lr_sigmoid       0.6      fill_capacity       24  5998      3598 3598       0                  0       0.599867
 groups_250        woe_lr_sigmoid       0.6  validation_cutoff       24  5998      3575 3598      55                  9       0.596032
 groups_250        woe_lr_sigmoid       0.7 cost_then_capacity       24  5998      3531 4198       0                  0       0.588696
 groups_250        woe_lr_sigmoid       0.7      fill_capacity       24  5998      4198 4198       0                  0       0.699900
 groups_250        woe_lr_sigmoid       0.7  validation_cutoff       24  5998      4167 4198      46                 11       0.694732
 groups_250        woe_lr_sigmoid       0.8 cost_then_capacity       24  5998      3531 4798       0                  0       0.588696
 groups_250        woe_lr_sigmoid       0.8      fill_capacity       24  5998      4798 4798       0                  0       0.799933
 groups_250        woe_lr_sigmoid       0.8  validation_cutoff       24  5998      4857 4798     103                 18       0.809770
 groups_250        woe_lr_sigmoid       0.9 cost_then_capacity       24  5998      3531 5398       0                  0       0.588696
 groups_250        woe_lr_sigmoid       0.9      fill_capacity       24  5998      5398 5398       0                  0       0.899967
 groups_250        woe_lr_sigmoid       0.9  validation_cutoff       24  5998      5440 5398      60                 16       0.906969
      whole         random_forest       0.5 cost_then_capacity        1  5998       635 2999       0                  0       0.105869
      whole         random_forest       0.5      fill_capacity        1  5998      2999 2999       0                  0       0.500000
      whole         random_forest       0.5  validation_cutoff        1  5998      2983 2999       0                  0       0.497332
      whole         random_forest       0.6 cost_then_capacity        1  5998       635 3598       0                  0       0.105869
      whole         random_forest       0.6      fill_capacity        1  5998      3598 3598       0                  0       0.599867
      whole         random_forest       0.6  validation_cutoff        1  5998      3614 3598      16                  1       0.602534
      whole         random_forest       0.7 cost_then_capacity        1  5998       635 4198       0                  0       0.105869
      whole         random_forest       0.7      fill_capacity        1  5998      4198 4198       0                  0       0.699900
      whole         random_forest       0.7  validation_cutoff        1  5998      4175 4198       0                  0       0.696065
      whole         random_forest       0.8 cost_then_capacity        1  5998       635 4798       0                  0       0.105869
      whole         random_forest       0.8      fill_capacity        1  5998      4798 4798       0                  0       0.799933
      whole         random_forest       0.8  validation_cutoff        1  5998      4839 4798      41                  1       0.806769
      whole         random_forest       0.9 cost_then_capacity        1  5998       635 5398       0                  0       0.105869
      whole         random_forest       0.9      fill_capacity        1  5998      5398 5398       0                  0       0.899967
      whole         random_forest       0.9  validation_cutoff        1  5998      5416 5398      18                  1       0.902968
      whole random_forest_sigmoid       0.5 cost_then_capacity        1  5998      2999 2999       0                  0       0.500000
      whole random_forest_sigmoid       0.5      fill_capacity        1  5998      2999 2999       0                  0       0.500000
      whole random_forest_sigmoid       0.5  validation_cutoff        1  5998      2983 2999       0                  0       0.497332
      whole random_forest_sigmoid       0.6 cost_then_capacity        1  5998      3317 3598       0                  0       0.553018
      whole random_forest_sigmoid       0.6      fill_capacity        1  5998      3598 3598       0                  0       0.599867
      whole random_forest_sigmoid       0.6  validation_cutoff        1  5998      3614 3598      16                  1       0.602534
      whole random_forest_sigmoid       0.7 cost_then_capacity        1  5998      3317 4198       0                  0       0.553018
      whole random_forest_sigmoid       0.7      fill_capacity        1  5998      4198 4198       0                  0       0.699900
      whole random_forest_sigmoid       0.7  validation_cutoff        1  5998      4175 4198       0                  0       0.696065
      whole random_forest_sigmoid       0.8 cost_then_capacity        1  5998      3317 4798       0                  0       0.553018
      whole random_forest_sigmoid       0.8      fill_capacity        1  5998      4798 4798       0                  0       0.799933
      whole random_forest_sigmoid       0.8  validation_cutoff        1  5998      4839 4798      41                  1       0.806769
      whole random_forest_sigmoid       0.9 cost_then_capacity        1  5998      3317 5398       0                  0       0.553018
      whole random_forest_sigmoid       0.9      fill_capacity        1  5998      5398 5398       0                  0       0.899967
      whole random_forest_sigmoid       0.9  validation_cutoff        1  5998      5416 5398      18                  1       0.902968
      whole                woe_lr       0.5 cost_then_capacity        1  5998      2999 2999       0                  0       0.500000
      whole                woe_lr       0.5      fill_capacity        1  5998      2999 2999       0                  0       0.500000
      whole                woe_lr       0.5  validation_cutoff        1  5998      2986 2999       0                  0       0.497833
      whole                woe_lr       0.6 cost_then_capacity        1  5998      3598 3598       0                  0       0.599867
      whole                woe_lr       0.6      fill_capacity        1  5998      3598 3598       0                  0       0.599867
      whole                woe_lr       0.6  validation_cutoff        1  5998      3575 3598       0                  0       0.596032
      whole                woe_lr       0.7 cost_then_capacity        1  5998      3601 4198       0                  0       0.600367
      whole                woe_lr       0.7      fill_capacity        1  5998      4198 4198       0                  0       0.699900
      whole                woe_lr       0.7  validation_cutoff        1  5998      4167 4198       0                  0       0.694732
      whole                woe_lr       0.8 cost_then_capacity        1  5998      3601 4798       0                  0       0.600367
      whole                woe_lr       0.8      fill_capacity        1  5998      4798 4798       0                  0       0.799933
      whole                woe_lr       0.8  validation_cutoff        1  5998      4857 4798      59                  1       0.809770
      whole                woe_lr       0.9 cost_then_capacity        1  5998      3601 5398       0                  0       0.600367
      whole                woe_lr       0.9      fill_capacity        1  5998      5398 5398       0                  0       0.899967
      whole                woe_lr       0.9  validation_cutoff        1  5998      5440 5398      42                  1       0.906969
      whole        woe_lr_sigmoid       0.5 cost_then_capacity        1  5998      2999 2999       0                  0       0.500000
      whole        woe_lr_sigmoid       0.5      fill_capacity        1  5998      2999 2999       0                  0       0.500000
      whole        woe_lr_sigmoid       0.5  validation_cutoff        1  5998      2986 2999       0                  0       0.497833
      whole        woe_lr_sigmoid       0.6 cost_then_capacity        1  5998      3531 3598       0                  0       0.588696
      whole        woe_lr_sigmoid       0.6      fill_capacity        1  5998      3598 3598       0                  0       0.599867
      whole        woe_lr_sigmoid       0.6  validation_cutoff        1  5998      3575 3598       0                  0       0.596032
      whole        woe_lr_sigmoid       0.7 cost_then_capacity        1  5998      3531 4198       0                  0       0.588696
      whole        woe_lr_sigmoid       0.7      fill_capacity        1  5998      4198 4198       0                  0       0.699900
      whole        woe_lr_sigmoid       0.7  validation_cutoff        1  5998      4167 4198       0                  0       0.694732
      whole        woe_lr_sigmoid       0.8 cost_then_capacity        1  5998      3531 4798       0                  0       0.588696
      whole        woe_lr_sigmoid       0.8      fill_capacity        1  5998      4798 4798       0                  0       0.799933
      whole        woe_lr_sigmoid       0.8  validation_cutoff        1  5998      4857 4798      59                  1       0.809770
      whole        woe_lr_sigmoid       0.9 cost_then_capacity        1  5998      3531 5398       0                  0       0.588696
      whole        woe_lr_sigmoid       0.9      fill_capacity        1  5998      5398 5398       0                  0       0.899967
      whole        woe_lr_sigmoid       0.9  validation_cutoff        1  5998      5440 5398      42                  1       0.906969
```

Group-packed batches are artificial workload partitions of the same 5,998 people, not new independent cohorts. Duplicate financial groups stay within batches; repeated schemes reuse people.
Unit-risk prefix rows enforce predicted expected defaults, not realized loss. Violations of realized counts do not imply the numerical predicted-risk constraint failed.
All costs are normalized retrospective classification scenarios, not causal approval profits. Credit limits are not treated as EAD. Row predictions, IDs and decision masks stay in ignored local files.
