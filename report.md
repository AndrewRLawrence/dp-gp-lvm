## Each fix alone and all fixes

| Experiment, model | Baseline | Item 1 | Item 2 | Items 1+2 | Item 7 | Item 5 | All fixes |
|---|---|---|---|---|---|---|---|
| Synthetic two-func, Bayesian GP-LVM | −882.8 ± 5.9 | — | −2104.7 ± 131.0 | — | — | −677.2 ± 18.2 | −692.7 ± 105.0 |
| Synthetic two-func, DP-GP-LVM | −1386.6 ± 20.8 | −1132.2 ± 119.8 | −902.6 ± 16.6 | −3415.7 ± 1015.3 | −1381.1 ± 21.3 | −1371.8 ± 21.6 | −668.0 ± 159.7 |
| Synthetic four-func, Bayesian GP-LVM | −1611.9 ± 17.6 | — | −2029.0 ± 61.8 | — | — | −1520.4 ± 12.6 | −1136.8 ± 28.9 |
| Synthetic four-func, DP-GP-LVM | −2821.9 ± 4.2 | −1809.5 ± 84.3 | −1643.5 ± 20.5 | −3299.6 ± 212.2 | −2799.9 ± 6.0 | −2786.0 ± 6.6 | −743.0 ± 65.8 |
| Frey faces prediction, Bayesian GP-LVM | −4285.3 ± 129.2 | — | −5975.7 ± 366.7 | — | — | −3319.9 ± 199.7 | −2975.1 ± 310.4 |
| Frey faces prediction, DP-GP-LVM | −8648.9 ± 75.1 | −9951.8 ± 1936.9 | −5720.9 ± 76.6 | −16083.6 ± 3064.2 | −8627.0 ± 75.2 | −8618.7 ± 75.0 | −2484.5 ± 400.7 |
| Frey faces split, MRD | −4457.5 ± 106.7 | — | −5023.5 ± 255.4 | — | — | −4070.4 ± 178.4 | −3326.2 ± 215.8 |
| Frey faces split, DP-GP-LVM | −8485.0 ± 57.4 | −10362.5 ± 1062.3 | −5621.5 ± 65.8 | −16062.1 ± 1902.0 | −8460.6 ± 58.3 | −8444.8 ± 58.8 | −3669.4 ± 534.5 |

## RMSE and predictive SD

| Experiment, model | RMSE (baseline) | RMSE (all fixes) | Predictive SD (baseline) | Predictive SD (all fixes) |
|---|---|---|---|---|
| Synthetic two-func, Bayesian GP-LVM | 1.19 | 0.68 | 1.03 | 0.45 |
| Synthetic two-func, DP-GP-LVM | 1.24 | 0.69 | 5.08 | 0.49 |
| Synthetic four-func, Bayesian GP-LVM | 0.99 | 0.69 | 1.29 | 0.58 |
| Synthetic four-func, DP-GP-LVM | 1.01 | 0.55 | 5.16 | 0.49 |
| Frey prediction, Bayesian GP-LVM | 1.08 | 0.71 | 0.93 | 0.58 |
| Frey prediction, DP-GP-LVM | 1.17 | 0.67 | 8.99 | 0.47 |
| Frey split, MRD | 1.10 | 0.83 | 1.68 | 0.75 |
| Frey split, DP-GP-LVM | 1.23 | 0.81 | 8.58 | 0.56 |

## DP-GP-LVM advantage, synthetic

| Data | Ñ missing | D̃ missing | BGP-LVM (baseline) | DP-GP-LVM (baseline) | DP − BGP (baseline) | BGP-LVM (all fixes) | DP-GP-LVM (all fixes) | DP − BGP (all fixes) |
|---|---|---|---|---|---|---|---|---|
| two-func | 20% | 30% | −286.7 ± 4.7 | −474.7 ± 12.9 | −187.9 ± 14.0 | −230.6 ± 32.5 | −214.4 ± 43.2 | 16.2 ± 25.9 |
| two-func | 20% | 20% | −188.7 ± 3.3 | −318.7 ± 7.6 | −130.0 ± 8.8 | −139.7 ± 16.9 | −117.4 ± 34.8 | 22.3 ± 38.5 |
| two-func | 20% | 10% | −97.5 ± 3.9 | −160.6 ± 4.8 | −63.1 ± 6.4 | −76.2 ± 12.5 | −74.6 ± 22.2 | 1.6 ± 31.9 |
| two-func | 10% | 30% | −153.1 ± 7.8 | −216.9 ± 4.9 | −63.7 ± 9.3 | −130.6 ± 44.0 | −130.8 ± 45.6 | −0.2 ± 9.9 |
| two-func | 10% | 20% | −104.1 ± 6.9 | −144.0 ± 3.5 | −39.9 ± 6.7 | −86.7 ± 29.0 | −94.7 ± 33.5 | −8.0 ± 10.7 |
| two-func | 10% | 10% | −52.7 ± 4.0 | −71.8 ± 2.5 | −19.1 ± 4.7 | −29.0 ± 4.4 | −36.1 ± 5.7 | −7.1 ± 3.4 |
| four-func | 20% | 30% | −546.7 ± 10.5 | −980.1 ± 2.0 | −433.4 ± 12.2 | −391.9 ± 14.2 | −257.7 ± 21.3 | 134.2 ± 13.1 |
| four-func | 20% | 20% | −361.8 ± 5.3 | −653.6 ± 1.1 | −291.8 ± 5.7 | −255.5 ± 14.9 | −174.3 ± 28.3 | 81.3 ± 18.7 |
| four-func | 20% | 10% | −183.2 ± 3.9 | −324.4 ± 1.4 | −141.2 ± 3.9 | −131.2 ± 11.7 | −95.6 ± 16.2 | 35.6 ± 7.0 |
| four-func | 10% | 30% | −263.9 ± 8.0 | −432.0 ± 1.5 | −168.1 ± 9.5 | −183.6 ± 9.3 | −111.8 ± 12.9 | 71.8 ± 6.0 |
| four-func | 10% | 20% | −167.7 ± 10.7 | −288.0 ± 1.6 | −120.3 ± 11.7 | −114.4 ± 9.4 | −67.0 ± 12.7 | 47.4 ± 4.1 |
| four-func | 10% | 10% | −88.5 ± 6.3 | −143.7 ± 0.9 | −55.1 ± 6.7 | −60.2 ± 3.9 | −36.6 ± 4.4 | 23.6 ± 2.6 |

## DP-GP-LVM advantage, Frey faces

| Data | Other model | DP − other (baseline) | DP − other (all fixes) |
|---|---|---|---|
| Frey faces prediction | Bayesian GP-LVM | −4363.7 ± 153.2 | 490.6 ± 101.4 |
| Frey faces split | MRD | −4027.5 ± 104.3 | −343.2 ± 337.2 |

## oi-VAE test log-likelihood

| Test sequence | Baseline | Item 7 | All fixes (items 3, 5, 7) |
|---|---|---|---|
| Sequence 11 | −436542 ± 38818 | −5837 ± 333 | −24443 ± 3011 |
| Sequence 12 | −340604 ± 30258 | −5198 ± 180 | −23740 ± 1781 |

## CMU swapped legs

| Model | Training | Objective (legacy DP terms) | Objective (fixed DP terms) | Groups | Blocks in each group |
|---|---|---|---|---|---|
| mask_size=3 | Baseline (legacy DP terms) | −2412.8 | −2423.4 | 1 | 62 |
| mask_size=3 | Item 4 (fixed DP terms) | −2342.7 | −2357.9 | 1 | 62 |
| mask_size=3 | Adjusted Rand index, baseline vs item 4 |  |  |  | 1.00 |
| mask_size=93 | Baseline (legacy DP terms) | −1764.7 | −1785.8 | 1 | 2 |
| mask_size=93 | Item 4 (fixed DP terms) | −1869.3 | −2113.0 | 1 | 2 |
| mask_size=93 | Adjusted Rand index, baseline vs item 4 |  |  |  | 1.00 |
