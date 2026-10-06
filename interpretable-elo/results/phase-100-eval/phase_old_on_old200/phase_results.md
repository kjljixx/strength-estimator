# Phase Elo gaps: chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted/model/weight_iter_159500.pt

Queries `query_sgf_chess`, 200 bootstrap resamples over games.
Phase Elo = Elo of the mean score over all moves in that phase, pooled over the bin's games.
Gap = phase Elo minus the average of the bin's three phase Elos (sums to zero within a bin); ± is the bootstrap standard error.
Centered gap subtracts each phase's mean gap over bins, leaving only how the gap changes with rating.
Error vs true rating = phase Elo minus the mean recorded rating of the bin's players.

# A. One whole-game calibration for all phases (`eval_phase_100/old_on_old200/calibration.json`; monotone, clamped)

## Summary over bins
| phase | mean_gap | mean_abs_gap | mean_error_vs_true_rating |
|---|---|---|---|
| opening | -83.792 | 83.792 | -135.133 |
| midgame | 126.757 | 148.637 | 75.416 |
| endgame | -42.966 | 95.516 | -94.308 |

## Gap vs phase average
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | -19.3 ± 7.5 | -19.3 ± 6.9 | +38.5 ± 13.5 |
| 1200-1400 | -25.7 ± 11.2 | -68.3 ± 12.8 | +94.0 ± 11.6 |
| 1400-1600 | -65.4 ± 9.7 | +7.7 ± 14.1 | +57.6 ± 15.0 |
| 1600-1800 | -111.3 ± 10.3 | +91.3 ± 11.8 | +20.0 ± 12.2 |
| 1800-2000 | -102.7 ± 8.8 | +136.0 ± 9.6 | -33.3 ± 10.7 |
| 2000-2200 | -153.6 ± 16.8 | +322.0 ± 30.7 | -168.4 ± 16.8 |
| 2200-2400 | -112.6 ± 7.1 | +293.8 ± 4.9 | -181.2 ± 8.1 |
| 2400-2600 | -79.7 ± 12.0 | +250.7 ± 6.9 | -171.0 ± 9.9 |

## Centered gap
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | +64.5 | -146.0 | +81.5 |
| 1200-1400 | +58.0 | -195.0 | +137.0 |
| 1400-1600 | +18.4 | -119.0 | +100.6 |
| 1600-1800 | -27.5 | -35.5 | +63.0 |
| 1800-2000 | -18.9 | +9.2 | +9.7 |
| 2000-2200 | -69.8 | +195.3 | -125.5 |
| 2200-2400 | -28.8 | +167.1 | -138.2 |
| 2400-2600 | +4.1 | +124.0 | -128.1 |

## Error vs true rating
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | -6.5 | -6.5 | +51.3 |
| 1200-1400 | -82.5 | -125.0 | +37.3 |
| 1400-1600 | -110.1 | -37.0 | +12.9 |
| 1600-1800 | -135.2 | +67.5 | -3.8 |
| 1800-2000 | -126.9 | +111.7 | -57.5 |
| 2000-2200 | -118.1 | +357.5 | -132.9 |
| 2200-2400 | -189.7 | +216.8 | -258.3 |
| 2400-2600 | -312.1 | +18.3 | -403.4 |

# B. Each phase calibrated with its own candidate-game scores (`candidate_sgf_chess`)

## Candidate score rise per 100 Elo (linear fit over buckets)
| phase | score_per_100_elo |
|---|---|
| opening | 0.054 |
| midgame | 0.088 |
| endgame | 0.049 |

## Summary over bins
| phase | mean_gap | mean_abs_gap | mean_error_vs_true_rating |
|---|---|---|---|
| opening | 14.350 | 26.664 | 5.496 |
| midgame | -15.325 | 29.943 | -24.178 |
| endgame | 0.975 | 33.039 | -7.879 |

## Gap vs phase average
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | +0.0 ± 1.2 | +0.0 ± 2.1 | +0.0 ± 1.8 |
| 1200-1400 | -26.9 ± 14.0 | -25.0 ± 10.1 | +51.9 ± 15.5 |
| 1400-1600 | +31.6 ± 10.0 | -56.9 ± 11.3 | +25.3 ± 13.5 |
| 1600-1800 | -2.5 ± 10.2 | +12.4 ± 11.9 | -9.9 ± 12.5 |
| 1800-2000 | -19.9 ± 11.2 | +6.7 ± 9.9 | +13.2 ± 17.8 |
| 2000-2200 | +1.7 ± 11.3 | -47.3 ± 8.5 | +45.6 ± 14.6 |
| 2200-2400 | +91.4 ± 22.5 | -51.8 ± 35.3 | -39.6 ± 24.4 |
| 2400-2600 | +39.4 ± 8.8 | +39.4 ± 7.8 | -78.8 ± 15.2 |

## Error vs true rating
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | -6.5 | -6.5 | -6.5 |
| 1200-1400 | -78.6 | -76.7 | +0.2 |
| 1400-1600 | +5.0 | -83.5 | -1.2 |
| 1600-1800 | -13.5 | +1.4 | -20.9 |
| 1800-2000 | -16.5 | +10.1 | +16.6 |
| 2000-2200 | +9.7 | -39.3 | +53.7 |
| 2200-2400 | +125.9 | -17.3 | -5.1 |
| 2400-2600 | +18.3 | +18.3 | -99.8 |

# C. Bucket prediction from one phase only (own calibration)

| phase | games_averaged | accuracy_exact | accuracy_within_one_bucket | mean_signed_error | mean_absolute_error | fraction_trials_with_phase_moves |
|---|---|---|---|---|---|---|
| opening | 10 | 0.442 | 0.880 | -2.123 | 144.509 | 1.000 |
| opening | 25 | 0.585 | 0.971 | -0.620 | 99.046 | 1.000 |
| opening | 100 | 0.809 | 1.000 | 2.009 | 58.520 | 1.000 |
| midgame | 10 | 0.365 | 0.799 | -26.297 | 179.356 | 1.000 |
| midgame | 25 | 0.508 | 0.925 | -27.668 | 122.235 | 1.000 |
| midgame | 100 | 0.721 | 0.992 | -25.238 | 73.672 | 1.000 |
| endgame | 10 | 0.303 | 0.665 | -4.335 | 249.872 | 1.000 |
| endgame | 25 | 0.381 | 0.806 | -9.764 | 176.601 | 1.000 |
| endgame | 100 | 0.584 | 0.974 | -11.023 | 96.211 | 1.000 |
