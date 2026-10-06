# Phase Elo gaps: chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted/model/weight_iter_159500.pt

Queries `query_sgf_chess`, 200 bootstrap resamples over games.
Phase Elo = Elo of the mean score over all moves in that phase, pooled over the bin's games.
Gap = phase Elo minus the average of the bin's three phase Elos (sums to zero within a bin); ± is the bootstrap standard error.
Centered gap subtracts each phase's mean gap over bins, leaving only how the gap changes with rating.
Error vs true rating = phase Elo minus the mean recorded rating of the bin's players.

# A. One whole-game calibration line for all phases (`eval_phase_100/old_on_old200/calibration.json`; fit quality {'score_per_100_elo': 0.06561479587941751, 'max_abs_bucket_residual_elo': 90.3896122613628, 'rms_bucket_residual_elo': 48.31063919257662})

## Summary over bins
| phase | mean_gap | mean_abs_gap | mean_error_vs_true_rating |
|---|---|---|---|
| opening | -89.321 | 96.054 | -118.830 |
| midgame | 145.928 | 188.233 | 116.419 |
| endgame | -56.607 | 115.166 | -86.116 |

## Gap vs phase average
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | +26.9 ± 10.2 | -108.2 ± 12.6 | +81.3 ± 13.1 |
| 1200-1400 | -23.5 ± 10.1 | -61.0 ± 11.3 | +84.5 ± 10.7 |
| 1400-1600 | -61.4 ± 9.1 | +7.4 ± 13.2 | +54.0 ± 13.9 |
| 1600-1800 | -107.0 ± 10.5 | +92.6 ± 12.9 | +14.4 ± 12.3 |
| 1800-2000 | -128.5 ± 10.8 | +179.7 ± 14.0 | -51.3 ± 12.7 |
| 2000-2200 | -129.6 ± 11.2 | +281.7 ± 12.5 | -152.1 ± 12.9 |
| 2200-2400 | -133.1 ± 11.5 | +370.1 ± 13.3 | -237.1 ± 13.0 |
| 2400-2600 | -158.4 ± 11.9 | +405.1 ± 13.7 | -246.7 ± 14.3 |

## Centered gap
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | +116.3 | -254.1 | +137.9 |
| 1200-1400 | +65.8 | -206.9 | +141.1 |
| 1400-1600 | +27.9 | -138.6 | +110.6 |
| 1600-1800 | -17.7 | -53.3 | +71.0 |
| 1800-2000 | -39.2 | +33.8 | +5.4 |
| 2000-2200 | -40.3 | +135.8 | -95.5 |
| 2200-2400 | -43.7 | +224.2 | -180.5 |
| 2400-2600 | -69.1 | +259.1 | -190.1 |

## Error vs true rating
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | +5.2 | -130.0 | +59.5 |
| 1200-1400 | -81.8 | -119.3 | +26.2 |
| 1400-1600 | -124.2 | -55.5 | -8.8 |
| 1600-1800 | -160.4 | +39.2 | -39.0 |
| 1800-2000 | -155.0 | +153.2 | -77.8 |
| 2000-2200 | -95.6 | +315.7 | -118.1 |
| 2200-2400 | -102.7 | +400.5 | -206.7 |
| 2400-2600 | -236.0 | +327.5 | -324.3 |

# B. Each phase with its own calibration line from candidate-game scores (`candidate_sgf_chess`)

## Per-phase calibration lines (score rise per 100 Elo and how far bucket means sit from the line)
| phase | score_per_100_elo | max_abs_bucket_residual_elo | rms_bucket_residual_elo |
|---|---|---|---|
| opening | 0.054 | 79.565 | 51.382 |
| midgame | 0.088 | 137.475 | 73.775 |
| endgame | 0.049 | 40.921 | 20.651 |

## Summary over bins
| phase | mean_gap | mean_abs_gap | mean_error_vs_true_rating |
|---|---|---|---|
| opening | 12.166 | 30.254 | -1.112 |
| midgame | -9.104 | 13.884 | -22.382 |
| endgame | -3.062 | 40.810 | -16.339 |

## Gap vs phase average
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | +32.3 ± 12.3 | -6.6 ± 11.5 | -25.7 ± 16.6 |
| 1200-1400 | -3.7 ± 10.7 | -20.5 ± 10.0 | +24.1 ± 14.2 |
| 1400-1600 | -16.0 ± 10.6 | -25.7 ± 11.7 | +41.6 ± 15.5 |
| 1600-1800 | -33.8 ± 10.9 | -18.1 ± 11.1 | +51.9 ± 14.4 |
| 1800-2000 | -18.9 ± 11.6 | -14.4 ± 10.9 | +33.3 ± 15.7 |
| 2000-2200 | +27.8 ± 12.2 | -6.7 ± 11.6 | -21.1 ± 15.6 |
| 2200-2400 | +62.2 ± 13.4 | +9.4 ± 13.4 | -71.6 ± 17.9 |
| 2400-2600 | +47.4 ± 13.2 | +9.7 ± 12.5 | -57.1 ± 16.9 |

## Error vs true rating
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | +5.0 | -33.9 | -53.0 |
| 1200-1400 | -58.8 | -75.6 | -31.0 |
| 1400-1600 | -68.4 | -78.1 | -10.8 |
| 1600-1800 | -71.6 | -55.9 | +14.2 |
| 1800-2000 | -24.2 | -19.7 | +28.0 |
| 2000-2200 | +87.9 | +53.4 | +39.0 |
| 2200-2400 | +120.4 | +67.6 | -13.3 |
| 2400-2600 | +0.7 | -37.0 | -103.8 |

# C. Bucket prediction from one phase only (own calibration)

| phase | games_averaged | accuracy_exact | accuracy_within_one_bucket | mean_signed_error | mean_absolute_error | fraction_trials_with_phase_moves |
|---|---|---|---|---|---|---|
| opening | 10 | 0.442 | 0.880 | 1.159 | 175.080 | 1.000 |
| opening | 25 | 0.585 | 0.971 | 0.538 | 118.220 | 1.000 |
| opening | 100 | 0.809 | 1.000 | -1.439 | 75.601 | 1.000 |
| midgame | 10 | 0.365 | 0.799 | -19.719 | 219.990 | 1.000 |
| midgame | 25 | 0.508 | 0.925 | -20.138 | 137.999 | 1.000 |
| midgame | 100 | 0.721 | 0.992 | -22.823 | 80.190 | 1.000 |
| endgame | 10 | 0.303 | 0.665 | -15.994 | 331.234 | 1.000 |
| endgame | 25 | 0.381 | 0.806 | -19.798 | 213.083 | 1.000 |
| endgame | 100 | 0.584 | 0.974 | -21.715 | 109.024 | 1.000 |
