# Phase Elo gaps: chess_phase_100/model/weight_iter_100000.pt

Queries `query_sgf_chess`, 200 bootstrap resamples over games.
Phase Elo = Elo of the mean score over all moves in that phase, pooled over the bin's games.
Gap = phase Elo minus the average of the bin's three phase Elos (sums to zero within a bin); ± is the bootstrap standard error.
Centered gap subtracts each phase's mean gap over bins, leaving only how the gap changes with rating.
Error vs true rating = phase Elo minus the mean recorded rating of the bin's players.

# A. One whole-game calibration for all phases (`eval_phase_100/new_on_old200/calibration.json`; monotone, clamped)

## Summary over bins
| phase | mean_gap | mean_abs_gap | mean_error_vs_true_rating |
|---|---|---|---|
| opening | -32.379 | 73.211 | -41.688 |
| midgame | 67.615 | 75.602 | 58.306 |
| endgame | -35.237 | 102.716 | -44.545 |

## Gap vs phase average
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | -31.9 ± 4.4 | -31.9 ± 4.4 | +63.9 ± 8.7 |
| 1200-1400 | -120.1 ± 8.6 | +13.7 ± 8.2 | +106.4 ± 9.9 |
| 1400-1600 | -102.1 ± 7.8 | +40.2 ± 5.2 | +61.8 ± 7.8 |
| 1600-1800 | -87.6 ± 5.6 | +49.8 ± 6.1 | +37.8 ± 7.6 |
| 1800-2000 | -68.4 ± 6.1 | +86.6 ± 5.7 | -18.2 ± 8.1 |
| 2000-2200 | -12.3 ± 6.7 | +106.7 ± 5.7 | -94.4 ± 7.6 |
| 2200-2400 | +60.9 ± 12.4 | +173.4 ± 7.9 | -234.3 ± 8.3 |
| 2400-2600 | +102.4 ± 4.1 | +102.4 ± 4.1 | -204.9 ± 8.1 |

## Centered gap
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | +0.4 | -99.6 | +99.1 |
| 1200-1400 | -87.8 | -53.9 | +141.7 |
| 1400-1600 | -69.7 | -27.4 | +97.1 |
| 1600-1800 | -55.2 | -17.8 | +73.0 |
| 1800-2000 | -36.0 | +18.9 | +17.1 |
| 2000-2200 | +20.1 | +39.1 | -59.2 |
| 2200-2400 | +93.3 | +105.8 | -199.1 |
| 2400-2600 | +134.8 | +34.8 | -169.6 |

## Error vs true rating
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | -6.5 | -6.5 | +89.4 |
| 1200-1400 | -127.6 | +6.3 | +99.0 |
| 1400-1600 | -95.4 | +46.9 | +68.5 |
| 1600-1800 | -110.9 | +26.4 | +14.4 |
| 1800-2000 | -97.1 | +57.8 | -46.9 |
| 2000-2200 | -18.6 | +100.4 | -100.8 |
| 2200-2400 | +104.2 | +216.8 | -191.0 |
| 2400-2600 | +18.3 | +18.3 | -289.0 |

# B. Each phase calibrated with its own candidate-game scores (`candidate_sgf_chess`)

## Candidate score rise per 100 Elo (linear fit over buckets)
| phase | score_per_100_elo |
|---|---|
| opening | 0.224 |
| midgame | 0.221 |
| endgame | 0.153 |

## Summary over bins
| phase | mean_gap | mean_abs_gap | mean_error_vs_true_rating |
|---|---|---|---|
| opening | 4.020 | 14.876 | 2.903 |
| midgame | 0.930 | 9.549 | -0.187 |
| endgame | -4.950 | 13.581 | -6.068 |

## Gap vs phase average
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | +0.0 ± 1.0 | +0.0 ± 1.2 | +0.0 ± 1.8 |
| 1200-1400 | -36.2 ± 7.7 | +5.1 ± 7.5 | +31.1 ± 10.6 |
| 1400-1600 | +12.8 ± 4.8 | -16.2 ± 5.9 | +3.4 ± 6.8 |
| 1600-1800 | -7.2 ± 4.7 | +21.3 ± 4.2 | -14.1 ± 5.6 |
| 1800-2000 | +7.0 ± 8.1 | -1.0 ± 7.4 | -6.0 ± 13.3 |
| 2000-2200 | +19.3 ± 6.8 | -12.2 ± 6.4 | -7.0 ± 10.3 |
| 2200-2400 | +21.0 ± 9.0 | -5.0 ± 10.2 | -15.9 ± 13.4 |
| 2400-2600 | +15.5 ± 6.1 | +15.5 ± 6.2 | -31.0 ± 12.3 |

## Error vs true rating
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | -6.5 | -6.5 | -6.5 |
| 1200-1400 | -45.6 | -4.3 | +21.8 |
| 1400-1600 | +36.1 | +7.1 | +26.7 |
| 1600-1800 | -25.7 | +2.8 | -32.6 |
| 1800-2000 | -11.2 | -19.2 | -24.2 |
| 2000-2200 | +16.9 | -14.6 | -9.4 |
| 2200-2400 | +40.8 | +14.8 | +3.9 |
| 2400-2600 | +18.3 | +18.3 | -28.2 |

# C. Bucket prediction from one phase only (own calibration)

| phase | games_averaged | accuracy_exact | accuracy_within_one_bucket | mean_signed_error | mean_absolute_error | fraction_trials_with_phase_moves |
|---|---|---|---|---|---|---|
| opening | 10 | 0.614 | 0.981 | 1.242 | 92.556 | 1.000 |
| opening | 25 | 0.780 | 0.999 | 4.337 | 61.890 | 1.000 |
| opening | 100 | 0.969 | 1.000 | 3.039 | 34.796 | 1.000 |
| midgame | 10 | 0.596 | 0.981 | 0.591 | 94.161 | 1.000 |
| midgame | 25 | 0.776 | 0.999 | 2.311 | 61.285 | 1.000 |
| midgame | 100 | 0.965 | 1.000 | 3.227 | 31.527 | 1.000 |
| endgame | 10 | 0.422 | 0.859 | -15.543 | 153.922 | 1.000 |
| endgame | 25 | 0.573 | 0.966 | -5.820 | 102.259 | 1.000 |
| endgame | 100 | 0.826 | 1.000 | -6.603 | 54.867 | 1.000 |
