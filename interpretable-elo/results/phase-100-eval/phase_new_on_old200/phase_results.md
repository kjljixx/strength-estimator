# Phase Elo gaps: chess_phase_100/model/weight_iter_100000.pt

Queries `query_sgf_chess`, 200 bootstrap resamples over games.
Phase Elo = Elo of the mean score over all moves in that phase, pooled over the bin's games.
Gap = phase Elo minus the average of the bin's three phase Elos (sums to zero within a bin); ± is the bootstrap standard error.
Centered gap subtracts each phase's mean gap over bins, leaving only how the gap changes with rating.
Error vs true rating = phase Elo minus the mean recorded rating of the bin's players.

# A. One whole-game calibration line for all phases (`eval_phase_100/new_on_old200/calibration.json`; fit quality {'score_per_100_elo': 0.19973642005247247, 'max_abs_bucket_residual_elo': 54.24292639884834, 'rms_bucket_residual_elo': 34.6965566332635})

## Summary over bins
| phase | mean_gap | mean_abs_gap | mean_error_vs_true_rating |
|---|---|---|---|
| opening | -38.832 | 78.035 | -41.779 |
| midgame | 74.666 | 77.504 | 71.719 |
| endgame | -35.834 | 107.649 | -38.780 |

## Gap vs phase average
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | -81.7 ± 6.8 | -11.4 ± 6.5 | +93.0 ± 7.5 |
| 1200-1400 | -95.6 ± 6.8 | +11.6 ± 6.4 | +84.0 ± 7.8 |
| 1400-1600 | -99.0 ± 7.1 | +36.0 ± 6.0 | +63.0 ± 9.0 |
| 1600-1800 | -107.2 ± 6.5 | +60.0 ± 6.6 | +47.2 ± 8.4 |
| 1800-2000 | -74.6 ± 6.5 | +95.6 ± 6.5 | -21.0 ± 8.7 |
| 2000-2200 | -9.3 ± 7.6 | +113.8 ± 5.9 | -104.5 ± 8.7 |
| 2200-2400 | +54.6 ± 7.1 | +146.6 ± 6.4 | -201.2 ± 8.2 |
| 2400-2600 | +102.3 ± 7.1 | +145.1 ± 6.7 | -247.3 ± 8.5 |

## Centered gap
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | -42.9 | -86.0 | +128.9 |
| 1200-1400 | -56.8 | -63.1 | +119.9 |
| 1400-1600 | -60.2 | -38.6 | +98.8 |
| 1600-1800 | -68.3 | -14.7 | +83.0 |
| 1800-2000 | -35.8 | +20.9 | +14.9 |
| 2000-2200 | +29.6 | +39.1 | -68.7 |
| 2200-2400 | +93.4 | +71.9 | -165.3 |
| 2400-2600 | +141.1 | +70.4 | -211.5 |

## Error vs true rating
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | -75.4 | -5.0 | +99.4 |
| 1200-1400 | -113.8 | -6.5 | +65.9 |
| 1400-1600 | -129.7 | +5.4 | +32.4 |
| 1600-1800 | -143.4 | +23.7 | +10.9 |
| 1800-2000 | -94.8 | +75.4 | -41.1 |
| 2000-2200 | +17.5 | +140.6 | -77.7 |
| 2200-2400 | +104.6 | +196.6 | -151.1 |
| 2400-2600 | +100.7 | +143.5 | -248.9 |

# B. Each phase with its own calibration line from candidate-game scores (`candidate_sgf_chess`)

## Per-phase calibration lines (score rise per 100 Elo and how far bucket means sit from the line)
| phase | score_per_100_elo | max_abs_bucket_residual_elo | rms_bucket_residual_elo |
|---|---|---|---|
| opening | 0.224 | 86.961 | 49.644 |
| midgame | 0.221 | 75.466 | 44.093 |
| endgame | 0.153 | 83.660 | 46.389 |

## Summary over bins
| phase | mean_gap | mean_abs_gap | mean_error_vs_true_rating |
|---|---|---|---|
| opening | 3.764 | 32.819 | 3.970 |
| midgame | 3.874 | 7.604 | 4.080 |
| endgame | -7.638 | 34.410 | -7.433 |

## Gap vs phase average
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | +44.7 ± 7.5 | -3.6 ± 6.5 | -41.2 ± 10.7 |
| 1200-1400 | +8.8 ± 6.6 | -4.0 ± 6.7 | -4.8 ± 9.8 |
| 1400-1600 | -18.0 ± 6.6 | -3.4 ± 7.2 | +21.4 ± 10.6 |
| 1600-1800 | -50.0 ± 6.4 | -4.0 ± 5.6 | +53.9 ± 8.9 |
| 1800-2000 | -41.9 ± 7.1 | +10.1 ± 6.7 | +31.8 ± 10.6 |
| 2000-2200 | -6.4 ± 7.4 | +7.0 ± 6.8 | -0.6 ± 10.4 |
| 2200-2400 | +32.4 ± 6.7 | +21.4 ± 6.4 | -53.8 ± 9.9 |
| 2400-2600 | +60.4 ± 7.4 | +7.4 ± 7.1 | -67.8 ± 9.7 |

## Error vs true rating
| bucket | opening | midgame | endgame |
|---|---|---|---|
| 1000-1200 | +49.7 | +1.4 | -36.2 |
| 1200-1400 | -6.2 | -19.1 | -19.8 |
| 1400-1600 | -42.3 | -27.6 | -2.9 |
| 1600-1800 | -75.9 | -29.9 | +28.0 |
| 1800-2000 | -54.1 | -2.1 | +19.5 |
| 2000-2200 | +24.7 | +38.0 | +30.4 |
| 2200-2400 | +80.6 | +69.6 | -5.7 |
| 2400-2600 | +55.4 | +2.4 | -72.9 |

# C. Bucket prediction from one phase only (own calibration)

| phase | games_averaged | accuracy_exact | accuracy_within_one_bucket | mean_signed_error | mean_absolute_error | fraction_trials_with_phase_moves |
|---|---|---|---|---|---|---|
| opening | 10 | 0.614 | 0.981 | 2.776 | 108.246 | 1.000 |
| opening | 25 | 0.780 | 0.999 | 5.156 | 77.980 | 1.000 |
| opening | 100 | 0.969 | 1.000 | 3.681 | 55.388 | 1.000 |
| midgame | 10 | 0.596 | 0.981 | 0.901 | 106.421 | 1.000 |
| midgame | 25 | 0.776 | 0.999 | 2.842 | 70.297 | 1.000 |
| midgame | 100 | 0.965 | 1.000 | 4.563 | 41.352 | 1.000 |
| endgame | 10 | 0.422 | 0.859 | -19.072 | 179.983 | 1.000 |
| endgame | 25 | 0.573 | 0.966 | -8.076 | 113.775 | 1.000 |
| endgame | 100 | 0.826 | 1.000 | -8.769 | 62.399 | 1.000 |
