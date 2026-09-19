| Player | Recorded rating (median) | Games | Predicted Elo | Error |
|---|---:|---:|---:|---:|
| PatMargency | 1541 | 82 | 1771 | +230 |
| Hamad86 | 1577 | 71 | 1533 | -44 |
| Claudio_Olmos | 1619 | 72 | 1776 | +157 |
| xxx123robert | 1633 | 79 | 1636 | +3 |
| TheBeginner76 | 1676 | 73 | 1587 | -89 |
| rezaesk58 | 1746 | 85 | 1682 | -64 |
| sg2210 | 1847 | 83 | 1813 | -34 |
| dalekhine | 1974.5 | 72 | 2109 | +135 |

617 player-games; 21,432 player moves; zero skipped games; zero clipped predictions.
Mean signed error: +36.7 Elo. Mean absolute error: 94.5 Elo. Median signed error: -15.9 Elo.
Errors use unrounded predictions minus median recorded rating; summary weights players equally.

Selection: eight most frequent players with median recorded rating in [1500, 2100), selected before model scoring. All their available corpus games were scored. sg2210 overlaps the previous cohort; seven players are new. Player-games can include the same game for both selected opponents.

Source: /workspace/training_sgf_chess_chain/games.txt. Training overlap remains unresolved; this is not independent validation and does not isolate the effect of time control.

Model: chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted, weight_iter_159500.pt; existing candidate_strengths.json calibration; move-weighted overall score.

Remote command (exit 0): `docker exec -w /workspace vigorous_khayyam python -u /tmp/run_higher_cohort.py`
