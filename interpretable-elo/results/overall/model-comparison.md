| Player | Current rapid | Median in PGN | Games | Win-chain estimate | Non-win-chain estimate | Non-win error vs current |
|---|---:|---:|---:|---:|---:|---:|
| pravan_sanghvi | 1125 | 1217 | 981 | ≤1100 | ≤1100 | ≤-25 |
| olgaastafjeva | 1217 | 1251 | 978 | 1158 | 1305 | +88 |
| Robsd | 1281 | 1254 | 1,007 | ≤1100 | 1133 | -148 |
| Gillou11 | 1397 | 1415 | 1,007 | ≤1100 | 1167 | -230 |
| EricRosen | 2571 | 2600.5 | 987 | 2058 | 2114 | -457 |

| Metric vs current rapid | Win-chain | Non-win-chain |
|---|---:|---:|
| Mean signed error | ≤-215 | ≤-155 |
| Mean absolute error | ≥215 | ≥190 |

Non-win-chain model: `chess_bt_b32_r8_p7_20bx256-d94989`, checkpoint `weight_iter_53500.pt`.

Its config uses `training_sgf_chess` and leaves `bt_use_win_chains` disabled. Each model uses its own calibration. Scores between models are not directly comparable.

`≤` and `≥` appear because pravan_sanghvi falls below the calibration range and is reported at its 1100 floor.
