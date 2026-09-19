| Player | Games | Recorded rating | Fixed estimate | Error | Previous broken estimate |
|---|---:|---:|---:|---:|---:|
| caolinita | 31 | 1691 | 1567 | -124 | 1100 |
| CellblockD | 303 | 1382 | 1336 | -46 | 1100 |
| Fins | 65 | 2542 | 2119 | -423 | 1630 |
| fiodor_nabokov | 330 | 1918.5 | 1924 | +5 | 1465 |
| getting_there | 73 | 1763 | 1761 | -2 | 1215 |
| TRG80 | 98 | 1928 | 1701 | -227 | 1194 |

Period: January 1–31, 2024. Event: rated Lichess blitz. Error = estimated Elo − median rating recorded in the scored games.

Corrected mean error: -136.2 Elo; median error: -85.3 Elo; mean absolute error: 137.8 Elo. Excluding Fins, whose 2542 rating is near the calibration ceiling and whose estimate remains the largest miss, mean error is -78.9 Elo.

Root cause: repository SGFs encode castling as king-to-rook-square (`e1h1`, `e1a1`, `e8h8`, `e8a8`). Standard UCI uses king-to-destination-square (`e1g1`, `e1c1`, `e8g8`, `e8c8`). The previous PGN converter emitted standard UCI into an engine expecting the repository convention.

Exact-game control for fiodor_nabokov:

| Encoding | Games | Moves | Model score | Estimated Elo |
|---|---:|---:|---:|---:|
| Legacy corpus converter | 291 | 10,686 | 0.447279 | 1916.8 |
| Previous PGN converter | 291 | 10,686 | 0.141859 | 1465.4 |
| Fixed PGN converter | 291 | 10,686 | 0.447279 | 1916.8 |

The fixed converter exactly reproduces the legacy model score on identical games.

Python query-evaluator reproduction on all 16,000 held-out games:

| Games averaged | Python accuracy | Native accuracy |
|---:|---:|---:|
| 1 | 24.6% | 24.2% |
| 10 | 44.8% | 46.8% |
| 25 | 59.4% | 60.3% |
| 50 | 71.7% | 72.3% |
| 75 | 79.4% | 79.7% |
| 100 | 83.4% | 82.8% |

Model: chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted, weight_iter_159500.pt. Calibration: candidate_strengths.json. Aggregation: player-move-weighted mean.
