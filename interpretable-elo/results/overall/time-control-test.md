| Player | Rapid games | Rapid rating | Rapid estimate | Rapid error | Blitz games | Blitz rating | Blitz estimate | Blitz error | Rapid − blitz error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| caolinita | 54 | 1754 | 1585 | -169 | 365 | 1711 | 1609 | -102 | -67 |
| CellblockD | 384 | 1506 | 1495 | -11 | 490 | 1555.5 | 1456 | -100 | +89 |
| Fins | 149 | 2637 | 2013 | -624 | 495 | 2546 | 2130 | -416 | -207 |
| fiodor_nabokov | 94 | 1923 | 1912 | -11 | 484 | 1889 | 1930 | +41 | -52 |
| getting_there | 273 | 1872 | 1790 | -82 | 49 | 1729 | 1843 | +114 | -196 |
| TRG80 | 127 | 1957.5 | 1523 | -434 | 463 | 1953 | 1699 | -254 | -181 |

Each player's rapid and blitz games use the same overlapping date window. Ratings are medians recorded in those games. Error = estimate − recorded rating. Judd0104 had no overlapping dates and was excluded.

Mean rapid error: -221.9 Elo. Mean blitz error: -119.4 Elo. Mean paired difference: -102.5 Elo; median paired difference: -123.9 Elo. Paired t-test: t=-2.184, df=5, p=0.081. Five of six players score lower in rapid, but this six-player result does not reach the conventional 0.05 significance threshold.

These are corrected results using repository castling notation. Earlier values were invalid because standard UCI castling corrupted the model's position sequence. Preserving PGN clock annotations changes estimates by effectively 0 Elo.

The calibration source is Lichess rated blitz from January 2024. Candidate/query games are random samples after same-bin filtering. Win-chain training then keeps only players that can form an eight-player directed win chain; it caps stored games per player at 32 and stored wins per player at 16, chooses one random non-chain-edge game per chain slot, samples at most 50,000 chains, and deduplicates the final game pool.

Model: chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted, weight_iter_159500.pt. Calibration: candidate_strengths.json. Aggregation: player-move-weighted mean.
