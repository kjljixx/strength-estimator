# Evaluation summary: phase-example model at 100,000 steps

Model: `chess_phase_100`, `weight_iter_100000` (trained on opening, midgame and endgame examples, 16 Elo bins of 100 points).
Old model: `chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted`, `weight_iter_159500` (trained on whole games, 200-point bins).
Scores are converted to Elo with a straight calibration line (score = a + b × Elo, fitted on the candidate games of each bin and then inverted). Nothing is clamped.

## 1. New model: mean signed Elo error

**Error here is measured against the ground truth:** signed error = estimated Elo − mean recorded rating of the sampled players (the `WR`/`BR` ratings in the SGFs). A positive value means the model rates players too high.

**How it is computed.** Held-out query games (2,000 per 100-point bin, 32,000 in total, never used in training). For each bin and each "games averaged" value N, 500 random groups of N player-games are drawn from the bin (with replacement). Each group's move-weighted mean score is converted to an Elo with the whole-game calibration line, and the error is that Elo minus the mean recorded rating of the group's players. The numbers below are averages over those groups, so they are per group of N games. N = 1 is a single game and is very noisy.

### 1a. Overall (all 16 bins, equal weight per bin)

| Games averaged | Mean signed error (Elo) | Mean absolute error (Elo) | Exact-bin accuracy |
|---|---:|---:|---:|
| 1 | +5.1 | 271.3 | 17.1% |
| 10 | +10.7 | 87.5 | 36.4% |
| 25 | +10.7 | 60.1 | 48.1% |
| 50 | +11.8 | 45.9 | 60.7% |
| 75 | +11.0 | 40.2 | 66.1% |
| 100 | +11.3 | 37.4 | 70.4% |

### 1b. Broken down by Elo bin

| Bin | 1 game averaged | 10 games averaged | 100 games averaged |
|---|---:|---:|---:|
| 1000–1100 | +9 | +35 | +38 |
| 1100–1200 | -25 | +7 | +13 |
| 1200–1300 | +14 | +15 | +22 |
| 1300–1400 | -5 | +22 | +6 |
| 1400–1500 | -21 | -3 | -5 |
| 1500–1600 | -41 | -16 | -7 |
| 1600–1700 | -7 | -25 | -15 |
| 1700–1800 | -35 | -24 | -24 |
| 1800–1900 | -4 | -6 | -7 |
| 1900–2000 | +16 | +18 | +22 |
| 2000–2100 | +5 | +30 | +23 |
| 2100–2200 | +83 | +58 | +55 |
| 2200–2300 | +77 | +54 | +65 |
| 2300–2400 | +63 | +55 | +50 |
| 2400–2500 | +6 | +2 | +4 |
| 2500–2600 | -53 | -52 | -59 |

Values for 25, 50 and 75 games averaged are in `new_on_new100/bucket_by_games.csv`. Pairwise ordering and the other metrics are in `new_on_new100/results.md`.

## 2. Phase gaps: new model vs old model, uncalibrated

**Read this first. This is not error against the true rating.** Each value is a *gap against the predicted Elo*: the phase's estimated Elo minus the average of that same bin's three predicted phase Elos (opening, midgame and endgame). The recorded rating is not used anywhere in this section. A positive value means the model rates that phase higher than it rates the bin's other phases on average. Within a bin the three values sum to zero.

**These are per model evaluation, not per game and not per player.** Each cell is one number for one bin and one phase from one evaluation run of one model: the mean score over all moves of that phase in all the bin's query games (about 40,000 moves for the opening of a bin), converted to an Elo, minus the bin's three-phase average. Games and players are not scored separately. A per-game or per-player version would be a different, noisier analysis.

**"Uncalibrated"** means one whole-game calibration line is used for all three phases, so a phase whose scores sit on a different scale shows up as a gap.

Bins are the old 200-point held-out bins, because the old model can only be scored fairly on games it did not train on (the new 100-point query games were probably in its training data).

### 2a. New model, uncalibrated

| Bin | Opening | Midgame | Endgame |
|---|---:|---:|---:|
| 1000–1200 | -82 | -11 | +93 |
| 1200–1400 | -96 | +12 | +84 |
| 1400–1600 | -99 | +36 | +63 |
| 1600–1800 | -107 | +60 | +47 |
| 1800–2000 | -75 | +96 | -21 |
| 2000–2200 | -9 | +114 | -104 |
| 2200–2400 | +55 | +147 | -201 |
| 2400–2600 | +102 | +145 | -247 |
| **Mean signed** | **-39** | **+75** | **-36** |
| Mean absolute | 78 | 78 | 108 |

### 2b. Old model, uncalibrated

| Bin | Opening | Midgame | Endgame |
|---|---:|---:|---:|
| 1000–1200 | +27 | -108 | +81 |
| 1200–1400 | -24 | -61 | +85 |
| 1400–1600 | -61 | +7 | +54 |
| 1600–1800 | -107 | +93 | +14 |
| 1800–2000 | -128 | +180 | -51 |
| 2000–2200 | -130 | +282 | -152 |
| 2200–2400 | -133 | +370 | -237 |
| 2400–2600 | -158 | +405 | -247 |
| **Mean signed** | **-89** | **+146** | **-57** |
| Mean absolute | 96 | 188 | 115 |

### Notes
- Mean signed is the average of the eight bin values in that column. Mean absolute ignores sign, so opposite-signed bins do not cancel.
- The calibration line is fitted on a 200-game candidate set per bin. That adds noise of roughly 10–20 Elo to every cell that is not shown by any standard error, so gaps below that size should not be read as meaningful.
- A straight line fits the score-to-Elo relationship imperfectly (bin means sit up to about 75 Elo from the whole-game line), which adds structured error to both models' tables.

## 3. New model: uncalibrated vs calibrated, 16 bins

**Same definition as section 2: this is a gap against the predicted Elo, not error against the true rating, and each cell is one value per model evaluation (one bin, one phase), not per game or per player.** Each value is the phase's estimated Elo minus the average of that bin's three phase estimates, in Elo. A positive value means the model rates that phase higher than the bin's other phases.

- **Uncalibrated:** one whole-game calibration line for all three phases.
- **Calibrated:** each phase has its own calibration line, fitted on that phase's average score in the candidate games of each bin.

This section uses the new model on its own 16 100-point held-out bins. The old model is not shown here: it has no 16-bin evaluation, because these query games were probably in its training data.

### 3a. Gaps by bin

| Bin | Uncal. open | Uncal. mid | Uncal. end | Cal. open | Cal. mid | Cal. end |
|---|---:|---:|---:|---:|---:|---:|
| 1000–1100 | -84 | -24 | +109 | +33 | -14 | -19 |
| 1100–1200 | -86 | -13 | +99 | +22 | -13 | -9 |
| 1200–1300 | -83 | +1 | +83 | +12 | -13 | +1 |
| 1300–1400 | -85 | +14 | +71 | +0 | -12 | +12 |
| 1400–1500 | -101 | +49 | +52 | -25 | +10 | +15 |
| 1500–1600 | -103 | +36 | +67 | -42 | -16 | +58 |
| 1600–1700 | -102 | +57 | +45 | -50 | -7 | +57 |
| 1700–1800 | -99 | +67 | +32 | -59 | -10 | +69 |
| 1800–1900 | -87 | +85 | +2 | -60 | -5 | +65 |
| 1900–2000 | -59 | +102 | -43 | -47 | -1 | +48 |
| 2000–2100 | -31 | +120 | -89 | -30 | +8 | +23 |
| 2100–2200 | +1 | +122 | -123 | -16 | -4 | +20 |
| 2200–2300 | +31 | +139 | -170 | +2 | +2 | -3 |
| 2300–2400 | +67 | +141 | -208 | +26 | -4 | -22 |
| 2400–2500 | +100 | +136 | -236 | +50 | -13 | -37 |
| 2500–2600 | +108 | +138 | -246 | +53 | -14 | -39 |
| **Mean signed** | **-38** | **+73** | **-35** | **-8** | **-7** | **+15** |
| Mean absolute | 77 | 78 | 105 | 33 | 9 | 31 |

### 3b. How well each straight line fits (new model, 16 bins)

| Calibration line | Score rise per 100 Elo | Largest bin distance from line (Elo) | RMS distance (Elo) |
|---|---:|---:|---:|
| Whole game | 0.203 | 75 | 41 |
| Opening | 0.229 | 131 | 64 |
| Midgame | 0.228 | 83 | 42 |
| Endgame | 0.155 | 127 | 64 |

### Notes
- Calibrating each phase separately removes most of the average gap (mean signed row) and shrinks the typical size, but a straight line leaves a bin-dependent pattern (mean absolute row) because the score-to-Elo relationship is not linear.
- The calibration lines come from 200 candidate games per bin, so gaps of about 10–20 Elo are within noise.
- Within each bin the three gaps of one calibration sum to zero.
