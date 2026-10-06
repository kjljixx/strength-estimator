from __future__ import annotations

import numpy as np

import eval_phase_bins
from phase_rules import PHASES

CURVE = (np.array([-1.0, 0.0, 1.0]), np.array([1000.0, 1500.0, 2000.0]))


def rows_with_phase_scores(scores, games=50, rating=1500):
  rows = []
  for _ in range(games):
    row = {"low": 1500, "high": 1600, "white_rating": rating, "black_rating": rating}
    for color in eval_phase_bins.COLORS:
      for phase, score in zip(PHASES, scores):
        row[f"{color}_{phase}_sum"] = score * 20
        row[f"{color}_{phase}_count"] = 20
    rows.append(row)
  return rows


def gaps(scores):
  results = eval_phase_bins.phase_gaps(rows_with_phase_scores(scores), CURVE, 20, np.random.default_rng(0))
  return {row["phase"]: row for row in results}


def test_equal_phase_scores_have_zero_gap():
  assert all(abs(row["gap_vs_phase_average"]) < 1e-9 for row in gaps((0.0, 0.0, 0.0)).values())


def test_higher_endgame_score_gives_positive_endgame_gap_and_zero_sum():
  by_phase = gaps((0.0, 0.0, 0.3))
  assert by_phase["endgame"]["gap_vs_phase_average"] > 0 > by_phase["opening"]["gap_vs_phase_average"]
  assert abs(sum(row["gap_vs_phase_average"] for row in by_phase.values())) < 1e-9
  assert abs(by_phase["endgame"]["error_vs_true_rating"] - 150) < 1e-9


def test_gap_is_zero_when_one_bin_only_after_centering():
  assert all(abs(row["gap_centered"]) < 1e-9 for row in gaps((0.0, 0.1, 0.2)).values())
