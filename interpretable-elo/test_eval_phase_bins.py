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
  results = eval_phase_bins.phase_gaps(rows_with_phase_scores(scores), [CURVE] * 3, 20, np.random.default_rng(0))
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


def synthetic_phase_rows(slopes, games=150, noise=0.2, seed=0):
  rng = np.random.default_rng(seed)
  rows = []
  for low in (1000, 1100, 1200, 1300):
    for _ in range(games):
      rating = low + 50
      row = {"low": low, "high": low + 100, "white_rating": rating, "black_rating": rating}
      for color in eval_phase_bins.COLORS:
        for phase, slope in zip(PHASES, slopes):
          row[f"{color}_{phase}_sum"] = (slope * (rating - 1000) / 100 + rng.normal(0, noise)) * 20
          row[f"{color}_{phase}_count"] = 20
      rows.append(row)
  return rows


def test_per_phase_calibration_removes_slope_driven_gaps():
  rows = synthetic_phase_rows((0.3, 0.3, 0.15))
  calibrations = eval_phase_bins.phase_calibrations(rows)
  curves = [eval_phase_bins.monotone_elo_curve(table) for table in calibrations]
  whole = eval_phase_bins.phase_gaps(rows, [curves[0]] * 3, 20, np.random.default_rng(0))
  per_phase = eval_phase_bins.phase_gaps(rows, curves, 20, np.random.default_rng(0))
  top = lambda results: next(r for r in results if r["bucket"] == "1300-1400" and r["phase"] == "endgame")
  assert top(whole)["gap_vs_phase_average"] < -40
  assert abs(top(per_phase)["gap_vs_phase_average"]) < 10


def test_phase_accuracy_is_lower_for_the_noisier_phase():
  rows = synthetic_phase_rows((0.3, 0.3, 0.1))
  calibrations = eval_phase_bins.phase_calibrations(rows)
  results = eval_phase_bins.phase_accuracy(rows, calibrations, (25,), 200, np.random.default_rng(1))
  accuracy = {row["phase"]: row["accuracy_exact"] for row in results}
  assert accuracy["opening"] > accuracy["endgame"] + 0.1
