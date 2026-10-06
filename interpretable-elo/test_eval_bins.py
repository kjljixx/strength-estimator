from __future__ import annotations

import numpy as np

import eval_bins

LOWS = (1000, 1100, 1200, 1300)


def synthetic_rows(rng, offset=0.0, noise=0.01):
  rows = []
  for low in LOWS:
    for _ in range(300):
      ratings = rng.integers(low, low + 100, size=2)
      scores = 0.001 * (ratings - 1000) + offset + rng.normal(0, noise, size=2)
      rows.append({"low": low, "high": low + 100,
                   "white_rating": ratings[0], "black_rating": ratings[1],
                   "white_score": scores[0], "white_moves": 40, "black_score": scores[1], "black_moves": 40})
  return rows


def evaluate(query_rows, calibration, sizes=(1, 100)):
  results = eval_bins.run_trials(query_rows, calibration, sizes, 200, np.random.default_rng(0))
  return results, *eval_bins.summarize(results, calibration, sizes)


def test_unbiased_scores_have_near_zero_signed_error():
  rng = np.random.default_rng(1)
  calibration = eval_bins.build_calibration(synthetic_rows(rng))
  _, overall, by_bucket = evaluate(synthetic_rows(rng), calibration)
  assert abs(overall[1]["mean_signed_error"]) < 5
  assert overall[1]["accuracy_exact"] > 0.9
  assert all(abs(row["mean_signed_error"]) < 8 for row in by_bucket if row["games_averaged"] == 100)


def test_score_offset_gives_positive_signed_error():
  rng = np.random.default_rng(2)
  calibration = eval_bins.build_calibration(synthetic_rows(rng))
  _, overall, _ = evaluate(synthetic_rows(rng, offset=0.03), calibration)
  assert overall[1]["mean_signed_error"] > 20


def test_pairwise_ordering_improves_with_gap_and_games():
  rng = np.random.default_rng(3)
  rows = synthetic_rows(rng, noise=0.08)
  calibration = eval_bins.build_calibration(rows)
  results, _, _ = evaluate(rows, calibration, sizes=(1, 100))
  pairwise = {(row["games_averaged"], row["bucket_gap"]): row["ordering_accuracy"]
              for row in eval_bins.pairwise_ordering(results, calibration, (1, 100))}
  assert pairwise[(1, 3)] > pairwise[(1, 1)] > 0.5
  assert pairwise[(100, 1)] > pairwise[(1, 1)]


def test_calibration_order_violation_is_reported():
  calibration = [{"low": 1000, "high": 1100, "mean_score": 0.2}, {"low": 1100, "high": 1200, "mean_score": 0.1}]
  assert len(eval_bins.calibration_monotonic(calibration)) == 1


def test_inverted_neighbour_buckets_do_not_explode_estimates():
  rng = np.random.default_rng(4)
  calibration = eval_bins.build_calibration(synthetic_rows(rng))
  calibration[0]["mean_score"], calibration[1]["mean_score"] = calibration[1]["mean_score"] - 0.0001, calibration[0]["mean_score"]
  _, overall, _ = evaluate(synthetic_rows(rng), calibration)
  assert all(row["mean_absolute_error"] < 200 for row in overall)


def test_linear_calibration_recovers_slope_and_inverts():
  calibration = [{"center": center, "mean_score": 0.002 * (center - 1500)} for center in (1000, 1100, 1200, 1300)]
  line = eval_bins.linear_calibration(calibration)
  assert abs(line[1] - 0.002) < 1e-12
  assert abs(float(eval_bins.elo_from_score(0.0, line)) - 1500) < 1e-6
  assert eval_bins.line_fit_quality(calibration, line)["max_abs_bucket_residual_elo"] < 1e-6
  assert abs(float(eval_bins.elo_from_score(0.5, line)) - 1750) < 1e-6
