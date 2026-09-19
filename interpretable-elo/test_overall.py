from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))
SPEC = importlib.util.spec_from_file_location("interpretable_elo_overall", MODULE_DIR / "overall.py")
overall = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(overall)


class OverallTest(unittest.TestCase):
  def test_color_move_count(self):
    self.assertEqual(overall.color_move_count(9, "white"), 5)
    self.assertEqual(overall.color_move_count(9, "black"), 4)

  def test_weighted_mean_weights_moves(self):
    self.assertEqual(overall.weighted_mean([(1.0, 10), (0.0, 30)]), 0.25)

  def test_estimate_elo_interpolates_neighboring_buckets(self):
    calibration = [
      {"low": 1000, "high": 1200, "center": 1100, "mean_score": -1.0},
      {"low": 1200, "high": 1400, "center": 1300, "mean_score": 1.0},
    ]
    result = overall.estimate_elo(0.0, calibration)
    self.assertEqual(result["estimated_elo"], 1200)
    self.assertEqual(result["nearest_bucket"], "1000-1200")
    self.assertEqual(result["estimate_status"], "interpolated")

  def test_scores_named_side_from_sgf_file(self):
    class Scorer:
      def score_sgf(self, _sgf):
        return {"white": 0.25, "black": -0.5}

    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "games.txt"
      path.write_text(
        "(;PW[Alice]PB[Bob]WR[1234]BR[1300];B[e2e4];W[e7e5])\n",
        encoding="utf-8",
      )
      result = overall.score_sgf_player(Scorer(), path, "alice")
    self.assertEqual(result["analyzed_games"], 1)
    self.assertEqual(result["moves"], 1)
    self.assertEqual(result["mean_score"], 0.25)
    self.assertEqual(result["median_recorded_rating"], 1234)


if __name__ == "__main__":
  unittest.main()
