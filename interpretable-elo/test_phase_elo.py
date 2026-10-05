from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
SPEC = importlib.util.spec_from_file_location("phase_elo", Path(__file__).with_name("phase_elo.py"))
phase_elo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(phase_elo)


class PhaseEloTest(unittest.TestCase):
  def test_candidate_boards_and_phase_totals(self):
    sgf = "(;GM[chess];B[e2e4];W[e7e5];B[g1f3];W[b8c6])"
    boards = phase_elo.candidate_boards(sgf)
    self.assertEqual(len(boards), 4)

    class Scorer:
      def score_sgf(self, prefix):
        return {"white": 0.5, "black": 0.25}

    totals = phase_elo.phase_totals(Scorer(), sgf, boards)
    self.assertEqual(totals["white"]["opening"], (1.0, 2))
    self.assertEqual(totals["black"]["opening"], (0.5, 2))

  def test_centering_preserves_total(self):
    raw = {"opening": 1700.0, "midgame": 1500.0, "endgame": 1300.0}
    reference = {"opening": 1800.0, "midgame": 1500.0, "endgame": 1200.0}
    mean = sum(reference.values()) / 3
    normalized = phase_elo.center_ratings(raw, {"phase_elo": reference, "mean_elo": mean})
    self.assertAlmostEqual(sum(normalized.values()) / 3, sum(raw.values()) / 3)
    self.assertEqual(phase_elo.center_ratings(reference, {"phase_elo": reference,
                                                         "mean_elo": mean}),
                     dict.fromkeys(reference, mean))
