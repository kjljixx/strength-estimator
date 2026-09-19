from __future__ import annotations

import importlib.util
import io
import unittest
from pathlib import Path

import chess.pgn


MODULE_PATH = Path(__file__).with_name("analyze.py")
SPEC = importlib.util.spec_from_file_location("interpretable_elo_analyze", MODULE_PATH)
analyze = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(analyze)


def parse_game(text: str):
  return chess.pgn.read_game(io.StringIO(text))


class AnalyzeTest(unittest.TestCase):
  def test_phase_boundaries(self):
    self.assertEqual(analyze.phase_for(10, 10, 30), "early")
    self.assertEqual(analyze.phase_for(11, 10, 30), "midgame")
    self.assertEqual(analyze.phase_for(31, 10, 30), "endgame")

  def test_instantaneous_score(self):
    self.assertAlmostEqual(analyze.instantaneous_score(0.6, 0.5, 3), 0.8)

  def test_player_match_and_sgf_conversion(self):
    game = parse_game(
      '[White "Alice"]\n[Black "Bob"]\n[Result "1-0"]\n\n'
      "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. O-O 1-0"
    )
    self.assertEqual(analyze.player_color(game, "alice"), "white")
    sgf, positions = analyze.game_to_sgf(game)
    self.assertIn(";B[e2e4];W[e7e5]", sgf)
    self.assertIn(";B[e1h1]", sgf)
    self.assertEqual(positions[0]["fen_before"], chess.pgn.Game().board().fen())


if __name__ == "__main__":
  unittest.main()
