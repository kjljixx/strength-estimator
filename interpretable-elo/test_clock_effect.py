"""Measure effect of preserving Lichess clock annotations during PGN conversion."""

import json
import math
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/tmp/interpretable-overall")
from overall import color_move_count, estimate_elo, player_color, weighted_mean
from build.chess import strength_py

DATA_DIR = Path("/tmp/time-control-test")
PLAYERS = ("caolinita", "CellblockD", "Fins", "fiodor_nabokov", "getting_there", "TRG80")


def game_to_sgf(game, include_clocks):
  board = game.board()
  moves = []
  for node in game.mainline():
    move = node.move
    item = f";{'B' if board.turn == chess.WHITE else 'W'}[{move.uci()}]"
    clock = node.clock()
    if include_clocks and clock is not None:
      item += f"TM[{int(clock)}]"
    moves.append(item)
    board.push(move)
  result = {"1-0": "1.0", "0-1": "-1.0"}.get(game.headers.get("Result"), "0.0")
  return f"(;GM[chess]RE[{result}]PW[{game.headers.get('White', '')}]PB[{game.headers.get('Black', '')}]{''.join(moves)})"


def score_file(scorer, path, player, include_clocks):
  scores = []
  ratings = []
  with path.open(encoding="utf-8", errors="replace") as stream:
    while game := chess.pgn.read_game(stream):
      color = player_color(game, player)
      if color is None or game.board().fen() != chess.Board().fen():
        continue
      rating = game.headers.get("WhiteElo" if color == "white" else "BlackElo", "")
      if rating.isdigit():
        ratings.append(int(rating))
      plies = game.end().ply()
      moves = color_move_count(plies, color)
      if plies < 2 or moves == 0:
        continue
      score = float(scorer.score_sgf(game_to_sgf(game, include_clocks))[color])
      if math.isfinite(score):
        scores.append((score, moves))
  return weighted_mean(scores), sorted(ratings)[len(ratings) // 2], len(scores)


def main():
  model = "/workspace/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted"
  scorer = strength_py.StrengthScorer(
    f"{model}/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted.cfg",
    f"{model}/model/weight_iter_159500.pt",
    0,
  )
  calibration = json.loads(Path("/tmp/interpretable-overall/candidate_strengths.json").read_text())
  print(f"Config: players={len(PLAYERS)}, compare=no_clocks_vs_clocks", flush=True)
  for player in PLAYERS:
    for time_control in ("rapid", "blitz"):
      path = DATA_DIR / f"{player}-{time_control}.pgn"
      no_clock_score, rating, games = score_file(scorer, path, player, False)
      clock_score, _, _ = score_file(scorer, path, player, True)
      no_clock_elo = estimate_elo(no_clock_score, calibration)["estimated_elo"]
      clock_elo = estimate_elo(clock_score, calibration)["estimated_elo"]
      print(json.dumps({
        "player": player,
        "time_control": time_control,
        "games": games,
        "rating": rating,
        "no_clock_elo": no_clock_elo,
        "clock_elo": clock_elo,
        "clock_change": clock_elo - no_clock_elo,
      }), flush=True)


if __name__ == "__main__":
  main()
