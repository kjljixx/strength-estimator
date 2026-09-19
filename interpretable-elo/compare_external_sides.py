"""Compare target, opponent, and pooled scores in filtered external games."""

import json
import sys
from pathlib import Path

import chess.pgn

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/tmp/interpretable-overall")
from analyze import game_to_sgf, player_color
from overall import color_move_count, estimate_elo, weighted_mean
from build.chess import strength_py

PLAYERS = ("caolinita", "CellblockD", "Fins", "fiodor_nabokov", "getting_there", "TRG80")
DATA_DIR = Path("/tmp/2024-test")


def main():
  model = "/workspace/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted"
  scorer = strength_py.StrengthScorer(
    f"{model}/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted.cfg",
    f"{model}/model/weight_iter_159500.pt",
    0,
  )
  calibration = json.loads(Path("/tmp/interpretable-overall/candidate_strengths.json").read_text())
  for player in PLAYERS:
    scores = {"target": [], "opponent": [], "pooled": []}
    with (DATA_DIR / f"{player}-filtered.pgn").open(encoding="utf-8", errors="replace") as stream:
      while game := chess.pgn.read_game(stream):
        target = player_color(game, player)
        if target is None:
          continue
        opponent = "black" if target == "white" else "white"
        sgf, positions = game_to_sgf(game)
        result = scorer.score_sgf(sgf)
        for color, role in ((target, "target"), (opponent, "opponent")):
          weight = color_move_count(len(positions), color)
          item = (float(result[color]), weight)
          scores[role].append(item)
          scores["pooled"].append(item)
    estimates = {
      role: estimate_elo(weighted_mean(values), calibration)["estimated_elo"]
      for role, values in scores.items()
    }
    print(json.dumps({"player": player, "games": len(scores["target"]), **estimates}), flush=True)


if __name__ == "__main__":
  main()
