"""Apply calibration-corpus eligibility rules to fresh blitz histories."""

import json
import statistics
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/tmp/interpretable-overall")
from overall import estimate_elo, player_color, score_player
from build.chess import strength_py

PLAYERS = ("caolinita", "CellblockD", "Fins", "fiodor_nabokov", "getting_there", "TRG80")
SOURCE_DIR = Path("/tmp/time-control-test")
OUTPUT_DIR = Path("/tmp/corpus-filter-test")


def rating(game, color):
  value = game.headers.get("WhiteElo" if color == chess.WHITE else "BlackElo", "")
  return int(value) if value.isdigit() else None


def eligible(game):
  white, black = rating(game, chess.WHITE), rating(game, chess.BLACK)
  return (
    game.board().fen() == chess.Board().fen()
    and game.end().ply() >= 22
    and white is not None
    and black is not None
    and 1000 <= white < 2600
    and 1000 <= black < 2600
    and (white - 1000) // 200 == (black - 1000) // 200
  )


def main():
  OUTPUT_DIR.mkdir(exist_ok=False)
  model = "/workspace/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted"
  scorer = strength_py.StrengthScorer(
    f"{model}/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted.cfg",
    f"{model}/model/weight_iter_159500.pt",
    0,
  )
  calibration = json.loads(Path("/tmp/interpretable-overall/candidate_strengths.json").read_text())
  print("Config: fresh blitz, same-bin=[1000,2600), minimum_plies=22", flush=True)
  changes = []
  for player in PLAYERS:
    source = SOURCE_DIR / f"{player}-blitz.pgn"
    target = OUTPUT_DIR / f"{player}-filtered.pgn"
    total = kept = 0
    gaps = []
    with source.open(encoding="utf-8", errors="replace") as input_stream, target.open("w", encoding="utf-8") as output_stream:
      exporter = chess.pgn.FileExporter(output_stream)
      while game := chess.pgn.read_game(input_stream):
        color = player_color(game, player)
        if color is None:
          continue
        total += 1
        player_rating = rating(game, chess.WHITE if color == "white" else chess.BLACK)
        opponent_rating = rating(game, chess.BLACK if color == "white" else chess.WHITE)
        if player_rating is not None and opponent_rating is not None:
          gaps.append(abs(player_rating - opponent_rating))
        if eligible(game):
          game.accept(exporter)
          kept += 1
    baseline = score_player(scorer, source, player)
    filtered = score_player(scorer, target, player)
    baseline_elo = estimate_elo(float(baseline["mean_score"]), calibration)["estimated_elo"]
    filtered_elo = estimate_elo(float(filtered["mean_score"]), calibration)["estimated_elo"]
    change = filtered_elo - baseline_elo
    changes.append(change)
    print(json.dumps({
      "player": player,
      "total_games": total,
      "kept_games": kept,
      "kept_fraction": kept / total,
      "median_absolute_rating_gap": statistics.median(gaps),
      "baseline_elo": baseline_elo,
      "filtered_elo": filtered_elo,
      "filter_change": change,
    }), flush=True)
  print(json.dumps({
    "mean_filter_change": statistics.mean(changes),
    "median_filter_change": statistics.median(changes),
  }), flush=True)


if __name__ == "__main__":
  main()
