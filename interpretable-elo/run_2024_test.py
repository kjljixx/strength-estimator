"""Score January 2024 blitz histories, with and without corpus filters."""

import csv
import json
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/tmp/interpretable-overall")
from overall import estimate_elo, score_player
from build.chess import strength_py

PLAYERS = ("caolinita", "CellblockD", "Fins", "fiodor_nabokov", "getting_there", "TRG80")
DATA_DIR = Path("/tmp/blitz-2024-01")
OUTPUT_DIR = Path("/tmp/2024-test-fixed")


def corpus_eligible(game):
  white = game.headers.get("WhiteElo", "")
  black = game.headers.get("BlackElo", "")
  if not white.isdigit() or not black.isdigit():
    return False
  white, black = int(white), int(black)
  return (
    game.board().fen() == chess.Board().fen()
    and game.end().ply() >= 22
    and 1000 <= white < 2600
    and 1000 <= black < 2600
    and (white - 1000) // 200 == (black - 1000) // 200
  )


def write_filtered(source, target):
  kept = 0
  with source.open(encoding="utf-8", errors="replace") as input_stream, target.open("w", encoding="utf-8") as output_stream:
    exporter = chess.pgn.FileExporter(output_stream)
    while game := chess.pgn.read_game(input_stream):
      if corpus_eligible(game):
        game.accept(exporter)
        kept += 1
  return kept


def main():
  OUTPUT_DIR.mkdir(exist_ok=False)
  model = "/workspace/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted"
  scorer = strength_py.StrengthScorer(
    f"{model}/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted.cfg",
    f"{model}/model/weight_iter_159500.pt",
    0,
  )
  calibration = json.loads(Path("/tmp/interpretable-overall/candidate_strengths.json").read_text())
  rows = []
  print("Config: period=2024-01, event=blitz, players=6", flush=True)
  for player in PLAYERS:
    source = DATA_DIR / f"{player}-blitz-2024-01.pgn"
    filtered = OUTPUT_DIR / f"{player}-filtered.pgn"
    kept = write_filtered(source, filtered)
    for sample, path in (("all", source), ("corpus_filtered", filtered)):
      if sample == "corpus_filtered" and kept == 0:
        continue
      result = score_player(scorer, path, player)
      estimate = estimate_elo(float(result["mean_score"]), calibration)
      row = {"player": player, "sample": sample, **result, **estimate}
      row["error"] = float(row["estimated_elo"]) - float(row["median_recorded_rating"])
      rows.append(row)
      print(json.dumps(row), flush=True)
  with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
  main()
