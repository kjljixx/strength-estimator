"""Compare rapid and blitz estimates for the same players and dates."""

import csv
import json
import statistics
import sys
from pathlib import Path

import chess.pgn

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/tmp/interpretable-overall")
from overall import estimate_elo, score_player
from build.chess import strength_py


PLAYERS = (
  "caolinita",
  "CellblockD",
  "Fins",
  "fiodor_nabokov",
  "getting_there",
  "Judd0104",
  "TRG80",
)
DATA_DIR = Path("/tmp/matched-pgn")
OUTPUT_DIR = Path("/tmp/time-control-test-fixed")


def read_games(path):
  games = []
  with path.open(encoding="utf-8", errors="replace") as stream:
    while game := chess.pgn.read_game(stream):
      games.append(game)
  return games


def game_date(game):
  return game.headers.get("UTCDate", game.headers.get("Date", "")).replace(".", "-")


def write_window(games, start, end, path):
  selected = [game for game in games if start <= game_date(game) <= end]
  with path.open("w", encoding="utf-8") as stream:
    exporter = chess.pgn.FileExporter(stream)
    for game in selected:
      game.accept(exporter)
  return selected


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
  print(f"Config: players={len(PLAYERS)}, data={DATA_DIR}, output={OUTPUT_DIR}", flush=True)
  for player in PLAYERS:
    rapid = read_games(DATA_DIR / f"{player}-rapid-500.pgn")
    blitz = read_games(DATA_DIR / f"{player}-blitz-500.pgn")
    start = max(min(map(game_date, rapid)), min(map(game_date, blitz)))
    end = min(max(map(game_date, rapid)), max(map(game_date, blitz)))
    if start > end:
      print(f"Skip {player}: no overlapping dates", flush=True)
      continue
    for time_control, games in (("rapid", rapid), ("blitz", blitz)):
      path = OUTPUT_DIR / f"{player}-{time_control}.pgn"
      selected = write_window(games, start, end, path)
      result = score_player(scorer, path, player)
      estimate = estimate_elo(float(result["mean_score"]), calibration)
      row = {
        "player": player,
        "time_control": time_control,
        "start": start,
        "end": end,
        **result,
        **estimate,
      }
      row["error"] = float(row["estimated_elo"]) - float(row["median_recorded_rating"])
      rows.append(row)
      print(json.dumps(row), flush=True)
  with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
  deltas = []
  for player in PLAYERS:
    pair = {row["time_control"]: row for row in rows if row["player"] == player}
    if len(pair) == 2:
      deltas.append(pair["rapid"]["error"] - pair["blitz"]["error"])
  print(json.dumps({
    "paired_players": len(deltas),
    "mean_rapid_minus_blitz_error": statistics.mean(deltas),
    "median_rapid_minus_blitz_error": statistics.median(deltas),
  }), flush=True)


if __name__ == "__main__":
  main()
