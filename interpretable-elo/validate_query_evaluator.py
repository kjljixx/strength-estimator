"""Reproduce native BT query evaluation through the Python scorer."""

import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/tmp/interpretable-overall")
from overall import color_move_count
from prediction_testing.sgf import move_count
from build.chess import strength_py

QUERY_DIR = Path("/workspace/query_sgf_chess")
CACHE = Path("/tmp/query-python-scores.csv")
TRIAL_SIZES = (1, 10, 25, 50, 75, 100)


def score_queries(scorer):
  rows = []
  for path in sorted(QUERY_DIR.glob("*.txt")):
    low, high = map(int, path.stem.split("_"))
    with path.open(encoding="utf-8", errors="replace") as stream:
      for index, sgf in enumerate(stream):
        plies = move_count(sgf)
        scores = scorer.score_sgf(sgf)
        rows.append({
          "low": low,
          "high": high,
          "index": index,
          "white_score": float(scores["white"]),
          "white_moves": color_move_count(plies, "white"),
          "black_score": float(scores["black"]),
          "black_moves": color_move_count(plies, "black"),
        })
        if len(rows) % 500 == 0:
          print(f"Scored {len(rows)} query games", flush=True)
  with CACHE.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
  return rows


def main():
  model = "/workspace/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted"
  scorer = strength_py.StrengthScorer(
    f"{model}/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted.cfg",
    f"{model}/model/weight_iter_159500.pt",
    0,
  )
  calibration = json.loads(Path("/tmp/interpretable-overall/candidate_strengths.json").read_text())
  candidates = {int(row["low"]): float(row["mean_score"]) for row in calibration}
  rows = score_queries(scorer)
  by_bucket = {}
  for row in rows:
    by_bucket.setdefault(row["low"], []).append(row)
  for low, games in sorted(by_bucket.items()):
    total = sum(
      game["white_score"] * game["white_moves"] + game["black_score"] * game["black_moves"]
      for game in games
    )
    moves = sum(game["white_moves"] + game["black_moves"] for game in games)
    mean = total / moves
    nearest = min(candidates, key=lambda bucket: abs(mean - candidates[bucket]))
    print(json.dumps({
      "bucket": f"{low}-{low + 200}",
      "games": len(games),
      "query_mean": mean,
      "candidate_mean": candidates[low],
      "nearest_bucket": f"{nearest}-{nearest + 200}",
    }), flush=True)
  rng = random.Random(0)
  for count in TRIAL_SIZES:
    bucket_accuracies = []
    for low, games in sorted(by_bucket.items()):
      correct = 0
      for _ in range(500):
        total = moves = 0
        for _ in range(count):
          game = rng.choice(games)
          color = rng.choice(("white", "black"))
          score = game[f"{color}_score"]
          weight = game[f"{color}_moves"]
          total += score * weight
          moves += weight
        predicted = min(candidates, key=lambda bucket: abs(total / moves - candidates[bucket]))
        correct += predicted == low
      bucket_accuracies.append(correct / 500)
    print(json.dumps({
      "games_per_trial": count,
      "accuracy": sum(bucket_accuracies) / len(bucket_accuracies),
      "by_bucket": bucket_accuracies,
    }), flush=True)


if __name__ == "__main__":
  main()
