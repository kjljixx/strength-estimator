"""Run a rating-selected cohort using the existing overall evaluator."""

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/tmp/interpretable-overall")
from overall import score_player, estimate_elo
from build.chess import strength_py

source = Path("/workspace/training_sgf_chess_chain/games.txt")
output = Path("/tmp/interpretable-higher-cohort")
output.mkdir(exist_ok=False)
print(f"Config: source={source}, output={output}, median_rating=[1500,2100), players=8", flush=True)
ratings = defaultdict(list)
pattern = re.compile(r"(?:PW|PB|WR|BR)\[([^]]*)\]")
with source.open() as stream:
  for line in stream:
    fields = pattern.findall(line.split(";B[", 1)[0])
    if len(fields) == 4:
      white, black, white_rating, black_rating = fields
      if white_rating.isdigit() and black_rating.isdigit():
        ratings[white].append(int(white_rating))
        ratings[black].append(int(black_rating))
eligible = [name for name, values in ratings.items() if 1500 <= statistics.median(values) < 2100]
players = sorted(eligible, key=lambda name: (-len(ratings[name]), name))[:8]
print("Selected: " + json.dumps({name: [len(ratings[name]), statistics.median(ratings[name])] for name in players}), flush=True)
cohort = output / "games.txt"
with source.open() as stream, cohort.open("w") as target:
  for line in stream:
    if any(f"PW[{name}]" in line or f"PB[{name}]" in line for name in players):
      target.write(line)
model = "/workspace/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted"
scorer = strength_py.StrengthScorer(f"{model}/chess_bt_b32_r8_p7_20bx256-7e7ac9unpolluted.cfg", f"{model}/model/weight_iter_159500.pt", 0)
if strength_py.get_bt_use_weight():
  raise ValueError("requires bt_use_weight=false")
calibration = json.loads(Path("/tmp/interpretable-overall/candidate_strengths.json").read_text())
with (output / "results.csv").open("w", newline="") as stream:
  writer = None
  for player in players:
    result = {"player": player, **score_player(scorer, cohort, player)}
    result.update(estimate_elo(result["mean_score"], calibration))
    result["error"] = result["estimated_elo"] - result["median_recorded_rating"]
    if writer is None:
      writer = csv.DictWriter(stream, fieldnames=result)
      writer.writeheader()
    writer.writerow(result)
    stream.flush()
    print(json.dumps(result), flush=True)
