#!/usr/bin/env python3
"""Score every prefix of one chess SGF game."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prediction_testing.model import StrengthDifferenceModel
from prediction_testing.sgf import game_prefix, move_count


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("sgf", type=Path)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--gpu-id", type=int, default=0)
  args = parser.parse_args()

  from build.chess import strength_py

  sgf = args.sgf.read_text(encoding="utf-8")
  scorer = strength_py.StrengthScorer(
    str(args.config),
    str(args.checkpoint),
    args.gpu_id,
  )
  if strength_py.get_bt_use_weight():
    raise ValueError("instantaneous scoring requires bt_use_weight=false")
  rows = []
  total_plies = move_count(sgf)
  previous_averages = {"white": 0.0, "black": 0.0}
  previous_counts = {"white": 0, "black": 0}
  latest_scores = {"white": 0.0, "black": 0.0}
  print(f"Scoring configuration: gpu_id={args.gpu_id}, total_plies={total_plies}")
  for observed_plies in range(2, total_plies + 1):
    strengths = scorer.score_sgf(game_prefix(sgf, observed_plies))
    counts = {
      "white": (observed_plies + 1) // 2,
      "black": observed_plies // 2,
    }
    for side in ("white", "black"):
      if counts[side] > previous_counts[side]:
        latest_scores[side] = (
          counts[side] * strengths[side]
          - previous_counts[side] * previous_averages[side]
        )
      previous_averages[side] = strengths[side]
      previous_counts[side] = counts[side]
    elo_difference = (
      strengths["white"] - strengths["black"]
    ) * StrengthDifferenceModel.default_score_to_elo_slope
    instantaneous_elo_difference = (
      latest_scores["white"] - latest_scores["black"]
    ) * StrengthDifferenceModel.default_score_to_elo_slope
    rows.append({
      "observed_plies": observed_plies,
      "observed_full_moves": observed_plies / 2,
      "white_current_strength": strengths["white"],
      "black_current_strength": strengths["black"],
      "white_current_game_elo_difference": elo_difference,
      "white_instantaneous_strength": latest_scores["white"],
      "black_instantaneous_strength": latest_scores["black"],
      "white_instantaneous_elo_difference": instantaneous_elo_difference,
    })
    if observed_plies % 10 == 0 or observed_plies == total_plies:
      print(f"Scored {observed_plies}/{total_plies} plies")

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
  print(f"Wrote {len(rows)} rows to {args.output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
