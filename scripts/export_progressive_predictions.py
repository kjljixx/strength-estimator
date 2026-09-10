#!/usr/bin/env python3
"""Export decisive-game progressive strength predictions as CSV."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prediction_testing.data_filter import PredictionDataFilter
from prediction_testing.model import StrengthDifferenceModel
from prediction_testing.schemas import ContextPolicy, GameResult
from prediction_testing.sgf import move_count


CSV_FIELDS = (
  "example_id",
  "game_id",
  "observed_plies",
  "observed_full_moves",
  "game_length_plies",
  "game_progress",
  "actual_result",
  "winner_color",
  "is_recorded_endpoint",
  "white_context_strength",
  "black_context_strength",
  "white_current_strength",
  "black_current_strength",
  "white_blended_strength",
  "black_blended_strength",
  "white_estimated_elo",
  "black_estimated_elo",
  "white_elo_difference",
  "context_only_winner_elo_difference",
  "winner_elo_difference",
  "white_win_probability",
  "draw_probability",
  "black_win_probability",
  "winner_probability",
  "white_current_game_fraction",
  "black_current_game_fraction",
  "current_move_weight",
  "typical_moves_per_player_per_game",
  "legacy_calibration",
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Export progressive strength predictions for decisive chess games",
  )
  parser.add_argument("game_paths", nargs="+", type=Path, help="SGF game files")
  parser.add_argument("--strength-config", type=Path, required=True)
  parser.add_argument("--strength-checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, default=Path("progressive_predictions.csv"))
  parser.add_argument("--gpu-id", type=int, default=0)
  parser.add_argument("--max-games-to-load", type=int)
  parser.add_argument("--context-size", type=int, default=8)
  parser.add_argument("--max-context-age-days", type=int)
  parser.add_argument("--exclude-same-day-context", action="store_true")
  parser.add_argument("--context-last-n-moves", type=int)
  parser.add_argument(
    "--legacy",
    action="store_true",
    help="Use the strength-to-Elo calibration introduced in commit 170eeb5",
  )
  parser.add_argument("--typical-moves-per-player-per-game", type=float, default=40)
  parser.add_argument("--current-move-weight", type=float, default=8)
  parser.add_argument("--seed", type=int, default=0)
  return parser


def validate_args(args: argparse.Namespace) -> None:
  if args.context_size <= 0:
    raise ValueError("--context-size must be positive")
  if args.max_context_age_days is not None and args.max_context_age_days < 0:
    raise ValueError("--max-context-age-days must be non-negative")
  if args.context_last_n_moves is not None and args.context_last_n_moves <= 0:
    raise ValueError("--context-last-n-moves must be positive")
  if args.typical_moves_per_player_per_game <= 0:
    raise ValueError("--typical-moves-per-player-per-game must be positive")
  if not math.isfinite(args.current_move_weight) or args.current_move_weight < 0:
    raise ValueError("--current-move-weight must be finite and non-negative")


def optional(metadata: dict, key: str) -> float | str:
  return metadata.get(key, "")


def trajectory_row(
  example,
  trajectory,
  baseline,
  game_length: int,
  args,
) -> dict[str, object]:
  prediction = trajectory.prediction
  metadata = dict(prediction.metadata)
  baseline_metadata = dict(baseline.prediction.metadata)
  winner_color = (
    "white" if example.prediction_game.result == GameResult.WHITE_WIN else "black"
  )
  direction = 1 if winner_color == "white" else -1
  white_elo_difference = (
    metadata["white_estimated_elo"] - metadata["black_estimated_elo"]
  )
  baseline_white_difference = (
    baseline_metadata["white_estimated_elo"]
    - baseline_metadata["black_estimated_elo"]
  )
  winner_probability = (
    prediction.white_win_probability
    if winner_color == "white"
    else prediction.black_win_probability
  )
  return {
    "example_id": example.example_id,
    "game_id": example.prediction_game.game_id,
    "observed_plies": trajectory.observed_plies,
    "observed_full_moves": trajectory.observed_plies / 2,
    "game_length_plies": game_length,
    "game_progress": trajectory.observed_plies / game_length if game_length else 0,
    "actual_result": example.prediction_game.result.name,
    "winner_color": winner_color,
    "is_recorded_endpoint": trajectory.is_recorded_endpoint,
    "white_context_strength": optional(metadata, "white_context_strength")
      or baseline_metadata["white_strength"],
    "black_context_strength": optional(metadata, "black_context_strength")
      or baseline_metadata["black_strength"],
    "white_current_strength": optional(metadata, "white_current_strength"),
    "black_current_strength": optional(metadata, "black_current_strength"),
    "white_blended_strength": metadata["white_strength"],
    "black_blended_strength": metadata["black_strength"],
    "white_estimated_elo": metadata["white_estimated_elo"],
    "black_estimated_elo": metadata["black_estimated_elo"],
    "white_elo_difference": white_elo_difference,
    "context_only_winner_elo_difference": direction * baseline_white_difference,
    "winner_elo_difference": direction * white_elo_difference,
    "white_win_probability": prediction.white_win_probability,
    "draw_probability": prediction.draw_probability,
    "black_win_probability": prediction.black_win_probability,
    "winner_probability": winner_probability,
    "white_current_game_fraction": optional(metadata, "white_current_game_fraction") or 0,
    "black_current_game_fraction": optional(metadata, "black_current_game_fraction") or 0,
    "current_move_weight": args.current_move_weight,
    "typical_moves_per_player_per_game": args.typical_moves_per_player_per_game,
    "legacy_calibration": metadata["legacy_calibration"],
  }


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  validate_args(args)
  print(
    "Export configuration: "
    f"context_size={args.context_size}, current_move_weight={args.current_move_weight}, "
    f"typical_moves_per_player_per_game={args.typical_moves_per_player_per_game}, "
    f"legacy_calibration={args.legacy}"
  )

  data_filter = PredictionDataFilter()
  catalog = data_filter.load_catalog(args.game_paths, args.max_games_to_load)
  dataset = data_filter.build_examples(catalog, ContextPolicy(
    context_size=args.context_size,
    max_context_age_days=args.max_context_age_days,
    exclude_same_day_context=args.exclude_same_day_context,
    seed=args.seed,
  ))
  from build.chess import strength_py

  scorer = strength_py.StrengthScorer(
    str(args.strength_config),
    str(args.strength_checkpoint),
    args.gpu_id,
  )
  model = StrengthDifferenceModel(
    scorer.score_sgf,
    catalog.load_sgf_by_id,
    context_last_n_moves=args.context_last_n_moves,
    legacy=args.legacy,
  )

  rows: list[dict[str, object]] = []
  decisive_examples = 0
  for index, example in enumerate(dataset.examples, start=1):
    if example.prediction_game.result == GameResult.DRAW:
      continue
    decisive_examples += 1
    trajectory = model.predict_progressive(
      example,
      typical_moves_per_player_per_game=args.typical_moves_per_player_per_game,
      current_move_weight=args.current_move_weight,
    )
    game_length = move_count(catalog.load_sgf_by_id(example.prediction_game.game_id))
    baseline = trajectory[0]
    for point in trajectory:
      rows.append(trajectory_row(example, point, baseline, game_length, args))
    if index % 100 == 0:
      print(f"Processed {index}/{len(dataset.examples)} prediction examples")

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
  print(
    f"Wrote {len(rows)} rows from {decisive_examples} decisive games to {args.output}; "
    f"excluded {len(dataset.examples) - decisive_examples} draws"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
