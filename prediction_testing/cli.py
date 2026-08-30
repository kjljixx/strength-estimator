from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_testing.data_filter import PredictionDataFilter
from prediction_testing.evaluator import PredictionEvaluator
from prediction_testing.model import EloBaselineModel
from prediction_testing.schemas import ContextPolicy, EvaluationConfig


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run chess outcome prediction evaluation")
  parser.add_argument("game_paths", nargs="+", type=Path, help="SGF game files")
  parser.add_argument("--max-games-to-load", type=int, default=None)
  parser.add_argument("--output-dir", type=Path, default=Path("prediction_output"))
  parser.add_argument("--context-size", type=int, default=8)
  parser.add_argument("--exclude-same-day-context", action="store_true")
  parser.add_argument("--seed", type=int, default=0)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  policy = ContextPolicy(
    context_size=args.context_size,
    exclude_same_day_context=args.exclude_same_day_context,
    seed=args.seed,
  )
  data_filter = PredictionDataFilter()
  catalog = data_filter.load_catalog(args.game_paths, args.max_games_to_load)
  print(f"Loaded {len(catalog.games)} games from {len(args.game_paths)} files")
  dataset = data_filter.build_examples(catalog, policy)
  print(f"Built {len(dataset.examples)} prediction examples with exclusions: {dataset.exclusion_counts}")
  print(dataset.examples[0].prediction_game)

  model = EloBaselineModel()
  result = PredictionEvaluator().run(
    model,
    dataset,
    EvaluationConfig(),
    args.output_dir,
  )
  summary = {
    "accepted_examples": len(dataset.examples),
    "accuracy": result.metric_report.accuracy,
    "log_loss": result.metric_report.log_loss,
  }
  print(json.dumps(summary, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
