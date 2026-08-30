from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Mapping


class GameResult(Enum):
  WHITE_WIN = "white_win"
  DRAW = "draw"
  BLACK_WIN = "black_win"


@dataclass(frozen=True)
class GameRecord:
  game_id: str
  played_at: datetime
  white_player: str
  black_player: str
  white_elo: int
  black_elo: int
  result: GameResult
  event: str = "Blitz"


@dataclass(frozen=True)
class ContextGame:
  game_id: str
  target_color: Literal["white", "black"]
  target_elo_at_game: int
  opponent_elo_at_game: int
  result_from_target_perspective: Literal["win", "draw", "loss"]
  played_at: datetime


@dataclass(frozen=True)
class PlayerContext:
  player_id: str
  games: tuple[ContextGame, ...]
  anchor_time: datetime


@dataclass(frozen=True)
class PredictionExample:
  example_id: str
  prediction_game: GameRecord
  white_context: PlayerContext
  black_context: PlayerContext


@dataclass(frozen=True)
class GamePrediction:
  example_id: str
  white_win_probability: float
  draw_probability: float
  black_win_probability: float
  predicted_result: GameResult
  metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextPolicy:
  context_size: int
  max_context_age_days: int | None = None
  min_context_age_seconds: int = 1
  allowed_events: tuple[str, ...] = ("Blitz",)
  min_elo: int = 1000
  max_elo: int = 3000
  require_both_ratings: bool = True
  exclude_same_day_context: bool = False
  require_distinct_context_games: bool = True
  max_prediction_games_per_player: int | None = None
  seed: int = 0


@dataclass(frozen=True)
class ExampleManifest:
  example_id: str
  prediction_game_id: str
  white_context_ids: tuple[str, ...]
  black_context_ids: tuple[str, ...]


@dataclass(frozen=True)
class FilteredPredictionDataset:
  catalog: "GameCatalog"
  examples: tuple[PredictionExample, ...]
  manifests: tuple[ExampleManifest, ...]
  exclusion_counts: Mapping[str, int]


@dataclass(frozen=True)
class EvaluationConfig:
  minimum_slice_size: int = 20
  report_by_context_size: bool = True
  report_by_elo: bool = True
  report_by_elo_difference: bool = True
  report_by_month: bool = True
  report_by_history_age: bool = True


@dataclass(frozen=True)
class MetricReport:
  n_examples: int
  accuracy: float
  log_loss: float
  brier_score: float
  slices: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
  model_name: str
  metric_report: MetricReport
  predictions: tuple[GamePrediction, ...]
